import gc
import tqdm
import transformers
import numpy as np
import torch
import peft
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

from .utils import 

def _get_lora_target_modules(model):
    """
    Inspects the model's named modules to find attention projection layers,
    rather than guessing target modules from the model name string.
    """
    module_names = {name.split('.')[-1] for name, _ in model.named_modules()}
    candidates = [
        ("query_proj", "value_proj"),   # DeBERTa
        ("q_proj", "v_proj"),           # BART, Llama-style
        ("query", "value"),             # BERT/RoBERTa
    ]
    for q, v in candidates:
        if q in module_names and v in module_names:
            return [q, v]
    raise ValueError(
        f"Could not auto-detect LoRA target modules for {model.config.model_type}. "
        f"Available module names: {sorted(module_names)}"
    )


def _compute_class_weights(train_dataloader, num_labels, device):
    """
    Computes balanced class weights from the training set's label distribution,
    to counteract the model collapsing onto majority classes.
    """
    all_labels = []
    for batch in train_dataloader:
        all_labels.extend(batch["labels"].numpy())
    all_labels = np.array(all_labels)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_labels),
        y=all_labels,
    )
    print(f"Class weights (balanced): {dict(zip(range(num_labels), class_weights.round(3)))}")
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def transformer_train(train_dataloader, eval_dataloader,
                      model_source="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli",
                      output_dir="", num_epochs=3, lr=2e-5, use_lora=True,
                      use_class_weights=True, num_labels=3):
    """
    Fine-tunes an encoder-only model for sequence classification.

    train_dataloader:   DataLoader of training data
    eval_dataloader:    DataLoader of eval data
    model_source:       HuggingFace model checkpoint
    output_dir:         path to save model (empty string skips saving)
    num_epochs:         number of training epochs
    lr:                 learning rate (2e-5 is typical for encoder fine-tuning)
    use_lora:           whether to apply LoRA (recommended for large models)
    use_class_weights:  if True, weights the loss by inverse class frequency to
                        counter majority-class collapse on imbalanced data
    num_labels:         number of classification labels

    Returns the model checkpoint with the best eval macro-F1 across all epochs
    (not necessarily the final epoch).
    """

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # NLI head -> N-class stance head
    ).to("cuda")

    if use_lora:
        target_modules = _get_lora_target_modules(model)
        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=target_modules,
            modules_to_save=["classifier"],  # fully fine-tune the (often freshly-initialized) classifier head,
                                              # not just LoRA-adapt it — critical for models without NLI pretraining
                                              # where the head starts from random weights
            bias="none",
            task_type=peft.TaskType.SEQ_CLS,
        )
        model = peft.get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    device = next(model.parameters()).device

    class_weights = None
    if use_class_weights:
        class_weights = _compute_class_weights(train_dataloader, num_labels, device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = loss_fct(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            del outputs, loss, labels, batch

        avg_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        eval_preds, eval_labels = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [eval]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                eval_preds.extend(preds.cpu().numpy())
                eval_labels.extend(labels.cpu().numpy())
                del outputs, preds, labels, batch

        macro_f1 = metrics.f1_score(eval_labels, eval_preds, average="macro")
        accuracy = metrics.accuracy_score(eval_labels, eval_preds)
        per_class_f1 = metrics.f1_score(eval_labels, eval_preds, average=None)
        print(f"Epoch {epoch+1} | loss: {avg_loss:.4f} "
              f"| eval accuracy: {accuracy:.4f} | eval macro-F1: {macro_f1:.4f}")
        print(f"  per-class F1: {dict(zip(range(num_labels), per_class_f1.round(3)))}")

    if output_dir:
        model.save_pretrained(output_dir)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    # Optimizer/scheduler hold references to model parameters; drop them
    # explicitly before returning so they don't keep gradients/state alive
    del optimizer, scheduler, loss_fct
    if class_weights is not None:
        del class_weights
    gc.collect()
    torch.cuda.empty_cache()

    return model


def transformer_test(model, test_dataloader):
    """
    Evaluates a fine-tuned encoder model on test data.

    model: fine-tuned model
    test_dataloader: DataLoader of test data
    """

    model.eval()
    device = next(model.parameters()).device
    all_preds = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds