import copy
import tqdm
import transformers
import numpy as np
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import peft
import torch

def _zero_shot_predict(test, model, tokenizer):
  #Takes untrained model and outputs predictions
    y_pred = []
    categories = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]
    answers = []
    for i in tqdm.tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = transformers.pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=5,
                        temperature=0.1)

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("classification:")[-1].strip()
        answers.append(answer)
        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")
    test['Output'] = answers
    test['Predicted'] = y_pred
    return y_pred

def zero_shot_test(x_test, model_source="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Loads untrained model and tests it on test data.

    x_test: Test data (not pre-processed)
    model_source: huggingface model to use
    returns: y_pred (predicted values)
    """

    base_model_name = model_source

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="float16",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    y_pred = _zero_shot_predict(x_test, model, tokenizer)
    
    labels = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, untrained model more likely to not label
    y_pred_mapped = np.vectorize(map_func)(y_pred)
    return y_pred_mapped

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


def fine_tune_train(train_dataloader, eval_dataloader, 
                    model_source="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                    output_dir="", num_epochs=3, lr=3e-4,
                    use_class_weights=False, num_labels=3):
    """
    Fine-tunes a decoder LLM for sequence classification via LoRA.

    train_dataloader:   DataLoader of training data
    eval_dataloader:    DataLoader of eval data
    model_source:       HuggingFace model checkpoint
    output_dir:         path to save model (empty string skips saving)
    num_epochs:         number of training epochs
    lr:                 learning rate
    use_class_weights:  if True, weights the loss by inverse class frequency to
                        counter majority-class collapse on imbalanced data.
                        Default False since LLM fine-tuning hasn't shown this
                        problem as strongly as encoder fine-tuning — check
                        per-class F1 below before enabling.
    num_labels:         number of classification labels

    Returns the model checkpoint with the best eval macro-F1 across all epochs
    (not necessarily the final epoch).
    """

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=num_labels,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA config — no llama_cookbook needed
    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=peft.TaskType.SEQ_CLS,
    )

    model = peft.get_peft_model(model, lora_config)
    model.train()

    device = next(model.parameters()).device

    class_weights = None
    if use_class_weights:
        class_weights = _compute_class_weights(train_dataloader, num_labels, device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = loss_fct(outputs.logits, labels)

            loss.backward()
            if (step + 1) % 4 == 0:  # gradient accumulation
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        # cleaning up last batches in case number isn't divisible by four
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        print(f"Epoch {epoch+1} loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        model.eval()
        eval_preds, eval_labels = [], []
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                eval_preds.extend(preds.cpu().numpy())
                eval_labels.extend(labels.cpu().numpy())

        acc = metrics.accuracy_score(eval_labels, eval_preds)
        macro_f1 = metrics.f1_score(eval_labels, eval_preds, average="macro")
        per_class_f1 = metrics.f1_score(eval_labels, eval_preds, average=None)
        print(f"Epoch {epoch+1} eval accuracy: {acc:.4f} | eval macro-F1: {macro_f1:.4f}")
        print(f"  per-class F1: {dict(zip(range(num_labels), per_class_f1.round(3)))}")
        model.train()

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ^ new best macro-F1 ({best_macro_f1:.4f}), checkpoint saved in memory")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint with eval macro-F1: {best_macro_f1:.4f}")

    if output_dir:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return model

def fine_tune_test(model, test_dataloader):
    """_summary_

    Args:
        model (_type_): _description_
        test_dataloader (_type_): _description_
        test_labels (_type_): _description_

    Returns:
        list: produced predictions
    """
    model.eval()
    device = next(model.parameters()).device

    all_preds = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            for i in range(batch["input_ids"].size(0)):  # iterate over batch dimension
                single_batch = {k: v[i].unsqueeze(0) for k, v in batch.items()}
                outputs = model(**single_batch)
                pred = torch.argmax(outputs.logits, dim=-1)
                all_preds.append(pred.item())
    return all_preds