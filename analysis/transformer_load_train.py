import tqdm
import transformers
import numpy as np
import torch
import peft
from sklearn import metrics


def transformer_train(train_dataloader, eval_dataloader,
                      model_source="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli",
                      output_dir="", num_epochs=3, lr=2e-5, use_lora=True):
    """
    Fine-tunes an encoder-only model for sequence classification.

    train_dataloader: DataLoader of training data
    eval_dataloader: DataLoader of eval data
    model_source: HuggingFace model checkpoint
    output_dir: path to save model (empty string skips saving)
    num_epochs: number of training epochs
    lr: learning rate (2e-5 is typical for encoder fine-tuning)
    use_lora: whether to apply LoRA (recommended for large models)
    """

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=3,
        ignore_mismatched_sizes=True,  # NLI head -> 3-class stance head
    ).to("cuda")

    if use_lora:
        if "DeBERTa" in model_source:
            target_modules = ["query_proj", "value_proj"]
        else: # RoBERTa/BERT based
            target_modules = ["query", "value"]

        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=target_modules,  # DeBERTa naming
            bias="none",
            task_type=peft.TaskType.SEQ_CLS,
        )
        model = peft.get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps,
    )

    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        eval_preds, eval_labels = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [eval]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                eval_preds.extend(preds.cpu().numpy())
                eval_labels.extend(batch["labels"].cpu().numpy())

        macro_f1 = metrics.f1_score(eval_labels, eval_preds, average="macro")
        accuracy = metrics.accuracy_score(eval_labels, eval_preds)
        print(f"Epoch {epoch+1} | loss: {avg_loss:.4f} "
              f"| eval accuracy: {accuracy:.4f} | eval macro-F1: {macro_f1:.4f}")

    if output_dir:
        model.save_pretrained(output_dir)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

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
