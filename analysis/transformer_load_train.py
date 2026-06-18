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
        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=["query_proj", "value_proj"],  # DeBERTa naming
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
        print(f"Epoch {epoch+1} | loss: {avg_loss:.4f} | eval accuracy: {accuracy:.4f} | eval macro-F1: {macro_f1:.4f}")

    if output_dir:
        model.save_pretrained(output_dir)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    return model


def transformer_test(model, test_dataloader, test_labels):
    """
    Evaluates a fine-tuned encoder model on test data.

    model: fine-tuned model
    test_dataloader: DataLoader of test data
    test_labels: ground truth integer labels
    """
    labels = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]

    model.eval()
    device = next(model.parameters()).device
    all_preds = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    accuracy = metrics.accuracy_score(test_labels, all_preds)
    macro_f1 = metrics.f1_score(test_labels, all_preds, average="macro")
    print(f"Test accuracy: {accuracy:.4f} | Test macro-F1: {macro_f1:.4f}")

    class_report = metrics.classification_report(
        test_labels, all_preds, target_names=labels, labels=list(range(len(labels)))
    )
    print("\nClassification Report:")
    print(class_report)

    conf_matrix = metrics.confusion_matrix(test_labels, all_preds, labels=list(range(len(labels))))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    return all_preds


def transformer_predict(model, prediction_dataloader, df, output_path="labeled_examples.csv"):
    """
    Runs inference on unlabeled data and saves results.

    model: fine-tuned model
    prediction_dataloader: DataLoader of unlabeled data
    df: original dataframe (for saving alongside predictions)
    output_path: path to save CSV output
    """
    labels = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]

    model.eval()
    device = next(model.parameters()).device
    all_preds = []

    with torch.no_grad():
        for batch in tqdm.tqdm(prediction_dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

            # Save incrementally in case of interruption
            df_small = df.iloc[:len(all_preds)].copy()
            df_small["label_int"] = all_preds
            df_small["label"] = [labels[p] for p in all_preds]
            df_small.to_csv(output_path, index=False)

    df["label_int"] = all_preds
    df["label"] = [labels[p] for p in all_preds]
    df.to_csv(output_path, index=False)

    return df, all_preds