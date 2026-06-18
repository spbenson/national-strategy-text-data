import tqdm
import transformers
import numpy as np
from sklearn import metrics
import peft
import torch

def zero_shot_predict(test, model, tokenizer):
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

    y_pred = zero_shot_predict(x_test, model, tokenizer)
    return y_pred

def fine_tune_train(train_dataloader, eval_dataloader, 
                    model_source="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                    output_dir="", num_epochs=1, lr=3e-4):

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=3,
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

    model = peft.prepare_model_for_kbit_training(model)
    model = peft.get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            if (step + 1) % 4 == 0:  # gradient accumulation
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        model.eval()
        eval_preds, eval_labels = [], []
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                eval_preds.extend(preds.cpu().numpy())
                eval_labels.extend(batch["labels"].cpu().numpy())
        
        acc = metrics.accuracy_score(eval_labels, eval_preds)
        print(f"Epoch {epoch+1} eval accuracy: {acc:.4f}")
        model.train()

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
                #print(outputs.logits)
                pred = torch.argmax(outputs.logits, dim=-1)
                all_preds.append(pred.item())

    print("Printing Fine-Tuned Model Results:")
    return all_preds

def llm_label_predictions(df,prediction_dataloader,model,output_path="labeled_examples.csv"):
    model.eval()
    device = next(model.parameters()).device

    all_preds = []

    with torch.no_grad():
        for batch in prediction_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            for i in range(batch["input_ids"].size(0)):  # iterate over batch dimension
                single_batch = {k: v[i].unsqueeze(0) for k, v in batch.items()}
                outputs = model(**single_batch)
                pred = torch.argmax(outputs.logits, dim=-1)
                all_preds.append(pred.item())
                df_small = df.iloc[0:len(all_preds)]
                df_small['labels'] = all_preds
                df_small.to_csv(output_path,index=False)
    df['labels'] = all_preds
    df.to_csv(output_path,index=False)
    return df, all_preds