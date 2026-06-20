import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

LABEL_NAMES = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]

# ---------------------------------------------------------------------------
# Shared calibration utilities
# ---------------------------------------------------------------------------

def collect_logits(model, dataloader):
    """
    Runs inference over a dataloader and collects raw logits and true labels.
    Used to fit temperature scaling on an eval set before prediction.

    model:      fine-tuned classification model (encoder or decoder)
    dataloader: eval DataLoader that includes labels

    Returns:
        all_logits: np.ndarray of shape (N, num_classes)
        all_labels: np.ndarray of shape (N,)
    """
    model.eval()
    device = next(model.parameters()).device
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Collecting logits"):
            labels = batch.get("labels")
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ["input_ids", "attention_mask"]}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().float().numpy())
            if labels is not None:
                all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    return all_logits, all_labels


def _nll(T, logits, labels):
    """Negative log likelihood at temperature T."""
    scaled = torch.tensor(logits, dtype=torch.float32) / T
    log_probs = F.log_softmax(scaled, dim=-1)
    nll = F.nll_loss(log_probs, torch.tensor(labels, dtype=torch.long))
    return nll.item()


def fit_temperature(logits, labels):
    """
    Finds the optimal temperature T that minimises NLL on a labelled set.
    T > 1 softens (reduces overconfidence), T < 1 sharpens the distribution.

    logits: np.ndarray of shape (N, num_classes) — raw model logits
    labels: np.ndarray of shape (N,)             — integer true labels

    Returns:
        optimal_T: float
    """
    result = minimize_scalar(
        _nll,
        args=(logits, labels),
        bounds=(0.1, 10.0),
        method='bounded',
    )
    optimal_T = result.x
    print(f"Optimal temperature: {optimal_T:.4f}")
    return optimal_T


def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error — measures gap between confidence and accuracy
    across confidence bins. Lower is better; 0 is perfectly calibrated.

    probs:  np.ndarray of shape (N, num_classes) — softmax probabilities
    labels: np.ndarray of shape (N,)             — integer true labels
    n_bins: number of equal-width confidence bins

    Returns:
        ece: float
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return ece


def calibrate(model, eval_dataloader):
    """
    Convenience wrapper: collects eval logits, fits temperature, and reports
    ECE before and after scaling.

    model:           fine-tuned classification model
    eval_dataloader: eval DataLoader that includes labels

    Returns:
        optimal_T: float — use this when calling predict functions
    """
    logits, labels = collect_logits(model, eval_dataloader)

    # ECE before scaling
    probs_raw = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    ece_before = compute_ece(probs_raw, labels)
    print(f"ECE before temperature scaling: {ece_before:.4f}")

    optimal_T = fit_temperature(logits, labels)

    # ECE after scaling
    probs_scaled = F.softmax(
        torch.tensor(logits, dtype=torch.float32) / optimal_T, dim=-1).numpy()
    ece_after = compute_ece(probs_scaled, labels)
    print(f"ECE after  temperature scaling: {ece_after:.4f}")

    return optimal_T


def logits_to_probs(logits, temperature=1.0):
    """Converts raw logits to calibrated softmax probabilities."""
    scaled = torch.tensor(logits, dtype=torch.float32) / temperature
    return F.softmax(scaled, dim=-1).numpy()


def _build_prob_df(df, all_logits, temperature):
    """
    Attaches raw logits, calibrated probabilities, hard label, and confidence
    to a copy of df. Used by both predict functions.
    """
    all_logits = np.array(all_logits)
    probs = logits_to_probs(all_logits, temperature)
    preds = probs.argmax(axis=1)

    df = df.copy()
    # Raw logits (saved for potential re-scaling later)
    for i, name in enumerate(LABEL_NAMES):
        col = name.lower().replace("/", "_")
        df[f"logit_{col}"] = all_logits[:, i]

    # Calibrated probabilities
    for i, name in enumerate(LABEL_NAMES):
        col = name.lower().replace("/", "_")
        df[f"prob_{col}"] = probs[:, i]

    df["label_int"] = preds
    df["label"] = [LABEL_NAMES[p] for p in preds]
    df["confidence"] = probs.max(axis=1)

    return df


# ---------------------------------------------------------------------------
# Transformer predict
# ---------------------------------------------------------------------------

def transformer_predict(model, prediction_dataloader, df,
                        output_path="labeled_examples.csv",
                        temperature=1.0):
    """
    Runs inference with a fine-tuned encoder model on unlabeled data.
    Saves raw logits, calibrated probabilities, hard labels, and confidence.

    model:                  fine-tuned encoder classification model
    prediction_dataloader:  DataLoader of unlabeled data (no labels column)
    df:                     original dataframe to attach predictions to
    output_path:            path to save CSV output
    temperature:            calibration temperature (from calibrate(); default 1.0 = no scaling)

    Returns:
        df:        dataframe with predictions attached
        all_preds: list of integer predicted class indices
    """
    model.eval()
    device = next(model.parameters()).device
    all_logits = []

    with torch.no_grad():
        for batch in tqdm.tqdm(prediction_dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ["input_ids", "attention_mask"]}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().float().numpy())

            # Incremental save in case of interruption
            partial_logits = np.concatenate(all_logits, axis=0)
            partial_df = _build_prob_df(df.iloc[:len(partial_logits)], partial_logits, temperature)
            partial_df.to_csv(output_path, index=False)

    all_logits = np.concatenate(all_logits, axis=0)
    df = _build_prob_df(df, all_logits, temperature)
    df.to_csv(output_path, index=False)

    all_preds = df["label_int"].tolist()
    return df, all_preds


# ---------------------------------------------------------------------------
# LLM predict
# ---------------------------------------------------------------------------

def llm_predict(model, prediction_dataloader, df,
                           output_path="labeled_examples.csv",
                           temperature=1.0):
    """
    Runs inference with a fine-tuned decoder LLM (used as sequence classifier)
    on unlabeled data. Saves raw logits, calibrated probabilities, hard labels,
    and confidence.

    model:                  fine-tuned decoder model (AutoModelForSequenceClassification)
    prediction_dataloader:  DataLoader of unlabeled data (no labels column)
    df:                     original dataframe to attach predictions to
    output_path:            path to save CSV output
    temperature:            calibration temperature (from calibrate(); default 1.0 = no scaling)

    Returns:
        df:        dataframe with predictions attached
        all_preds: list of integer predicted class indices
    """
    model.eval()
    device = next(model.parameters()).device
    all_logits = []

    with torch.no_grad():
        for batch in tqdm.tqdm(prediction_dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ["input_ids", "attention_mask"]}
            # Process batch-at-a-time for decoder models to avoid memory issues
            for i in range(batch["input_ids"].size(0)):
                single = {k: v[i].unsqueeze(0) for k, v in batch.items()}
                outputs = model(**single)
                all_logits.append(outputs.logits.cpu().float().numpy())

            # Incremental save
            partial_logits = np.concatenate(all_logits, axis=0)
            partial_df = _build_prob_df(df.iloc[:len(partial_logits)], partial_logits, temperature)
            partial_df.to_csv(output_path, index=False)

    all_logits = np.concatenate(all_logits, axis=0)
    df = _build_prob_df(df, all_logits, temperature)
    df.to_csv(output_path, index=False)

    all_preds = df["label_int"].tolist()
    return df, all_preds