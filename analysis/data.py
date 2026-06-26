import pathlib
import gdown
import pandas as pd
from sklearn import model_selection
import datasets
import transformers
from torch.utils import data
import numpy as np

from .prompts import generate_large_prompt, generate_gpt_prompt, generate_simple_prompt, generate_encoder_prompt, generate_nli_premise
from .utils import LABEL_MAPPING

# Models that were pretrained on NLI and benefit from text-pair input formatting
NLI_MODEL_KEYWORDS = ["mnli", "nli", "fever", "anli", "debate"]

def is_nli_model(model_source):
    """Returns True if the model was pretrained on NLI data."""
    return any(kw in model_source.lower() for kw in NLI_MODEL_KEYWORDS)

def import_coded_data(data_path):
    """Imports coded data from Google Drive if not already present locally."""
    dir_path = pathlib.Path(data_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    coded_data_link = "https://drive.google.com/file/d/1oUXoIgs7M465Kvef8gjOIet-1gRDcRkv/view?usp=sharing"
    coded_data_path = data_path + "/coded_natsec.csv"
    if not pathlib.Path(coded_data_path).is_file():
        gdown.download(coded_data_link, output=coded_data_path, quiet=False, fuzzy=True)

def import_uncoded_data(data_path):
    """Imports uncoded data from Google Drive if not already present locally."""
    dir_path = pathlib.Path(data_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    uncoded_data_link = "https://drive.google.com/file/d/17rX-0ew_fWR2LH2vJjApRRb9jHpGJjb-/view?usp=sharing"
    uncoded_data_path = data_path + "/uncoded_natsec.csv"
    if not pathlib.Path(uncoded_data_path).is_file():
        gdown.download(uncoded_data_link, output=uncoded_data_path, quiet=False, fuzzy=True)


def create_label(df):
    """Takes dataframe of human coded sheet and outputs it with one label column."""
    result = []
    count = 0
    for row in df.itertuples(index=True):
        found = False

        if str(row.ALIGNED) == 'X' or str(row.ALIGNED) == '1':
            if found:
                print(count)
                raise ValueError("duplicate labeling")
            result.append("Aligned")
            found = True
        if str(row.NOT_ALIGNED) == 'X' or str(row.NOT_ALIGNED) == '1':
            if found:
                print(count)
                raise ValueError("duplicate labeling")
            result.append("Not_Aligned")
            found = True
        if str(row.NEUTRAL_IRRELEVANT) == 'X' or str(row.NEUTRAL_IRRELEVANT) == '1':
            if found:
                print(count)
                raise ValueError("duplicate labeling")
            result.append("Neutral/Irrelevant")
            found = True
        if str(row.BORDER_CASE) == 'X' or str(row.BORDER_CASE) == '1':
            if found:
                print(count)
                raise ValueError("duplicate labeling")
            result.append("Border Case")
            found = True
        if not found:
            print(count)
            raise ValueError("missing labeling")
        count += 1

    return pd.DataFrame(result, columns=['LABEL'])


def _apply_prompt(x, prompt_type, use_nli_format):
    """
    Applies prompt formatting to a dataframe split.
    For NLI format, generates premise/hypothesis columns.
    For standard format, generates a single text column.
    """
    if use_nli_format:
        x = x.copy()
        x.loc[:, 'premise'] = x.apply(generate_nli_premise, axis=1)
        x.loc[:, 'hypothesis'] = x['TEXT']
    else:
        prompt_funcs = {
            "long": generate_large_prompt,
            "gpt": generate_gpt_prompt,
            "simple": generate_simple_prompt,
            "encoder": generate_encoder_prompt,
            # NLI format is handled seperately
        }
        if prompt_type not in prompt_funcs:
            raise ValueError(f"Unknown prompt_type '{prompt_type}'. Choose from: {list(prompt_funcs)}")
        x = x.copy()
        x.loc[:, 'text'] = x.apply(prompt_funcs[prompt_type], axis=1)
    return x


def prepare_data_simple(data_path, prompt_type="long", use_nli_format=False):
    """
    Step 1 of training data preparation. Outputs data in df format with
    prompt-fit text and labels.

    data_path:      path to data CSV file
    prompt_type:    prompt template to use ('long', 'gpt', 'simple') — ignored if use_nli_format=True
    use_nli_format: if True, generates 'premise'/'hypothesis' columns instead of 'text'
                    (recommended for NLI-pretrained models like DeBERTa-mnli, Political DEBATE)
    """
    df = pd.read_csv(data_path, encoding_errors='ignore')
    columns = ['ALIGNED', 'NOT_ALIGNED', 'NEUTRAL_IRRELEVANT', 'BORDER_CASE']
    big_y = df[columns]

    x = df[['TEXT', 'Country', 'TARGET']][0:]
    y = create_label(big_y)
    flat_y = np.ravel(y)
    x['Label'] = flat_y

    # Remove border case entries
    x = x[x['Label'] != 'Border Case']
    flat_y = flat_y[flat_y != 'Border Case']

    # Train/eval/test split
    x_train, x_eval, y_train, y_eval = model_selection.train_test_split(
        x, flat_y, test_size=0.3, train_size=0.7, random_state=42)
    x_eval, x_test, y_eval, y_true = model_selection.train_test_split(
        x_eval, y_eval, test_size=0.5, train_size=0.5, random_state=42)

    x_train = _apply_prompt(x_train, prompt_type, use_nli_format)
    x_eval = _apply_prompt(x_eval, prompt_type, use_nli_format)
    x_test = _apply_prompt(x_test, prompt_type, use_nli_format)

    return x_train, x_eval, x_test


def _build_dataset(x, use_nli_format):
    """Converts a dataframe split to a HuggingFace Dataset with the right columns."""
    cols = ["premise", "hypothesis", "labels"] if use_nli_format else ["text", "labels"]
    return datasets.Dataset.from_pandas(x[cols].reset_index(drop=True), preserve_index=False)

def _make_tokenize_fn(tokenizer, use_nli_format, max_length):
    """Returns a tokenization function appropriate for the input format."""
    def tokenize_nli(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
    def tokenize_standard(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding='longest',
            max_length=max_length,
        )
    return tokenize_nli if use_nli_format else tokenize_standard


def _drop_text_columns(tokenized_dataset, use_nli_format):
    """Removes text columns after tokenization."""
    cols_to_drop = ["premise", "hypothesis"] if use_nli_format else ["text"]
    return tokenized_dataset.remove_columns(
        [c for c in cols_to_drop if c in tokenized_dataset.column_names]
    )


def prepare_data_final(x_train, x_eval, x_test,
                       model_source="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       batch_size=10, is_encoder_model = False, use_nli_format=None, max_length=None):
    """
    Final data preparation for fine-tuning. Outputs data as DataLoaders.

    x_train, x_eval, x_test: dataframes from prepare_data_simple
    model_source:   HuggingFace model to use for tokenizer
    batch_size:     batch size for DataLoaders
    use_nli_format: if True, tokenizes as text pairs (premise + hypothesis).
                    Defaults to auto-detect based on model_source.
    max_length:     max token length. Defaults to 512 for encoders, 1024 for decoders.
    """
    # Auto-detect NLI format and max_length if not specified
    if use_nli_format is None:
        use_nli_format = is_nli_model(model_source)
    if max_length is None:
        max_length = 512 if is_encoder_model else 1024
    
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)

    model_max = getattr(tokenizer, 'model_max_length', 512)
    if model_max < max_length:
        print(f"Model parameters change max length from {max_length} to {model_max}")
        max_length = model_max

    # Convert labels to int
    for split in [x_train, x_eval, x_test]:
        split.loc[:, 'labels'] = split.apply(lambda r: LABEL_MAPPING[r['Label']], axis=1)

    # Convert to HuggingFace datasets
    train_data = _build_dataset(x_train, use_nli_format)
    eval_data = _build_dataset(x_eval, use_nli_format)
    test_data = _build_dataset(x_test, use_nli_format)

    # Decoder-only models need pad token set manually
    if not is_encoder_model:
        if "gemma" in model_source.lower():
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = _make_tokenize_fn(tokenizer, use_nli_format, max_length)

    tokenized_train = _drop_text_columns(train_data.map(tokenize_fn, batched=True), use_nli_format)
    tokenized_eval = _drop_text_columns(eval_data.map(tokenize_fn, batched=True), use_nli_format)
    tokenized_test = _drop_text_columns(test_data.map(tokenize_fn, batched=True), use_nli_format)

    # Build DataLoaders
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dataloader = data.DataLoader(
        tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    eval_dataloader = data.DataLoader(
        tokenized_eval, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    test_dataloader = data.DataLoader(
        tokenized_test, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    test_labels = x_test['labels'].tolist()
    
    return train_dataloader, eval_dataloader, test_dataloader, test_labels

def prepare_data_train(data_path, prompt_type="long",
                       model_source="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli",
                       batch_size=10, is_encoder_model=False):
    """
    End-to-end training data preparation. Combines prepare_data_simple and
    prepare_data_final into a single call.

    data_path:     path to labeled CSV file
    prompt_type:   prompt template ('long', 'gpt', 'simple') — ignored for NLI models
    model_source:  HuggingFace model to use for tokenizer
    batch_size:    batch size for DataLoaders
    encoder_model: True if using an encoder-only model (DeBERTa, RoBERTa, etc.),
                   False for decoder-only (Llama, Mistral, etc.)

    Returns: train_dataloader, eval_dataloader, test_dataloader, test_labels
    """
    use_nli_format = is_nli_model(model_source)
    max_length = 512 if is_encoder_model else 1024

    x_train, x_eval, x_test = prepare_data_simple(
        data_path,
        prompt_type=prompt_type,
        use_nli_format=use_nli_format,
    )

    return prepare_data_final(
        x_train, x_eval, x_test,
        model_source=model_source,
        batch_size=batch_size,
        is_encoder_model=is_encoder_model,
        use_nli_format=use_nli_format,
        max_length=max_length,
    )


def load_prediction_data(path, prompt_type="long",
                         model_source="meta-llama/Meta-Llama-3.1-8B-Instruct", batch_size=10,
                         is_encoder_model=False, max_length=None):
    """
    Loads and tokenizes unlabeled data for inference.

    path:           path to unlabeled CSV file
    prompt_type:    prompt template ('long', 'gpt', 'simple') — ignored if use_nli_format=True
    model_source:   HuggingFace model to use for tokenizer
    use_nli_format: if True, tokenizes as text pairs. Defaults to auto-detect.
    max_length:     max token length. Defaults to 512 for encoders, 1024 for decoders.
    batch_size:     batch size for DataLoader
    """
    use_nli_format = is_nli_model(model_source)
    max_length = 512 if is_encoder_model else 1024

    df = pd.read_csv(path, encoding_errors='ignore')

    x = df[['Text', 'Country', 'Target']].copy()
    x['TEXT'] = x['Text']      # align column names to match prompt functions
    x['TARGET'] = x['Target']

    x = _apply_prompt(x, prompt_type, use_nli_format)

    if use_nli_format:
        prediction_data = datasets.Dataset.from_pandas(
            x[["premise", "hypothesis"]].reset_index(drop=True), preserve_index=False)
    else:
        prediction_data = datasets.Dataset.from_pandas(
            x[["text"]].reset_index(drop=True), preserve_index=False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
    model_max = getattr(tokenizer, 'model_max_length', 512)
    if model_max < max_length:
        print(f"Model parameters change max length from {max_length} to {model_max}")
        max_length = model_max

    if not is_encoder_model:
        if "gemma" in model_source.lower():
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = _make_tokenize_fn(tokenizer, use_nli_format, max_length)
    tokenized_prediction = _drop_text_columns(
        prediction_data.map(tokenize_fn, batched=True), use_nli_format)

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    prediction_dataloader = data.DataLoader(
        tokenized_prediction, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    return df, prediction_dataloader
