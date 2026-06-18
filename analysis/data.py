import pathlib
import gdown
import pandas as pd
from sklearn import model_selection
import datasets
import transformers
from torch.utils import data
import numpy as np

from .prompts import generate_large_prompt, generate_gpt_prompt, generate_simple_prompt

def import_data():
    """ Imports data from google drive
    """
    dir_path = pathlib.Path("data")
    dir_path.mkdir(parents=True, exist_ok=True)
    coded_data_link = "https://drive.google.com/file/d/1oUXoIgs7M465Kvef8gjOIet-1gRDcRkv/view?usp=sharing"
    coded_data_path = "data/coded_natsec.csv"
    if not pathlib.Path(coded_data_path).is_file():
        gdown.download(coded_data_link, output=coded_data_path, quiet=False)

    uncoded_data_link = "https://drive.google.com/file/d/17rX-0ew_fWR2LH2vJjApRRb9jHpGJjb-/view?usp=sharing"
    uncoded_data_path = "data/uncoded_natsec.csv"
    if not pathlib.Path(uncoded_data_path).is_file():
        gdown.download(uncoded_data_link, output=uncoded_data_path, quiet=False)

def create_label(df):
    #Takes dataframe of human coded sheet and outputs it with one label column
    result = []
    count = 0
    for row in df.itertuples(index=True):
        found = False

        if str(row.ALIGNED) == 'X' or str(row.ALIGNED) == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append("Aligned")
            found = True
        if str(row.NOT_ALIGNED)  == 'X' or str(row.NOT_ALIGNED)  == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append("Not_Aligned")
            found = True
        if str(row.NEUTRAL_IRRELEVANT)  == 'X' or str(row.NEUTRAL_IRRELEVANT)  == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append("Neutral/Irrelevant")
            found = True
        if str(row.BORDER_CASE)  == 'X' or str(row.BORDER_CASE)  == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append("Border Case")
            found = True
        if not found:
            print(count)
            raise Exception("missing labeling")
        count += 1

    df_little = pd.DataFrame(result,columns=['LABEL'])
    return df_little

def prepare_data_simple(data_path,prompt_type="long"):
    """
    Step 1 of training data preperation. Outputs data in df format with
    prompt-fit text and labels.

    data_path: path to data file
    prompt_type: function to use to generate prompt
    """
    #Read in coding data and convert to dataframe
    df = pd.read_csv(data_path,encoding_errors='ignore')
    columns = ['ALIGNED','NOT_ALIGNED','NEUTRAL_IRRELEVANT','BORDER_CASE']
    big_y = df[columns]


    # Split the data into text and label
    x = df[['TEXT','Country','TARGET']][0:]
    y = create_label(big_y)
    flat_y = np.ravel(y)
    x['Label'] = flat_y

    #Remove "border case" entries
    x = x[x['Label'] != 'Border Case']
    flat_y = flat_y[flat_y != 'Border Case']

    #train-test split
    x_train, x_eval, y_train, y_eval = model_selection.train_test_split(x, flat_y, test_size=0.3, train_size = .7, random_state=42)
    x_eval, x_test, y_eval, y_true = model_selection.train_test_split(x_eval, y_eval, test_size=0.5, train_size = .5, random_state=42)

    # Generate data with prompts
    if prompt_type == "long":
        x_train.loc[:,'text'] = x_train.apply(generate_large_prompt, axis=1)
        x_eval.loc[:,'text'] = x_eval.apply(generate_large_prompt, axis=1)
        x_test.loc[:,'text'] = x_test.apply(generate_large_prompt, axis=1)
    if prompt_type == "gpt":
        x_train.loc[:,'text'] = x_train.apply(generate_gpt_prompt, axis=1)
        x_eval.loc[:,'text'] = x_eval.apply(generate_gpt_prompt, axis=1)
        x_test.loc[:,'text'] = x_test.apply(generate_gpt_prompt, axis=1)
    if prompt_type == "simple":
        x_train.loc[:,'text'] = x_train.apply(generate_simple_prompt, axis=1)
        x_eval.loc[:,'text'] = x_eval.apply(generate_simple_prompt, axis=1)
        x_test.loc[:,'text'] = x_test.apply(generate_simple_prompt, axis=1)

    return x_train, x_eval, x_test


def prepare_data_final(x_train,x_eval,x_test,model_source="meta-llama/Meta-Llama-3.1-8B-Instruct",batch_size=10):
    """
    Final data preperation for fine-tuning.
    Outputs data as dataloaders.

    x_train: dataframe of training data
    x_eval: dataframe of eval data
    x_test: dataframe of test data
    model_source: huggingface model to use for tokenizer
    batch_size: batch size for dataloader
    """

    #convert y to int
    labels = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]
    mapping = {label: idx for idx, label in enumerate(labels)}
    x_train.loc[:,'labels'] = x_train.apply(lambda x: mapping[x['Label']], axis=1)
    x_eval.loc[:,'labels'] = x_eval.apply(lambda x: mapping[x['Label']], axis=1)
    x_test.loc[:,'labels'] = x_test.apply(lambda x: mapping[x['Label']], axis=1)

    # Convert to datasets
    train_data = datasets.Dataset.from_pandas(x_train[["text",'labels']])
    eval_data = datasets.Dataset.from_pandas(x_eval[["text",'labels']])
    test_data = datasets.Dataset.from_pandas(x_test[["text",'labels']])

    # Initialize the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)

    # Handle specific tokenizer for google/gemma-3-4b-it
    if "gemma" in model_source:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token


    # Define a tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='longest')

    # Apply the tokenization function to the dataset
    tokenized_dataset = train_data.map(tokenize_function, batched=True)
    tokenized_train = tokenized_dataset.remove_columns(["text"])

    tokenized_dataset = eval_data.map(tokenize_function, batched=True)
    tokenized_eval = tokenized_dataset.remove_columns(["text"])

    tokenized_dataset = test_data.map(tokenize_function, batched=True)
    tokenized_test = tokenized_dataset.remove_columns(["text"])

    # Turn data to dataloader
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dataloader = data.DataLoader(tokenized_train, batch_size=batch_size, shuffle=True,
                                  collate_fn = data_collator)
    eval_dataloader = data.DataLoader(tokenized_eval, batch_size=batch_size, shuffle=True,
                                  collate_fn = data_collator)
    test_dataloader = data.DataLoader(tokenized_test, batch_size=batch_size, shuffle=False,
                                  collate_fn = data_collator) #turn shuffle off for test

    test_labels = x_test['labels']

    return train_dataloader, eval_dataloader, test_dataloader,test_labels

def load_prediction_data(path,prompt_type="long",model_source="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    #Read in coding data and convert to dataframe
    df = pd.read_csv(path,encoding_errors='ignore')

    # Split the data into text and label
    x = df[['Text','Country','Target']][0:]
    x['TExT'] = x['Text']
    x['TARGET'] = x['Target']

    # Generate data with prompts
    if prompt_type == "long":
        x.loc[:,'text'] = x.apply(generate_large_prompt, axis=1)
    if prompt_type == "gpt":
        x.loc[:,'text'] = x.apply(generate_gpt_prompt, axis=1)
    if prompt_type == "simple":
        x.loc[:,'text'] = x.apply(generate_simple_prompt, axis=1)

    # Convert to datasets
    prediction_data = datasets.Dataset.from_pandas(x[["text"]])

    # Initialize the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)

    # Handle specific tokenizer for google/gemma-3-4b-it
    if "gemma" in model_source:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token


    # Define a tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='longest')

    tokenized_dataset = prediction_data.map(tokenize_function, batched=True)
    tokenized_prediction = tokenized_dataset.remove_columns(["text"])

    # Turn data to dataloader
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    batch_size = 10
    prediction_dataloader = data.DataLoader(tokenized_prediction, batch_size=batch_size, shuffle=False,
                                  collate_fn = data_collator) #turn shuffle off for test

    return df, prediction_dataloader