import torch
from datasets import Dataset
from transformers import AutoTokenizer

def make(train_texts, train_labels, valid_texts, valid_labels, test_texts, model_name):

    # load tokenizer
    if "bertweet" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            normalization=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

    # tokenization function
    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=64
        )
        # Remove token_type_ids (BERTweet, RoBERTa etc.)
        tokens.pop("token_type_ids", None)
        return tokens

    # create datasets
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    valid_ds = Dataset.from_dict({"text": valid_texts, "label": valid_labels})
    test_ds  = Dataset.from_dict({"text": test_texts})

    # apply tokenization
    train_ds = train_ds.map(tokenize, batched=True, load_from_cache_file=False)
    valid_ds = valid_ds.map(tokenize, batched=True, load_from_cache_file=False)
    test_ds  = test_ds.map(tokenize, batched=True, load_from_cache_file=False)

    # remove text column
    train_ds = train_ds.remove_columns(["text"])
    valid_ds = valid_ds.remove_columns(["text"])
    test_ds  = test_ds.remove_columns(["text"])

    # format for PyTorch
    train_ds = train_ds.with_format("torch")
    valid_ds = valid_ds.with_format("torch")
    test_ds  = test_ds.with_format("torch")

    return tokenizer, train_ds, valid_ds, test_ds
