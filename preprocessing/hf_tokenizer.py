import torch
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np

def make(train_texts, train_labels, valid_texts, valid_labels, test_texts, model_name):
    """
    Create tokenizer and tokenized HuggingFace datasets for training, validation, and testing.

    Args:
        train_texts (list of str): Raw training sentences
        train_labels (list of int): Labels corresponding to each training text
        valid_texts (list of str): Raw validation sentences
        valid_labels (list of int): Labels corresponding to each validation text
        test_texts (list of str): Raw test sentences (labels not required)
        model_name (str): Name of the transformer model used to load the tokenizer

    Returns:
        tokenizer (transformers.AutoTokenizer): Loaded tokenizer adapted to the specified model
        train_ds (datasets.Dataset): Tokenized and PyTorch-formatted training dataset
        valid_ds (datasets.Dataset): Tokenized and PyTorch-formatted validation dataset
        test_ds (datasets.Dataset): Tokenized and PyTorch-formatted test dataset
    """
    # load tokenizer
    if "bertweet" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            normalization=False
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
