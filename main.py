from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison
import torch
import pandas as pd
import os
# TO RUN THIS FILE: python -m main, in your terminal

# Ensure determinism
torch.use_deterministic_algorithms(True)
torch.backends.mps.allow_tf32 = False

# # Choose your embedding, model and data size here :)
embeddings = ["tfidf", "fasttext", "word2vec", "glove", "vinai/bertweet-base", "roberta-base", "cardiffnlp/twitter-roberta-base", "vinai/bertweet-large"]
models = ["hf", "logreg", "mlp", "random_forest", "svm"]
# data size maximums: train_size=2_250_000, valid_size=250_000 (by default if you don't specify)

# With this seeup you will obtain our highest reproducible results
# The run is long, it took around 12 hours
# AICrowd submission ID: #304074, score: 0.900, secondary score: 0.902
embedding = "vinai/bertweet-base"
model = "hf"
train_size = 2_250_000
valid_size = 250_000

# Uncomment to run the pipeline with the chosen setup
# pipe = Pipeline(
#     embedding=embedding,
#     model=model,
#     train_size=train_size,
#     valid_size=valid_size,
#     alpha=None,
#     gamma=None,
#     max_iter=None, # logreg + mlp model
#     hidden_layer_sizes=None,
#     activation=None,
#     random_state=42 # mlp model params

# )

# pipe.run()

# In order to avoid this long run, you can use the following lines, which load our best model (from the pipeline above) and evaluate it on the test set
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "saved_models/hf/vinai/bertweet-base_2250000"
TEST_PATH = "twitter-datasets/test_data.txt"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Charge tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# Charge test data, same as in pipeline
test_texts = []
with open(TEST_PATH) as f:
    for line in f:
        line = line.strip()
        _, tweet = line.split(",", 1)
        test_texts.append(tweet)

# Batch inference
all_preds = []
batch_size = 64

for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i + batch_size]

    encodings = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    encodings.pop("token_type_ids", None)
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        logits = model(**encodings).logits
        preds = logits.argmax(dim=-1)

    all_preds.extend(preds.cpu().tolist())

# Save submission file
from helpers.utils import save_submit
submit_path = save_submit(all_preds, name="Final_Submission_bertweet_base_full")


# To ensure that the script exits cleanly on completion and so ensure determinism
os._exit(0)