from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison
import torch
import os

torch.use_deterministic_algorithms(True)
torch.backends.mps.allow_tf32 = False

# # Choose your embedding, model and data size here :)
embeddings = ["tfidf", "fasttext", "word2vec", "glove", "vinai/bertweet-base", "roberta-base", "cardiffnlp/twitter-roberta-base", "vinai/bertweet-large"]
models = ["hf", "logreg", "mlp", "random_forest", "svm"]
# data size maximums: train_size=2_250_000, valid_size=250_000 (by default if you don't specify)
embedding = "vinai/bertweet-base"
model = "hf"
train_size = 2000
valid_size = 200

Pipeline(
    embedding=embedding,
    model=model,
    train_size=train_size,
    valid_size=valid_size,
    alpha=None,
    gamma=None,
    max_iter=None, # logreg + mlp model ###sara###
    hidden_layer_sizes=None,
    activation=None,
    random_state=42 # mlp model params ###sara###

).run() 

# plot_comparison()

# To ensure that the script exits cleanly on completion -> ensure determinism
os._exit(0)