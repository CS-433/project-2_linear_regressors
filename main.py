from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison

# Choose your embedding, model and data size here :)
embeddings = ["tfidf", "fasttext", "word2vec", "glove", "vinai/bertweet-base", "roberta-base", "cardiffnlp/twitter-roberta-base", "vinai/bertweet-large"]
models = ["hf", "logreg", "mlp", "random_forest", "svm"]
# data size maximums: train_size=2_500_000, valid_size=250_000 (by default if you don't specify)
embedding = embeddings[3]
model = models[0]
train_size = 20_000
valid_size = 2_000


Pipeline(
    embedding=embedding,
    model=model,
    train_size=train_size,
    valid_size=valid_size
).run()

plot_comparison()