from pipelines.pipeline_generic import Pipeline
from helpers.plots import plot_comparison

# # Choose your embedding, model and data size here :)
# embeddings = ["tfidf", "fasttext", "word2vec", "glove", "vinai/bertweet-base", "roberta-base", "cardiffnlp/twitter-roberta-base", "vinai/bertweet-large"]
# models = ["hf", "logreg", "mlp", "random_forest", "svm"]
# # data size maximums: train_size=2_250_000, valid_size=250_000 (by default if you don't specify)
# embedding = embeddings[4]
# model = models[0]
# train_size = 2_250_000
# valid_size = 250_000


# Pipeline(
#     embedding=embedding,
#     model=model,
#     train_size=train_size,
#     valid_size=valid_size
# ).run()

# plot_comparison()

### CHANGE SUBMISSSION FILE FROM 0,1 TO -1,1
import pandas as pd
subm = pd.read_csv("results/submission/submit_vinai/bertweet-base_full.csv")
subm_new = subm.copy()
subm_new['Prediction'] = subm_new['Prediction'].apply(lambda x: -1 if x == 0 else 1)
subm_new.to_csv("results/submission/submit_vinai/bertweet-base_full_adjusted.csv", index=False)