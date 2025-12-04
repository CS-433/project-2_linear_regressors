# grid_search/parameters_model_selection.py

import itertools
import pandas as pd
from pipelines.pipeline_generic import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import os

# Hyperparameter definitions for grid search
grid_params = {
    "logreg": {
        "alpha": [0.1, 1, 5, 10]
    },
    "mlp": {
        "alpha": [0.01, 0.1, 1],
        "hidden_layer_sizes": [(128,), (256,), (256, 128), (128, 128)],
        "activation": ["relu", "tanh", "logistic"]
    },
    "svm": {
        "C": [0.01, 0.1, 1, 10],
        "loss": ["hinge", "squared_hinge"]
    },
    "random_forest": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [10, 20, 30, None]
    }
}

# Embeddings to test
embeddings = ["word2vec", "fasttext", "glove", "tfidf"]

all_results = []

# Grid search over models, embeddings, and hyperparameters
for embedding in embeddings:
    for model_name, params_dict in grid_params.items():
        keys, values = zip(*params_dict.items())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            print(f"\n=== Testing model={model_name}, embedding={embedding}, params={params} ===")

            # Initialize pipeline with current parameters
            if model_name == "logreg":
                pipe = Pipeline(
                    embedding=embedding,
                    model="logreg",
                    train_size=20000,
                    valid_size=2000,
                    alpha=params["alpha"],
                    max_iter=1000
                )
            elif model_name == "mlp":
                pipe = Pipeline(
                    embedding=embedding,
                    model="mlp",
                    train_size=20000,
                    valid_size=2000,
                    alpha=params["alpha"],
                    hidden_layer_sizes=params["hidden_layer_sizes"],
                    activation=params["activation"],
                    max_iter=1200
                )
            elif model_name == "svm":
                pipe = Pipeline(
                    embedding=embedding,
                    model="svm",
                    train_size=20000,
                    valid_size=2000,
                    C=params["C"],
                    loss=params["loss"],
                    max_iter=1000
                )
            elif model_name == "random_forest":
                pipe = Pipeline(
                    embedding=embedding,
                    model="random_forest",
                    train_size=20000,
                    valid_size=2000,
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"]
                )

            # Run and evaluate
            pipe.run()
            preds = pipe.evaluate()

            results_entry = {
                "model": model_name,
                "embedding": embedding,
            }
            results_entry.update(params)
            results_entry.update({
                "accuracy": accuracy_score(pipe.y_valid, preds),
                "f1": f1_score(pipe.y_valid, preds, zero_division=0)
            })
            all_results.append(results_entry)

# Save all results
df = pd.DataFrame(all_results)
df.to_csv("grid_search/parameters_model_selection.csv", index=False)
print("\nGrid search completed! Results saved to grid_search/parameters_model_selection.csv")

save_path = "grid_search/parameters_model_selection.csv"

# After grid search, extract best F1 scores for each model-embedding combo and the hyperparameters that achieved them
best_path_f1 = "grid_search/best_model_selection_f1.csv"

if os.path.exists(save_path):
    print("\nExisting results detected. Loading file instead of recomputing...")
    df = pd.read_csv(save_path)
    best_rows = []
    grouped = df.groupby(["model", "embedding"])

for (model, embedding), group in grouped:
    best_f1 = group["f1"].max()
    best_rows.append({
        "model": model,
        "embedding": embedding,
        "best_f1": best_f1,
        "best_params": group.loc[group["f1"].idxmax()].drop(["model", "embedding", "accuracy", "f1"]).to_dict()
    })

df_best_f1 = pd.DataFrame(best_rows)
df_best_f1.to_csv(best_path_f1, index=False)
print("Best-F1 summary saved to:", best_path_f1)

# Print the 3 best f1 scores overall
top3_f1 = df_best_f1.nlargest(3, 'best_f1')[['model', 'embedding', 'best_f1', 'best_params']]
print("\nTop 3 F1 scores overall:")
print(top3_f1.to_string(index=False))

'''
Results printed in the terminal:

Top 3 F1 scores overall:
        model embedding  best_f1
          mlp     tfidf 0.799801
random_forest     tfidf 0.792400
          svm     tfidf 0.792188
'''

# After grid search, extract best accuracy scores for each model-embedding combo and the hyperparameters that achieved them
best_path_acc = "grid_search/best_model_selection_acc.csv"

if os.path.exists(save_path):
    print("\nExisting results detected. Loading file instead of recomputing...")
    df = pd.read_csv(save_path)
    best_rows = []
    grouped = df.groupby(["model", "embedding"])

for (model, embedding), group in grouped:
    best_acc = group["accuracy"].max()
    best_rows.append({
        "model": model,
        "embedding": embedding,
        "best_acc": best_acc,
        "best_params": group.loc[group["accuracy"].idxmax()].drop(["model", "embedding", "accuracy", "f1"]).to_dict()
    })

df_best_acc = pd.DataFrame(best_rows)
df_best_acc.to_csv(best_path_acc, index=False)
print("Best-Accuracy summary saved to:", best_path_acc)

# Print the 3 best accuracy scores overall
top3_acc = df_best_acc.nlargest(3, 'best_acc')[['model', 'embedding', 'best_acc', 'best_params']]
print("\nTop 3 Accuracy scores overall:")       
print(top3_acc.to_string(index=False))
'''
Results printed in the terminal:

Top 3 Accuracy scores overall:
 model embedding  best_acc
   mlp     tfidf    0.7985
   svm     tfidf    0.7925
logreg     tfidf    0.7890
'''