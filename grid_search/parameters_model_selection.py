# grid_search/parameters_model_selection.py

import itertools
import pandas as pd
from pipelines.pipeline_generic import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Définition des hyperparamètres à tester pour chaque modèle
'''
grid_params = {"mlp": {
        "alpha": [0.02],
        "hidden_layer_sizes": [(128,)],
        "activation": ["relu"]}}

embeddings = ["word2vec"]
'''
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
        "max_depth": [10, 20, 30, None]  # None = pas de limite
    }
}

# Embeddings à tester
embeddings = ["word2vec", "fasttext", "glove", "tfidf"]

all_results = []


for embedding in embeddings:
    for model_name, params_dict in grid_params.items():
        keys, values = zip(*params_dict.items())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            print(f"\n=== Testing model={model_name}, embedding={embedding}, params={params} ===")

            # Construction de la pipeline avec uniquement les paramètres définis
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

            # Run et évaluation
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

# Sauvegarde
df = pd.DataFrame(all_results)
df.to_csv("grid_search/parameters_model_selection.csv", index=False)
print("\nGrid search completed! Results saved to grid_search/model_selection.csv")