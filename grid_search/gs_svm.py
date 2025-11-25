# grid_search/gs_svm.py

import itertools
import pandas as pd
from pipelines.pipeline_generic import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def grid_search_svm():
    """
    Grid search over SVM hyperparameters:
    - C (regularization)
    - loss (hinge or squared_hinge)
    """

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "loss": ["hinge", "squared_hinge"]
    }

    results = []

    # Boucle sur toutes les combinaisons possibles
    for C, loss in itertools.product(param_grid["C"], param_grid["loss"]):
        print(f"\n=== Testing C={C}, loss={loss} ===")

        pipe = Pipeline(
            embedding="word2vec",
            model="svm",
            train_size=20000,
            valid_size=2000,
            C=C,
            loss=loss,
            max_iter=1200  # ou autre valeur si nécessaire
        )

        pipe.run()
        preds = pipe.evaluate()

        results.append({
            "C": C,
            "loss": loss,
            "accuracy": accuracy_score(pipe.y_valid, preds),
            "f1": f1_score(pipe.y_valid, preds, zero_division=0)
        })

    # Sauvegarde
    df = pd.DataFrame(results)
    df.to_csv("grid_search/gs_svm_results.csv", index=False)
    #print("\nGrid search completed! Results saved to grid_search/gs_svm_results.csv")


if __name__ == "__main__":
    grid_search_svm()
