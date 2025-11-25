# grid_search/gs_random_forest.py

import itertools
import pandas as pd
from pipelines.pipeline_generic import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def grid_search_rf():
    """
    Grid search over Random Forest hyperparameters:
    - n_estimators (number of trees)
    - max_depth (maximum depth of each tree)
    """

    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [10, 20, 30, None]  # None = pas de limite
    }

    results = []

    for n_estimators, max_depth in itertools.product(
        param_grid["n_estimators"], param_grid["max_depth"]
    ):
        print(f"\n=== Testing n_estimators={n_estimators}, max_depth={max_depth} ===")

        pipe = Pipeline(
            embedding="word2vec",
            model="random_forest",
            train_size=20000,
            valid_size=2000,
            n_estimators=n_estimators,
            max_depth=max_depth
        )

        pipe.run()
        preds = pipe.evaluate()

        results.append({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "accuracy": accuracy_score(pipe.y_valid, preds),
            "f1": f1_score(pipe.y_valid, preds, zero_division=0)
        })

    # Sauvegarde
    df = pd.DataFrame(results)
    df.to_csv("grid_search/gs_forest_results.csv", index=False)
    #print("\nGrid search completed! Results saved to analysis/rf_grid_search_results.csv")


if __name__ == "__main__":
    grid_search_rf()
