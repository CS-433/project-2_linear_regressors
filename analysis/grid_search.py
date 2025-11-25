# analysis/grid_search.py

import itertools
import pandas as pd
from pipelines.pipeline_generic import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def grid_search_logreg():
    """
    Perform a manual grid search over the hyperparameters of the logistic regression model.
    """

    param_grid = {
        "alpha": [0.1, 1, 5, 10],
        "gamma": [0.0, 0.5, 1.0] # a changer, en fait gamma n'est pas utilisé pour la logreg
    }

    results = []

    # Iterate over all combinations of hyperparameters
    for alpha, gamma in itertools.product(
        param_grid["alpha"],
        param_grid["gamma"]
    ):
        print(f"\n=== Testing alpha={alpha}, gamma={gamma} ===")

        # Build and run the pipeline
        pipe = Pipeline(
            embedding="word2vec",
            model="logreg",
            train_size=20000,
            valid_size=2000,
            alpha=alpha,
            gamma=gamma,
            max_iter=1200
        )

        pipe.run()

        # Store metrics

        preds = pipe.evaluate()

        results.append({
            "alpha": alpha,
            "gamma": gamma,
            "accuracy": accuracy_score(pipe.y_valid, preds),
            "f1": f1_score(pipe.y_valid, preds, zero_division=0)
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("analysis/logreg_grid_search_results.csv", index=False)

    print("\nGrid search completed! Results saved to analysis/logreg_grid_search_results.csv")

if __name__ == "__main__":
    grid_search_logreg()
