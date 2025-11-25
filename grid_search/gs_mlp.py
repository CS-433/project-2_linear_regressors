# grid_search/gs_mlp.py

import itertools
import pandas as pd
from pipelines.pipeline_generic import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def grid_search_mlp():
    """
    Grid search over MLP hyperparameters:
    - alpha (regularization)
    - hidden_layer_sizes (number of layers and neurons)
    - activation function
    """

    param_grid = {
        "alpha": [0.01, 0.1, 1],
        "hidden_layer_sizes": [(128,), (256,), (256, 128), (128, 128)],
        "activation": ["relu", "tanh", "logistic"]
    }

    results = []

    # Boucle sur toutes les combinaisons possibles
    for alpha, hidden_layers, activation in itertools.product(
        param_grid["alpha"],
        param_grid["hidden_layer_sizes"],
        param_grid["activation"]
    ):
        print(f"\n=== Testing alpha={alpha}, hidden_layers={hidden_layers}, activation={activation} ===")

        pipe = Pipeline(
            embedding="word2vec",
            model="mlp",
            train_size=20000,
            valid_size=2000,
            alpha=alpha,
            max_iter=1200, 
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            random_state=42
        )

        pipe.run()
        preds = pipe.evaluate()

        results.append({
            "alpha": alpha,
            "hidden_layer_sizes": hidden_layers,
            "activation": activation,
            "accuracy": accuracy_score(pipe.y_valid, preds),
            "f1": f1_score(pipe.y_valid, preds, zero_division=0)
        })

    # Sauvegarde
    df = pd.DataFrame(results)
    df.to_csv("grid_search/gs_mlp_results.csv", index=False)
    #print("\nGrid search completed! Results saved to grid_search/gs_mlp_results.csv")


if __name__ == "__main__":
    grid_search_mlp()
