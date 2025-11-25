# models/mlp.py

from sklearn.neural_network import MLPClassifier

def make(hidden_layer_sizes, activation, alpha, max_iter, random_state=42, shuffle=True, learning_rate_init=0.001, learning_rate='constant'):
    """
    Create and return an MLPModel instance.

    Parameters
    ----------
    hidden_layer_sizes : tuple
        Size of hidden layers, e.g., (100,) for one hidden layer with 100 units.
    activation : str
        Activation function ('relu', 'tanh', 'logistic').
    alpha : float
        L2 penalty (regularization term).
    max_iter : int
        Maximum number of iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    MLPModel
        An initialized MLP wrapper.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=max_iter,
        shuffle=shuffle,
        learning_rate_init=learning_rate_init,
        learning_rate=learning_rate
    )
    return model


class MLPModel:
    """
    Wrapper class around scikit-learn's MLPClassifier.
    """

    def __init__(self, hidden_layer_sizes, activation, alpha, max_iter, random_state):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            solver="adam"
        )

    def train(self, X, y):
        """Fit the MLP model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)

