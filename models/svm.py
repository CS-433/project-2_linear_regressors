# models/svm.py

from sklearn.svm import LinearSVC

def make(C, loss, max_iter):
    """
    Create and return a SVMModel instance.

    Parameters
    ----------
    C : float
        Regularization strength (inverse regularization).
    loss : str
        Loss function ("hinge" or "squared_hinge").
    max_iter : int
        Maximum number of optimization iterations.

    Returns
    -------
    SVMModel
        An initialized linear SVM model wrapper.
    """
    return SVMModel(C=C, loss=loss, max_iter=max_iter)


class SVMModel:
    """
    Wrapper class around scikit-learn's LinearSVC.

    Parameters
    ----------
    C : float
        Inverse regularization (higher C → lower regularization).
    loss : str
        "hinge" or "squared_hinge".
    max_iter : int
        Max number of iterations for training.
    """

    def __init__(self, C, loss, max_iter):
        self.C = C
        self.loss = loss
        self.model = LinearSVC(
            C=C,
            loss=loss,
            max_iter=max_iter
        )

    def train(self, X, y):
        """Fit the SVM model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)

