from sklearn.svm import LinearSVC

# Make function to create SVMModel
def make(C, loss, max_iter):
    """
    Create and return a SVMModel instance.

    Args:
        C (float): Regularization strength (inverse regularization)
        loss (str): Loss function ("hinge" or "squared_hinge")
        max_iter (int): Maximum number of optimization iterations

    Returns:
        SVMModel: An initialized linear SVM model wrapper
    """
    return SVMModel(C=C, loss=loss, max_iter=max_iter)

# Class definition for SVMModel
class SVMModel:

    def __init__(self, C, loss, max_iter):
        """
        Initialize the SVM model with given hyperparameters.
        Args:
            C (float): Regularization strength (inverse regularization)
            loss (str): Loss function ("hinge" or "squared_hinge")
            max_iter (int): Maximum number of optimization iterations

        Returns:
            None
        """
        self.C = C
        self.loss = loss
        self.model = LinearSVC(
            C=C,
            loss=loss,
            max_iter=max_iter
        )

    def train(self, X, y):
        """
        Fit the SVM model.
        
        Args:
            X (array-like, shape (n_samples, n_features)) : Input data matrix
            y (array-like, shape (n_samples,)) : Binary class labels

        Returns:
            None
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X (array-like, shape (n_samples, n_features)) : Input data matrix for prediction

        Returns:
            (array, shape (n_samples,)): Predicted class labels
        """
        return self.model.predict(X)

