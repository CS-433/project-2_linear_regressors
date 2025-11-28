from sklearn.linear_model import LogisticRegression

# Make function to create LogRegModel
def make(alpha, max_iter):
    """
    Create and return a LogRegModel instance.

    Args:
        alpha (float): Regularization strength (inversely mapped to C = 1/alpha)
        gamma (float): Unused hyperparameter in this implementation, but kept for compatibility with the required interface
        max_iter (int): Maximum number of iterations allowed for the optimizer

    Returns:
        LogRegModel: An initialized logistic regression model wrapper.
    """
    return LogRegModel(alpha=alpha, max_iter=max_iter)

# Class definition for LogRegModel
class LogRegModel:
    """
    Wrapper class around scikit-learn's LogisticRegression model.

    Args:
        alpha (float): Regularization strength (inverse of C)
        gamma (float): Additional hyperparameter (not used internally)
        max_iter (int): Maximum number of iterations for the optimization algorithm
    """

    def __init__(self, alpha, max_iter):
        self.alpha = alpha
        self.model = LogisticRegression(
            C=1.0 / alpha,      # scikit-learn uses C = 1/lambda (regularization)
            max_iter=max_iter,  # maximum optimization iterations
            solver="lbfgs"      # optimization algorithm
        )

    def train(self, X, y):
        """
        Fit the logistic regression model.

        Args:
            X (array-like, shape (n_samples, n_features)) : Input data matrix
            y (array-like, shape (n_samples,)) : Binary class labels

        Returns:
            None
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels for new samples.

        Args:
            X (array-like, shape (n_samples, n_features)) : Input data matrix for prediction
        
        Returns:
            (array, shape (n_samples,)): Predicted class labels
        """
        return self.model.predict(X)


