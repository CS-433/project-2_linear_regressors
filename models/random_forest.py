from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Make function to create RandomForestModel
def make(n_estimators=None, max_depth=None, random_state=42):
    """
    Create and return a RandomForestModel instance.

    Args:
        n_estimators (int): Number of trees in the forest
        max_depth (int or None): Maximum depth of each tree
        random_state (int): Random seed for reproducibility

    Returns:
        RandomForestModel: An initialized Random Forest wrapper
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    return model

# Class definition for RandomForestModel
class RandomForestModel:
    """
    Wrapper class around scikit-learn's RandomForestClassifier.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the Random Forest model with given hyperparameters.

        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int or None): Maximum depth of each tree
            random_state (int): Random seed for reproducibility

        Returns:
            None
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X, y):
        """
        Fit the Random Forest model.

        Args:
            X (array-like, shape (n_samples, n_features)) : Input data matrix
            y (array-like, shape (n_samples,)) : Binary class labels

        Returns:
            None
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Predict class labels.
        Args:
            X (array-like, shape (n_samples, n_features)) : Input data matrix for prediction
        Returns:
            (array, shape (n_samples,)): Predicted class labels
        """
        return self.model.predict(X)
