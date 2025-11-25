# models/random_forest.py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------------
# Fonction make
# -------------------------
def make(n_estimators=None, max_depth=None, random_state=42):
    """
    Create and return a RandomForestModel instance.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    RandomForestModel
        An initialized Random Forest wrapper.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    return model


# -------------------------
# Classe wrapper
# -------------------------
class RandomForestModel:
    """
    Wrapper class around scikit-learn's RandomForestClassifier.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X, y):
        """Fit the Random Forest model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
