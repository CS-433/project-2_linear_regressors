# Sara
from sklearn.linear_model import LogisticRegression

def make(alpha, gamma, max_iter):
    return LogRegModel(alpha=alpha, gamma=gamma, max_iter=max_iter)

class LogRegModel:
    def __init__(self, alpha, gamma, max_iter):
        self.alpha = alpha
        self.gamma = gamma
        self.model = LogisticRegression(
            C=1.0 / alpha,
            max_iter=max_iter,
            solver="lbfgs"
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

