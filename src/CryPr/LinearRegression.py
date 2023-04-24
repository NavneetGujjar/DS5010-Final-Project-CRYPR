import numpy as np
import pandas as pd

class LinearRegression: #Linear Regression

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = np.dot(X, self.weights) + self.bias
        cost = (1 / (2 * len(X))) * np.sum((y_pred - y) ** 2)
        return cost

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X = X.to_numpy()
        y = y.to_numpy()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.to_numpy()
        y_pred = np.dot(X, self.weights) + self.bias
        return pd.Series(y_pred, index=X.index)

    def train_and_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.Series:
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        return y_pred