import numpy as np
import pandas as pd

class ARIMA: #Arima time series

    def __init__(self, p: int, d: int, q: int):
        self.p = p
        self.d = d
        self.q = q
        self.phi = None
        self.theta = None

    def _arma(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        eps = np.random.normal(size=n)
        ar_params = np.random.uniform(-1, 1, self.p)
        ma_params = np.random.uniform(-1, 1, self.q)

        ar = np.zeros(n)
        ma = np.zeros(n)

        for t in range(n):
            ar[t] = np.sum(ar_params * y[t - self.p:t][::-1]) if t - self.p >= 0 else 0
            ma[t] = np.sum(ma_params * eps[t - self.q:t][::-1]) if t - self.q >= 0 else 0

        return ar + ma + eps

    def _difference(self, y: np.ndarray) -> np.ndarray:
        for _ in range(self.d):
            y = np.diff(y)
        return y

    def fit(self, y: pd.Series) -> None:
        y = y.to_numpy()
        y_diff = self._difference(y)
        y_arma = self._arma(y_diff)
        self.phi, self.theta = self._estimate_parameters(y_diff, y_arma)

    def _estimate_parameters(self, y_diff: np.ndarray, y_arma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Placeholder for parameter estimation, which should be replaced by a more advanced method
        return np.random.uniform(-1, 1, self.p), np.random.uniform(-1, 1, self.q)

    def predict(self, steps: int) -> pd.Series:
        y_pred = np.zeros(steps)
        y_pred[:self.p] = self.phi[:self.p]

        for t in range(self.p, steps):
            y_pred[t] = np.sum(self.phi * y_pred[t - self.p:t][::-1]) + np.random.normal()

        for _ in range(self.d):
            y_pred = np.cumsum(y_pred)

        return pd.Series(y_pred)

    def train_and_predict(self, y_train: pd.Series, steps: int) -> pd.Series:
        self.fit(y_train)
        y_pred = self.predict(steps)
        return y_pred
    
    