import numpy as np
import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class LSTMTimeSeries:

    def __init__(self, look_back: int = 1, epochs: int = 100, batch_size: int = 1, verbose: int = 0):
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, dataset: np.ndarray) -> tuple:
        data_x, data_y = [], []
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i + self.look_back), 0]
            data_x.append(a)
            data_y.append(dataset[i + self.look_back, 0])
        return np.array(data_x), np.array(data_y)

    def fit(self, series: pd.Series) -> None:
        dataset = series.values.reshape(-1, 1)
        dataset = self.scaler.fit_transform(dataset)
        train_x, train_y = self.create_dataset(dataset)

        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))

        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, self.look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=self.verbose)
        self.model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, validation_split=0.1, callbacks=[early_stop])

    def predict(self, series: pd.Series, steps: int) -> pd.Series:
        dataset = series.values.reshape(-1, 1)
        dataset = self.scaler.transform(dataset)

        test_x, _ = self.create_dataset(dataset)
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        y_pred = self.model.predict(test_x)
        y_pred = self.scaler.inverse_transform(y_pred).flatten()

        return pd.Series(y_pred[-steps:], index=series.index[-steps:])

    def train_and_predict(self, train_series: pd.Series, steps: int) -> pd.Series:
        self.fit(train_series)
        y_pred = self.predict(train_series, steps)
        return y_pred
