import pandas as pd
from prophet import Prophet

class TimeSeriesProphet: #FB Prophet

    def __init__(self, growth: str = 'linear', seasonality_mode: str = 'additive', **kwargs):
        self.model = Prophet(growth=growth, seasonality_mode=seasonality_mode, **kwargs)

    def add_seasonality(self, name: str, period: float, fourier_order: int, prior_scale: float = 10.0):
        self.model.add_seasonality(name=name, period=period, fourier_order=fourier_order, prior_scale=prior_scale)

    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df)

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        forecast = self.model.predict(future)
        return forecast

    def train_and_predict(self, df_train: pd.DataFrame, df_future: pd.DataFrame) -> pd.DataFrame:
        self.fit(df_train)
        forecast = self.predict(df_future)
        return forecast
