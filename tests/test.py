from CryPr.DataRetrieval import retrieve_file_data
from CryPr.DataPreprocessing import DataCleaner
from CryPr.FeatureEngineering import FeatureEngineer
from CryPr.LinearRegression import LinearRegression
from CryPr.LSTM import LSTMTimeSeries
from CryPr.ARIMA import ARIMA
from CryPr.Prophet import TimeSeriesProphet

import datetime, pytz

def dateparse(time_in_secs):
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))


file_path = "C:/Users/yuvra/OneDrive/Documents/CRYPR/data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
data = retrieve_file_data(file_path)
print(data)


data_cleaner = DataCleaner(data)
clean_data = data_cleaner.process_data(method="mean", columns_to_drop=["Open"], missing_values_threshold=0.5)
print(clean_data)

feature_engineer = FeatureEngineer(clean_data)
engineered_data = feature_engineer.feature_engineering(
    log_transform_columns=["Weighted_Price"],
    standard_scale_columns=["High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)"]
)
print(engineered_data)

# Assuming 'X_train', 'y_train', and 'X_test' are pandas DataFrames containing the training and test data
linear_regression = LinearRegression(learning_rate=0.01, iterations=1000)

# Train the model on the training data and predict the target values for the test data
y_pred = linear_regression.train_and_predict(X_train, y_train, X_test)

print(y_pred)
