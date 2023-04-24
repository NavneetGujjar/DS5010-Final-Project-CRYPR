# **CryPr  - A Cryptocurrency Price Prediction Package**

## Overview
**Crypr** is a cryptocurrency price forecasting package that uses Machine Learning algorithms to predict future price trends of cryptocurrencies. The package allows users to analyze and forecast price movements of cryptocurrencies such as Bitcoin, Ethereum etc. 
With Crypr, users can retrieve historical price data and use it to train Machine Learning models that will provide price predictions based on current market scenarios. The package provides machine learning models such as Linear Regression, ARIMA, LSTM, and Prophet, which can be used to predict cryptocurrency prices over different time horizons. 

## Functionality
**Crypr** – a cryptocurrency price forecasting package involves several modules, classes, and functions that work together to provide users with accurate predictions of cryptocurrency prices.

•	_Data Retrieval Module_: This module is responsible for retrieving historical cryptocurrency price data from various sources such as API endpoints and CSV files. It will include function retrieve_file_data() to get the data and store it in a suitable format. 

•	_Data Preprocessing Module_: The data retrieved from different sources can be inconsistent and incomplete, so this module will handle the pre-processing of data to prepare it for use in Machine Learning models. Functions like clean_raw_data() and process_clean_data() will be implemented to clean and prepare the data for analysis.

•	_Feature Engineering Module_: This module will include functions that create additional features or variables from the existing data that can be useful in predicting cryptocurrency prices. Functions such as feature_engineering() will be implemented to create additional features. 

•	_Machine Learning Model Module_: This module will contain different machine learning models that can be used for predicting cryptocurrency prices. The models will be trained using historical data and used to forecast prices over a given time horizon. Examples of models to be implemented include Linear Regression, ARIMA, LSTM, and Prophet. 


## Installing the package
```
pip install -i https://test.pypi.org/simple/ CryPr==0.0.1
```
Package details : [CryPr Home](https://test.pypi.org/project/CryPr/0.0.1/)

## Load Package Functionality 
```
$from CryPr.DataRetrieval import retrieve_file_data
$from CryPr.DataPreprocessing import DataCleaner
$from CryPr.FeatureEngineering import FeatureEngineer
$from CryPr.LinearRegression import LinearRegression
$from CryPr.LSTM import LSTMTimeSeries
$from CryPr.ARIMA import ARIMA
$from CryPr.Prophet import TimeSeriesProphet
```

# Usage
## Import Data 
```
$file_path = "example.csv"  # Change this to the path of your file (e.g., example.xlsx or example.json)
$data = retrieve_file_data(file_path)
$print(data.tail(10))
```
This function uses the Pandas library to read data from files with different formats. It checks the file extension and calls the appropriate function from the Pandas library to load the data. Note that this function assumes that the input file has a valid extension and is formatted correctly.

### Subsample of data 
```
$data = data.sample(n=10000, random_state=1)
$data.shape
```
## Preprocess Data
```
# Assuming 'data' is a pandas DataFrame containing the raw data
$data_cleaner = DataCleaner(data)
# Process the data using the mean method, optionally drop specific columns, and set a missing values threshold
$clean_data = data_cleaner.process_data(method="mean", columns_to_drop=["column_name"], missing_values_threshold=0.5)
$print(clean_data.tail(5))
```
This class defines a DataCleaner class that takes a pandas DataFrame as input. It provides methods for cleaning and processing the data, 
such as dropping columns with too many missing values, filling missing values, and dropping specified columns.

## Feature Engineering
```
# Assuming 'data' is a pandas DataFrame containing the data
$feature_engineer = FeatureEngineer(data)
# Apply log transformation, standard scaling, and create polynomial features
$engineered_data = feature_engineer.feature_engineering(
    log_transform_columns=["column1", "column2"],
    standard_scale_columns=["column3", "column4"],
    polynomial_features_columns=["column5", "column6"],
    polynomial_degree=2)
$print(engineered_data)
```
This class defines a FeatureEngineer class that takes a pandas DataFrame as input. It provides methods for feature engineering, such as applying log transformation, standard scaling, and creating polynomial features. 
The feature_engineering method allows you to perform these operations in a single step by providing lists of column names for each transformation.

## Machine Learning - Time Series Forecasting Models
### Linear Regression Usage
```
# Assuming 'X_train', 'y_train', and 'X_test' are pandas DataFrames containing the training and test data
$linear_regression = LinearRegression(learning_rate=0.01, iterations=1000)
# Train the model on the training data and predict the target values for the test data
$y_pred = linear_regression.train_and_predict(X_train, y_train, X_test)
$print(y_pred)
```
This class defines a LinearRegression class that implements the gradient descent algorithm for linear regression. The fit method trains the model, the predict method predicts target values, and the train_and_predict method performs both operations in a single step. 
The class uses NumPy for matrix operations and supports pandas DataFrames as input.

### ARIMA - Usage
```
# Assuming 'y_train' is a pandas Series containing the time series data
$arima = ARIMA(p=1, d=1, q=1)
# Train the model on the training data and predict the next 'steps' values
$steps = 10
$y_pred = arima.train_and_predict(y_train, steps)
$print(y_pred)
```
This class defines an ARIMA class that implements a simple ARIMA model using autoregression, differencing, and moving average components. The fit method trains the model, the predict method predicts future values, and the train_and_predict method performs both operations in a single step. The class uses NumPy for matrix operations and supports pandas Series.

### Prophet
```
# Assuming 'df_train' and 'df_future' are pandas DataFrames containing the training and future data, respectively
# 'ds' column should contain datetime values, and 'y' column should contain the target values in 'df_train'
$time_series_prophet = TimeSeriesProphet()
# Train the model on the training data and predict the target values for the future data
$forecast = time_series_prophet.train_and_predict(df_train, df_future)
$print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
```
This class defines a TimeSeriesProphet class that uses the FB Prophet library to perform time series forecasting. The class provides methods to fit the model to data, predict future values, and perform both operations in a single step. Additionally, there is a method to add custom seasonality to the model.

### LSTM 
```
# Assuming 'train_series' is a pandas Series containing the time series data
$lstm_time_series = LSTMTimeSeries(look_back=3, epochs=100, batch_size=1, verbose=1)
# Train the model on the training data and predict the next 'steps' values
$steps = 10
$y_pred = lstm_time_series.train_and_predict(train_series, steps)
$print(y_pred)
```
This class defines an LSTMTimeSeries class that uses the Keras library to create an LSTM model for time series forecasting. The class provides methods to fit the model to data, predict future values, and perform both operations in a single step. The look_back parameter determines how many previous time steps are used.
