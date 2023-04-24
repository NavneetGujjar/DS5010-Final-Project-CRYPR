# CryPr  - A Cryptocurrency Price Forecasting Package

## Installing the package
pip install -i https://test.pypi.org/simple/ CryPr==0.0.1

## Load Package Functionality 
$from CryPr.DataRetrieval import retrieve_file_data
$from CryPr.DataPreprocessing import DataCleaner
$from CryPr.FeatureEngineering import FeatureEngineer
$from CryPr.LinearRegression import LinearRegression
$from CryPr.LSTM import LSTMTimeSeries
$from CryPr.ARIMA import ARIMA
$from CryPr.Prophet import TimeSeriesProphet

## Import Data - Usage

$file_path = "example.csv"  # Change this to the path of your file (e.g., example.xlsx or example.json)
$data = retrieve_file_data(file_path)
$print(data.tail(10))

This function uses the Pandas library to read data from files with different formats. It checks the file extension and calls the appropriate function from the Pandas library to load the data. Note that this function assumes that the input file has a valid extension and is formatted correctly.

### Subsample of data 

$data = data.sample(n=10000, random_state=1)
$data.shape

## Preprocess Data - Usage
Assuming 'data' is a pandas DataFrame containing the raw data

$data_cleaner = DataCleaner(data)

Process the data using the mean method, optionally drop specific columns, and set a missing values threshold

$clean_data = data_cleaner.process_data(method="mean", columns_to_drop=["column_name"], missing_values_threshold=0.5)
$print(clean_data.tail(5))

This class defines a DataCleaner class that takes a pandas DataFrame as input. It provides methods for cleaning and processing the data, 
such as dropping columns with too many missing values, filling missing values, and dropping specified columns.

## Feature Engineering
Assuming 'data' is a pandas DataFrame containing the data

$feature_engineer = FeatureEngineer(data)

Apply log transformation, standard scaling, and create polynomial features

$engineered_data = feature_engineer.feature_engineering(
    log_transform_columns=["column1", "column2"],
    standard_scale_columns=["column3", "column4"],
    polynomial_features_columns=["column5", "column6"],
    polynomial_degree=2
)

$print(engineered_data)

This class defines a FeatureEngineer class that takes a pandas DataFrame as input. It provides methods for feature engineering, such as applying log transformation, standard scaling, and creating polynomial features. 
The feature_engineering method allows you to perform these operations in a single step by providing lists of column names for each transformation.

## Linear Regression Usage
Assuming 'X_train', 'y_train', and 'X_test' are pandas DataFrames containing the training and test data

$linear_regression = LinearRegression(learning_rate=0.01, iterations=1000)

Train the model on the training data and predict the target values for the test data

$y_pred = linear_regression.train_and_predict(X_train, y_train, X_test)

$print(y_pred)

This class defines a LinearRegression class that implements the gradient descent algorithm for linear regression. The fit method trains the model, the predict method predicts target values, and the train_and_predict method performs both operations in a single step. 
The class uses NumPy for matrix operations and supports pandas DataFrames as input.

## ARIMA - Usage
Assuming 'y_train' is a pandas Series containing the time series data

$arima = ARIMA(p=1, d=1, q=1)

Train the model on the training data and predict the next 'steps' values

$steps = 10

$y_pred = arima.train_and_predict(y_train, steps)

$print(y_pred)



