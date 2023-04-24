import pandas as pd
from typing import List

class DataCleaner: #Preprocess

    def __init__(self, data: pd.DataFrame):
        self.raw_data = data
        self.clean_data = None

    def drop_missing_values(self, threshold: float = 0.5) -> None:
        """
        Drop columns with a proportion of missing values greater than the specified threshold.

        :param threshold: float, proportion of missing values required to drop a column (default: 0.5)
        """
        missing_values_ratio = self.raw_data.isnull().mean()
        columns_to_drop = missing_values_ratio[missing_values_ratio > threshold].index
        self.raw_data.drop(columns=columns_to_drop, inplace=True)

    def fill_missing_values(self, method: str = "mean") -> None:
        """
        Fill missing values using the specified method.

        :param method: str, method to fill missing values, options are "mean", "median", "mode", or "zero" (default: "mean")
        """
        if method == "mean":
            self.raw_data.fillna(self.raw_data.mean(), inplace=True)
        elif method == "median":
            self.raw_data.fillna(self.raw_data.median(), inplace=True)
        elif method == "mode":
            self.raw_data.fillna(self.raw_data.mode().iloc[0], inplace=True)
        elif method == "zero":
            self.raw_data.fillna(0, inplace=True)
        else:
            raise ValueError("Invalid method. Supported methods are 'mean', 'median', 'mode', and 'zero'.")

    def drop_columns(self, columns_to_drop: List[str]) -> None:
        """
        Drop specified columns from the DataFrame.

        :param columns_to_drop: List[str], list of column names to drop from the DataFrame
        """
        self.raw_data.drop(columns=columns_to_drop, inplace=True)

    def process_data(self, method: str = "mean", columns_to_drop: List[str] = None, missing_values_threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean and process raw data.

        :param method: str, method to fill missing values, options are "mean", "median", "mode", or "zero" (default: "mean")
        :param columns_to_drop: List[str], optional list of column names to drop from the DataFrame
        :param missing_values_threshold: float, optional proportion of missing values required to drop a column (default: 0.5)
        :return: pd.DataFrame containing the cleaned and processed data
        """
        self.drop_missing_values(threshold=missing_values_threshold)
        self.fill_missing_values(method=method)

        if columns_to_drop:
            self.drop_columns(columns_to_drop)

        self.clean_data = self.raw_data.copy()
        return self.clean_data