import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class FeatureEngineer: #Feature Engineering

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def log_transform(self, columns: list) -> None:
        """
        Apply log transformation to specified columns.

        :param columns: list, column names to apply log transformation to
        """
        for col in columns:
            if np.all(self.data[col] > 0):
                self.data[col] = np.log(self.data[col])
            else:
                raise ValueError(f"Log transformation cannot be applied to non-positive values in column '{col}'.")

    def standard_scale(self, columns: list) -> None:
        """
        Standardize specified columns by applying the StandardScaler.

        :param columns: list, column names to apply standard scaling to
        """
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def polynomial_features(self, columns: list, degree: int = 2) -> None:
        """
        Create polynomial features for specified columns.

        :param columns: list, column names to create polynomial features for
        :param degree: int, the degree of the polynomial features (default: 2)
        """
        poly = PolynomialFeatures(degree, include_bias=False)
        poly_data = poly.fit_transform(self.data[columns])

        poly_columns = poly.get_feature_names(columns)
        poly_df = pd.DataFrame(poly_data, columns=poly_columns, index=self.data.index)

        self.data.drop(columns=columns, inplace=True)
        self.data = pd.concat([self.data, poly_df], axis=1)

    def feature_engineering(self, log_transform_columns: list = None, standard_scale_columns: list = None,
                            polynomial_features_columns: list = None, polynomial_degree: int = 2) -> pd.DataFrame:
        """
        Perform feature engineering on the data.

        :param log_transform_columns: list, optional column names to apply log transformation to
        :param standard_scale_columns: list, optional column names to apply standard scaling to
        :param polynomial_features_columns: list, optional column names to create polynomial features for
        :param polynomial_degree: int, optional degree of the polynomial features (default: 2)
        :return: pd.DataFrame containing the engineered features
        """
        if log_transform_columns:
            self.log_transform(log_transform_columns)

        if standard_scale_columns:
            self.standard_scale(standard_scale_columns)

        if polynomial_features_columns:
            self.polynomial_features(polynomial_features_columns, degree=polynomial_degree)

        return self.data
