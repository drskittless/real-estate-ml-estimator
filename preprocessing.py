import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.median(numeric_only=True))

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_clean

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, drop_first=True)

    def scale_features(self, X: pd.DataFrame):
        return self.scaler.fit_transform(X)

    def split(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)