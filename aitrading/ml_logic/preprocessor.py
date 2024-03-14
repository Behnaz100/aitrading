from typing import Tuple

import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from aitrading.ml_logic.encoders import str_to_datetime, df_to_windowed_df, windowed_df_to_date_x_y


def preprocess_date(df: pd.DataFrame, min_date: str, max_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # check if X is a DataFrame and include the necessary columns
    # such as Date, Close, Volume
    if not isinstance(df,
                      pd.DataFrame) or "Date" not in df.columns or "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError(f"Input must be a DataFrame with columns: Date, Close, Volume, got {df.columns}")

    df_selected = df[['Date', 'Close', 'Volume']]
    df_selected['Date'] = df_selected['Date'].apply(str_to_datetime)
    df_selected.index = df_selected.pop('Date')
    df_selected['dir'] = np.where(df_selected['Close'].diff() >= 0, 1, 0)
    print(type(min_date))
    print(type(max_date))
    windowed_df = df_to_windowed_df(df_selected,
                                    min_date,
                                    max_date,
                                    n=3)

    dates, X, y = windowed_df_to_date_x_y(windowed_df)

    return dates, X, y


# split the data into train and test and validation
def split_data(dates, X, y, split_ratios=(0.8, 0.1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, val_ratio = split_ratios
    q_train = int(len(dates) * train_ratio)
    q_val = q_train + int(len(dates) * val_ratio)

    dates_train, X_train, y_train = dates[:q_train], X[:q_train], y[:q_train]
    dates_val, X_val, y_val = dates[q_train:q_val], X[q_train:q_val], y[q_train:q_val]
    dates_test, X_test, y_test = dates[q_val:], X[q_val:], y[q_val:]

    return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test
