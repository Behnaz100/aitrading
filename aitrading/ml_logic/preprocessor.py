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
        raise ValueError(f"Input must be a DataFrame with columns: Date, Close, Volume, got {X.columns}")

    df_selected = df[['Date', 'Close', 'Volume']]
    df_selected['Date'] = df_selected['Date'].apply(str_to_datetime)
    df_selected.index = df_selected.pop('Date')
    df_selected['dir'] = np.where(df_selected['Close'].diff() >= 0, 1, 0)

    windowed_df = df_to_windowed_df(df_selected,
                                    min_date,
                                    max_date,
                                    n=3)

    dates, X, y = windowed_df_to_date_x_y(windowed_df)

    return dates, X, y

# split the data into train and test and validation
