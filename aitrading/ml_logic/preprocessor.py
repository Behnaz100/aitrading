from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


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
def split_data(dates, X, y, split_ratios=(0.8, 0.1)) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, val_ratio = split_ratios
    q_train = int(len(dates) * train_ratio)
    q_val = q_train + int(len(dates) * val_ratio)

    dates_train, X_train, y_train = dates[:q_train], X[:q_train], y[:q_train]
    dates_val, X_val, y_val = dates[q_train:q_val], X[q_train:q_val], y[q_train:q_val]
    dates_test, X_test, y_test = dates[q_val:], X[q_val:], y[q_val:]

    return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test


def windowed_df_to_date_x_y(windowed_dataframe):
    # df_as_np = windowed_dataframe.to_numpy()

    dates = windowed_dataframe.index

    middle_matrix = windowed_dataframe.iloc[:, :-1]
    # X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = windowed_dataframe.iloc[:, -1]

    return dates, middle_matrix, Y


def create_binary_target_column(dataframe, column_name):
    dataframe['shifted_column'] = dataframe[column_name].shift(1)
    dataframe['target'] = np.where(dataframe[column_name] > dataframe['shifted_column'], 1, 0)
    dataframe.drop(columns=['shifted_column'], inplace=True)
    return dataframe


def custom_df_to_windowed_df(dataframe, first_date_str, last_date_str, included_columns, n=3):
    """
    Create a windowed dataframe within a specified date range for selected columns in the dataframe,
    with dynamic column naming based on original feature names and their position in the window. The function
    increments the target date by 5 minutes for each step. The 'Target Date' becomes the index of the returned DataFrame.

    Parameters:
    - dataframe: The input DataFrame with a datetime index.
    - first_date_str, last_date_str: The date range for creating the windowed DataFrame, as strings.
    - included_columns: List of column names to be included in the windowed DataFrame.
    - n: The number of previous records to include in each window.

    Returns:
    - A DataFrame where each row contains n previous records of selected data leading up to a target date,
      with columns dynamically named based on their original feature names and window position. The 'Target Date'
      column is used as the index.
    """
    print(f'Creating windowed dataframe from {first_date_str} to {last_date_str}')
    try:
        first_date = pd.to_datetime(first_date_str)
        last_date = pd.to_datetime(last_date_str)
    except ValueError as e:
        print(f"Error converting dates: {e}")
        return pd.DataFrame()

    dataframe.sort_index(inplace=True)  # Ensure the DataFrame's index is sorted

    target_date = first_date

    dates, windowed_data = [], []

    def generate_column_names(features, n):
        column_names = []
        for feature in features:
            for i in range(n, 0, -1):
                column_names.append(f'{feature}_T-{i}')
            column_names.append(f'{feature}_Target')
        return column_names

    while target_date <= last_date:
        df_subset = dataframe.loc[:target_date, included_columns].tail(n + 1)

        if len(df_subset) < n + 1:
            print(f'Warning: Window of size {n} is too large for date {target_date}. Skipping.')
            target_date += timedelta(minutes=5)  # Increment the target date by 5 minutes
            continue

        window = df_subset.to_numpy()
        window_flat = window[:-1].flatten()
        y = window[-1, :]
        windowed_data.append(np.concatenate([window_flat, y]))

        dates.append(target_date)

        target_date += timedelta(minutes=5)  # Increment the target date by 5 minutes

    column_names = generate_column_names(included_columns, n)
    ret_df = pd.DataFrame(windowed_data, columns=column_names)
    # print(create_binary_target_column(ret_df, f'{included_columns[0]}_Target')['target'])
    ret_df['dir'] = create_binary_target_column(ret_df, f'{included_columns[0]}_Target')['target']
    ret_df['Target Date'] = dates
    ret_df.set_index('Target Date', inplace=True)  # Set 'Target Date' as the index

    return ret_df
