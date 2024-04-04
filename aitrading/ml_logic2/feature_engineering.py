import pandas as pd
import numpy as np

from aitrading.ml_logic2.utility import calculate_rsi


def enrich_dataframe(df: pd.DataFrame, predicate: bool = False, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Enriches the input DataFrame with financial analytics features. Validates the presence of required columns
    including a dynamic datetime column, 'close', and 'low' before proceeding. Adds time-based features, moving
    averages, price change, RSI, and more.

    Parameters:
    - df: pandas DataFrame.
    - datetime_col: str, name of the column in df that contains datetime information.

    Returns:
    - DataFrame enriched with additional financial features if validation passes. Otherwise, raises ValueError.
    """
    # Validation: Check for necessary columns
    required_columns = [datetime_col, 'close', 'low']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    preprocessing_df = df.copy()
    # Convert datetime column to pandas Timestamp and sort the DataFrame
    preprocessing_df['timestamp'] = pd.to_datetime(df[datetime_col])
    preprocessing_df.sort_values(by='timestamp', inplace=True)

    # Calculate the target variable
    if not predicate:
        preprocessing_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Time-based features
    preprocessing_df['hour'] = preprocessing_df['timestamp'].dt.hour
    # preprocessing_df['part_of_day'] = preprocessing_df.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
    preprocessing_df['day_of_week'] = preprocessing_df['timestamp'].dt.dayofweek
    preprocessing_df['sin_day'] = np.sin(2 * np.pi * preprocessing_df['day_of_week'] / 7)
    preprocessing_df['cos_day'] = np.cos(2 * np.pi * preprocessing_df['day_of_week'] / 7)

    # Price change and rolling window features
    preprocessing_df['price_change_5_intervals'] = df['close'].diff(periods=5)
    preprocessing_df['rolling_avg_price_10_intervals'] = df['close'].rolling(window=5).mean()
    preprocessing_df['rolling_avg_price_10_close_intervals'] = df['low'].rolling(window=5).mean()

    # Moving averages
    preprocessing_df['ma_30m'] = df['close'].rolling(window=5).mean()
    preprocessing_df['ma_1h'] = df['close'].rolling(window=12).mean()

    # Exponential moving averages
    preprocessing_df['ema_30min'] = df['close'].ewm(span=6, adjust=False).mean()
    preprocessing_df['ema_1h'] = df['close'].ewm(span=288, adjust=False).mean()

    # Relative Strength Index (RSI)
    preprocessing_df['rsi'] = calculate_rsi(df['close'], window=5)

    # Drop rows with NaN values
    preprocessing_df.dropna(inplace=True)

    return preprocessing_df