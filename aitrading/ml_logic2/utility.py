import requests
import pandas as pd

from aitrading import params


def calculate_rsi(data, window=5):
    """
    Calculates the Relative Strength Index (RSI) for the given data.

    Parameters:
    - data: pandas Series of price data.
    - window: int, the period over which to calculate RSI.

    Returns:
    - pandas Series with the RSI.
    """
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def data_tradermade_source():
    import datetime
    now = datetime.datetime.now() - datetime.timedelta(hours=1)
    now = now.strftime("%Y-%m-%d-%H:%M")
    # compute for last 50 minutes
    start = datetime.datetime.now() - datetime.timedelta(hours=2)
    start = start.strftime("%Y-%m-%d-%H:%M")

    print(f"➡️ request data from  {start} to {now}")

    if params.api_key is None:
        raise ValueError("Please set the API key in the environment")

    if params.api_key == "should_be_replaced_with_your_api_key":
        raise ValueError("Please set the API key in the environment")

    parameters = {
        'currency': 'EURUSD',
        'api_key': f"{params.api_key}",
        'start_date': f"{start}",
        'end_date': f"{now}",
        'format': 'records',
        'interval': 'minute',
        'period': 5
    }

    url = f"https://marketdata.tradermade.com/api/v1/timeseries"
    res = requests.get(url, params= parameters).json()
    df = pd.DataFrame(res['quotes'])
    print(f"➡️ data_tradermade_source {len(df)}")
    return pd.DataFrame(df)


def load_data_source(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
