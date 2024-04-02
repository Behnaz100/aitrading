import requests
import pandas as pd


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
    print(f"start {start}, now {now}")

    url = f"https://marketdata.tradermade.com/api/v1/timeseries?currency=EURUSD&api_key=zNv02V5PnDlZwEQypn1j&start_date={start}&end_date={now}&format=records&interval=minute&period=5"
    res = requests.get(url).json()
    return pd.DataFrame(res['quotes'])


def load_data_source(path: str) -> pd.DataFrame:
    # return pd.read_csv("/Users/kassraniroumand/code/aitrading/aitrading/data/eurousd_df_clean_2.csv")
    return pd.read_csv(path)
