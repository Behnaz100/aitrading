import pandas as pd
from pathlib import Path


def get_data_from_local(file_name) -> pd.DataFrame:
    # if path exists, read from path
    file_path = Path.cwd().parent / 'data' / file_name

    if Path(file_path).exists():
        return pd.read_csv(file_path, index_col="datetime", parse_dates=True)
    else:
        raise FileNotFoundError(f'File {file_path} not found')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)

    # Remove buggy transactions

    # Remove geographically irrelevant transactions (rows)


    return df