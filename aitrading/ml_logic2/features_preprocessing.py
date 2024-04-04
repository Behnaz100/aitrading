import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
import numpy as np

from aitrading import params
from aitrading.ml_logic2 import repository, feature_engineering


def preprocess_features_and_target(X_train, y_train, X_val, y_val, X_test, y_test,
                                   numeric_features=None,
                                   categorical_features=None,
                                   categorical=False):
    """
    Preprocesses features by scaling numerical features and encoding categorical features.
    Also, reshapes the data for neural network models and converts the target variable into categorical format.

    Parameters:
    - X_train, X_val, X_test: pandas DataFrame or numpy array, features for training, validation, and test sets.
    - y_train, y_val, y_test: pandas Series or numpy array, target for training, validation, and test sets.

    Returns:
    - Preprocessed and reshaped features and target variables for training, validation, and test sets.
    """
    # Define numeric and categorical features
    if numeric_features is None:
        numeric_features = [
            'close', 'price_change_5_intervals', 'rolling_avg_price_10_close_intervals', 'close',
            'rolling_avg_price_10_intervals', 'sin_day', 'cos_day', 'ma_1h', 'ema_30min', 'ema_1h', 'rsi'
        ]
    # numeric_features = [
    #     'close', 'price_change_5_intervals', 'rolling_avg_price_10_close_intervals', 'close',
    #     'rolling_avg_price_10_intervals', 'sin_day', 'cos_day', 'ma_1h', 'ema_30min', 'ema_1h', 'rsi'
    # ]
    if categorical_features is None:
        categorical_features = []  # Update this based on your dataset

    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError("One or more input datasets are empty. Please check your data inputs.")

    if not set(numeric_features).issubset(X_train.columns):
        missing_features = set(numeric_features) - set(X_train.columns)
        raise ValueError(f"Missing features in X_train: {missing_features}")

    # Initialize the StandardScaler and OneHotEncoder
    scaler = StandardScaler()
    encoder = OneHotEncoder(drop='first', sparse=False)

    # Scale numeric features
    X_train_num = scaler.fit_transform(X_train[numeric_features])
    X_val_num = scaler.transform(X_val[numeric_features])
    X_test_num = scaler.transform(X_test[numeric_features])

    # save the scaler for later use with timestamp
    repository.saving_scaler(scaler)

    # scaler = joblib.load('scaler.pkl')
    # encoder = joblib.load('encoder.pkl')

    # Encode categorical features
    X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
    repository.saving_scaler(encoder)
    X_val_cat_encoded = encoder.transform(X_val[categorical_features])
    X_test_cat_encoded = encoder.transform(X_test[categorical_features])

    # Concatenate numeric and encoded categorical features
    X_train_preprocessed = np.concatenate((X_train_num, X_train_cat_encoded), axis=1)
    X_val_preprocessed = np.concatenate((X_val_num, X_val_cat_encoded), axis=1)
    X_test_preprocessed = np.concatenate((X_test_num, X_test_cat_encoded), axis=1)

    # Reshape for neural network compatibility
    X_train_reshaped = X_train_preprocessed.reshape((X_train_preprocessed.shape[0], 1, X_train_preprocessed.shape[1]))
    X_val_reshaped = X_val_preprocessed.reshape((X_val_preprocessed.shape[0], 1, X_val_preprocessed.shape[1]))
    X_test_reshaped = X_test_preprocessed.reshape((X_test_preprocessed.shape[0], 1, X_test_preprocessed.shape[1]))

    # Convert target variables to categorical
    if categorical:
        y_train_categorical = to_categorical(y_train)
        y_val_categorical = to_categorical(y_val)
        y_test_categorical = to_categorical(y_test)
    else:
        y_val_categorical = y_val
        y_train_categorical = y_train
        y_test_categorical = y_test

    # Print dataset sizes
    print(f"Training set size: {len(X_train_reshaped)}")
    print(f"Validation set size: {len(X_val_reshaped)}")
    print(f"Test set size: {len(X_test_reshaped)}")

    return X_train_reshaped, y_train_categorical, X_val_reshaped, y_val_categorical, X_test_reshaped, y_test_categorical


def preprocess_predict_feature(df: pd.DataFrame, scaler, encoder, numeric_features=None,
                               categorical_features=None) -> np.ndarray:
    if df.empty:
        raise ValueError("Input dataset is empty. Please check your data inputs.")

    if numeric_features is None:
        numeric_features = [
            'close', 'price_change_5_intervals', 'rolling_avg_price_10_close_intervals', 'close',
            'rolling_avg_price_10_intervals', 'sin_day', 'cos_day', 'ma_1h', 'ema_30min', 'ema_1h', 'rsi'
        ]
    if categorical_features is None:
        categorical_features = []  # Update this based on your dataset

    on_hour_final_df = feature_engineering.enrich_dataframe(df=df, predicate=True, datetime_col='date')

    # Scale numeric features
    X_predict_num = scaler.transform(on_hour_final_df[numeric_features])

    # Encode categorical features
    X_predict_cat_encoded = encoder.transform(on_hour_final_df[categorical_features])

    # Concatenate numeric and encoded categorical features
    X_train_preprocessed = np.concatenate((X_predict_num, X_predict_cat_encoded), axis=1)

    # Reshape for neural network compatibility
    X_train_reshaped = X_train_preprocessed.reshape((X_train_preprocessed.shape[0], 1, X_train_preprocessed.shape[1]))

    return X_train_reshaped
