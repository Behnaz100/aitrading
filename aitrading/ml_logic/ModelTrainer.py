import json
import pathlib
from pprint import pprint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D
from keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from aitrading import params
from aitrading.ml_logic.preprocessor import split_data
import pandas as pd
import os
from pathlib import Path
from aitrading.ml_logic.preprocessor import custom_df_to_windowed_df, windowed_df_to_date_x_y
from pandas import to_datetime


def create_binary_target_column(dataframe, column_name):
    dataframe['shifted_column'] = dataframe[column_name]
    dataframe['target'] = np.where(dataframe[column_name] > dataframe['shifted_column'], 1, 0)
    dataframe.drop(columns=['shifted_column'], inplace=True)
    return dataframe


class ModelTrainer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelTrainer, cls).__new__(cls)
        return cls._instance

    def __init__(self, dataframe):
        self.update_df = None
        self.history = None
        self.dates_test = None
        self.y_val = None
        self.X_val = None
        self.dates_val = None
        self.y_train = None
        self.X_train = None
        self.dates_train = None
        self.windowed_df = None
        self.model = None
        self.dataframe = dataframe
        self.n = 0
        self.cache = {}
        self.model_cache = {}

    def prepare_data(self, first_date_str, last_date_str, included_columns=None, target_column=None, n=3):
        cache_key = json.dumps((first_date_str, last_date_str, tuple(included_columns) if included_columns else None))
        # params.LOCAL_DATA_PATH
        cache_dir = "data_cache"
        pathlib.Path()
        cache_file_path = os.path.join(params.LOCAL_DATA_PATH,  f"{cache_key}.pkl")
        print(cache_file_path)
        print(cache_key)

        # return
        if cache_key in self.cache:
            print("Using cached data")
            self.dates_train, self.X_train, self.y_train, self.dates_val, self.X_val, self.y_val, self.dates_test, self.X_test, self.y_test = \
                self.cache[cache_key]
        else:
            print("not using cached data")

            if included_columns is None:
                print("✅ using all columns")
                included_columns = self.dataframe.columns.tolist()

            # create_binary_target_column(self.windowed_df, f'{included_columns[0]}_Target')
            self.windowed_df = custom_df_to_windowed_df(self.dataframe, first_date_str, last_date_str, included_columns,
                                                        n)

            self.n = n
            # create_binary_target_column(self.windowed_df, f'{included_columns[0]}_Target')
            (self.dates_train, self.X_train, self.y_train, self.dates_val, self.X_val, self.y_val, self.dates_test,
             self.X_test, self.y_test) = split_data(
                self.windowed_df.index,
                self.windowed_df.iloc[:, 0:-2],
                self.windowed_df.iloc[:, -1],
                # dates, x, y
            )
            self.cache[cache_key] = (
                self.dates_train, self.X_train, self.y_train, self.dates_val, self.X_val, self.y_val, self.dates_test,
                self.X_test, self.y_test)

    def train_model(self, model, epochs=100):
        # model should be compiled before calling this method
        if model is None:
            raise ValueError("Model should be compiled before training")
        model_key = model.to_json()
        if model_key in self.model_cache:
            print("Using cached model")
            self.model = self.model_cache[model_key]
        else:
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=40)
            self.history = model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                                     epochs=epochs, callbacks=[early_stopping])
            self.model = model
            self.model_cache[model_key] = model

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model should be trained before evaluation")
        return self.model.evaluate(self.X_test, self.y_test)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model should be trained before prediction")
        return self.model.predict(X)

    # save model to file
    def save_model(self, file_path=None):
        """Save the trained model to a file."""
        # from tensorflow.python.saved_model import saved_model

        if file_path is None:
            file_path = Path.cwd().parent / "training_outputs" / 'models'
        else:
            file_path = Path(file_path)
            # saved_model.save(self.model, path)
        self.model.save(file_path)
