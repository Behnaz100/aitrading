import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

from keras.src.callbacks import EarlyStopping
from keras.src.layers import BatchNormalization, Bidirectional
from tensorflow.python.keras.regularizers import l2

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras import Model
from tensorflow.python.keras import optimizers
from typing import Tuple

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    # check if the input shape is a tuple and has 2 dimensions (rows, columns) and
    # second element is 1

    if not isinstance(input_shape, tuple) or len(input_shape) != 2 or input_shape[1] != 1:
        raise ValueError(f"Input shape must be a tuple with 2 dimensions and second element 1, got {input_shape}")

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',  # Use binary crossentropy for binary classification
                  metrics=['accuracy'])  # Use accuracy as a metric for classification tasks

    print("✅ Model compiled")

    return model


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None,  # overrides validation_split
        validation_split=0.3
) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics


