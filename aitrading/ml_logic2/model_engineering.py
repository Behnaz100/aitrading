from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, GRU, Dense, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, min_lr=0.0001)
adam = Adam(learning_rate=0.001)

def create_compile_model(input_shape):
    """
    Creates and compiles a neural network model.

    Parameters:
    - input_shape: tuple, shape of the input data (num_timesteps, num_features).

    Returns:
    - Compiled model ready for training.
    """
    model = Sequential([
        # Input(shape=),
        Bidirectional(LSTM(256, return_sequences=True, activation='tanh',input_shape=input_shape)),
        Dropout(0.5),
        GRU(128, return_sequences=False, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification tasks
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Trains the neural network model with the given training and validation data.

    Parameters:
    - model: The compiled neural network model to be trained.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - class_weights_dict: Dictionary defining weights for each class for handling imbalanced data.

    Returns:
    - history: History object containing training and validation metrics.
    """

    history = model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        # callbacks=[early_stopping, reduce_lr]
    )

    return history


