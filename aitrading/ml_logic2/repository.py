import os
import pathlib
import time

import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow import keras
from keras.models import load_model
from aitrading import params
import glob


# from aitrading.params import *

def save_model_default_path(model2: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """
    if model2 is None:
        print("❎ No model to save")
        return None

    print("✅ Saving model...")
    base_model_path = pathlib.Path(params.LOCAL_PATH) / "models"
    base_model_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"model_{timestamp}.keras"

    model_path = pathlib.Path(base_model_path / model_filename)

    # model.save(model_path)
    # save_model(model, model_path)
    model2.save(model_path)
    print(f"✅ Model saved at: {model_path}")


def load_latest_model(base_path=params.LOCAL_PATH) -> keras.Model:
    """
    Loads the latest keras model based on its timestamp from a specified base path.

    Parameters:
    - base_path: str or pathlib.Path, the base directory where the models are stored.

    Returns:
    - A loaded keras.Model object if found, None otherwise.
    """
    # Convert base_path to Path object if it's a str
    base_path = os.path.join(base_path, "models") if isinstance(base_path, str) else base_path / "models"

    # Pattern to match model files
    pattern = os.path.join(base_path, "model_*.keras")

    # Find all model files matching the pattern
    model_files = glob.glob(pattern)
    if not model_files:
        print("No model files found.")
        return None

    # Sort the files by their modification time (latest first)
    latest_model_file = max(model_files, key=os.path.getmtime)
    print(f"✅ Loading latest model: {latest_model_file}")

    return load_model(latest_model_file)


def saving_scaler(scaler: StandardScaler | OneHotEncoder, base_path=params.LOCAL_PATH):
    if scaler is None:
        print("❎ No scaler to save")
        return None

    if not isinstance(scaler, StandardScaler) and not isinstance(scaler, OneHotEncoder):
        print("❎ Invalid type")
        return None

    base_model_path = pathlib.Path(params.LOCAL_PATH) / "models"
    scaler_filename = None
    if isinstance(scaler, OneHotEncoder):
        print("✅ Saving OneHotEncoder ")
        base_model_path = base_model_path / "encoder"
        scaler_filename = f"encoder"

    if isinstance(scaler, StandardScaler):
        print("✅ Saving StandardScaler ")
        base_model_path = base_model_path / "scaler"
        scaler_filename = f"scaler"

    base_model_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    scaler_filename += f"_{timestamp}"
    scaler_path = pathlib.Path(base_model_path / scaler_filename)
    joblib.dump(scaler, scaler_path)
    print(f"✅ scaler saved at: {scaler_path}")


def load_scaler_or_encoder(type: str, base_path=params.LOCAL_PATH) -> StandardScaler | OneHotEncoder | None:
    """
    Loads a StandardScaler or OneHotEncoder object from the specified file path.

    Parameters:
    - file_path: str - The path to the saved scaler or encoder file.

    Returns:
    - The loaded StandardScaler or OneHotEncoder object, or None if the file doesn't exist.
    """

    # if not file_path.exists():
    #     print("❎ The specified file does not exist.")
    #     return None
    base_scaler_path2 = os.path.join(base_path, "models") if isinstance(base_path, str) else base_path / "models"
    base_scaler_path = pathlib.Path(base_scaler_path2)
    scaler_filename = None
    pattern = None
    if type == "encoder":
        print("✅ loading OneHotEncoder ")
        base_model_path2 = base_scaler_path / "encoder"
        scaler_filename = f"encoder"
        pattern = os.path.join(base_model_path2, "encoder_*")

    if type == "scaler":
        print("✅ loading StandardScaler ")
        base_model_path2 = base_scaler_path / "scaler"
        scaler_filename = f"scaler"
        pattern = os.path.join(base_model_path2, "scaler_*")

    scaler_files = glob.glob(pattern)
    if not scaler_files:
        print("No scaler or encoder files found.")
        return None

    try:
        latest_scaler_file = max(scaler_files, key=os.path.getmtime)
        print(f"✅ Loading latest scaler/encoder: {latest_scaler_file}")
        scaler_or_encoder = joblib.load(latest_scaler_file)
        if isinstance(scaler_or_encoder, StandardScaler):
            print("✅ StandardScaler loaded successfully.")
        elif isinstance(scaler_or_encoder, OneHotEncoder):
            print("✅ OneHotEncoder loaded successfully.")
        else:
            print("❎ Loaded object is not a StandardScaler or OneHotEncoder.")
            return None
        return scaler_or_encoder
    except Exception as e:
        print(f"❎ An error occurred while loading the scaler/encoder: {e}")
        return None
