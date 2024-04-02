import os
import pathlib
import time
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
