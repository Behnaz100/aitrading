import glob
import os
import pathlib
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras

from aitrading import params
from aitrading.params import *


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
    with open(params_path, "wb") as file:
        pickle.dump(params, file)

    # Save metrics locally

    metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
    with open(metrics_path, "wb") as file:
        pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    # local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    # local_model_paths = glob.glob(f"{local_model_directory}/*")
    #
    # if not local_model_paths:
    #     return None
    #
    # most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
    #
    # print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    # Load the model from disk
    # model_path = os.path.join(Path.cwd() / "training_outputs" / 'models' / 'model2.keras')
    # / Users / kassraniroumand / code / aitrading / aitrading / training_outputs / models / model2.keras
    model_path = pathlib.Path(Path.cwd() / "aitrading" / 'models' / 'model2.keras')
    assert os.path.isfile(model_path)

    # print("os.getcwd()",model_path)
    # print("--->", pathlib.Path(Path.cwd()) / "training_outputs" / 'models' / 'model2.keras')

    if not model_path.exists():
        print(Fore.RED + f"❌ No model found in {model_path}" + Style.RESET_ALL)
        return None

    # print("model_path", model_path)
    print(f"✅ Model found in local {model_path}")
    with open(model_path, "rb") as file_:
        if file_:
            latest_model = keras.models.load_model(model_path)
            print("latest_model", latest_model)
            print("✅ Model loaded from local disk")
            return latest_model
    # latest_model = keras.models.load_model(model_path)

    # print("latest_model", latest_model)
    # print("✅ Model loaded from local disk")
    return None
