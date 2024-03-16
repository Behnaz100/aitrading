import os
from pathlib import Path
import numpy as np

##################  VARIABLES  ##################
api_key=os.environ.get("api_key")


##################  CONSTANTS  #####################
LOCAL_DATA_PATH = Path(os.getcwd()).parent / "data"
LOCAL_REGISTRY_PATH = Path(os.getcwd()).parent / "training_outputs"



################## VALIDATIONS #################

