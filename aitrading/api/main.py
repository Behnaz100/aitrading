import os
from pathlib import Path
import tradermade as tm

import pandas as pd

from aitrading import params
# $WIPE_BEGIN

from aitrading.ml_logic.registry import load_model
# $WIPE_END

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

BASE_PATH = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_PATH / "templates"))

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
app.state.model = load_model()
# $WIPE_END


# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
        pickup_datetime: str,  # 2013-07-06 17:18:00
        pickup_longitude: float,    # -73.950655
        pickup_latitude: float,     # 40.783282
        dropoff_longitude: float,   # -73.984365
        dropoff_latitude: float,    # 40.769802
        passenger_count: int
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # $CHA_BEGIN

    # 💡 Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    # X_pred = pd.DataFrame(locals(), index=[0])
    #
    # # Convert to US/Eastern TZ-aware!
    # X_pred['pickup_datetime'] = pd.Timestamp(pickup_datetime, tz='US/Eastern')
    #
    # model = app.state.model
    # assert model is not None
    #
    # X_processed = preprocess_features(X_pred)
    # y_pred = model.predict(X_processed)
    #
    # # ⚠️ fastapi only accepts simple Python data types as a return value
    # # among them dict, list, str, int, float, bool
    # # in order to be able to convert the api response to JSON
    # return dict(fare_amount=float(y_pred))
    # $CHA_END
    return 1

@app.get("/")
async def root(request: Request):
    tm.set_rest_api_key(params.api_key)
    # get time in 2024-03-13-00:00 format and 5 days back

    today = pd.Timestamp.now().strftime("%Y-%m-%d-00:00")
    five_days_back = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d-00:00")

    hs = tm.timeseries(currency='EURUSD', start=five_days_back, end=today, interval="daily",
                  fields=["close"])  # returns timeseries data for the currency requested interval is daily, hourly, minute - fields is optional

    hs_lst = hs.to_dict('records')
    prediction = 1

    # $CHA_BEGIN
    return templates.TemplateResponse("index.html", {"request": request, "name": "ai-trading", "history_data": hs.to_dict('records'), "prediction": prediction})
    # $CHA_END
