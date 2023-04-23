# Get the grandparent directory of the current script (../../.py)
import sys
import os
from pathlib import Path
grandparent_dir = str(Path(__file__).resolve().parents[2]) # change to parent 2 (two) levels
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
os.chdir(grandparent_dir)

import src.util as utils
import src.features.preprocessing as preprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import pandas as pd

# Load config
config = utils.load_config()

# Load your models
rf_clf = utils.pickle_load(config['production_model_path'])
knn_clf = utils.pickle_load(config['knn_model_path'])

# Initialize FastAPI app
app = FastAPI()

# Create a Pydantic model for input validation
class SensorData(BaseModel):
    MQ2: int
    MQ3: int
    MQ5: int
    MQ6: int
    MQ7: int
    MQ8: int
    MQ135: int

    # data input validation (range and type) for all (*) fields
    @validator("*")
    def check_range(cls, v):
        if not (config['range_sensor_val'][0] <= v <= config['range_sensor_val'][1]):
            raise ValueError("Value must be integer between 0 and 1000")
        return v


# Define the FastAPI endpoint
@app.post("/predict_gas")
async def predict_gas(data: SensorData):
    # Convert the input data into a 2D array
    input_data = np.array([
        [
            data.MQ2,
            data.MQ3,
            data.MQ5,
            data.MQ6,
            data.MQ7,
            data.MQ8,
            data.MQ135
        ]
    ])

    scaler = utils.pickle_load(config['scaler_path'])
    input_data = scaler.transform(input_data)

    # Make predictions using kNN models (based on our model evaluation performance)

    # rf_prediction = rf_clf.predict(input_data)
    # rf_prob = rf_clf.predict_proba(input_data)

    knn_prediction = knn_clf.predict(input_data)
    knn_prob = knn_clf.predict_proba(input_data)

    # Return the predicted gas type and probability for both models
    return {
        # "rf_prediction": config['encoder_classes'][rf_prediction[0].item()], # item() to convert np.int32 to native python data type
        # "rf_probability": max(rf_prob[0]).item(),
        "knn_prediction": config['encoder_classes'][knn_prediction[0].item()],
        "knn_probability": max(knn_prob[0]).item(),
    }

# Run the app using Uvicorn (only when running the script directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
