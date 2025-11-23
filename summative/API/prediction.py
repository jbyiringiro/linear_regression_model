# prediction.py (place inside summative/API)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

app = FastAPI(title="Wheelchair Battery Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# If model/scaler are placed elsewhere, update the path accordingly.
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("best_model.pkl or scaler.pkl missing in the API folder. Place them here before running the server.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class PredictRequest(BaseModel):
    Age: int = Field(..., ge=1, le=120)
    Height: float = Field(..., ge=0.5, le=2.5)
    Weight: float = Field(..., ge=2.0, le=300.0)
    Bmi: float = Field(..., ge=10.0, le=80.0)
    daily_distance_km: float = Field(..., ge=0.0, le=50.0)
    terrain_factor: float = Field(..., ge=1.0, le=2.5)

class PredictResponse(BaseModel):
    predicted_battery_Wh: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array([[req.Age, req.Height, req.Weight, req.Bmi, req.daily_distance_km, req.terrain_factor]])
    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling error: {e}")
    pred = model.predict(x_scaled)[0]
    return {"predicted_battery_Wh": float(pred)}