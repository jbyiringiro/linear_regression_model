# prediction.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

app = FastAPI(title="Wheelchair Battery Prediction API",
              description="Predict battery energy (Wh) for personalized powered wheelchair.")

# CORS - allow all origins for demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler from same directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model or scaler file missing - put best_model.pkl and scaler.pkl in the API folder.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Input schema with types and ranges (modify ranges realistically)
class PredictRequest(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Age in years (1-120)")
    Height: float = Field(..., ge=0.5, le=2.5, description="Height in meters (0.5-2.5)")
    Weight: float = Field(..., ge=2.0, le=300.0, description="Weight in kg (2-300)")
    Bmi: float = Field(..., ge=10.0, le=80.0, description="BMI (10-80)")
    daily_distance_km: float = Field(..., ge=0.0, le=50.0, description="Daily distance in km (0-50)")
    terrain_factor: float = Field(..., ge=1.0, le=2.5, description="Terrain factor (1.0 flat - 1.6 rough)")

class PredictResponse(BaseModel):
    predicted_battery_Wh: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Create numpy array and scale
    x = np.array([[req.Age, req.Height, req.Weight, req.Bmi, req.daily_distance_km, req.terrain_factor]])
    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling error: {e}")

    pred = model.predict(x_scaled)[0]
    return {"predicted_battery_Wh": float(pred)}