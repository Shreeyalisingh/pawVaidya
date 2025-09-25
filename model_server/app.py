from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
import onnxruntime as ort
import numpy as np
import json
import os

# ---------- Config & Globals ----------
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/diabetes_prediction")
PORT = int(os.getenv("PORT", "5000"))

app = FastAPI(title="Diabetes Prediction API (FastAPI)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

client: MongoClient = None
db = None
predictions_col = None

session: ort.InferenceSession = None
model_info = {}

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "diabetes_model.onnx"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

# ---------- Schemas ----------
class PredictIn(BaseModel):
    pregnancies: Optional[float] = 0
    glucose: float = Field(..., description="Required")
    bloodPressure: Optional[float] = 0
    skinThickness: Optional[float] = 0
    insulin: Optional[float] = 0
    bmi: float = Field(..., description="Required")
    diabetesPedigreeFunction: Optional[float] = 0
    age: float = Field(..., description="Required")

class PredictOut(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    accuracy: Optional[float] = None

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
def on_startup():
    global client, db, predictions_col, session, model_info

    # Mongo
    client = MongoClient(MONGODB_URI)

    # Use default DB if URI includes one, else explicit name
    try:
        default_db = client.get_default_database()  # works only if URI has /dbname
    except Exception:
        default_db = None

    db = default_db if default_db is not None else client["diabetes_prediction"]
    predictions_col = db["predictions"]
    predictions_col.create_index([("timestamp", DESCENDING)])

    # --- ONNX model loading (your current code) ---
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    if not MODEL_INFO_PATH.exists():
        raise RuntimeError(f"model_info.json not found at: {MODEL_INFO_PATH}")

    session = ort.InferenceSession(str(MODEL_PATH))
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        model_info = json.load(f)
@app.on_event("shutdown")
def on_shutdown():
    if client:
        client.close()

# ---------- Helpers ----------
def scale_inputs(raw: List[float]) -> np.ndarray:
    means = np.array(model_info["scaler_mean"], dtype=np.float32)
    scales = np.array(model_info["scaler_scale"], dtype=np.float32)
    x = np.array(raw, dtype=np.float32)
    return (x - means) / scales

def run_inference(scaled_row: np.ndarray) -> (int, float):
    # Build input tensor
    input_name = session.get_inputs()[0].name
    input_tensor = scaled_row.astype(np.float32)[None, :]  # shape (1, N)

    results = session.run(None, {input_name: input_tensor})

    # Try to interpret common output forms
    out = results[0]
    if hasattr(out, "tolist"):
        out = np.array(out)
    data = out.ravel()

    if data.size == 1:
        # Could be a score/logit or 0/1 directly
        val = float(data[0])
        prob = 1.0 / (1.0 + np.exp(-val)) if val > 1 else max(0.0, min(1.0, val))
        pred = 1 if prob > 0.5 else 0
    elif data.size == 2:
        # Probabilities [p0, p1]
        prob = float(data[1])
        pred = 1 if prob > 0.5 else 0
    else:
        # Fallback: take first value as score
        val = float(data[0])
        prob = 1.0 / (1.0 + np.exp(-val)) if val > 1 else max(0.0, min(1.0, val))
        pred = 1 if prob > 0.5 else 0

    return int(pred), float(prob)

def risk_bucket(p: float) -> str:
    return "High" if p > 0.7 else "Medium" if p > 0.3 else "Low"

# ---------- Routes ----------
@app.get("/api/health")
def health():
    return {
        "status": "OK",
        "model_loaded": session is not None,
        "features": model_info.get("feature_names", []),
    }

@app.post("/api/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    # Validate required fields (glucose, bmi, age) via Pydantic already
    # Order must match training:
    raw = [
        payload.pregnancies or 0,
        payload.glucose,
        payload.bloodPressure or 0,
        payload.skinThickness or 0,
        payload.insulin or 0,
        payload.bmi,
        payload.diabetesPedigreeFunction or 0,
        payload.age,
    ]

    try:
        scaled = scale_inputs(raw)
        pred, prob = run_inference(scaled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {e}")

    doc = {
        "inputs": payload.dict(),
        "prediction": pred,
        "probability": prob,
        "timestamp": datetime.utcnow(),
    }
    try:
        predictions_col.insert_one(doc)
    except Exception:
        # Don’t fail the API if logging fails
        pass

    return {
        "prediction": pred,
        "probability": prob,
        "risk_level": risk_bucket(prob),
        "accuracy": model_info.get("accuracy"),
    }

@app.get("/api/predictions")
def get_predictions():
    try:
        cur = predictions_col.find({}, {"_id": False}).sort("timestamp", DESCENDING).limit(50)
        return list(cur)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
