"""
api.py — FastAPI microservice for real-time and batch anomaly detection.

Exposes:
  • /health          — health check
  • /metadata        — model + pipeline metadata
  • /predict         — real-time anomaly detection (single record)
  • /predict_batch   — batch anomaly detection (array of records)

Uses:
  • UnifiedPreprocessor — for preprocessing API input
  • Pre-trained ML models (random_forest.pkl, logistic_regression.pkl)

Author: Dennis Fashimpaur
"""

import uvicorn
from fastapi import FastAPI
from pandas import DataFrame
from pydantic import BaseModel
import joblib
from typing import List, Optional

from src.Preprocessors.unified_preprocessor import UnifiedPreprocessor

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(
    title="Anomaly Detection API",
    description="Real-time microservice for unified transaction anomaly detection.",
    version="1.0.0"
)

# -------------------------------
# Load preprocessor and models
# -------------------------------
print("🔧 Loading preprocessing pipeline and models...")

preprocessor = UnifiedPreprocessor()

MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/logistic_regression.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ API Startup Complete.")

# -------------------------------
# Pydantic schemas
# -------------------------------
class Transaction(BaseModel):
    timestamp: Optional[str] = None
    customer_id: Optional[int] = None
    Amount: Optional[float] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    status: Optional[int] = None


class BatchRequest(BaseModel):
    records: List[Transaction]


# -------------------------------
# Helper function
# -------------------------------
def preprocess_input(records: List[Transaction]) -> DataFrame:
    """
    Convert API payload → DataFrame → unified preprocessing.
    """
    df = DataFrame([r.dict() for r in records])
    df_processed = preprocessor.preprocess_runtime(df)
    return df_processed


# -------------------------------
# Routes
# -------------------------------
# -----------------------------------------------------
# Root Landing Page
# -----------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to the Anomaly Detection API!",
        "documentation": "/docs",
        "metadata": "/metadata"
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Anomaly Detection API is running."}


@app.get("/metadata")
def metadata():
    return {
        "model_type": type(model).__name__,
        "scaler_type": type(scaler).__name__,
        "preprocessor": "UnifiedPreprocessor",
        "description": "API for unified transaction anomaly detection pipeline."
    }


# -------------------------------
# Single prediction
# -------------------------------
@app.post("/predict")
def predict(record: Transaction):
    df = preprocess_input([record])
    pred = model.predict(df)[0]
    score = model.decision_function(df)[0]

    return {
        "prediction": int(pred),
        "anomaly_score": float(score)
    }


# -------------------------------
# Batch prediction
# -------------------------------
@app.post("/predict_batch")
def predict_batch(batch: BatchRequest):
    df = preprocess_input(batch.records)
    preds = model.predict(df)
    scores = model.decision_function(df)

    return {
        "count": len(preds),
        "predictions": preds.tolist(),
        "anomaly_scores": scores.tolist()
    }


# -------------------------------
# Local development entry point
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api:app",  # points to this file
        host="0.0.0.0",
        port=8000,
        reload=True
    )
