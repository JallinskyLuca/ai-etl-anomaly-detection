"""
api.py — FastAPI microservice for real-time and batch anomaly detection.

This service exposes:
  • /health          — health check
  • /metadata        — model + pipeline metadata
  • /predict         — real-time anomaly detection (single record)
  • /predict_batch   — batch anomaly detection (array of records)

It loads:
  • UnifiedPreprocessor — used for preprocessing incoming data
  • Trained ML model    — saved during Day 5 training

The API is aligned with the project's README description:
  - End-to-end ETL pipeline
  - Data cleaning, feature engineering, scaling
  - Unified dataset structure
  - Anomaly prediction via ML model

Author: Dennis Fashimpaur
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Optional

from src.Preprocessors.unified_preprocessor import UnifiedPreprocessor

# -----------------------------------------------------
# Initialize FastAPI App
# -----------------------------------------------------
app = FastAPI(
    title="Anomaly Detection API",
    description="Real-time microservice for unified transaction anomaly detection.",
    version="1.0.0"
)

# -----------------------------------------------------
# Load Preprocessor + Model
# -----------------------------------------------------
print("🔧 Loading preprocessing pipeline and model...")

preprocessor = UnifiedPreprocessor()

MODEL_PATH = "models/trained_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ API Startup Complete.")


# -----------------------------------------------------
# Pydantic Request Schemas
# -----------------------------------------------------
class Transaction(BaseModel):
    timestamp: Optional[str] = None
    customer_id: Optional[int] = None
    Amount: Optional[float] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    status: Optional[int] = None


class BatchRequest(BaseModel):
    records: List[Transaction]


# -----------------------------------------------------
# Helper Function
# -----------------------------------------------------
def preprocess_input(records: List[Transaction]) -> pd.DataFrame:
    """
    Convert request payload → DataFrame → unified preprocessing.
    """
    df = pd.DataFrame([r.dict() for r in records])
    df_processed = preprocessor.preprocess_runtime(df)
    return df_processed


# -----------------------------------------------------
# Routes
# -----------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Anomaly Detection API is running."}


@app.get("/metadata")
def metadata():
    return {
        "model_type": str(type(model)),
        "scaler_type": str(type(scaler)),
        "preprocessor": "UnifiedPreprocessor",
        "description": "API for unified transaction anomaly detection pipeline."
    }


# ------------------ Single Prediction ------------------
@app.post("/predict")
def predict(record: Transaction):
    df = preprocess_input([record])
    pred = model.predict(df)[0]
    score = model.decision_function(df)[0]

    return {
        "prediction": int(pred),
        "anomaly_score": float(score)
    }


# ------------------ Batch Prediction -------------------
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


# -----------------------------------------------------
# Local Development Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
