"""
Production REST API for Heart Disease Risk Prediction
Tasks 6 & 8 of AIMLCZG523 MLOps Assignment 01

Features:
- Built with FastAPI
- Robust Pydantic request validation
- /health, /predict, /predict/batch, and Prometheus /metrics endpoints
- Structured request logging with audit trail
- Prometheus instrumentation for latency, throughput, and error rates
"""

import os
import time
import uuid
import logging
import sys
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import joblib
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Set up structured logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "api_requests.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("heart_disease_api")

# Custom Prometheus business metrics
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total number of heart disease model predictions classified",
    ["risk_category"]
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Time spent in model inference pipeline",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Pydantic Schemas
class HeartPatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Patient age in years (1-120)", json_schema_extra={"example": 58})
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)", json_schema_extra={"example": 1})
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic)", json_schema_extra={"example": 2})
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure in mm Hg (50-260)", json_schema_extra={"example": 135.0})
    chol: float = Field(..., ge=50, le=650, description="Serum cholestoral in mg/dl (50-650)", json_schema_extra={"example": 240.0})
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", json_schema_extra={"example": 0})
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: hypertrophy)", json_schema_extra={"example": 1})
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved in bpm (40-250)", json_schema_extra={"example": 160.0})
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)", json_schema_extra={"example": 0})
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest", json_schema_extra={"example": 1.2})
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0, 1, 2)", json_schema_extra={"example": 1})
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3) colored by flourosopy", json_schema_extra={"example": 0})
    thal: int = Field(..., ge=1, le=3, description="Thalassemia (1 = normal, 2 = fixed defect, 3 = reversable defect)", json_schema_extra={"example": 2})

class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    prediction: int
    risk_label: str
    confidence: float
    probability_distribution: dict
    model_version: str

# Model Loading Helper
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.joblib")
)
model_pipeline = None

def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Model successfully loaded from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
            model_pipeline = None
    else:
        logger.warning(f"Model artifact not found at {MODEL_PATH}. Prediction endpoint will return 503 until trained.")

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    load_model()
    yield

# Application Factory
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-grade MLOps Serving API with Prometheus Monitoring and Automated Inference Pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Prometheus Instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        f"req_id={request_id} client={request.client.host if request.client else 'unknown'} "
        f"method={request.method} path={request.url.path} status={response.status_code} "
        f"duration_ms={duration_ms}ms"
    )
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Heart Disease Risk Prediction MLOps API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "metrics_endpoint": "/metrics"
    }

@app.get("/health", tags=["Monitoring"])
def health_check():
    if model_pipeline is None:
        # Try loading once more
        load_model()
    is_ready = model_pipeline is not None
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "HEALTHY" if is_ready else "DEGRADED",
            "model_loaded": is_ready,
            "model_path": MODEL_PATH,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: HeartPatientInput):
    global model_pipeline
    if model_pipeline is None:
        load_model()
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet initialized or loaded."
        )

    req_id = str(uuid.uuid4())
    start_time = time.time()

    # Convert to DataFrame matching training schema
    input_dict = patient.model_dump()
    df_input = pd.DataFrame([input_dict])

    try:
        # Run inference
        pred = int(model_pipeline.predict(df_input)[0])
        probas = model_pipeline.predict_proba(df_input)[0]
        confidence = float(probas[pred])
        prob_dist = {
            "no_disease": round(float(probas[0]), 4),
            "disease_detected": round(float(probas[1]), 4)
        }
        latency = time.time() - start_time

        # Update business metrics
        risk_category = "High Risk" if pred == 1 else "Low Risk"
        PREDICTION_COUNTER.labels(risk_category=risk_category).inc()
        INFERENCE_LATENCY.observe(latency)

        logger.info(f"prediction_success req_id={req_id} risk={risk_category} confidence={confidence:.4f}")

        return PredictionResponse(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prediction=pred,
            risk_label=risk_category,
            confidence=round(confidence, 4),
            probability_distribution=prob_dist,
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Inference error req_id={req_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
def predict_batch(patients: List[HeartPatientInput]):
    if not patients:
        raise HTTPException(status_code=400, detail="Empty patient batch provided.")
    results = [predict(p) for p in patients]
    return results
