"""
Production Model Serving API with FastAPI.
Implements /predict, /predict/batch, /health, and Prometheus /metrics endpoints.
Includes structured logging, latency profiling, and error handling.
"""

import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import (
    HeartDiseaseInput, PredictionResponse, 
    BatchHeartDiseaseInput, BatchPredictionResponse,
    HealthResponse
)
from app.logger import logger, log_prediction

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_disease_pipeline.joblib")
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-ready MLOps API for clinical heart disease risk classification (BITS Pilani AIMLCZG523).",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend / dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model container
model_pipeline = None


def load_or_initialize_model():
    """Load serialized scikit-learn pipeline or initialize if absent."""
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info(f"[+] Loaded production model pipeline from {MODEL_PATH}")
            return model_pipeline
        except Exception as e:
            logger.error(f"[-] Error loading model file: {e}")
    
    # Auto-train on startup if model artifact is not yet generated
    logger.warning("[!] Model artifact not found. Triggering baseline training...")
    try:
        from src.train import train_and_track
        model_pipeline, _ = train_and_track()
        logger.info("[+] Baseline training finished successfully on startup.")
    except Exception as e:
        logger.error(f"[-] Could not auto-train pipeline: {e}")
        # Build basic fallback pipeline
        from src.feature_pipeline import build_model_pipeline
        from sklearn.ensemble import RandomForestClassifier
        model_pipeline = build_model_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
        from src.data_loader import get_train_test_data
        X_train, _, y_train, _ = get_train_test_data()
        model_pipeline.fit(X_train, y_train)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model_pipeline, MODEL_PATH)
        logger.info("[+] Fallback model initialized and saved.")
    return model_pipeline


@app.on_event("startup")
def startup_event():
    """Application startup hook to load ML model and initialize Prometheus instrumentator."""
    load_or_initialize_model()
    # Expose Prometheus /metrics endpoint
    Instrumentator().instrument(app).expose(app)
    logger.info("[+] Prometheus metrics instrumented and exposed at /metrics")


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Heart Disease Prediction Service is operational.",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Liveness and readiness probe endpoint for Kubernetes / Docker."""
    is_loaded = model_pipeline is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        service="heart-disease-prediction-api",
        model_loaded=is_loaded,
        model_version=MODEL_VERSION
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_heart_disease(patient: HeartDiseaseInput, request: Request):
    """
    Predict heart disease risk for an individual patient.
    Accepts clinical patient features and returns binary risk, confidence score, and risk category.
    """
    global model_pipeline
    if model_pipeline is None:
        load_or_initialize_model()
        if model_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model pipeline is currently not loaded."
            )
            
    start_time = time.time()
    input_dict = patient.model_dump()
    df_input = pd.DataFrame([input_dict])
    
    try:
        pred_label_raw = model_pipeline.predict(df_input)[0]
        prob_score = model_pipeline.predict_proba(df_input)[0][1] if hasattr(model_pipeline, "predict_proba") else float(pred_label_raw)
    except Exception as e:
        logger.error(f"[-] Inference computation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
        
    duration_ms = (time.time() - start_time) * 1000.0
    pred_int = int(pred_label_raw)
    
    # Clinical risk classification
    if prob_score >= 0.70:
        risk_level = "High"
        label_str = "High Risk - Presence of Heart Disease"
    elif prob_score >= 0.40:
        risk_level = "Moderate"
        label_str = "Moderate Risk - Borderline Findings"
    else:
        risk_level = "Low"
        label_str = "Low Risk - Absence of Heart Disease"

    client_ip = request.client.host if request.client else "unknown"
    log_prediction(client_ip, input_dict, pred_int, prob_score, duration_ms)

    return PredictionResponse(
        heart_disease_risk=pred_int,
        prediction_label=label_str,
        probability=round(float(prob_score), 4),
        risk_level=risk_level,
        model_version=MODEL_VERSION,
        inference_time_ms=round(duration_ms, 2)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_heart_disease(batch: BatchHeartDiseaseInput, request: Request):
    """Predict heart disease risk for a batch of multiple patient records."""
    global model_pipeline
    if model_pipeline is None:
        load_or_initialize_model()

    start_time = time.time()
    records = [p.model_dump() for p in batch.patients]
    df_batch = pd.DataFrame(records)

    try:
        preds = model_pipeline.predict(df_batch)
        probs = model_pipeline.predict_proba(df_batch)[:, 1] if hasattr(model_pipeline, "predict_proba") else preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

    duration_ms = (time.time() - start_time) * 1000.0
    avg_duration_ms = duration_ms / max(len(records), 1)

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        pred_int = int(pred)
        prob_flt = float(prob)
        if prob_flt >= 0.70:
            r_level, l_str = "High", "High Risk - Presence of Heart Disease"
        elif prob_flt >= 0.40:
            r_level, l_str = "Moderate", "Moderate Risk - Borderline Findings"
        else:
            r_level, l_str = "Low", "Low Risk - Absence of Heart Disease"

        results.append(PredictionResponse(
            heart_disease_risk=pred_int,
            prediction_label=l_str,
            probability=round(prob_flt, 4),
            risk_level=r_level,
            model_version=MODEL_VERSION,
            inference_time_ms=round(avg_duration_ms, 2)
        ))

    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"AUDIT_LOG: Batch of {len(records)} records processed from {client_ip} in {duration_ms:.2f}ms")

    return BatchPredictionResponse(
        total_records=len(results),
        predictions=results
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
