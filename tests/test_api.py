"""
Unit and Integration Tests for FastAPI Model Serving Endpoint
Task 5 & Task 6 of AIMLCZG523 MLOps Assignment 01
"""

import os
import pytest
from fastapi.testclient import TestClient
from src.app import app, load_model
from src.train import train_and_track
from data.download_data import fetch_uci_dataset, clean_and_preprocess

@pytest.fixture(scope="session", autouse=True)
def ensure_model_is_trained():
    """Ensure data is downloaded and best model is trained before API tests execute."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    data_dir = os.path.join(project_root, "data")
    model_file = os.path.join(project_root, "models", "best_model.joblib")

    if not os.path.exists(model_file):
        raw_df = fetch_uci_dataset(data_dir)
        clean_and_preprocess(raw_df, data_dir)
        train_and_track()

    # Trigger model reload in app
    load_model()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    """Test root endpoint returns 200 and API discovery links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["version"] == "1.0.0"
    assert "docs_url" in data

def test_health_check(client):
    """Test /health endpoint returns 200 and HEALTHY state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert "timestamp" in data

def test_predict_endpoint_valid_patient(client):
    """Test /predict with a realistic patient record."""
    payload = {
        "age": 57,
        "sex": 1,
        "cp": 2,
        "trestbps": 130.0,
        "chol": 236.0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 174.0,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 1,
        "thal": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "prediction" in res
    assert res["prediction"] in [0, 1]
    assert res["risk_label"] in ["Low Risk", "High Risk"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "request_id" in res
    assert "probability_distribution" in res
    assert "X-Request-ID" in response.headers

def test_predict_endpoint_validation_error(client):
    """Test that pydantic rejects invalid patient input (age out of range)."""
    invalid_payload = {
        "age": -5,  # Invalid: age < 1
        "sex": 1,
        "cp": 2,
        "trestbps": 130.0,
        "chol": 236.0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 174.0,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 1,
        "thal": 2
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
    error_detail = response.json()
    assert "detail" in error_detail

def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint returns 200 and standard Prometheus exposition format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "# HELP" in content
    assert "# TYPE" in content
    assert "http_requests_total" in content or "http_request_duration_seconds" in content

def test_batch_predict_endpoint(client):
    """Test batch prediction with multiple patients."""
    batch_payload = [
        {
            "age": 52, "sex": 1, "cp": 0, "trestbps": 125.0, "chol": 212.0,
            "fbs": 0, "restecg": 1, "thalach": 168.0, "exang": 0,
            "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
        },
        {
            "age": 45, "sex": 0, "cp": 1, "trestbps": 110.0, "chol": 190.0,
            "fbs": 0, "restecg": 0, "thalach": 180.0, "exang": 0,
            "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2
        }
    ]
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    for r in results:
        assert r["prediction"] in [0, 1]
        assert "risk_label" in r
