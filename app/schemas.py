"""
Pydantic Request and Response Schemas for Heart Disease Inference Service.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class HeartDiseaseInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age of the patient in years", examples=[58])
    sex: int = Field(..., ge=0, le=1, description="Patient sex (1 = male, 0 = female)", examples=[1])
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", examples=[2])
    trestbps: int = Field(..., ge=60, le=260, description="Resting blood pressure in mm Hg", examples=[130])
    chol: int = Field(..., ge=80, le=600, description="Serum cholestoral in mg/dl", examples=[240])
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", examples=[0])
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)", examples=[1])
    thalach: int = Field(..., ge=50, le=230, description="Maximum heart rate achieved", examples=[165])
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes, 0 = no)", examples=[0])
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest", examples=[1.2])
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)", examples=[1])
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-3) colored by flourosopy", examples=[0])
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (1 = normal, 2 = fixed defect, 3 = reversable defect)", examples=[2])

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 240,
                "fbs": 0,
                "restecg": 1,
                "thalach": 165,
                "exang": 0,
                "oldpeak": 1.2,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        }


class PredictionResponse(BaseModel):
    heart_disease_risk: int = Field(..., description="0 = Low Risk (Absence), 1 = High Risk (Presence)")
    prediction_label: str = Field(..., description="Human-readable clinical risk classification")
    probability: float = Field(..., description="Estimated probability of heart disease [0.0 - 1.0]")
    risk_level: str = Field(..., description="Categorical risk stratification (Low / Moderate / High)")
    model_version: str = Field(..., description="Active production model version")
    inference_time_ms: float = Field(..., description="Model inference latency in milliseconds")


class BatchHeartDiseaseInput(BaseModel):
    patients: List[HeartDiseaseInput]


class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model_version: str
