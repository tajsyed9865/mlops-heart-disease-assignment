"""
Feature Engineering and Preprocessing Pipeline
Task 2 & Task 4 of AIMLCZG523 MLOps Assignment 01

Defines scikit-learn ColumnTransformer for scaling numeric features
and encoding categorical features to guarantee zero data leakage and 100% reproducibility.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES: List[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

ALL_FEATURE_NAMES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_NAME: str = "target"

def get_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    1. StandardScales numeric features (zero mean, unit variance).
    2. OneHotEncodes categorical features with handle_unknown='ignore'
       to gracefully handle unexpected categories during serving.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits input dataframe into feature matrix X and target vector y."""
    feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_NAME].copy() if TARGET_NAME in df.columns else None
    return X, y
