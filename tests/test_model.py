"""
Unit Tests for Model Preprocessing, Training, and Evaluation
Task 5 of AIMLCZG523 MLOps Assignment 01
"""

import os
import tempfile
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

from src.features import get_preprocessor, split_features_target, ALL_FEATURE_NAMES
from data.download_data import fetch_uci_dataset, clean_and_preprocess

@pytest.fixture(scope="session")
def dataset():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    data_dir = os.path.join(project_root, "data")
    clean_csv = os.path.join(data_dir, "heart_disease_cleaned.csv")
    if not os.path.exists(clean_csv):
        raw_df = fetch_uci_dataset(data_dir)
        return clean_and_preprocess(raw_df, data_dir)
    return pd.read_csv(clean_csv)

def test_preprocessor_transformation(dataset):
    """Ensure ColumnTransformer handles features and outputs finite numeric array."""
    X, _ = split_features_target(dataset)
    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed is not None
    assert isinstance(X_transformed, np.ndarray)
    assert not np.isnan(X_transformed).any(), "Transformed feature matrix contains NaNs"
    assert np.isfinite(X_transformed).all(), "Transformed feature matrix contains infinite values"
    # One-hot encoded categorical columns expand feature dimension
    assert X_transformed.shape[1] >= len(ALL_FEATURE_NAMES)

def test_model_pipeline_training(dataset):
    """Verify that a full Pipeline fits and predicts without leakage."""
    X, y = split_features_target(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probas.shape == (len(X_test), 2)
    # Probabilities must sum to 1
    assert np.allclose(probas.sum(axis=1), 1.0)

def test_model_performance_threshold(dataset):
    """Ensure trained model meets assignment production performance threshold (ROC-AUC > 0.75)."""
    X, y = split_features_target(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    probas = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probas)
    acc = accuracy_score(y_test, pipeline.predict(X_test))

    print(f"[TEST] Model Test ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
    assert roc_auc >= 0.75, f"ROC-AUC {roc_auc:.4f} is below the required 0.75 threshold"
    assert acc >= 0.70, f"Accuracy {acc:.4f} is below the 0.70 threshold"

def test_pipeline_serialization_and_reproducibility(dataset):
    """Verify that serializing to joblib and reloading produces identical outputs."""
    X, y = split_features_target(dataset)
    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X, y)
    orig_preds = pipeline.predict_proba(X[:10])

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        joblib.dump(pipeline, tmp.name)
        loaded = joblib.load(tmp.name)
        loaded_preds = loaded.predict_proba(X[:10])

        assert np.allclose(orig_preds, loaded_preds, atol=1e-7), "Reloaded model predictions deviate!"
    os.remove(tmp.name)
