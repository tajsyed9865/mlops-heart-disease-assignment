"""
Unit Tests for Data Ingestion and Preprocessing
Task 5 of AIMLCZG523 MLOps Assignment 01
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.features import ALL_FEATURE_NAMES, TARGET_NAME, split_features_target
from data.download_data import fetch_uci_dataset, clean_and_preprocess

@pytest.fixture(scope="session")
def project_dirs():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    data_dir = os.path.join(project_root, "data")
    return {"root": project_root, "data": data_dir}

@pytest.fixture(scope="session")
def dataset(project_dirs):
    data_dir = project_dirs["data"]
    clean_csv = os.path.join(data_dir, "heart_disease_cleaned.csv")
    if not os.path.exists(clean_csv):
        raw_df = fetch_uci_dataset(data_dir)
        clean_df = clean_and_preprocess(raw_df, data_dir)
        return clean_df
    return pd.read_csv(clean_csv)

def test_dataset_columns_exist(dataset):
    """Verify that all 13 predictor features plus target column exist."""
    for col in ALL_FEATURE_NAMES:
        assert col in dataset.columns, f"Missing expected feature column: {col}"
    assert TARGET_NAME in dataset.columns, f"Missing target column: {TARGET_NAME}"

def test_no_null_values(dataset):
    """Ensure cleaned dataset has zero missing values."""
    null_counts = dataset.isna().sum().sum()
    assert null_counts == 0, f"Found {null_counts} missing values in cleaned dataset."

def test_target_binary_values(dataset):
    """Verify that the target column contains strictly binary values (0 or 1)."""
    unique_vals = set(dataset[TARGET_NAME].unique())
    assert unique_vals.issubset({0, 1}), f"Target contains invalid values: {unique_vals}"

def test_split_features_target(dataset):
    """Verify split_features_target splits X and y with correct shapes."""
    X, y = split_features_target(dataset)
    assert X.shape[1] == len(ALL_FEATURE_NAMES)
    assert len(X) == len(y)
    assert TARGET_NAME not in X.columns
