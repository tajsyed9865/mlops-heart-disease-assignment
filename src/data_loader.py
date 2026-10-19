"""
Data Loading and Validation Module.
Loads cleaned dataset, validates column schemas, and splits train/test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "heart_disease.csv")

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal"
]
TARGET_COLUMN = "target"

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_dataset(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset from path and validate schema."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Validation checks
    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
    # Fill any missing values if present
    df = df.dropna().reset_index(drop=True)
    return df


def get_train_test_data(file_path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """Return stratified train and test splits for features and target."""
    df = load_dataset(file_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
