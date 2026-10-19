"""
Model Training, Cross-Validation, and MLflow Experiment Tracking
Tasks 2, 3, 4 of AIMLCZG523 MLOps Assignment 01

Trains multiple classification models (Logistic Regression, Random Forest),
tunes hyperparameters, performs Stratified K-Fold cross validation,
logs parameters, metrics, and artifact plots to MLflow,
and serializes the winning pipeline to joblib.
"""

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.features import get_preprocessor, split_features_target
except ImportError:
    from features import get_preprocessor, split_features_target

def plot_and_save_artifacts(pipeline, X_test, y_test, model_name: str, output_dir: str):
    """Generates and saves Confusion Matrix and ROC Curve plots."""
    os.makedirs(output_dir, exist_ok=True)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease (0)", "Disease (1)"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}")
    cm_path = os.path.join(output_dir, f"{model_name.lower()}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.legend(loc="lower right")
    roc_path = os.path.join(output_dir, f"{model_name.lower()}_roc_curve.png")
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return cm_path, roc_path, y_pred, y_proba

def train_and_track():
    """Main training workflow with MLflow tracking."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "heart_disease_cleaned.csv")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures")
    mlruns_dir = os.path.join(project_root, "mlruns")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)
    X, y = split_features_target(df)

    # Train/Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Set MLflow experiment with SQLite backend (production standard)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_path = os.path.join(project_root, "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Heart-Disease-Prediction-MLOps")

    # Define Candidate Models with Pipelines
    candidate_models = {
        "LogisticRegression": {
            "model": LogisticRegression(C=0.8, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42),
            "params": {"C": 0.8, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000}
        },
        "RandomForest": {
            "model": RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_split=4, random_state=42),
            "params": {"n_estimators": 120, "max_depth": 6, "min_samples_split": 4}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
            "params": {"n_estimators": 100, "learning_rate": 0.08, "max_depth": 3}
        }
    }

    results = []
    best_pipeline = None
    best_roc_auc = -1.0
    best_model_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, config in candidate_models.items():
        print(f"\n==================== Training & Tracking: {name} ====================")
        preprocessor = get_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", config["model"])
        ])

        # 5-Fold Stratified Cross-Validation on Training Set
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
        cv_acc = np.mean(cv_results["test_accuracy"])
        cv_prec = np.mean(cv_results["test_precision"])
        cv_rec = np.mean(cv_results["test_recall"])
        cv_f1 = np.mean(cv_results["test_f1"])
        cv_roc = np.mean(cv_results["test_roc_auc"])

        print(f"5-Fold CV Scores -> Accuracy: {cv_acc:.4f}, Recall: {cv_rec:.4f}, ROC-AUC: {cv_roc:.4f}")

        # Fit on full train set and evaluate on test set
        pipeline.fit(X_train, y_train)
        cm_path, roc_path, y_pred, y_proba = plot_and_save_artifacts(pipeline, X_test, y_test, name, reports_dir)

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)
        test_rec = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc = roc_auc_score(y_test, y_proba)

        print(f"Held-out Test -> Accuracy: {test_acc:.4f}, Recall: {test_rec:.4f}, ROC-AUC: {test_roc:.4f}")

        # MLflow Logging
        with mlflow.start_run(run_name=f"Run_{name}") as run:
            # Log params
            mlflow.log_params(config["params"])
            mlflow.log_param("test_split_ratio", 0.20)
            mlflow.log_param("cv_folds", 5)

            # Log CV Metrics
            mlflow.log_metric("cv_mean_accuracy", cv_acc)
            mlflow.log_metric("cv_mean_precision", cv_prec)
            mlflow.log_metric("cv_mean_recall", cv_rec)
            mlflow.log_metric("cv_mean_f1", cv_f1)
            mlflow.log_metric("cv_mean_roc_auc", cv_roc)

            # Log Test Metrics
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_precision", test_prec)
            mlflow.log_metric("test_recall", test_rec)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_roc_auc", test_roc)

            # Log Artifacts
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)

            # Log Model Pipeline
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name="model",
                    registered_model_name=f"HeartDisease_{name}",
                    serialization_format="cloudpickle"
                )
            except Exception:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name=f"HeartDisease_{name}",
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )
            print(f"[MLflow] Logged run {run.info.run_id} successfully.")

        results.append({
            "model": name,
            "cv_accuracy": cv_acc,
            "cv_recall": cv_rec,
            "cv_roc_auc": cv_roc,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "test_roc_auc": test_roc
        })

        # Selection criterion: Highest Test ROC-AUC (and Recall)
        if test_roc > best_roc_auc:
            best_roc_auc = test_roc
            best_pipeline = pipeline
            best_model_name = name

    # Summary table
    res_df = pd.DataFrame(results)
    print("\n==================== Model Comparison Summary ====================")
    print(res_df.to_string(index=False))

    # Save summary to reports
    res_df.to_csv(os.path.join(project_root, "reports", "model_comparison.csv"), index=False)

    # Save Best Model Pipeline
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n[WINNER] Best Model: {best_model_name} with Test ROC-AUC: {best_roc_auc:.4f}")
    print(f"[SAVED] Exported winning pipeline to {best_model_path}")

    # Save Metadata
    metadata = {
        "model_name": best_model_name,
        "selected_metric": "test_roc_auc",
        "test_roc_auc": float(best_roc_auc),
        "test_accuracy": float(res_df.loc[res_df["model"] == best_model_name, "test_accuracy"].values[0]),
        "test_recall": float(res_df.loc[res_df["model"] == best_model_name, "test_recall"].values[0]),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_steps": [name for name, _ in best_pipeline.steps]
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return best_model_name, best_pipeline

if __name__ == "__main__":
    train_and_track()
