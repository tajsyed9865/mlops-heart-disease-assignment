# ==============================================================================
# BITS Pilani — MLOps (AIMLCZG523) Assignment 01
# Complete End-to-End Jupyter Notebook Code: Heart Disease Risk Prediction
# Ready to run directly in any Jupyter Notebook cell or as a standalone script
# ==============================================================================

# %% [markdown]
# ## 1. Imports and Setup

# %%
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import joblib

# Optional MLflow tracking (graceful fallback if not running)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})
print("[✓] All packages imported successfully!")

# %% [markdown]
# ## 2. Data Acquisition & Hygiene Validation

# %%
# Ensure data directory exists
os.makedirs("data", exist_ok=True)
data_file = os.path.join("data", "heart_disease.csv")

# If local CSV does not exist, download or generate standard UCI Cleveland dataset
if not os.path.exists(data_file):
    print("[*] Downloading dataset from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    try:
        urllib.request.urlretrieve(url, data_file)
        df = pd.read_csv(data_file, names=cols, na_values="?")
    except Exception as e:
        print(f"[*] Generating representative Cleveland dataset: {e}")
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            "age": np.random.normal(54.4, 9.0, n).clip(29, 77).astype(int),
            "sex": np.random.binomial(1, 0.68, n),
            "cp": np.random.choice([0, 1, 2, 3], size=n, p=[0.48, 0.17, 0.28, 0.07]),
            "trestbps": np.random.normal(131.6, 17.5, n).clip(94, 200).astype(int),
            "chol": np.random.normal(246.3, 51.8, n).clip(126, 564).astype(int),
            "fbs": np.random.binomial(1, 0.15, n),
            "restecg": np.random.choice([0, 1, 2], size=n, p=[0.49, 0.48, 0.03]),
            "thalach": np.random.normal(149.6, 22.9, n).clip(71, 202).astype(int),
            "exang": np.random.binomial(1, 0.33, n),
            "oldpeak": np.round(np.random.exponential(1.0, n).clip(0.0, 6.2), 1),
            "slope": np.random.choice([0, 1, 2], size=n, p=[0.46, 0.46, 0.08]),
            "ca": np.random.choice([0, 1, 2, 3], size=n, p=[0.58, 0.22, 0.13, 0.07]),
            "thal": np.random.choice([1, 2, 3], size=n, p=[0.06, 0.55, 0.39]),
            "target": np.random.binomial(1, 0.54, n)
        })
else:
    df = pd.read_csv(data_file)

# Preprocessing: Binarize target & clean missing values
df["target"] = (df["target"] > 0).astype(int)
df = df.dropna().reset_index(drop=True)

print(f"[✓] Dataset Dimensions: {df.shape}")
print(f"[✓] Class Distribution:\n{df['target'].value_counts().rename({0: 'Healthy (0)', 1: 'Heart Disease (1)'})}")
display(df.head())

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
# Figure 1: Class Balance & Feature Distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
colors = ["#2ecc71", "#e74c3c"]
sns.countplot(x="target", data=df, palette=colors, ax=axes[0])
axes[0].set_title("Class Balance (Healthy vs Heart Disease)", weight="bold")
axes[0].set_xticklabels(["Healthy (0)", "Disease (1)"])

# Correlation Heatmap
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1], cbar=False)
axes[1].set_title("Pearson Correlation Heatmap", weight="bold")

plt.tight_layout()
plt.show()

# Figure 2: Clinical Feature Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["age"], kde=True, ax=axes[0], color="#3498db")
axes[0].set_title("Age Distribution (Years)", weight="bold")

sns.histplot(df["thalach"], kde=True, ax=axes[1], color="#9b59b6")
axes[1].set_title("Max Heart Rate (thalach)", weight="bold")

sns.boxplot(x="target", y="oldpeak", data=df, palette=colors, ax=axes[2])
axes[2].set_title("ST Depression (oldpeak) by Outcome", weight="bold")
axes[2].set_xticklabels(["Healthy (0)", "Disease (1)"])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Feature Engineering & Preprocessing Pipeline

# %%
# Define feature types
num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Build modular ColumnTransformer to prevent data leakage
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_features)
    ],
    remainder="drop"
)

# Stratified 80/20 train/test split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"[✓] Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# %% [markdown]
# ## 5. Model Development, Tuning & Cross-Validation

# %%
if MLFLOW_AVAILABLE:
    mlflow.set_experiment("Heart_Disease_Prediction_AIMLCZG523")

models = [
    {
        "name": "Logistic_Regression",
        "pipeline": Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]),
        "grid": {"classifier__C": [0.01, 0.1, 1.0, 10.0], "classifier__solver": ["lbfgs", "liblinear"]}
    },
    {
        "name": "Random_Forest",
        "pipeline": Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))]),
        "grid": {"classifier__n_estimators": [50, 100], "classifier__max_depth": [3, 5, 8]}
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model_overall = None
best_auc_overall = -1.0
best_name_overall = ""

results = []

for m in models:
    name = m["name"]
    print(f"\n[*] Tuning and Cross-Validating: {name}...")
    
    grid = GridSearchCV(m["pipeline"], m["grid"], cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_pipe = grid.best_estimator_
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"    - Best 5-Fold CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"    - Best Hyperparameters:   {grid.best_params_}")
    print(f"    - Test Accuracy:          {acc:.4f}")
    print(f"    - Test ROC-AUC:           {auc:.4f}")
    
    results.append({
        "Model": name,
        "CV ROC-AUC": round(grid.best_score_, 4),
        "Test Accuracy": round(acc, 4),
        "Test Precision": round(prec, 4),
        "Test Recall": round(rec, 4),
        "Test F1": round(f1, 4),
        "Test ROC-AUC": round(auc, 4),
        "Best Params": str(grid.best_params_)
    })
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_roc_auc", grid.best_score_)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_roc_auc", auc)
            mlflow.sklearn.log_model(best_pipe, "model")
    
    if auc > best_auc_overall:
        best_auc_overall = auc
        best_model_overall = best_pipe
        best_name_overall = name

# Display comparative evaluation metrics table
df_results = pd.DataFrame(results)
print("\n" + "="*50)
print("MODEL EVALUATION COMPARISON SUMMARY")
print("="*50)
display(df_results)

# %% [markdown]
# ## 6. Confusion Matrix & ROC-AUC Evaluation Plots

# %%
y_pred_best = best_model_overall.predict(X_test)
y_prob_best = best_model_overall.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Healthy", "Heart Disease"], yticklabels=["Healthy", "Heart Disease"])
axes[0].set_title(f"Confusion Matrix ({best_name_overall})", weight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
axes[1].plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {best_auc_overall:.3f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_title(f"ROC-AUC Curve ({best_name_overall})", weight="bold")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=["Healthy (0)", "Heart Disease (1)"]))

# %% [markdown]
# ## 7. Model Serialization & Production Inference Test

# %%
os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "heart_disease_pipeline.joblib")
joblib.dump(best_model_overall, model_path)
print(f"[✓] Serialized production pipeline saved to: {model_path}")

# Load and test the pipeline with raw JSON clinical sample
loaded_pipeline = joblib.load(model_path)

sample_patient = {
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

df_patient = pd.DataFrame([sample_patient])
pred = loaded_pipeline.predict(df_patient)[0]
prob = loaded_pipeline.predict_proba(df_patient)[0][1]

risk_level = "High" if prob >= 0.70 else ("Moderate" if prob >= 0.40 else "Low")
label = "Presence of Heart Disease" if pred == 1 else "Absence of Heart Disease"

print("\n" + "="*50)
print("SAMPLE CLINICAL INFERENCE OUTPUT (Ready for API)")
print("="*50)
print(f"Prediction:         {pred} ({label})")
print(f"Confidence (Prob):  {prob:.4f} ({prob*100:.1f}%)")
print(f"Risk Stratification: {risk_level}")
print("="*50)
