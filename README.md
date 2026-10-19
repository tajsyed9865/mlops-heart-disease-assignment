# End-to-End MLOps Pipeline: Heart Disease Risk Prediction
**Course**: Machine Learning Operations (MLOps) - AIMLCZG523 (Assignment 01)  
**Institution**: BITS Pilani  
**Dataset**: UCI Heart Disease (Cleveland Dataset)

---

## 📋 Project Overview

This repository contains a production-ready, scalable, reproducible, and fully monitored machine learning solution adhering to modern MLOps best practices. It covers the complete lifecycle:
1. **Data Acquisition & EDA**: Automated data ingestion from UCI ML repository, missing value imputation, statistical profiling, and publication-quality exploratory visualizations.
2. **Feature Engineering & Preprocessing**: Leakage-free `ColumnTransformer` with `StandardScaler` for continuous features and `OneHotEncoder` for categorical variables.
3. **Model Development & Comparison**: Logistic Regression, Random Forest, and Gradient Boosting evaluated with 5-Fold Stratified Cross-Validation (Accuracy, Precision, Recall, F1, ROC-AUC).
4. **Experiment Tracking**: Comprehensive MLflow tracking capturing hyperparameters, cross-validation metrics, confusion matrix & ROC curve artifacts, and serialized pipelines.
5. **Model Packaging & Serialization**: End-to-end serialized `sklearn.pipeline.Pipeline` with `joblib` guaranteeing deterministic inference.
6. **Automated Testing & CI/CD**: Pytest suite for data, model, and API endpoints integrated into a GitHub Actions CI/CD workflow (Linting, Testing, Training, Container build).
7. **Production Containerization**: Multi-stage lightweight Docker image serving a high-performance FastAPI application with non-root security.
8. **Kubernetes Orchestration**: Production deployment manifests, LoadBalancer service, Horizontal Pod Autoscaler (HPA), and Prometheus scrape configurations.
9. **Observability & Monitoring**: Prometheus endpoint (`/metrics`) instrumented with latency histograms, throughput counters, and structured request logging.

---

## 📁 Repository Directory Structure

```text
mlops-heart-disease-assignment/
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD automation workflow
├── data/
│   ├── download_data.py          # Automated UCI ingestion & preprocessing
│   ├── heart_disease_raw.csv     # Raw dataset
│   └── heart_disease_cleaned.csv # Cleaned & binarized dataset
├── k8s/
│   ├── deployment.yaml           # Kubernetes 2-replica deployment manifest
│   ├── service.yaml              # LoadBalancer / NodePort service
│   ├── hpa.yaml                  # Horizontal Pod Autoscaler manifest
│   └── prometheus-config.yaml    # Prometheus scrape configuration
├── models/
│   ├── best_model.joblib         # Winning serialized model pipeline
│   └── model_metadata.json       # Version, hyperparameters & metrics
├── reports/
│   ├── figures/                  # EDA & evaluation charts
│   │   ├── 01_class_balance.png
│   │   ├── 02_numeric_histograms.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_categorical_risk_factors.png
│   │   ├── 05_age_vs_thalach.png
│   │   ├── logisticregression_confusion_matrix.png
│   │   ├── logisticregression_roc_curve.png
│   │   ├── randomforest_confusion_matrix.png
│   │   └── randomforest_roc_curve.png
│   └── model_comparison.csv     # Benchmarking results table
├── src/
│   ├── __init__.py
│   ├── features.py               # Preprocessing & feature pipelines
│   ├── eda.py                    # Exploratory Data Analysis & visual plots
│   ├── train.py                  # Training, cross-validation & MLflow logging
│   └── app.py                    # FastAPI serving app with Prometheus metrics
├── tests/
│   ├── __init__.py
│   ├── test_data.py              # Schema, null, and split tests
│   ├── test_model.py             # Reproducibility, transformation & threshold tests
│   └── test_api.py               # API health, /predict, and validation tests
├── Dockerfile                    # Containerization definition
├── .dockerignore                 # Excluded build files
├── requirements.txt              # Pinned Python package dependencies
├── run_pipeline.py               # 1-click end-to-end execution runner
├── README.md                     # Project documentation & run guide
└── REPORT.md                     # Comprehensive 10-page academic submission report
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd mlops-heart-disease-assignment

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Data -> EDA -> Train -> Test)
```bash
python run_pipeline.py
```

### 3. Launch MLflow Dashboard
Inspect logged runs, parameters, metrics, and artifact plots:
```bash
mlflow ui --backend-store-uri file:///$(pwd)/mlruns --port 5000
```
Open your browser at `http://localhost:5000`.

### 4. Run API Locally
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger UI Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Prometheus Metrics: `http://localhost:8000/metrics`

### 5. Test Inference Endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 58,
       "sex": 1,
       "cp": 2,
       "trestbps": 135.0,
       "chol": 240.0,
       "fbs": 0,
       "restecg": 1,
       "thalach": 160.0,
       "exang": 0,
       "oldpeak": 1.2,
       "slope": 1,
       "ca": 0,
       "thal": 2
     }'
```

---

## 🐳 Docker Containerization

### Build Image
```bash
docker build -t heart-disease-api:v1.0.0 .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name heart-disease-service heart-disease-api:v1.0.0
```

### Verify Running Container
```bash
curl http://localhost:8000/health
```

---

## ☸️ Kubernetes Deployment

Deploy to local Minikube or cloud Kubernetes (GKE, EKS, AKS):
```bash
# Apply deployment and load balancer service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check pod and service status
kubectl get pods -l app=heart-disease-api
kubectl get svc heart-disease-api-service
```

---

## 🧪 Automated Testing
Run the full test suite with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
