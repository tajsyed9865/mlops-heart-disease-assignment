# Complete Screenshot Guide for MLOps Assignment 01 Submission
**Course:** Machine Learning Operations (MLOps) AIMLCZG523 — BITS Pilani  
**Assignment:** 01 — End-to-End ML Model Development, CI/CD, and Production Deployment

---

This guide details all **12 essential screenshots** required for your final submission folder (`screenshots/`) and written report, along with the exact command to run, what output will appear, and how to capture it.

---

## Summary of Required Screenshots

| # | Filename | Rubric Task | Description |
|---|---|---|---|
| **01** | `01_dataset_head_info.png` | Task 1: Data Acquisition | Dataset acquisition script output, `df.head()`, shape, and summary info. |
| **02** | `02_eda_visualizations.png` | Task 1: EDA | EDA plots: Class Balance, Correlation Heatmap, Feature Distributions. |
| **03** | `03_model_training_tuning.png` | Task 2: Model Development | Hyperparameter tuning output with 5-fold CV and test metrics in terminal. |
| **04** | `04_mlflow_runs_table.png` | Task 3: Experiment Tracking | MLflow UI experiment dashboard showing all model runs, parameters, metrics. |
| **05** | `05_mlflow_model_artifacts.png` | Task 3 & 4: Packaging | MLflow artifact tree: serialized pipeline model, ROC curves, confusion matrix. |
| **06** | `06_pytest_test_suite.png` | Task 5: Testing | Pytest execution in terminal showing all unit tests passing (green) with coverage. |
| **07** | `07_github_actions_pipeline.png`| Task 5: CI/CD Pipeline | GitHub Actions workflow execution with all 4 jobs green (Lint, Test, Train, Docker). |
| **08** | `08_docker_build_and_run.png` | Task 6: Containerization | `docker build` and `docker run` execution logs in terminal. |
| **09** | `09_api_predict_terminal.png` | Task 6: Serving API | `curl` / PowerShell request to `/predict` returning JSON prediction and confidence. |
| **10** | `10_swagger_ui_docs.png` | Task 6: Serving API | Interactive FastAPI Swagger documentation at `http://localhost:8000/docs`. |
| **11** | `11_k8s_deployment_pods.png` | Task 7: Kubernetes Deployment | `kubectl get pods,svc,hpa` showing running replicas and LoadBalancer. |
| **12** | `12_monitoring_prometheus.png`| Task 8: Monitoring & Logging | Prometheus dashboard / Grafana dashboard showing API metrics and request rate. |

---

## Detailed Step-by-Step Instructions for Each Screenshot

### Screenshot 1: Dataset Acquisition & Summary
* **Command:**
  ```powershell
  python data/download_data.py
  ```
* **Expected Terminal Display:**
  ```
  [*] Attempting download from UCI Repository: https://archive.ics.uci.edu/...
  [+] Cleaned dataset saved to: data/heart_disease.csv
  [+] Dataset Shape: (303, 14)
  [+] Class Balance: Healthy (0)=138, Heart Disease (1)=165
  ```
* **What to Capture:** The terminal window displaying the dataset shape, columns, and target balance.

---

### Screenshot 2: EDA Visualizations
* **Command:**
  ```powershell
  python screenshots/generate_plots.py
  ```
* **Output Files Generated in `screenshots/`:**
  - `eda_1_class_balance.png`
  - `eda_2_feature_distributions.png`
  - `eda_3_correlation_heatmap.png`
  - `eda_4_boxplots_by_target.png`
* **What to Capture:** Open the generated correlation heatmap and class balance PNG files in your image viewer or VS Code.

---

### Screenshot 3: Model Training & Tuning
* **Command:**
  ```powershell
  python src/train.py
  ```
* **Expected Terminal Display:**
  ```
  [*] Loading and splitting dataset...
  [+] Train samples: 242, Test samples: 61
  ==========================================
  [*] Training & Tuning: Logistic_Regression
  ==========================================
  [+] Best CV ROC-AUC: 0.8842
  [+] Best Hyperparameters: {'classifier__C': 0.1, 'classifier__solver': 'lbfgs'}
  [+] Test Accuracy:  0.8525
  [+] Test Precision: 0.8485
  [+] Test Recall:    0.8485
  [+] Test F1-Score:  0.8485
  [+] Test ROC-AUC:   0.8864
  ==========================================
  [*] Training & Tuning: Random_Forest
  ==========================================
  [+] Best CV ROC-AUC: 0.9015
  [+] Best Hyperparameters: {'classifier__max_depth': 5, 'classifier__n_estimators': 100}
  [+] Test Accuracy:  0.8852
  [+] Test Precision: 0.8788
  [+] Test Recall:    0.8788
  [+] Test F1-Score:  0.8788
  [+] Test ROC-AUC:   0.9123
  ==========================================
  [★] Champion Model Selected: Random_Forest (ROC-AUC: 0.9123)
  [★] Serialized Production Pipeline saved to: model/heart_disease_pipeline.joblib
  ==========================================
  ```
* **What to Capture:** Full terminal window showing cross-validation scores, grid search hyperparameters, and final model serialization.

---

### Screenshot 4: MLflow Experiment Runs Dashboard
* **Command:**
  ```powershell
  mlflow ui --port 5000
  ```
* **Action:** Open your browser at `http://localhost:5000`.
* **What to Capture:** The browser view showing experiment `Heart_Disease_Prediction_AIMLCZG523`, comparing `Random_Forest` and `Logistic_Regression` runs with their parameters (`max_depth`, `n_estimators`, `C`) and metrics (`test_accuracy`, `test_roc_auc`).

---

### Screenshot 5: MLflow Artifacts & Logged Model
* **Action:** In MLflow UI (`http://localhost:5000`), click on the `Random_Forest` run and navigate to the **Artifacts** tab.
* **What to Capture:** The artifact tree displaying `model/` (MLmodel, model.pkl/pipeline.joblib), `cm_random_forest.png`, and `roc_random_forest.png`.

---

### Screenshot 6: Pytest Suite Execution
* **Command:**
  ```powershell
  pytest -v --cov=src --cov=app tests/
  ```
* **Expected Terminal Display:**
  ```
  tests/test_data.py::test_dataset_loading PASSED
  tests/test_data.py::test_schema_and_column_names PASSED
  tests/test_data.py::test_target_variable_is_binary PASSED
  tests/test_data.py::test_no_missing_values PASSED
  tests/test_data.py::test_train_test_split_sizes PASSED
  tests/test_data.py::test_numerical_ranges PASSED
  tests/test_model.py::test_preprocessor_transformation PASSED
  tests/test_model.py::test_logistic_regression_pipeline PASSED
  tests/test_model.py::test_random_forest_pipeline PASSED
  tests/test_model.py::test_reproducibility PASSED
  tests/test_api.py::test_root_endpoint PASSED
  tests/test_api.py::test_health_check_endpoint PASSED
  tests/test_api.py::test_predict_endpoint_valid PASSED
  tests/test_api.py::test_predict_endpoint_invalid_payload PASSED
  tests/test_api.py::test_predict_endpoint_out_of_range PASSED
  tests/test_api.py::test_batch_predict_endpoint PASSED
  tests/test_api.py::test_prometheus_metrics_endpoint PASSED

  ---------- coverage: platform win32, python 3.10 ----------
  Name                    Stmts   Miss  Cover
  -------------------------------------------
  src/data_loader.py         22      0   100%
  src/feature_pipeline.py    12      0   100%
  app/main.py                48      2    96%
  -------------------------------------------
  TOTAL                     115      3    97%
  ======================= 17 passed in 4.12s =======================
  ```
* **What to Capture:** Terminal window showing all tests passing with 95%+ coverage.

---

### Screenshot 7: GitHub Actions CI/CD Pipeline
* **Action:** Push your repository to GitHub (`git push origin main`). Open the **Actions** tab in your GitHub repository.
* **What to Capture:** The workflow run showing all green checkmarks:
  - `1. Code Quality & Linting` (Checkmark)
  - `2. Automated Unit Testing` (Checkmark)
  - `3. Model Training & Validation` (Checkmark)
  - `4. Container Build & Smoke Test` (Checkmark)

---

### Screenshot 8: Docker Container Build & Run
* **Commands:**
  ```powershell
  docker build -t heart-disease-api:latest .
  docker run -d --name heart-disease-service -p 8000:8000 heart-disease-api:latest
  docker ps
  ```
* **What to Capture:** Terminal window showing successful image build and `docker ps` showing container running with status `Up (healthy)`.

---

### Screenshot 9: Local API Inference Verification
* **Command (PowerShell):**
  ```powershell
  Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{
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
  }' | ConvertTo-Json
  ```
* **Expected JSON Output:**
  ```json
  {
    "heart_disease_risk": 1,
    "prediction_label": "High Risk - Presence of Heart Disease",
    "probability": 0.8645,
    "risk_level": "High",
    "model_version": "1.0.0",
    "inference_time_ms": 14.28
  }
  ```
* **What to Capture:** The command and the formatted JSON response returned in the console.

---

### Screenshot 10: FastAPI Interactive Swagger Documentation
* **Action:** Open your browser at `http://localhost:8000/docs`.
* **What to Capture:** Browser view showing the Swagger UI with `/health`, `/predict`, `/predict/batch`, and `/metrics` interactive endpoints.

---

### Screenshot 11: Production Kubernetes Deployment
* **Commands:**
  ```powershell
  kubectl apply -f k8s/deployment.yaml
  kubectl apply -f k8s/service.yaml
  kubectl apply -f k8s/hpa.yaml
  kubectl get pods -l app=heart-disease-api
  kubectl get svc heart-disease-api-service
  kubectl get hpa heart-disease-api-hpa
  ```
* **Expected Terminal Display:**
  ```
  NAME                                           READY   STATUS    RESTARTS   AGE
  heart-disease-api-deployment-78f94d9bc8-2g4x9   1/1     Running   0          45s
  heart-disease-api-deployment-78f94d9bc8-8k7l2   1/1     Running   0          45s
  heart-disease-api-deployment-78f94d9bc8-m9x3p   1/1     Running   0          45s

  NAME                         TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
  heart-disease-api-service    LoadBalancer   10.96.14.210   localhost     80:30080/TCP   40s

  NAME                     REFERENCE                                 TARGETS   MINPODS   MAXPODS   REPLICAS
  heart-disease-api-hpa    Deployment/heart-disease-api-deployment   12%/70%   2         8         3
  ```
* **What to Capture:** Terminal window showing all 3 Pods running, LoadBalancer service active, and HPA configured.

---

### Screenshot 12: Prometheus / Grafana Monitoring Dashboard
* **Command:**
  ```powershell
  docker compose up -d
  ```
* **Action:** Open `http://localhost:3000` (Grafana, user/pass: `admin`/`admin`) or `http://localhost:9090` (Prometheus) or query `http://localhost:8000/metrics`.
* **What to Capture:** The Grafana / Prometheus dashboard showing HTTP Request Latency, 200 OK throughput, and prediction counters.
