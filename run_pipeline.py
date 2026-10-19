"""
End-to-End MLOps Pipeline Runner
AIMLCZG523 Assignment 01

Executes the complete pipeline in sequence:
1. Data Acquisition & Preprocessing
2. Exploratory Data Analysis & Plots
3. Model Training, 5-Fold Cross Validation, & MLflow Tracking
4. Pytest Test Suite Execution
5. Verification summary for submission
"""

import os
import sys
import subprocess
import time

def run_step(step_name: str, command: list):
    print(f"\n========================================================")
    print(f"  STEP: {step_name}")
    print(f"========================================================")
    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    res = subprocess.run(command, check=True, env=env)
    duration = time.time() - start
    print(f"--> [COMPLETED] {step_name} in {duration:.2f}s (Exit Code: {res.returncode})\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    python_cmd = sys.executable

    print("\n********************************************************")
    print("  AIMLCZG523 MLOps Assignment 01 - Pipeline Execution   ")
    print("********************************************************\n")

    # Step 1: Data Acquisition
    run_step("1. Data Acquisition & Cleaning", [python_cmd, "data/download_data.py"])

    # Step 2: Exploratory Data Analysis
    run_step("2. Exploratory Data Analysis & Figures", [python_cmd, "src/eda.py"])

    # Step 3: Model Training & MLflow Tracking
    run_step("3. Model Training, Cross-Validation & MLflow Logging", [python_cmd, "src/train.py"])

    # Step 4: Unit Testing with Pytest
    run_step("4. Automated Unit & Integration Testing", [python_cmd, "-m", "pytest", "tests/", "-v"])

    print("********************************************************")
    print("  ALL PIPELINE STEPS EXECUTED SUCCESSFULLY!             ")
    print("********************************************************")
    print("\nNext Steps for Serving & Production:")
    print("1. Start FastAPI server locally:")
    print("   uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload")
    print("\n2. Launch MLflow UI:")
    print("   mlflow ui --backend-store-uri file:///$(pwd)/mlruns --port 5000")
    print("\n3. Build and run Docker container:")
    print("   docker build -t heart-disease-api:v1.0.0 .")
    print("   docker run -d -p 8000:8000 --name heart-api heart-disease-api:v1.0.0")
    print("\n4. Deploy to Kubernetes:")
    print("   kubectl apply -f k8s/deployment.yaml")
    print("   kubectl apply -f k8s/service.yaml")
    print("   kubectl apply -f k8s/hpa.yaml")
    print("   kubectl get pods -l app=heart-disease-api")

if __name__ == "__main__":
    main()
