.PHONY: help setup data train plots test run docker-build docker-run compose-up compose-down k8s-deploy k8s-delete

help:
	@echo "MLOps Assignment 01 Automation Commands:"
	@echo "  make setup        - Install all Python dependencies"
	@echo "  make data         - Download and clean Heart Disease dataset"
	@echo "  make train        - Run model training with MLflow tracking"
	@echo "  make plots        - Generate EDA & evaluation charts"
	@echo "  make test         - Run full pytest test suite with coverage"
	@echo "  make run          - Launch FastAPI model serving API locally"
	@echo "  make docker-build - Build Docker container image"
	@echo "  make docker-run   - Run Docker container on port 8000"
	@echo "  make compose-up   - Launch API + Prometheus + Grafana stack"
	@echo "  make compose-down - Stop Docker Compose stack"
	@echo "  make k8s-deploy   - Deploy API to Kubernetes cluster"
	@echo "  make k8s-delete   - Tear down Kubernetes deployment"

setup:
	pip install --upgrade pip
	pip install -r requirements.txt

data:
	python data/download_data.py

train:
	python src/train.py

plots:
	python screenshots/generate_plots.py

test:
	pytest -v --cov=src --cov=app --cov-report=term-missing tests/

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run -d --name heart-disease-service -p 8000:8000 heart-disease-api:latest

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

k8s-deploy:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml

k8s-delete:
	kubectl delete -f k8s/service.yaml
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/hpa.yaml
