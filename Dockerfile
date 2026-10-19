# Multi-stage / Production Dockerfile for Heart Disease MLOps Serving
# AIMLCZG523 Assignment 01 - Task 6
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/best_model.joblib

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user for security best practices
RUN groupadd -r mlopsgroup && useradd -r -g mlopsgroup mlopsuser

# Copy application source code, models, and data
COPY src /app/src
COPY models /app/models
COPY data /app/data

# Create logs directory with write permissions for non-root user
RUN mkdir -p /app/logs && chown -R mlopsuser:mlopsgroup /app

# Switch to non-root user
USER mlopsuser

# Expose API port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI Server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
