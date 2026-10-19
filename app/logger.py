"""
Structured Logging Configuration.
Logs all API requests, latency, payload summary, and prediction outcomes.
"""

import sys
import logging
from datetime import datetime

# Configure standard root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("heart_disease_api")


def log_prediction(client_ip: str, features: dict, prediction: int, prob: float, latency_ms: float):
    """Logs structured prediction events for operational auditing and drift monitoring."""
    log_record = {
        "event": "INFERENCE_COMPLETED",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "client_ip": client_ip,
        "prediction": prediction,
        "probability": round(prob, 4),
        "latency_ms": round(latency_ms, 2),
        "age": features.get("age"),
        "sex": features.get("sex"),
        "cp": features.get("cp"),
        "trestbps": features.get("trestbps"),
        "chol": features.get("chol")
    }
    logger.info(f"AUDIT_LOG: {log_record}")
