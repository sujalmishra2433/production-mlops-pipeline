"""Integration tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from mlops_platform.inference.predict import ModelInferenceService


@pytest.fixture
def client():
    """Create FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health liveness probe returns 200 healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp_utc" in data


def test_metrics_endpoint(client):
    """Verify /metrics exposes Prometheus metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_predict_validation_error(client):
    """Verify malformed payload returns 422 Unprocessable Entity."""
    response = client.post("/predict", json={"tenure": -5})
    assert response.status_code == 422


def test_predict_endpoint_with_mock_model(client, trained_pipeline, sample_customer_dict, monkeypatch):
    """Verify /predict returns valid prediction when model is loaded."""
    service = ModelInferenceService.get_instance()
    monkeypatch.setattr(service, "model", trained_pipeline)
    monkeypatch.setattr(service, "model_version", "1")
    monkeypatch.setattr(service, "model_name", "test_classifier")

    response = client.post("/predict", json=sample_customer_dict)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"]["customer_id"] == "CUST-99999"
    assert data["result"]["churn_prediction"] in [0, 1]
    assert 0.0 <= data["result"]["churn_probability"] <= 1.0
    assert data["result"]["risk_tier"] in ["Low", "Medium", "High"]
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_predict_batch_endpoint_with_mock_model(client, trained_pipeline, sample_customer_dict, monkeypatch):
    """Verify /predict/batch processes multiple records."""
    service = ModelInferenceService.get_instance()
    monkeypatch.setattr(service, "model", trained_pipeline)
    monkeypatch.setattr(service, "model_version", "1")
    monkeypatch.setattr(service, "model_name", "test_classifier")

    payload = {"records": [sample_customer_dict, sample_customer_dict]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["record_count"] == 2
    assert len(data["results"]) == 2


def test_readiness_probe_with_mock_model(client, trained_pipeline, monkeypatch):
    """Verify /ready reports ready when model is active."""
    service = ModelInferenceService.get_instance()
    monkeypatch.setattr(service, "model", trained_pipeline)
    monkeypatch.setattr(service, "model_version", "1")

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
