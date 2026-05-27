"""Tests for the FastAPI application."""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from api.main import PredictionRequest, PredictionResponse, app


@pytest.fixture
def client():
    """Create a test client for the API.

    Returns:
        FastAPI TestClient instance.
    """
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock sklearn-like classifier for prediction tests.

    Returns:
        Mock model with predict and predict_proba.
    """
    model = Mock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.9, 0.1]])
    return model


@pytest.fixture
def mock_scaler():
    """Mock feature scaler for prediction tests.

    Returns:
        Mock scaler with transform.
    """
    scaler = Mock()
    scaler.transform.return_value = np.random.randn(1, 29)
    return scaler


def test_health_endpoint(client):
    """Return healthy status from /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_success(client, mock_model, mock_scaler):
    """Return a valid prediction when model and scaler are loaded."""
    with patch("api.main.load_champion_model") as mock_load:
        mock_load.return_value = (mock_model, mock_scaler)

        request_data = {
            "V1": -1.3598,
            "V2": 0.0724,
            "V3": 2.5363,
            "V4": 1.0985,
            "V5": -0.8723,
            "V6": 0.5785,
            "V7": -1.5285,
            "V8": 0.3525,
            "V9": 0.1234,
            "V10": -0.3456,
            "V11": 0.4567,
            "V12": -0.5678,
            "V13": 0.6789,
            "V14": -0.7890,
            "V15": 0.8901,
            "V16": -0.9012,
            "V17": 0.0123,
            "V18": -0.1234,
            "V19": 0.2345,
            "V20": -0.3456,
            "V21": 0.4567,
            "V22": -0.5678,
            "V23": 0.6789,
            "V24": -0.7890,
            "V25": 0.8901,
            "V26": -0.9012,
            "V27": 0.0123,
            "V28": -0.1234,
            "Amount": 149.62,
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "is_fraud" in data
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability"], float)
        assert isinstance(data["is_fraud"], bool)


def test_predict_endpoint_missing_field(client):
    """Return 422 when required fields are missing."""
    request_data = {
        "V1": -1.3598,
        "V2": 0.0724,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422


def test_predict_endpoint_invalid_type(client):
    """Return 422 when a field has an invalid type."""
    request_data = {
        "V1": "invalid",
        "V2": 0.0724,
        "V3": 2.5363,
        "V4": 1.0985,
        "V5": -0.8723,
        "V6": 0.5785,
        "V7": -1.5285,
        "V8": 0.3525,
        "V9": 0.1234,
        "V10": -0.3456,
        "V11": 0.4567,
        "V12": -0.5678,
        "V13": 0.6789,
        "V14": -0.7890,
        "V15": 0.8901,
        "V16": -0.9012,
        "V17": 0.0123,
        "V18": -0.1234,
        "V19": 0.2345,
        "V20": -0.3456,
        "V21": 0.4567,
        "V22": -0.5678,
        "V23": 0.6789,
        "V24": -0.7890,
        "V25": 0.8901,
        "V26": -0.9012,
        "V27": 0.0123,
        "V28": -0.1234,
        "Amount": 149.62,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422


def test_model_info_endpoint(client, mock_model, mock_scaler):
    """Return champion metadata from /model-info."""
    with patch("api.main.load_champion_model") as mock_load:
        mock_load.return_value = (mock_model, mock_scaler)

        metadata = {
            "run_id": "test_run_123",
            "pr_auc": 0.8542,
            "model_type": "RandomForestClassifier",
            "metrics": {"accuracy": 0.99},
            "params": {"n_estimators": 100},
        }

        with patch("api.main.load_champion_metadata") as mock_meta:
            mock_meta.return_value = metadata

            response = client.get("/model-info")

            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == "test_run_123"
            assert data["pr_auc"] == 0.8542
            assert data["model_type"] == "RandomForestClassifier"


def test_model_info_endpoint_no_model(client):
    """Return 503 when champion model is not loaded."""
    with patch("api.main.load_champion_model") as mock_load:
        mock_load.return_value = (None, None)

        response = client.get("/model-info")
        assert response.status_code == 503


def test_prediction_request_schema():
    """Validate PredictionRequest Pydantic schema."""
    data = {
        "V1": -1.3598,
        "V2": 0.0724,
        "V3": 2.5363,
        "V4": 1.0985,
        "V5": -0.8723,
        "V6": 0.5785,
        "V7": -1.5285,
        "V8": 0.3525,
        "V9": 0.1234,
        "V10": -0.3456,
        "V11": 0.4567,
        "V12": -0.5678,
        "V13": 0.6789,
        "V14": -0.7890,
        "V15": 0.8901,
        "V16": -0.9012,
        "V17": 0.0123,
        "V18": -0.1234,
        "V19": 0.2345,
        "V20": -0.3456,
        "V21": 0.4567,
        "V22": -0.5678,
        "V23": 0.6789,
        "V24": -0.7890,
        "V25": 0.8901,
        "V26": -0.9012,
        "V27": 0.0123,
        "V28": -0.1234,
        "Amount": 149.62,
    }

    request = PredictionRequest(**data)
    assert request.V1 == -1.3598
    assert request.Amount == 149.62


def test_prediction_response_schema():
    """Validate PredictionResponse Pydantic schema."""
    response = PredictionResponse(
        prediction=0,
        probability=0.0234,
        is_fraud=False,
    )

    assert response.prediction == 0
    assert response.probability == 0.0234
    assert response.is_fraud is False
