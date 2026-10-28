"""Unit tests for inference schemas, prediction validity, and probability bounds."""

import pytest
from pydantic import ValidationError

from mlops_platform.inference.schemas import (
    BatchPredictionRequest,
    CustomerFeatures,
    PredictionResult,
)


def test_customer_features_schema_valid(sample_customer_dict):
    """Verify that valid dictionary parses into CustomerFeatures without error."""
    customer = CustomerFeatures(**sample_customer_dict)
    assert customer.customer_id == "CUST-99999"
    assert customer.tenure == 24
    assert customer.monthly_charges == 65.50


def test_customer_features_schema_rejects_negative_tenure(sample_customer_dict):
    """Verify that negative tenure raises Pydantic ValidationError."""
    corrupted = sample_customer_dict.copy()
    corrupted["tenure"] = -5
    with pytest.raises(ValidationError):
        CustomerFeatures(**corrupted)


def test_customer_features_schema_rejects_negative_charges(sample_customer_dict):
    """Verify that negative charges raises Pydantic ValidationError."""
    corrupted = sample_customer_dict.copy()
    corrupted["monthly_charges"] = -10.0
    with pytest.raises(ValidationError):
        CustomerFeatures(**corrupted)


def test_customer_features_schema_rejects_invalid_contract(sample_customer_dict):
    """Verify that invalid contract category raises Pydantic ValidationError."""
    corrupted = sample_customer_dict.copy()
    corrupted["contract_type"] = "Permanent"
    with pytest.raises(ValidationError):
        CustomerFeatures(**corrupted)


def test_prediction_result_schema_bounds():
    """Verify that probability outside [0, 1] is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        PredictionResult(
            customer_id="CUST-1",
            churn_prediction=1,
            churn_probability=1.5,
            risk_tier="High",
        )


def test_batch_prediction_request_length_validation(sample_customer_dict):
    """Verify that empty batch list is rejected."""
    with pytest.raises(ValidationError):
        BatchPredictionRequest(records=[])
