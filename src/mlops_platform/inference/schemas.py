"""Pydantic v2 schemas for API requests, responses, and validation contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Features schema representing a single customer."""

    customer_id: str = Field(default="CUST-SAMPLE", description="Unique customer identifier")
    tenure: int = Field(..., ge=0, le=120, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0.0, le=500.0, description="Monthly billing amount in USD")
    total_charges: float = Field(..., ge=0.0, le=50000.0, description="Total lifetime billing amount in USD")
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract term type"
    )
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ] = Field(..., description="Payment billing method")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service type"
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tech support add-on subscription status"
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Online security add-on subscription status"
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ..., description="Paperless billing preference"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST-10492",
                "tenure": 12,
                "monthly_charges": 79.85,
                "total_charges": 958.20,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "tech_support": "No",
                "online_security": "No",
                "paperless_billing": "Yes",
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""

    records: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of customer feature records"
    )


class PredictionResult(BaseModel):
    """Individual prediction outcome."""

    customer_id: str
    churn_prediction: int = Field(..., description="0 = Retained, 1 = Churned")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn")
    risk_tier: Literal["Low", "Medium", "High"] = Field(..., description="Risk classification")


class PredictionResponse(BaseModel):
    """Single prediction API response."""

    success: bool = True
    result: PredictionResult
    model_name: str
    model_version: str
    request_id: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction API response."""

    success: bool = True
    results: list[PredictionResult]
    record_count: int
    model_name: str
    model_version: str
    request_id: str
    total_latency_ms: float


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    model_version: str
    run_id: str
    alias: str
    expected_features: list[str]
    status: str
    promoted_at_utc: str | None = None


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = "healthy"
    timestamp_utc: str


class ReadyResponse(BaseModel):
    """Readiness probe response."""

    status: str
    model_loaded: bool
    model_version: str | None = None
    message: str
