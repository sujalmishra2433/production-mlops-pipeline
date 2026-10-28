"""Pytest fixtures and test data configurations."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_platform.data.ingestion import generate_synthetic_churn_data
from mlops_platform.data.preprocessing import build_preprocessor_pipeline


@pytest.fixture(scope="session")
def sample_churn_df() -> pd.DataFrame:
    """Return a small deterministic DataFrame for unit tests."""
    return generate_synthetic_churn_data(n_samples=100, seed=42)


@pytest.fixture(scope="session")
def sample_customer_dict() -> dict:
    """Return a valid customer dictionary for API & inference testing."""
    return {
        "customer_id": "CUST-99999",
        "tenure": 24,
        "monthly_charges": 65.50,
        "total_charges": 1572.00,
        "contract_type": "One year",
        "payment_method": "Credit card",
        "internet_service": "DSL",
        "tech_support": "Yes",
        "online_security": "Yes",
        "paperless_billing": "No",
    }


@pytest.fixture(scope="session")
def trained_pipeline(sample_churn_df) -> Pipeline:
    """Return a fitted model pipeline for fast unit testing."""
    num_cols = ["tenure", "monthly_charges", "total_charges"]
    cat_cols = [
        "contract_type",
        "payment_method",
        "internet_service",
        "tech_support",
        "online_security",
        "paperless_billing",
    ]
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    feature_cols = num_cols + cat_cols
    X = sample_churn_df[feature_cols]
    y = sample_churn_df["churn"]
    pipe.fit(X, y)
    return pipe
