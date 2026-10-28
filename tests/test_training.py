"""Unit tests for training, feature engineering, and model quality gates."""

import numpy as np
import pandas as pd

from mlops_platform.data.preprocessing import split_data
from mlops_platform.features.build_features import ChurnFeatureEngineer
from mlops_platform.registry.register_model import check_quality_gate


def test_split_data_reproducibility_and_stratification(sample_churn_df):
    """Verify that train/val/test splits preserve class ratio and are reproducible."""
    train1, val1, test1 = split_data(sample_churn_df, test_size=0.20, val_size=0.20, random_state=42)
    train2, val2, test2 = split_data(sample_churn_df, test_size=0.20, val_size=0.20, random_state=42)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)

    # Check stratification
    base_churn = sample_churn_df["churn"].mean()
    assert abs(train1["churn"].mean() - base_churn) < 0.10


def test_feature_engineering_transformer(sample_churn_df):
    """Verify that ChurnFeatureEngineer computes engineered columns without leakage."""
    engineer = ChurnFeatureEngineer()
    transformed = engineer.fit_transform(sample_churn_df)

    assert "charge_ratio" in transformed.columns
    assert "monthly_to_total_ratio" in transformed.columns
    assert "has_support_services" in transformed.columns
    assert "is_long_term_contract" in transformed.columns

    # Verify no NaN or Inf created
    assert not transformed["charge_ratio"].isnull().any()
    assert not np.isinf(transformed["charge_ratio"]).any()


def test_quality_gate_passes_on_strong_metrics(tmp_path):
    """Verify quality gate passes when candidate beats baseline and exceeds thresholds."""
    metrics = {
        "val_roc_auc": 0.85,
        "val_f1_score": 0.75,
        "val_latency_ms": 1.5,
    }
    baseline_metrics = {
        "val_roc_auc": 0.80,
        "val_f1_score": 0.70,
    }
    passed, failures = check_quality_gate(metrics, baseline_metrics=baseline_metrics)
    assert passed is True
    assert len(failures) == 0


def test_quality_gate_fails_on_low_auc():
    """Verify quality gate rejects candidate with ROC-AUC below 0.70."""
    metrics = {
        "val_roc_auc": 0.65,
        "val_f1_score": 0.72,
        "val_latency_ms": 2.0,
    }
    passed, failures = check_quality_gate(metrics)
    assert passed is False
    assert any("ROC-AUC" in f for f in failures)


def test_quality_gate_fails_on_low_f1():
    """Verify quality gate rejects candidate with F1 below 0.60."""
    metrics = {
        "val_roc_auc": 0.82,
        "val_f1_score": 0.52,
        "val_latency_ms": 2.0,
    }
    passed, failures = check_quality_gate(metrics)
    assert passed is False
    assert any("F1" in f for f in failures)


def test_quality_gate_fails_on_nan_or_missing_metric():
    """Verify quality gate rejects candidate with NaN or missing metrics."""
    metrics = {
        "val_roc_auc": float("nan"),
        "val_f1_score": 0.72,
        "val_latency_ms": 2.0,
    }
    passed, failures = check_quality_gate(metrics)
    assert passed is False
    assert any("missing or non-finite" in f for f in failures)
