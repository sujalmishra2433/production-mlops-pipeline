"""Unit tests for data drift detection and ground-truth performance evaluation."""

import json

import numpy as np
import pandas as pd

from mlops_platform.monitoring.drift import calculate_psi, detect_drift
from mlops_platform.monitoring.performance import evaluate_production_ground_truth
from scripts.generate_sample_data import generate_drifted_data


def test_psi_identical_distributions():
    """Verify PSI of identical series is close to zero."""
    vals = np.random.normal(50, 10, size=500)
    psi = calculate_psi(vals, vals)
    assert psi < 0.05


def test_psi_shifted_distributions():
    """Verify PSI of significantly shifted series is large (> 0.20)."""
    ref = np.random.normal(50, 10, size=500)
    cur = np.random.normal(85, 10, size=500)
    psi = calculate_psi(ref, cur)
    assert psi > 0.20


def test_detect_drift_no_drift(sample_churn_df, tmp_path):
    """Verify that identical current and reference datasets show zero drift."""
    report_file = tmp_path / "drift_no_drift.json"
    report = detect_drift(
        reference_df=sample_churn_df,
        current_df=sample_churn_df,
        report_path=report_file,
    )
    assert report["dataset_drift_detected"] is False
    assert report["drifted_features_count"] == 0
    assert report_file.exists()


def test_detect_drift_flagged_on_shifted_data(sample_churn_df, tmp_path):
    """Verify that drifted synthetic data triggers dataset_drift_detected."""
    drifted_df = generate_drifted_data(n_samples=200, seed=123)
    report_file = tmp_path / "drift_shifted.json"
    report = detect_drift(
        reference_df=sample_churn_df,
        current_df=drifted_df,
        report_path=report_file,
    )
    assert report["drifted_features_count"] > 0
    assert report["drift_share"] > 0.0


def test_ground_truth_performance_evaluation(tmp_path):
    """Verify joining ground-truth labels with prediction logs computes performance."""
    log_file = tmp_path / "predictions.jsonl"
    report_file = tmp_path / "perf_report.json"

    # Write dummy prediction logs
    log_entries = [
        {"customer_id": "C-1", "prediction": 0, "probability": 0.12, "model_version": "1"},
        {"customer_id": "C-2", "prediction": 1, "probability": 0.88, "model_version": "1"},
        {"customer_id": "C-3", "prediction": 1, "probability": 0.75, "model_version": "1"},
        {"customer_id": "C-4", "prediction": 0, "probability": 0.20, "model_version": "1"},
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    ground_truth = pd.DataFrame(
        [
            {"customer_id": "C-1", "churn": 0},
            {"customer_id": "C-2", "churn": 1},
            {"customer_id": "C-3", "churn": 0},
            {"customer_id": "C-4", "churn": 0},
        ]
    )

    report = evaluate_production_ground_truth(
        ground_truth_df=ground_truth,
        prediction_log_path=log_file,
        output_report_path=report_file,
    )

    assert report["evaluated_records"] == 4
    assert "accuracy" in report["observed_metrics"]
    assert report["observed_metrics"]["accuracy"] == 0.75
    assert report_file.exists()
