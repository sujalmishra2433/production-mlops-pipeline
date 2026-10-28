"""Ground-truth evaluation and production performance monitoring."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from mlops_platform.utils.logging import logger


def evaluate_production_ground_truth(
    ground_truth_df: pd.DataFrame,
    prediction_log_path: str | Path = "reports/prediction_traffic.jsonl",
    output_report_path: str | Path = "reports/performance_drift_report.json",
    join_key: str = "customer_id",
    target_col: str = "churn",
) -> dict[str, Any]:
    """Calculate actual observed model performance by joining delayed ground-truth labels with predictions.

    NOTE: True model performance CANNOT be calculated from inferences alone.
    Ground-truth labels collected over time (e.g. after customer 30-day billing cycle)
    must be ingested to compute actual production accuracy, precision, recall, and ROC-AUC.

    Args:
        ground_truth_df: DataFrame containing customer_id and actual churn status.
        prediction_log_path: Path to persisted prediction logs.
        output_report_path: Destination path for performance monitoring JSON report.
        join_key: Column name used to join ground truth with logged inferences.
        target_col: Column name containing true binary churn labels.

    Returns:
        dict[str, Any]: Production performance evaluation metrics.
    """
    logger.info("Evaluating production model against observed ground truth (%d records)", len(ground_truth_df))

    log_path = Path(prediction_log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Prediction log not found at {log_path}")

    # Read predictions
    predictions: list[dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    predictions.append(json.loads(line))
                except Exception:
                    continue

    if not predictions:
        raise ValueError("Prediction log is empty. No inferences available to evaluate.")

    preds_df = pd.DataFrame(predictions)

    # Join on customer_id
    merged = pd.merge(
        ground_truth_df[[join_key, target_col]],
        preds_df[[join_key, "prediction", "probability", "model_version"]],
        on=join_key,
        how="inner",
    )

    if len(merged) == 0:
        logger.warning("No matches found between ground-truth and prediction logs on %s", join_key)
        return {
            "status": "no_matched_records",
            "message": "Ground-truth IDs did not match any logged prediction IDs.",
        }

    y_true = merged[target_col].astype(int)
    y_pred = merged["prediction"].astype(int)
    y_prob = merged["probability"].astype(float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "evaluated_records": len(merged),
        "unique_model_versions": merged["model_version"].unique().tolist(),
        "observed_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
        },
        "true_churn_rate": float(y_true.mean()),
        "predicted_churn_rate": float(y_pred.mean()),
        "note": "Performance metrics verified against verified delayed ground-truth feedback.",
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Production ground truth performance report saved to %s", out_file)
    return report
