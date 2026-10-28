"""Model evaluation, metrics computation, and artifact visualization."""

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlops_platform.utils.logging import logger


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    dataset_name: str = "validation",
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics and latency for a binary classifier.

    Args:
        model: Trained scikit-learn model or pipeline.
        X: Feature matrix.
        y: True binary labels.
        dataset_name: Name of the evaluated dataset partition.

    Returns:
        dict[str, Any]: Dictionary of evaluation metrics.
    """
    logger.info("Evaluating model on %s set (samples=%d)", dataset_name, len(X))

    # Measure latency
    start_time = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - start_time
    latency_ms = (inference_time / len(X)) * 1000.0

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, y_prob))
        loss = float(log_loss(y, y_prob))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        auc = float(roc_auc_score(y, scores))
        loss = None
    else:
        auc = None
        loss = None

    precision = float(precision_score(y, y_pred, zero_division=0))
    recall = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        f"{dataset_name}_accuracy": accuracy,
        f"{dataset_name}_precision": precision,
        f"{dataset_name}_recall": recall,
        f"{dataset_name}_f1_score": f1,
        f"{dataset_name}_roc_auc": auc,
        f"{dataset_name}_log_loss": loss,
        f"{dataset_name}_latency_ms": latency_ms,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    logger.info(
        "[%s] F1: %.4f | ROC-AUC: %.4f | Precision: %.4f | Recall: %.4f | Latency: %.2f ms/rec",
        dataset_name.upper(),
        f1,
        auc if auc is not None else 0.0,
        precision,
        recall,
        latency_ms,
    )
    return metrics


def save_confusion_matrix_plot(
    cm: np.ndarray,
    output_path: str | Path = "reports/confusion_matrix.png",
    labels: tuple[str, str] = ("Retained", "Churned"),
) -> str:
    """Generate and save confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array.
        output_path: Destination image path.
        labels: Class label names.

    Returns:
        str: Absolute path of saved plot.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title("Model Confusion Matrix (Validation Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(dest, dpi=150)
    plt.close()

    logger.info("Confusion matrix saved to %s", dest)
    return str(dest.resolve())


def save_evaluation_report(
    report: dict[str, Any],
    output_path: str | Path = "reports/evaluation_metrics.json",
) -> str:
    """Save evaluation metrics to JSON.

    Args:
        report: Metrics dictionary.
        output_path: Path to save JSON report.

    Returns:
        str: Absolute path of saved report.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report saved to %s", dest)
    return str(dest.resolve())
