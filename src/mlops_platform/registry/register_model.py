"""Quality gates enforcement and MLflow model registry promotion."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops_platform.utils.config import load_config
from mlops_platform.utils.logging import logger


class QualityGateFailureError(Exception):
    """Raised when a candidate model fails automated promotion quality gates."""

    pass


def check_quality_gate(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    config_path: str | Path = "configs/config.yaml",
) -> tuple[bool, list[str]]:
    """Evaluate candidate model against strict quality gates.

    Checks:
    1. Metrics exist and are finite numbers.
    2. Candidate ROC-AUC >= min_roc_auc threshold.
    3. Candidate F1 >= min_f1_score threshold.
    4. Inference latency <= max_inference_latency_ms.
    5. Candidate beats or meets baseline model metrics (if enabled).

    Args:
        metrics: Candidate model evaluation metrics dictionary.
        baseline_metrics: Optional baseline model metrics dictionary.
        config_path: Path to configuration YAML.

    Returns:
        tuple[bool, list[str]]: (passed, failure_reasons)
    """
    config = load_config(config_path)
    gate_cfg = config.get("quality_gate", {})

    min_auc = gate_cfg.get("min_roc_auc", 0.70)
    min_f1 = gate_cfg.get("min_f1_score", 0.60)
    max_latency = gate_cfg.get("max_inference_latency_ms", 50.0)
    must_beat = gate_cfg.get("must_beat_baseline", True)

    failures: list[str] = []

    # Identify val metrics
    auc = metrics.get("val_roc_auc")
    f1 = metrics.get("val_f1_score")
    latency = metrics.get("val_latency_ms", 0.0)

    # 1. Finite check
    for name, val in [("val_roc_auc", auc), ("val_f1_score", f1)]:
        if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            failures.append(f"Metric '{name}' is missing or non-finite ({val})")

    if failures:
        return False, failures

    # 2. Minimum absolute thresholds
    if auc is not None and auc < min_auc:
        failures.append(f"Validation ROC-AUC ({auc:.4f}) is below minimum threshold ({min_auc:.4f})")

    if f1 is not None and f1 < min_f1:
        failures.append(f"Validation F1 ({f1:.4f}) is below minimum threshold ({min_f1:.4f})")

    if latency > max_latency:
        failures.append(
            f"Validation inference latency ({latency:.2f} ms) exceeds limit ({max_latency:.2f} ms)"
        )

    # 3. Compare with baseline
    if must_beat and baseline_metrics:
        baseline_f1 = baseline_metrics.get("val_f1_score", 0.0)
        if f1 is not None and f1 < baseline_f1:
            failures.append(
                f"Candidate F1 ({f1:.4f}) did not beat baseline F1 ({baseline_f1:.4f})"
            )

    passed = len(failures) == 0
    return passed, failures


def run_serialization_smoke_test(
    model_uri: str,
    sample_features: pd.DataFrame,
) -> bool:
    """Verify that a logged MLflow model artifact can be reloaded and executes inference.

    Args:
        model_uri: MLflow model URI (e.g. 'runs:/<run_id>/model').
        sample_features: DataFrame containing sample rows.

    Returns:
        bool: True if reload and inference succeed.
    """
    logger.info("Running serialization smoke test on %s", model_uri)
    _TRUSTED_TYPES = [
        "mlops_platform.features.build_features.ChurnFeatureEngineer",
        "numpy.dtype",
        "numpy.float64",
        "numpy.int64",
    ]
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri, dst_path=None)
        preds = loaded_model.predict(sample_features)
        if len(preds) != len(sample_features):
            logger.error("Smoke test failed: output shape mismatch (%d != %d)", len(preds), len(sample_features))
            return False

        if hasattr(loaded_model, "predict_proba"):
            probs = loaded_model.predict_proba(sample_features)
            if not np.all((probs >= 0.0) & (probs <= 1.0)):
                logger.error("Smoke test failed: probabilities out of [0, 1] range")
                return False

        logger.info("Serialization smoke test passed successfully.")
        return True
    except Exception as exc:
        logger.error("Serialization smoke test failed with exception: %s", exc)
        return False


def register_and_promote_model(
    run_id: str,
    val_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    sample_features: pd.DataFrame | None = None,
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any]:
    """Execute quality gate check, smoke test, and promote model to MLflow Model Registry.

    Args:
        run_id: Winning MLflow run ID.
        val_metrics: Candidate validation metrics.
        baseline_metrics: Optional baseline metrics.
        sample_features: Sample feature data for serialization smoke test.
        config_path: Path to configuration YAML.

    Returns:
        dict[str, Any]: Model promotion details and metadata.

    Raises:
        QualityGateFailureError: If quality gate or smoke test fails.
    """
    config = load_config(config_path)
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    registry_name = mlflow_cfg.get("model_registry_name", "customer_churn_classifier")
    champion_alias = mlflow_cfg.get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Quality gate check
    passed, failure_reasons = check_quality_gate(
        val_metrics, baseline_metrics=baseline_metrics, config_path=config_path
    )
    if not passed:
        msg = f"Model promotion rejected! Quality gate failures: {'; '.join(failure_reasons)}"
        logger.error(msg)
        raise QualityGateFailureError(msg)

    # 2. Serialization smoke test
    model_uri = f"runs:/{run_id}/model"
    if sample_features is not None:
        if not run_serialization_smoke_test(model_uri, sample_features):
            msg = "Model promotion rejected! Serialization smoke test failed."
            logger.error(msg)
            raise QualityGateFailureError(msg)

    # 3. MLflow Model Registry Registration
    logger.info("Registering model in MLflow Model Registry: '%s'", registry_name)
    model_version_info = mlflow.register_model(
        model_uri=model_uri,
        name=registry_name,
    )

    version = str(model_version_info.version)
    logger.info("Registered model version: %s", version)

    # Set champion alias and tag
    try:
        client.set_registered_model_alias(
            name=registry_name,
            alias=champion_alias,
            version=version,
        )
        logger.info("Assigned alias '%s' to version %s", champion_alias, version)
    except Exception as e:
        logger.warning("Could not set alias on model registry (%s). Setting tag instead.", e)
        client.set_model_version_tag(
            name=registry_name,
            version=version,
            key="stage",
            value="Production",
        )

    # Save promotion metadata artifact for tracking and audit
    metadata = {
        "model_registry_name": registry_name,
        "promoted_version": version,
        "run_id": run_id,
        "model_uri": model_uri,
        "alias": champion_alias,
        "promoted_at_utc": datetime.now(UTC).isoformat(),
        "validation_metrics": {k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
        "rollback_procedure": (
            "To rollback, run: python -m mlops_platform.registry.register_model "
            "--rollback-to-version <PREVIOUS_VERSION>"
        ),
    }

    meta_path = Path("reports/champion_metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model successfully promoted to Champion (Version %s)!", version)
    return metadata
