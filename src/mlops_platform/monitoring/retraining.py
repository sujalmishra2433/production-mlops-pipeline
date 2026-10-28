"""Automated retraining workflow with transactional quality gate safety."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_platform.data.ingestion import ingest_data
from mlops_platform.data.validation import validate_dataset
from mlops_platform.registry.register_model import register_and_promote_model
from mlops_platform.training.train import run_training
from mlops_platform.utils.config import load_config
from mlops_platform.utils.logging import logger


def trigger_retraining(
    data_path: str | Path | None = None,
    config_path: str | Path = "configs/config.yaml",
    force_promotion: bool = False,
    trigger_reason: str = "manual_trigger",
) -> dict[str, Any]:
    """Execute end-to-end retraining with transactional safety guarantees.

    Guarantees:
    - Retraining strictly validates latest data before training.
    - If validation or quality gate fails, previous champion model remains active.
    - Only promoted if candidate model passes quality gates and meets threshold.

    Args:
        data_path: Optional path to updated data. If None, uses configured raw data path.
        config_path: Path to YAML configuration.
        force_promotion: If True, bypasses baseline comparison (still enforces absolute gates).
        trigger_reason: Identifier for what triggered this retraining run.

    Returns:
        dict[str, Any]: Status summary of retraining operation.
    """
    logger.info("Starting automated retraining workflow [Reason: %s]", trigger_reason)
    config = load_config(config_path)

    # 1. Step: Ingest / Verify Data
    raw_path = Path(data_path or config["data"]["raw_data_path"])
    if not raw_path.exists():
        logger.info("Raw data missing. Ingesting fresh dataset...")
        df, meta = ingest_data(output_path=raw_path, n_samples=5000, seed=config["project"]["random_seed"])
    else:
        df = pd.read_csv(raw_path)

    # 2. Step: Data Quality Validation
    logger.info("Validating new dataset before training...")
    try:
        is_valid, val_report = validate_dataset(
            df=df,
            include_target=True,
            report_path=config["monitoring"].get("validation_report_path", "reports/validation_report.json"),
            raise_on_error=True,
        )
    except Exception as e:
        logger.error("Retraining ABORTED: Data validation failed (%s). Existing champion PRESERVED.", e)
        return {
            "status": "failed",
            "stage": "data_validation",
            "message": f"Data validation failed: {str(e)}",
            "champion_preserved": True,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    # 3. Step: Execute Training & Model Evaluation
    logger.info("Executing model training and candidate selection...")
    try:
        training_results = run_training(config_path=config_path, tune=False)
    except Exception as e:
        logger.error("Retraining ABORTED: Training pipeline crashed (%s). Existing champion PRESERVED.", e)
        return {
            "status": "failed",
            "stage": "training",
            "message": f"Training failed: {str(e)}",
            "champion_preserved": True,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    winner_name = training_results["winner_name"]
    winner_run_id = training_results["winner_run_id"]
    val_metrics = training_results["val_metrics"]

    baseline_metrics = None
    if not force_promotion and "logistic_regression" in training_results["candidates"]:
        baseline_metrics = training_results["candidates"]["logistic_regression"]["val_metrics"]

    # 4. Step: Quality Gate & Safe Registry Promotion
    logger.info("Enforcing quality gates on candidate model '%s' (run_id: %s)...", winner_name, winner_run_id)
    feature_cols = training_results["feature_names"]
    sample_features = df[feature_cols].iloc[:10]

    try:
        promotion_meta = register_and_promote_model(
            run_id=winner_run_id,
            val_metrics=val_metrics,
            baseline_metrics=baseline_metrics,
            sample_features=sample_features,
            config_path=config_path,
        )
        logger.info(
            "Retraining SUCCEEDED! New model promoted: Version %s [F1: %.4f, AUC: %.4f]",
            promotion_meta["promoted_version"],
            val_metrics.get("val_f1_score", 0.0),
            val_metrics.get("val_roc_auc", 0.0),
        )
        return {
            "status": "success",
            "stage": "promoted",
            "trigger_reason": trigger_reason,
            "promoted_version": promotion_meta["promoted_version"],
            "model_name": promotion_meta["model_registry_name"],
            "run_id": winner_run_id,
            "metrics": {k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.warning(
            "Retraining BLOCKED BY QUALITY GATE: %s. Previous champion PRESERVED safely.", e
        )
        return {
            "status": "blocked_by_quality_gate",
            "stage": "quality_gate",
            "message": str(e),
            "champion_preserved": True,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }


if __name__ == "__main__":
    result = trigger_retraining(trigger_reason="cli_direct_execution")
    print(json.dumps(result, indent=2))
