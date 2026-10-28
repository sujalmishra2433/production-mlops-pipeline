"""End-to-end MLOps pipeline runner script."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mlops_platform.data.ingestion import ingest_data
from mlops_platform.data.validation import validate_dataset
from mlops_platform.registry.register_model import register_and_promote_model
from mlops_platform.training.train import run_training
from mlops_platform.utils.config import load_config
from mlops_platform.utils.logging import logger


def main() -> int:
    """Run full automated MLOps pipeline."""
    parser = argparse.ArgumentParser(description="Execute End-to-End MLOps Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic records if generating new dataset",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Force re-generation/re-download of raw dataset",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning during training",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = Path(config["data"]["raw_data_path"])

    print("=" * 70)
    print(">> PRODUCTION MLOPS PIPELINE: AUTOMATED ML LIFECYCLE")
    print("=" * 70)

    # Stage 1: Data Ingestion
    print("\n[Stage 1/5] Ingesting Dataset...")
    if args.force_ingest or not raw_path.exists():
        df, meta = ingest_data(
            output_path=raw_path,
            n_samples=args.samples,
            seed=config["project"]["random_seed"],
        )
    else:
        df = pd.read_csv(raw_path)
        logger.info("Using existing raw dataset at %s (%d rows)", raw_path, len(df))

    # Stage 2: Data Quality Validation
    print("\n[Stage 2/5] Running Pandera Schema & Quality Validation...")
    val_report_path = config["monitoring"].get("validation_report_path", "reports/validation_report.json")
    is_valid, report = validate_dataset(
        df=df,
        include_target=True,
        report_path=val_report_path,
        raise_on_error=True,
    )
    print(f"  [OK] Data Validation: PASSED (Rows: {len(df)}, Columns: {len(df.columns)})")

    # Stage 3 & 4: Training, Evaluation & Experiment Tracking
    print("\n[Stage 3/5] Preprocessing & Training Candidate Models (MLflow Tracking)...")
    training_results = run_training(config_path=args.config, tune=args.tune)

    winner = training_results["winner_name"]
    val_metrics = training_results["val_metrics"]
    test_metrics = training_results["test_metrics"]
    winner_run_id = training_results["winner_run_id"]

    print("\n[Stage 4/5] Evaluating Candidates & Comparing Against Baseline...")
    print("  Candidate Comparison:")
    for name, cand in training_results["candidates"].items():
        score = cand["score"]
        print(f"    - {name:22s} | Score: {score:.4f} | Run ID: {cand['run_id']}")
    print(f"  [WINNER] Winner Selected: {winner.upper()} (Run ID: {winner_run_id})")
    print(f"     Val F1: {val_metrics.get('val_f1_score', 0):.4f} | Val ROC-AUC: {val_metrics.get('val_roc_auc', 0):.4f}")
    print(f"     Test F1: {test_metrics.get('test_f1_score', 0):.4f} | Test ROC-AUC: {test_metrics.get('test_roc_auc', 0):.4f}")

    # Stage 5: Quality Gate & Model Registry Promotion
    print("\n[Stage 5/5] Enforcing Quality Gates & Promoting to MLflow Registry...")
    feature_cols = training_results["feature_names"]
    sample_features = df[feature_cols].iloc[:10]
    baseline_metrics = training_results["candidates"].get("logistic_regression", {}).get("val_metrics")

    promotion_meta = register_and_promote_model(
        run_id=winner_run_id,
        val_metrics=val_metrics,
        baseline_metrics=baseline_metrics,
        sample_features=sample_features,
        config_path=args.config,
    )

    print("\n" + "=" * 70)
    print("[DONE] PIPELINE EXECUTION COMPLETE: MODEL PROMOTED TO CHAMPION")
    print("=" * 70)
    print(f"  Model Name       : {promotion_meta['model_registry_name']}")
    print(f"  Promoted Version : {promotion_meta['promoted_version']}")
    print(f"  Alias            : {promotion_meta['alias']}")
    print(f"  MLflow Run ID    : {promotion_meta['run_id']}")
    print(f"  Artifact Path    : {promotion_meta['model_uri']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
