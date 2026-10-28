"""Model training and MLflow experiment tracking."""

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_platform.data.preprocessing import (
    build_preprocessor_pipeline,
    save_processed_data,
    split_data,
)
from mlops_platform.training.evaluate import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_evaluation_report,
)
from mlops_platform.training.tune import tune_logistic_regression, tune_random_forest
from mlops_platform.utils.config import load_config
from mlops_platform.utils.logging import logger


def run_training(
    config_path: str | Path = "configs/config.yaml",
    tune: bool = False,
) -> dict[str, Any]:
    """Execute complete training and model selection pipeline with MLflow tracking.

    Args:
        config_path: Path to configuration YAML.
        tune: Whether to run hyperparameter tuning.

    Returns:
        dict[str, Any]: Summary of winning model, run ID, and metrics.
    """
    config = load_config(config_path)

    # 1. Setup MLflow
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "customer_churn_prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured: URI=%s | Experiment=%s", tracking_uri, experiment_name)

    # 2. Load dataset
    raw_path = Path(config["data"]["raw_data_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    df = pd.read_csv(raw_path)
    target_col = config["data"]["target_column"]
    num_features = config["data"]["features"]["numerical"]
    cat_features = config["data"]["features"]["categorical"]
    seed = config["project"]["random_seed"]

    # Read dataset metadata if available
    meta_path = raw_path.with_suffix(".meta.json")
    dataset_fingerprint = "unknown"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            dataset_fingerprint = meta.get("fingerprint_sha256", "unknown")

    # 3. Split dataset
    train_df, val_df, test_df = split_data(
        df=df,
        target_col=target_col,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=seed,
    )

    save_processed_data(
        train_df,
        val_df,
        test_df,
        processed_dir=config["data"]["processed_dir"],
        reference_path=config["data"]["reference_data_path"],
    )

    feature_cols = num_features + cat_features
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    candidates: dict[str, dict[str, Any]] = {}
    selection_metric = config["training"].get("selection_metric", "f1_score")
    val_metric_key = f"val_{selection_metric}"

    # ----------------------------------------------------
    # Model 1: Logistic Regression (Baseline)
    # ----------------------------------------------------
    with mlflow.start_run(run_name="baseline_logistic_regression") as run_lr:
        logger.info("Training Candidate: Baseline Logistic Regression")
        preprocessor_lr = build_preprocessor_pipeline(num_features, cat_features)

        if tune:
            pipeline_lr, best_params_lr = tune_logistic_regression(
                preprocessor_lr, X_train, y_train, random_state=seed
            )
            mlflow.log_params(best_params_lr)
        else:
            lr_cfg = config["training"].get("logistic_regression", {})
            pipeline_lr = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_lr),
                    (
                        "classifier",
                        LogisticRegression(
                            C=lr_cfg.get("C", 1.0),
                            max_iter=lr_cfg.get("max_iter", 1000),
                            solver=lr_cfg.get("solver", "lbfgs"),
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline_lr.fit(X_train, y_train)
            mlflow.log_params(lr_cfg)

        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("role", "baseline")
        mlflow.set_tag("dataset_fingerprint", dataset_fingerprint)

        # Evaluate on train and validation
        train_metrics_lr = evaluate_model(pipeline_lr, X_train, y_train, dataset_name="train")
        val_metrics_lr = evaluate_model(pipeline_lr, X_val, y_val, dataset_name="val")

        for k, v in {**train_metrics_lr, **val_metrics_lr}.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # Signature & logging model artifact
        signature_lr = infer_signature(X_val[:5], pipeline_lr.predict(X_val[:5]))
        mlflow.sklearn.log_model(
            sk_model=pipeline_lr,
            artifact_path="model",
            signature=signature_lr,
            input_example=X_val[:3],
            skops_trusted_types=[
                "mlops_platform.features.build_features.ChurnFeatureEngineer",
                "numpy.dtype",
                "numpy.float64",
                "numpy.int64",
                "sklearn.tree._tree.Tree",
            ],
        )

        candidates["logistic_regression"] = {
            "run_id": run_lr.info.run_id,
            "pipeline": pipeline_lr,
            "val_metrics": val_metrics_lr,
            "score": val_metrics_lr.get(val_metric_key, 0.0),
        }

    # ----------------------------------------------------
    # Model 2: Random Forest (Challenger)
    # ----------------------------------------------------
    with mlflow.start_run(run_name="challenger_random_forest") as run_rf:
        logger.info("Training Candidate: Challenger Random Forest")
        preprocessor_rf = build_preprocessor_pipeline(num_features, cat_features)

        if tune:
            pipeline_rf, best_params_rf = tune_random_forest(
                preprocessor_rf, X_train, y_train, random_state=seed
            )
            mlflow.log_params(best_params_rf)
        else:
            rf_cfg = config["training"].get("random_forest", {})
            pipeline_rf = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_rf),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=rf_cfg.get("n_estimators", 100),
                            max_depth=rf_cfg.get("max_depth", 8),
                            min_samples_split=rf_cfg.get("min_samples_split", 5),
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline_rf.fit(X_train, y_train)
            mlflow.log_params(rf_cfg)

        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("role", "challenger")
        mlflow.set_tag("dataset_fingerprint", dataset_fingerprint)

        # Evaluate on train and validation
        train_metrics_rf = evaluate_model(pipeline_rf, X_train, y_train, dataset_name="train")
        val_metrics_rf = evaluate_model(pipeline_rf, X_val, y_val, dataset_name="val")

        for k, v in {**train_metrics_rf, **val_metrics_rf}.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # Signature & logging model artifact
        signature_rf = infer_signature(X_val[:5], pipeline_rf.predict(X_val[:5]))
        mlflow.sklearn.log_model(
            sk_model=pipeline_rf,
            artifact_path="model",
            signature=signature_rf,
            input_example=X_val[:3],
            skops_trusted_types=[
                "mlops_platform.features.build_features.ChurnFeatureEngineer",
                "numpy.dtype",
                "numpy.float64",
                "numpy.int64",
                "sklearn.tree._tree.Tree",
            ],
        )

        candidates["random_forest"] = {
            "run_id": run_rf.info.run_id,
            "pipeline": pipeline_rf,
            "val_metrics": val_metrics_rf,
            "score": val_metrics_rf.get(val_metric_key, 0.0),
        }

    # ----------------------------------------------------
    # Model Selection based on validation score
    # ----------------------------------------------------
    winner_name = max(candidates, key=lambda k: candidates[k]["score"])
    winner = candidates[winner_name]
    logger.info(
        "Candidate comparison based on %s: %s (Winner: %s with %.4f)",
        val_metric_key,
        {k: round(v["score"], 4) for k, v in candidates.items()},
        winner_name,
        winner["score"],
    )

    # Evaluate winner on unseen test set
    test_metrics = evaluate_model(winner["pipeline"], X_test, y_test, dataset_name="test")

    # Generate and save confusion matrix on validation
    val_cm = winner["val_metrics"]["confusion_matrix"]
    cm_matrix = np.array(
        [
            [val_cm["true_negatives"], val_cm["false_positives"]],
            [val_cm["false_negatives"], val_cm["true_positives"]],
        ]
    )
    cm_plot_path = save_confusion_matrix_plot(
        cm_matrix, output_path=config["monitoring"].get("confusion_matrix_path", "reports/confusion_matrix.png")
    )

    evaluation_report = {
        "selection_metric": val_metric_key,
        "winner_name": winner_name,
        "winner_run_id": winner["run_id"],
        "dataset_fingerprint": dataset_fingerprint,
        "candidates": {
            name: {
                "run_id": data["run_id"],
                "val_metrics": {k: v for k, v in data["val_metrics"].items() if k != "confusion_matrix"},
            }
            for name, data in candidates.items()
        },
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
    }
    eval_report_path = save_evaluation_report(
        evaluation_report, output_path=config["monitoring"].get("evaluation_report_path", "reports/evaluation_metrics.json")
    )

    # Log test metrics and report into winner run
    with mlflow.start_run(run_id=winner["run_id"]):
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.log_artifact(cm_plot_path)
        mlflow.log_artifact(eval_report_path)

    return {
        "winner_name": winner_name,
        "winner_run_id": winner["run_id"],
        "winner_pipeline": winner["pipeline"],
        "val_metrics": winner["val_metrics"],
        "test_metrics": test_metrics,
        "candidates": candidates,
        "feature_names": feature_cols,
    }
