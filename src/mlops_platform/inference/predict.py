"""Inference engine with model loading, prediction caching, and request logging."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops_platform.inference.schemas import CustomerFeatures, PredictionResult
from mlops_platform.utils.config import load_config
from mlops_platform.utils.logging import logger


class ModelInferenceService:
    """Production model inference service with cached model instance and telemetry."""

    _instance = None

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.model: Any = None
        self.model_version: str = "unloaded"
        self.model_name: str = self.config.get("mlflow", {}).get(
            "model_registry_name", "customer_churn_classifier"
        )
        self.run_id: str = "unknown"
        self.alias: str = self.config.get("mlflow", {}).get("champion_alias", "champion")
        self.feature_columns: list[str] = (
            self.config.get("data", {}).get("features", {}).get("numerical", [])
            + self.config.get("data", {}).get("features", {}).get("categorical", [])
        )
        self.log_file = Path(
            self.config.get("monitoring", {}).get(
                "prediction_log_path", "reports/prediction_traffic.jsonl"
            )
        )
        self.load_model()

    @classmethod
    def get_instance(cls, config_path: str | Path = "configs/config.yaml") -> "ModelInferenceService":
        """Singleton accessor for inference service."""
        if cls._instance is None:
            cls._instance = cls(config_path=config_path)
        return cls._instance

    def load_model(self) -> bool:
        """Load the approved champion model from MLflow Model Registry or local fallback.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        mlflow_cfg = self.config.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        logger.info("Attempting to load model '%s' from MLflow tracking URI: %s", self.model_name, tracking_uri)

        # 1. Try loading from model alias (e.g. models:/customer_churn_classifier@champion)
        try:
            model_uri = f"models:/{self.model_name}@{self.alias}"
            self.model = mlflow.sklearn.load_model(model_uri)
            model_version_details = client.get_model_version_by_alias(self.model_name, self.alias)
            self.model_version = str(model_version_details.version)
            self.run_id = model_version_details.run_id
            logger.info(
                "Loaded Champion model version %s (run_id: %s) via alias @%s",
                self.model_version,
                self.run_id,
                self.alias,
            )
            return True
        except Exception as e:
            logger.warning("Could not load model via alias '%s': %s", self.alias, e)

        # 2. Try loading latest version from registry
        try:
            versions = client.get_latest_versions(self.model_name)
            if versions:
                latest = versions[-1]
                model_uri = f"models:/{self.model_name}/{latest.version}"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = str(latest.version)
                self.run_id = latest.run_id
                logger.info(
                    "Loaded latest registered model version %s (run_id: %s)",
                    self.model_version,
                    self.run_id,
                )
                return True
        except Exception as e:
            logger.warning("Could not load model via get_latest_versions: %s", e)

        # 3. Fallback to champion_metadata.json if present
        meta_path = Path("reports/champion_metadata.json")
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                run_id = meta.get("run_id")
                if run_id:
                    model_uri = f"runs:/{run_id}/model"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = str(meta.get("promoted_version", "1"))
                    self.run_id = run_id
                    logger.info("Loaded model from fallback champion metadata: %s", model_uri)
                    return True
            except Exception as e:
                logger.warning("Could not load model from champion metadata fallback: %s", e)

        logger.error("No champion model found in MLflow registry or local metadata!")
        self.model = None
        self.model_version = "unloaded"
        return False

    def is_ready(self) -> bool:
        """Check if model is loaded and responds to sample inference."""
        if self.model is None:
            return False

        sample_df = pd.DataFrame(
            [
                {
                    "tenure": 12,
                    "monthly_charges": 70.0,
                    "total_charges": 840.0,
                    "contract_type": "Month-to-month",
                    "payment_method": "Electronic check",
                    "internet_service": "Fiber optic",
                    "tech_support": "No",
                    "online_security": "No",
                    "paperless_billing": "Yes",
                }
            ]
        )
        try:
            preds = self.model.predict(sample_df)
            return len(preds) == 1
        except Exception as e:
            logger.error("Readiness probe check failed: %s", e)
            return False

    def predict_features(
        self,
        features_list: list[CustomerFeatures],
        request_id: str = "req-default",
    ) -> list[PredictionResult]:
        """Perform predictions on a list of CustomerFeatures records.

        Args:
            features_list: List of customer features.
            request_id: Tracing request identifier.

        Returns:
            list[PredictionResult]: Predictions with probabilities and risk tiers.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Ensure training pipeline has been executed.")

        # Convert to DataFrame
        records = [f.model_dump() for f in features_list]
        df = pd.DataFrame(records)

        # Retain customer_ids for mapping
        cust_ids = df["customer_id"].tolist()

        # Extract features expected by pipeline
        X = df[self.feature_columns]

        # Predict
        preds = self.model.predict(X)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = np.array([1.0 if p == 1 else 0.0 for p in preds])

        results: list[PredictionResult] = []
        log_entries: list[dict[str, Any]] = []
        now_iso = datetime.now(UTC).isoformat()

        for cid, pred, prob, rec in zip(cust_ids, preds, probs, records, strict=False):
            prob_float = float(round(prob, 4))
            pred_int = int(pred)

            if prob_float < 0.35:
                tier = "Low"
            elif prob_float < 0.70:
                tier = "Medium"
            else:
                tier = "High"

            res = PredictionResult(
                customer_id=cid,
                churn_prediction=pred_int,
                churn_probability=prob_float,
                risk_tier=tier,
            )
            results.append(res)

            # Build telemetry log entry for drift monitoring
            log_entries.append(
                {
                    "request_id": request_id,
                    "timestamp_utc": now_iso,
                    "model_version": self.model_version,
                    "customer_id": cid,
                    "features": rec,
                    "prediction": pred_int,
                    "probability": prob_float,
                    "risk_tier": tier,
                }
            )

        # Append to prediction log file for drift analysis
        self._log_predictions(log_entries)
        return results

    def _log_predictions(self, log_entries: list[dict[str, Any]]) -> None:
        """Asynchronously or synchronously persist prediction telemetry."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                for entry in log_entries:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("Failed to write prediction telemetry log: %s", e)
