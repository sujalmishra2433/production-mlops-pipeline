"""Configuration management with YAML loading and environment variable overrides."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file with environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Environment variable overrides
    if "MLFLOW_TRACKING_URI" in os.environ:
        config.setdefault("mlflow", {})["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

    if "MLFLOW_EXPERIMENT_NAME" in os.environ:
        config.setdefault("mlflow", {})["experiment_name"] = os.environ["MLFLOW_EXPERIMENT_NAME"]

    if "MODEL_REGISTRY_NAME" in os.environ:
        config.setdefault("mlflow", {})["model_registry_name"] = os.environ["MODEL_REGISTRY_NAME"]

    if "MIN_ROC_AUC_THRESHOLD" in os.environ:
        try:
            config.setdefault("quality_gate", {})["min_roc_auc"] = float(
                os.environ["MIN_ROC_AUC_THRESHOLD"]
            )
        except ValueError:
            pass

    if "MIN_F1_THRESHOLD" in os.environ:
        try:
            config.setdefault("quality_gate", {})["min_f1_score"] = float(
                os.environ["MIN_F1_THRESHOLD"]
            )
        except ValueError:
            pass

    return config


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    # Find directory containing pyproject.toml
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "configs").exists():
            return current
        current = current.parent
    return Path.cwd()
