"""Structured logging setup for MLOps platform."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Any

import yaml


def setup_logger(
    name: str = "mlops_platform",
    config_path: str | Path | None = None,
    default_level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a structured logger instance.

    Args:
        name: Name of the logger.
        config_path: Optional path to logging YAML configuration.
        default_level: Fallback log level if config file not found.

    Returns:
        logging.Logger: Configured logger.
    """
    if config_path is None:
        # Check standard config directory
        candidate = Path("configs/logging.yaml")
        if candidate.exists():
            config_path = candidate

    configured = False
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config: dict[str, Any] = yaml.safe_load(f)
            logging.config.dictConfig(config)
            configured = True
        except Exception as e:
            # Fall back to basic configuration
            logging.basicConfig(level=default_level)
            logger = logging.getLogger(name)
            logger.warning("Failed to load logging config from %s: %s", config_path, e)
            return logger

    if not configured:
        log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, log_level_env, default_level)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return logging.getLogger(name)


logger = setup_logger()
