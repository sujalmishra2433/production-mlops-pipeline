"""Data drift and prediction drift detection engine."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mlops_platform.utils.config import load_config
from mlops_platform.utils.logging import logger


def calculate_psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two continuous or discrete series.

    Args:
        reference: Baseline reference array.
        current: Current production traffic array.
        num_bins: Number of quantiles/bins.
        epsilon: Small smoothing constant to prevent zero division.

    Returns:
        float: Computed PSI score.
    """
    ref_clean = pd.Series(reference).dropna()
    cur_clean = pd.Series(current).dropna()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Determine bin edges from reference quantiles
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles * 100)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Adjust edges slightly to include bounds
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = pd.cut(ref_clean, bins=bin_edges).value_counts(sort=False).values
    cur_counts = pd.cut(cur_clean, bins=bin_edges).value_counts(sort=False).values

    ref_pct = (ref_counts / len(ref_clean)) + epsilon
    cur_pct = (cur_counts / len(cur_clean)) + epsilon

    # Re-normalize
    ref_pct = ref_pct / np.sum(ref_pct)
    cur_pct = cur_pct / np.sum(cur_pct)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(round(psi_val, 4))


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config_path: str | Path = "configs/config.yaml",
    report_path: str | Path = "reports/drift_report.json",
) -> dict[str, Any]:
    """Detect statistical data drift and distribution shift between reference and current traffic.

    Args:
        reference_df: Baseline reference dataframe.
        current_df: Current inference traffic dataframe.
        config_path: Path to YAML configuration.
        report_path: Path to output drift report JSON.

    Returns:
        dict[str, Any]: Drift detection summary and per-feature metrics.
    """
    config = load_config(config_path)
    mon_cfg = config.get("monitoring", {})
    p_value_threshold = mon_cfg.get("drift_p_value_threshold", 0.05)
    drift_share_threshold = mon_cfg.get("drift_share_threshold", 0.30)
    psi_threshold = mon_cfg.get("psi_threshold", 0.20)

    num_features = config.get("data", {}).get("features", {}).get("numerical", [])
    cat_features = config.get("data", {}).get("features", {}).get("categorical", [])

    feature_drift_results: dict[str, Any] = {}
    drifted_count = 0
    total_features = 0

    # 1. Numerical Feature Drift (Kolmogorov-Smirnov test & PSI)
    for col in num_features:
        if col in reference_df.columns and col in current_df.columns:
            total_features += 1
            ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna()
            cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna()

            ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
            psi_val = calculate_psi(ref_vals, cur_vals)

            is_drifted = bool(ks_pval < p_value_threshold or psi_val > psi_threshold)
            if is_drifted:
                drifted_count += 1

            feature_drift_results[col] = {
                "type": "numerical",
                "drift_detected": is_drifted,
                "ks_statistic": float(round(ks_stat, 4)),
                "p_value": float(round(ks_pval, 6)),
                "psi": psi_val,
                "ref_mean": float(round(ref_vals.mean(), 2)),
                "current_mean": float(round(cur_vals.mean(), 2)),
            }

    # 2. Categorical Feature Drift (Chi-Square contingency & distribution shift)
    for col in cat_features:
        if col in reference_df.columns and col in current_df.columns:
            total_features += 1
            ref_counts = reference_df[col].value_counts()
            cur_counts = current_df[col].value_counts()

            # Align categories
            all_cats = list(set(ref_counts.index).union(set(cur_counts.index)))
            aligned_ref = [ref_counts.get(c, 0) + 1 for c in all_cats]
            aligned_cur = [cur_counts.get(c, 0) + 1 for c in all_cats]

            chi2_stat, chi2_pval, _, _ = stats.chi2_contingency([aligned_ref, aligned_cur])
            is_drifted = bool(chi2_pval < p_value_threshold)
            if is_drifted:
                drifted_count += 1

            feature_drift_results[col] = {
                "type": "categorical",
                "drift_detected": is_drifted,
                "chi2_statistic": float(round(chi2_stat, 4)),
                "p_value": float(round(chi2_pval, 6)),
                "distribution_shift": {
                    c: {
                        "ref_share": float(round(ref_counts.get(c, 0) / len(reference_df), 3)),
                        "cur_share": float(round(cur_counts.get(c, 0) / len(current_df), 3)),
                    }
                    for c in all_cats[:5]
                },
            }

    drift_share = float(round(drifted_count / total_features, 4)) if total_features > 0 else 0.0
    dataset_drift_detected = drift_share >= drift_share_threshold

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset_drift_detected": dataset_drift_detected,
        "drift_share": drift_share,
        "drift_share_threshold": drift_share_threshold,
        "drifted_features_count": drifted_count,
        "total_features_evaluated": total_features,
        "reference_records": len(reference_df),
        "current_records": len(current_df),
        "features": feature_drift_results,
    }

    # Persist report
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Drift analysis complete: Dataset Drift=%s (Share: %.1f%%, Threshold: %.1f%%)",
        dataset_drift_detected,
        drift_share * 100,
        drift_share_threshold * 100,
    )
    return report


def analyze_prediction_logs(
    config_path: str | Path = "configs/config.yaml",
) -> dict[str, Any] | None:
    """Read prediction traffic log and evaluate drift against saved reference dataset.

    Args:
        config_path: Path to configuration YAML.

    Returns:
        dict[str, Any] | None: Drift report or None if insufficient log records.
    """
    config = load_config(config_path)
    log_file = Path(
        config.get("monitoring", {}).get(
            "prediction_log_path", "reports/prediction_traffic.jsonl"
        )
    )
    ref_file = Path(
        config.get("data", {}).get(
            "reference_data_path", "data/reference/reference_features.parquet"
        )
    )

    if not log_file.exists():
        logger.warning("No prediction traffic log found at %s", log_file)
        return None

    if not ref_file.exists():
        logger.warning("Reference dataset not found at %s. Run training first.", ref_file)
        return None

    reference_df = pd.read_parquet(ref_file)

    # Read log entries
    records: list[dict[str, Any]] = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    records.append(entry.get("features", {}))
                except Exception:
                    continue

    if len(records) < 20:
        logger.info("Insufficient traffic records for reliable drift analysis (%d < 20)", len(records))
        return None

    current_df = pd.DataFrame(records)
    report_file = config.get("monitoring", {}).get(
        "drift_report_path", "reports/drift_report.json"
    )
    return detect_drift(
        reference_df=reference_df,
        current_df=current_df,
        config_path=config_path,
        report_path=report_file,
    )
