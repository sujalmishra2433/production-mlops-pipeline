"""Dataset ingestion and deterministic synthetic data generator."""

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops_platform.utils.logging import logger


def generate_synthetic_churn_data(
    n_samples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic, deterministic customer churn data.

    Args:
        n_samples: Number of customer records to generate.
        seed: Random seed for exact reproducibility.

    Returns:
        pd.DataFrame: Tabular customer churn dataset.
    """
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]

    # Feature distributions
    tenure = rng.integers(1, 73, size=n_samples)  # 1 to 72 months

    contract_types = ["Month-to-month", "One year", "Two year"]
    contract_probs = [0.55, 0.25, 0.20]
    contract = rng.choice(contract_types, size=n_samples, p=contract_probs)

    internet_services = ["DSL", "Fiber optic", "No"]
    internet_probs = [0.35, 0.45, 0.20]
    internet = rng.choice(internet_services, size=n_samples, p=internet_probs)

    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_probs = [0.35, 0.20, 0.25, 0.20]
    payment = rng.choice(payment_methods, size=n_samples, p=payment_probs)

    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])

    # Dependent categorical features based on internet service
    tech_support = []
    online_security = []
    for inet in internet:
        if inet == "No":
            tech_support.append("No internet service")
            online_security.append("No internet service")
        else:
            tech_support.append(rng.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_security.append(rng.choice(["Yes", "No"], p=[0.30, 0.70]))

    # Monthly charges depend on internet service
    monthly_charges = np.zeros(n_samples)
    for i, inet in enumerate(internet):
        if inet == "No":
            monthly_charges[i] = rng.uniform(18.0, 26.0)
        elif inet == "DSL":
            monthly_charges[i] = rng.uniform(40.0, 75.0)
        else:  # Fiber optic
            monthly_charges[i] = rng.uniform(70.0, 118.0)

    monthly_charges = np.round(monthly_charges, 2)

    # Total charges roughly tenure * monthly_charges + small noise
    total_charges = np.round(
        tenure * monthly_charges + rng.normal(0, 15, size=n_samples), 2
    )
    # Ensure total charges is non-negative and >= monthly charges
    total_charges = np.maximum(total_charges, monthly_charges)

    # Deterministic Churn probability modeling
    # Month-to-month, Fiber optic, High monthly charge, Low tenure increase churn
    log_odds = -1.2  # Base rate
    log_odds += np.where(contract == "Month-to-month", 1.4, -1.0)
    log_odds += np.where(internet == "Fiber optic", 0.6, -0.4)
    log_odds += np.where(payment == "Electronic check", 0.5, -0.2)
    log_odds += np.where(np.array(tech_support) == "No", 0.4, -0.3)
    log_odds += (monthly_charges - 65.0) / 35.0 * 0.5
    log_odds -= (tenure - 32.0) / 24.0 * 1.1

    churn_prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (rng.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "churn": churn,
        }
    )

    return df


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hash of a dataframe content."""
    df_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(df_bytes).hexdigest()


def ingest_data(
    output_path: str | Path = "data/raw/customer_churn.csv",
    n_samples: int = 5000,
    seed: int = 42,
    source_url: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest dataset from URL or deterministic synthetic generator.

    Args:
        output_path: Where to save raw ingested CSV.
        n_samples: Number of records if synthetic generator is used.
        seed: Random seed.
        source_url: Optional URL to fetch real data from.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: The ingested DataFrame and its metadata.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame | None = None
    source = "synthetic_deterministic"

    if source_url:
        try:
            logger.info("Attempting to download dataset from %s", source_url)
            # Timeout protection
            with urllib.request.urlopen(source_url, timeout=10) as response:
                df = pd.read_csv(response)
            source = f"external_download: {source_url}"
            logger.info("Successfully downloaded dataset with %d rows", len(df))
        except Exception as e:
            logger.warning(
                "External download failed (%s). Falling back to synthetic generator.", e
            )
            df = None

    if df is None:
        logger.info(
            "Generating deterministic synthetic dataset (n=%d, seed=%d)", n_samples, seed
        )
        df = generate_synthetic_churn_data(n_samples=n_samples, seed=seed)

    # Save raw CSV
    df.to_csv(dest, index=False)

    fingerprint = compute_dataset_fingerprint(df)
    metadata = {
        "source": source,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "fingerprint_sha256": fingerprint,
        "saved_path": str(dest.resolve()),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    # Save metadata JSON alongside dataset
    meta_path = dest.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset ingested and saved to %s (Fingerprint: %s)", dest, fingerprint[:12])
    return df, metadata
