"""CLI script to generate synthetic churn data or simulate distribution shift."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_platform.data.ingestion import compute_dataset_fingerprint, generate_synthetic_churn_data
from mlops_platform.utils.logging import logger


def generate_drifted_data(
    n_samples: int = 1000,
    seed: int = 999,
) -> pd.DataFrame:
    """Generate dataset with severe distribution shift for testing drift detection.

    Simulated shifts:
    - tenure: shifted towards very new customers (e.g., tenure mostly 1-6 months).
    - monthly_charges: significantly inflated (e.g., $100 - $180).
    - contract_type: 90% month-to-month contracts.
    """
    rng = np.random.default_rng(seed)
    customer_ids = [f"CUST-DRIFT-{i:05d}" for i in range(1, n_samples + 1)]

    # Distribution Shift: Tenure much lower
    tenure = rng.integers(1, 12, size=n_samples)

    # Distribution Shift: Contracts heavily skewed
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.90, 0.08, 0.02])

    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.20, 0.75, 0.05])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
        p=[0.70, 0.10, 0.10, 0.10],
    )
    paperless = rng.choice(["Yes", "No"], size=n_samples, p=[0.85, 0.15])

    tech_support = ["No" if inet != "No" else "No internet service" for inet in internet]
    online_security = ["No" if inet != "No" else "No internet service" for inet in internet]

    # Distribution Shift: Monthly charges inflated
    monthly_charges = np.round(rng.uniform(85.0, 145.0, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges, 2)

    # Higher churn
    churn = rng.choice([0, 1], size=n_samples, p=[0.40, 0.60])

    return pd.DataFrame(
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
            "paperless_billing": paperless,
            "churn": churn,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic or drifted churn data")
    parser.add_argument("--samples", type=int, default=5000, help="Number of records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/raw/customer_churn.csv", help="Output CSV path")
    parser.add_argument("--drift", action="store_true", help="Generate severely drifted dataset")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.drift:
        logger.info("Generating DRIFTED dataset (samples=%d, seed=%d)", args.samples, args.seed)
        df = generate_drifted_data(n_samples=args.samples, seed=args.seed)
    else:
        logger.info("Generating standard synthetic dataset (samples=%d, seed=%d)", args.samples, args.seed)
        df = generate_synthetic_churn_data(n_samples=args.samples, seed=args.seed)

    df.to_csv(out_path, index=False)
    fingerprint = compute_dataset_fingerprint(df)
    print(f"Dataset generated: {out_path} ({len(df)} rows, Fingerprint: {fingerprint[:12]})")


if __name__ == "__main__":
    main()
