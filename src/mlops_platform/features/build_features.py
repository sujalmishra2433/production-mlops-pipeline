"""Feature engineering transformer integrated into scikit-learn pipeline."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer for churn prediction.

    Engineers features without leaking test set statistics:
    - charge_ratio: ratio of total_charges to (tenure * monthly_charges + 1)
    - monthly_to_total_ratio: ratio of monthly_charges to total_charges
    - has_support_services: flag indicating if customer has tech support or online security
    - is_long_term_contract: flag indicating multi-year commitment
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ChurnFeatureEngineer":  # noqa: ARG002
        """Fit transformer (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe by adding engineered features.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with engineered features.
        """
        df = X.copy()

        # 1. Charge consistency ratio
        expected_total = df["tenure"] * df["monthly_charges"]
        df["charge_ratio"] = np.round(
            df["total_charges"] / (expected_total + 1.0),
            4,
        )

        # 2. Monthly to total ratio
        df["monthly_to_total_ratio"] = np.round(
            df["monthly_charges"] / (df["total_charges"] + 1.0),
            4,
        )

        # 3. Support security indicator
        has_tech = df["tech_support"] == "Yes"
        has_sec = df["online_security"] == "Yes"
        df["has_support_services"] = (has_tech | has_sec).astype(int)

        # 4. Long term contract flag
        df["is_long_term_contract"] = df["contract_type"].isin(
            ["One year", "Two year"]
        ).astype(int)

        return df
