"""Data preprocessing pipelines and leakage-free dataset splitting."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_platform.features.build_features import ChurnFeatureEngineer
from mlops_platform.utils.logging import logger


def split_data(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset reproducibly into train, validation, and test sets with stratification.

    Ensures zero leakage: transformations must only be fit on train.

    Args:
        df: Input DataFrame.
        target_col: Name of the binary target column.
        test_size: Fraction of data reserved for final test set.
        val_size: Fraction of data reserved for validation set.
        random_state: Random state seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    logger.info(
        "Splitting dataset (rows=%d) into train, val (%.2f), and test (%.2f) sets",
        len(df),
        val_size,
        test_size,
    )

    stratify_target = df[target_col] if target_col in df.columns else None

    # First split off the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Calculate adjusted validation size relative to remaining train_val
    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_train_val = (
        train_val_df[target_col] if target_col in train_val_df.columns else None
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    logger.info(
        "Splits generated: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(train_df),
        len(train_df) / len(df) * 100,
        len(val_df),
        len(val_df) / len(df) * 100,
        len(test_df),
        len(test_df) / len(df) * 100,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_preprocessor_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Construct an end-to-end scikit-learn preprocessing pipeline.

    Combines:
    1. Feature engineering (calculating ratios and flags)
    2. ColumnTransformer (imputing, scaling numerics, encoding categoricals)

    Args:
        numerical_features: List of raw numerical feature names.
        categorical_features: List of raw categorical feature names.

    Returns:
        Pipeline: Preprocessing pipeline ready for fitting.
    """
    # Raw numerical features + engineered numerical features
    all_numerical = numerical_features + ["charge_ratio", "monthly_to_total_ratio"]
    # Raw categorical features + engineered binary flags
    all_categorical = categorical_features + ["has_support_services", "is_long_term_contract"]

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, all_numerical),
            ("cat", cat_transformer, all_categorical),
        ],
        remainder="drop",
    )

    preprocessor = Pipeline(
        steps=[
            ("feature_engineer", ChurnFeatureEngineer()),
            ("transformer", transformer),
        ]
    )

    return preprocessor


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
    reference_path: str | Path = "data/reference/reference_features.parquet",
) -> dict[str, str]:
    """Save processed splits to parquet files and store baseline reference data.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        processed_dir: Output directory for train/val/test splits.
        reference_path: Path for baseline reference dataset for drift monitoring.

    Returns:
        dict[str, str]: Saved file paths.
    """
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    ref_path = Path(reference_path)
    ref_path.parent.mkdir(parents=True, exist_ok=True)

    train_file = proc_path / "train.parquet"
    val_file = proc_path / "val.parquet"
    test_file = proc_path / "test.parquet"

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)

    # Save training features as reference dataset for drift monitoring (excluding target/id)
    feature_cols = [c for c in train_df.columns if c not in ["customer_id", "churn"]]
    train_df[feature_cols].to_parquet(ref_path, index=False)

    logger.info("Saved train (%s), val (%s), test (%s)", train_file, val_file, test_file)
    logger.info("Saved baseline reference features to %s", ref_path)

    return {
        "train": str(train_file),
        "val": str(val_file),
        "test": str(test_file),
        "reference": str(ref_path),
    }
