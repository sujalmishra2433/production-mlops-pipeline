"""Data validation and quality schema checks using Pandera."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

from mlops_platform.utils.logging import logger


class DataValidationError(Exception):
    """Raised when dataset fails critical schema or data quality checks."""

    pass


def get_feature_schema(include_target: bool = True) -> DataFrameSchema:
    """Build Pandera schema for churn features.

    Args:
        include_target: Whether to include the 'churn' column in the schema.

    Returns:
        DataFrameSchema: Pandera DataFrame schema.
    """
    columns = {
        "customer_id": Column(
            pa.String,
            nullable=False,
            description="Unique customer identifier",
        ),
        "tenure": Column(
            pa.Int,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(120)],
            nullable=False,
            description="Months with company",
        ),
        "monthly_charges": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0.0), Check.less_than_or_equal_to(500.0)],
            nullable=False,
            description="Monthly subscription charge",
        ),
        "total_charges": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0.0), Check.less_than_or_equal_to(50000.0)],
            nullable=False,
            description="Total lifetime charges",
        ),
        "contract_type": Column(
            pa.String,
            checks=Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False,
            description="Billing contract term",
        ),
        "payment_method": Column(
            pa.String,
            checks=Check.isin(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
            ),
            nullable=False,
            description="Payment method",
        ),
        "internet_service": Column(
            pa.String,
            checks=Check.isin(["DSL", "Fiber optic", "No"]),
            nullable=False,
            description="Internet service type",
        ),
        "tech_support": Column(
            pa.String,
            checks=Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            description="Tech support add-on status",
        ),
        "online_security": Column(
            pa.String,
            checks=Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
            description="Online security add-on status",
        ),
        "paperless_billing": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
            description="Paperless billing preference",
        ),
    }

    if include_target:
        columns["churn"] = Column(
            pa.Int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Target churn indicator: 1 if churned, 0 otherwise",
        )

    return DataFrameSchema(
        columns=columns,
        strict=False,  # Allow additional meta columns if present
        coerce=True,   # Coerce numeric dtypes if safely convertible
    )


def validate_dataset(
    df: pd.DataFrame,
    include_target: bool = True,
    report_path: str | Path | None = "reports/validation_report.json",
    raise_on_error: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Validate DataFrame against schema and generate a quality report.

    Args:
        df: Input DataFrame.
        include_target: Whether target column is expected.
        report_path: Optional path to write machine-readable JSON report.
        raise_on_error: Whether to raise DataValidationError if validation fails.

    Returns:
        tuple[bool, dict[str, Any]]: Validation status and detailed report.
    """
    logger.info("Starting dataset validation (rows=%d, cols=%d)", len(df), len(df.columns))

    schema = get_feature_schema(include_target=include_target)
    errors: list[str] = []
    is_valid = True

    # 1. Check duplicate customer_ids
    duplicate_count = 0
    if "customer_id" in df.columns:
        duplicate_count = int(df["customer_id"].duplicated().sum())
        if duplicate_count > 0:
            msg = f"Found {duplicate_count} duplicate customer_id records"
            errors.append(msg)
            is_valid = False

    # 2. Check null counts
    null_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0}
    if null_counts:
        msg = f"Null values detected in columns: {null_counts}"
        errors.append(msg)
        is_valid = False

    # 3. Pandera Schema Validation
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        is_valid = False
        failure_cases = err.failure_cases.to_dict(orient="records")
        for failure in failure_cases[:10]:  # Cap first 10 for readability
            col = failure.get("column", "unknown")
            check = failure.get("check", "unknown")
            errors.append(f"Schema failure in '{col}': {check}")
    except Exception as exc:
        is_valid = False
        errors.append(f"Unexpected validation error: {str(exc)}")

    # 4. Target distribution summary if present
    target_summary: dict[str, Any] = {}
    if include_target and "churn" in df.columns:
        counts = df["churn"].value_counts().to_dict()
        target_summary = {
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "churn_rate": float(df["churn"].mean()),
        }

    # Build report
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "is_valid": is_valid,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "duplicate_ids": duplicate_count,
        "null_counts": null_counts,
        "target_summary": target_summary,
        "error_count": len(errors),
        "errors": errors,
    }

    if report_path:
        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Validation report written to %s", out_path)

    if not is_valid and raise_on_error:
        error_summary = "; ".join(errors[:5])
        logger.error("Dataset validation failed: %s", error_summary)
        raise DataValidationError(f"Data validation failed: {error_summary}")

    logger.info("Dataset validation passed successfully.")
    return is_valid, report
