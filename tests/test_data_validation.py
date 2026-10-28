"""Unit tests for Pandera schema and data quality validation."""

import pytest

from mlops_platform.data.validation import (
    DataValidationError,
    validate_dataset,
)


def test_validation_passes_on_valid_data(sample_churn_df, tmp_path):
    """Verify that pristine synthetic dataset passes all schema checks."""
    report_file = tmp_path / "val_report.json"
    is_valid, report = validate_dataset(
        df=sample_churn_df,
        include_target=True,
        report_path=report_file,
        raise_on_error=True,
    )
    assert is_valid is True
    assert report["is_valid"] is True
    assert report["error_count"] == 0
    assert report_file.exists()


def test_validation_fails_on_missing_column(sample_churn_df):
    """Verify that dropping a required column causes validation to fail."""
    corrupted_df = sample_churn_df.drop(columns=["monthly_charges"])
    with pytest.raises(DataValidationError):
        validate_dataset(corrupted_df, include_target=True, report_path=None, raise_on_error=True)


def test_validation_fails_on_out_of_range_numerical(sample_churn_df):
    """Verify that negative tenure triggers range failure."""
    corrupted_df = sample_churn_df.copy()
    corrupted_df.loc[0, "tenure"] = -10
    with pytest.raises(DataValidationError):
        validate_dataset(corrupted_df, include_target=True, report_path=None, raise_on_error=True)


def test_validation_fails_on_invalid_categorical(sample_churn_df):
    """Verify that unexpected categorical category triggers schema failure."""
    corrupted_df = sample_churn_df.copy()
    corrupted_df.loc[0, "contract_type"] = "Five year plan"
    with pytest.raises(DataValidationError):
        validate_dataset(corrupted_df, include_target=True, report_path=None, raise_on_error=True)


def test_validation_fails_on_duplicate_ids(sample_churn_df):
    """Verify that duplicate customer_id triggers duplicate ID check failure."""
    corrupted_df = sample_churn_df.copy()
    corrupted_df.loc[1, "customer_id"] = corrupted_df.loc[0, "customer_id"]
    with pytest.raises(DataValidationError):
        validate_dataset(corrupted_df, include_target=True, report_path=None, raise_on_error=True)


def test_validation_fails_on_null_values(sample_churn_df):
    """Verify that unexpected null values trigger null check failure."""
    corrupted_df = sample_churn_df.copy()
    corrupted_df.loc[2, "monthly_charges"] = None
    with pytest.raises(DataValidationError):
        validate_dataset(corrupted_df, include_target=True, report_path=None, raise_on_error=True)
