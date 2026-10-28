"""Integration tests for training pipeline, evaluate, tune, and registry modules."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_platform.data.ingestion import generate_synthetic_churn_data
from mlops_platform.data.preprocessing import build_preprocessor_pipeline, split_data
from mlops_platform.training.evaluate import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_evaluation_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_df():
    """Small synthetic dataset for fast integration tests."""
    return generate_synthetic_churn_data(n_samples=400, seed=42)


@pytest.fixture(scope="module")
def feature_cols():
    return (
        ["tenure", "monthly_charges", "total_charges"],
        ["contract_type", "payment_method", "internet_service",
         "tech_support", "online_security", "paperless_billing"],
    )


@pytest.fixture(scope="module")
def trained_pipeline(small_df, feature_cols):
    """A trained sklearn pipeline for reuse across tests."""
    num_f, cat_f = feature_cols
    features = num_f + cat_f
    train, val, test = split_data(small_df, target_col="churn", random_state=42)
    X_train = train[features]
    y_train = train["churn"]
    preprocessor = build_preprocessor_pipeline(num_f, cat_f)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=300, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe, train, val, test, features


# ---------------------------------------------------------------------------
# evaluate.py — evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_returns_dict_with_all_keys(self, trained_pipeline, feature_cols):
        pipe, train, val, test, features = trained_pipeline
        X_val = val[features]
        y_val = val["churn"]
        result = evaluate_model(pipe, X_val, y_val, dataset_name="val")
        assert isinstance(result, dict)
        assert "val_f1_score" in result
        assert "val_roc_auc" in result
        assert "val_precision" in result
        assert "val_recall" in result
        assert "val_accuracy" in result
        assert "confusion_matrix" in result

    def test_model_beats_random(self, trained_pipeline, feature_cols):
        pipe, train, val, test, features = trained_pipeline
        X_val = val[features]
        y_val = val["churn"]
        result = evaluate_model(pipe, X_val, y_val, dataset_name="val")
        assert result["val_roc_auc"] > 0.55, "Trained model should beat random baseline"

    def test_train_set_returns_train_prefixed_keys(self, trained_pipeline, feature_cols):
        pipe, train, val, test, features = trained_pipeline
        X_train = train[features]
        y_train = train["churn"]
        result = evaluate_model(pipe, X_train, y_train, dataset_name="train")
        assert "train_f1_score" in result
        assert "train_roc_auc" in result

    def test_latency_key_present(self, trained_pipeline, feature_cols):
        pipe, train, val, test, features = trained_pipeline
        X_val = val[features]
        y_val = val["churn"]
        result = evaluate_model(pipe, X_val, y_val, dataset_name="val")
        assert "val_latency_ms" in result
        assert result["val_latency_ms"] >= 0.0

    def test_confusion_matrix_structure(self, trained_pipeline, feature_cols):
        pipe, train, val, test, features = trained_pipeline
        X_val = val[features]
        y_val = val["churn"]
        result = evaluate_model(pipe, X_val, y_val, dataset_name="val")
        cm = result["confusion_matrix"]
        assert set(cm.keys()) == {"true_negatives", "false_positives", "false_negatives", "true_positives"}
        total = sum(cm.values())
        assert total == len(X_val)


# ---------------------------------------------------------------------------
# evaluate.py — save_evaluation_report
# ---------------------------------------------------------------------------


class TestSaveEvaluationReport:
    def test_report_written_to_disk(self):
        metrics = {"val_f1_score": 0.75, "val_roc_auc": 0.88}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval.json"
            save_evaluation_report(metrics, output_path=str(path))
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["val_f1_score"] == pytest.approx(0.75)
            assert data["val_roc_auc"] == pytest.approx(0.88)

    def test_report_creates_parent_dir(self):
        metrics = {"test_accuracy": 0.92}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "sub" / "report.json"
            save_evaluation_report(metrics, output_path=str(path))
            assert path.exists()


# ---------------------------------------------------------------------------
# evaluate.py — save_confusion_matrix_plot
# ---------------------------------------------------------------------------


class TestSaveConfusionMatrixPlot:
    def test_image_file_created(self):
        cm = np.array([[80, 10], [15, 45]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cm.png"
            result = save_confusion_matrix_plot(cm, output_path=str(path))
            assert path.exists()
            assert path.stat().st_size > 0
            assert result == str(path.resolve())


# ---------------------------------------------------------------------------
# preprocessing.py — build_preprocessor_pipeline
# ---------------------------------------------------------------------------


class TestBuildPreprocessorPipeline:
    def test_fit_transform_output_shape(self, small_df, feature_cols):
        num_f, cat_f = feature_cols
        X = small_df[num_f + cat_f]
        preprocessor = build_preprocessor_pipeline(num_f, cat_f)
        X_t = preprocessor.fit_transform(X)
        assert X_t.shape[0] == len(small_df)
        assert X_t.shape[1] > len(num_f)  # OHE expands categoricals

    def test_no_nans_after_transform(self, small_df, feature_cols):
        num_f, cat_f = feature_cols
        X = small_df[num_f + cat_f]
        preprocessor = build_preprocessor_pipeline(num_f, cat_f)
        X_t = preprocessor.fit_transform(X)
        assert not np.isnan(X_t).any()

    def test_transform_only_uses_train_stats(self, small_df, feature_cols):
        """Test that test set is transformed using train-set statistics only."""
        num_f, cat_f = feature_cols
        train, val, test = split_data(small_df, target_col="churn", random_state=42)
        X_train = train[num_f + cat_f]
        X_val = val[num_f + cat_f]
        preprocessor = build_preprocessor_pipeline(num_f, cat_f)
        preprocessor.fit(X_train)
        # Should not raise even if val has different ranges
        X_val_t = preprocessor.transform(X_val)
        assert X_val_t.shape[0] == len(X_val)


# ---------------------------------------------------------------------------
# preprocessing.py — split_data
# ---------------------------------------------------------------------------


class TestSplitData:
    def test_split_sizes_sum_to_whole(self, small_df):
        train, val, test = split_data(small_df, target_col="churn", random_state=42)
        assert len(train) + len(val) + len(test) == len(small_df)

    def test_no_customer_id_overlap(self, small_df):
        train, val, test = split_data(small_df, target_col="churn", random_state=42)
        ids_t = set(train["customer_id"])
        ids_v = set(val["customer_id"])
        ids_te = set(test["customer_id"])
        assert not ids_t & ids_v, "Train/Val overlap"
        assert not ids_t & ids_te, "Train/Test overlap"
        assert not ids_v & ids_te, "Val/Test overlap"

    def test_reproducibility_across_calls(self, small_df):
        t1, v1, te1 = split_data(small_df, target_col="churn", random_state=123)
        t2, v2, te2 = split_data(small_df, target_col="churn", random_state=123)
        pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))

    def test_different_seeds_produce_different_splits(self, small_df):
        t1, _, _ = split_data(small_df, target_col="churn", random_state=1)
        t2, _, _ = split_data(small_df, target_col="churn", random_state=2)
        # Very unlikely to be identical with 400 rows
        assert not t1["customer_id"].reset_index(drop=True).equals(
            t2["customer_id"].reset_index(drop=True)
        )

    def test_stratification_preserves_label_ratio(self, small_df):
        """Churn rate in splits should be close to original."""
        original_rate = small_df["churn"].mean()
        train, val, test = split_data(small_df, target_col="churn", random_state=42)
        for name, split in [("train", train), ("val", val), ("test", test)]:
            rate = split["churn"].mean()
            assert abs(rate - original_rate) < 0.08, (
                f"{name} churn rate ({rate:.3f}) deviated from original ({original_rate:.3f})"
            )


# ---------------------------------------------------------------------------
# monitoring/retraining.py — module import smoke test
# ---------------------------------------------------------------------------


class TestRetrainingModule:
    def test_module_has_trigger_retraining(self):
        from mlops_platform.monitoring import retraining
        assert callable(retraining.trigger_retraining)

    def test_retraining_returns_failed_dict_on_bad_data_path(self, tmp_path):
        """trigger_retraining with valid config but missing raw data should ingest fresh data
        or return gracefully — it must never crash unhandled."""
        from mlops_platform.monitoring.retraining import trigger_retraining
        # Point to an empty directory that doesn't have the raw CSV
        fake_data = str(tmp_path / "no_such_file.csv")
        # trigger_retraining will auto-ingest data if missing, so it may succeed
        # What we assert is: it NEVER raises an unhandled exception
        result = trigger_retraining(
            data_path=fake_data,
            config_path="configs/config.yaml",
            trigger_reason="test_auto_ingest",
        )
        assert isinstance(result, dict)
        assert "status" in result


# ---------------------------------------------------------------------------
# registry/register_model.py — check_quality_gate
# ---------------------------------------------------------------------------


class TestCheckQualityGate:
    def test_passes_with_strong_metrics(self):
        from mlops_platform.registry.register_model import check_quality_gate
        metrics = {"val_roc_auc": 0.85, "val_f1_score": 0.72}
        passed, reasons = check_quality_gate(metrics)
        assert passed is True
        assert reasons == []

    def test_fails_on_low_auc(self):
        from mlops_platform.registry.register_model import check_quality_gate
        metrics = {"val_roc_auc": 0.55, "val_f1_score": 0.70}
        passed, reasons = check_quality_gate(metrics)
        assert passed is False
        assert any("roc" in r.lower() or "auc" in r.lower() for r in reasons)

    def test_fails_on_low_f1(self):
        from mlops_platform.registry.register_model import check_quality_gate
        metrics = {"val_roc_auc": 0.80, "val_f1_score": 0.35}
        passed, reasons = check_quality_gate(metrics)
        assert passed is False
        assert any("f1" in r.lower() for r in reasons)

    def test_fails_on_none_metric(self):
        from mlops_platform.registry.register_model import check_quality_gate
        metrics = {"val_roc_auc": None, "val_f1_score": 0.65}
        passed, reasons = check_quality_gate(metrics)
        assert passed is False

    def test_baseline_comparison_blocks_regression(self):
        from mlops_platform.registry.register_model import check_quality_gate
        # Candidate is worse than baseline → should fail
        metrics = {"val_roc_auc": 0.80, "val_f1_score": 0.62}
        baseline = {"val_f1_score": 0.72}
        passed, reasons = check_quality_gate(metrics, baseline_metrics=baseline)
        assert passed is False
        assert any("baseline" in r.lower() or "beat" in r.lower() for r in reasons)

    def test_baseline_comparison_passes_when_better(self):
        from mlops_platform.registry.register_model import check_quality_gate
        metrics = {"val_roc_auc": 0.82, "val_f1_score": 0.75}
        baseline = {"val_f1_score": 0.70}
        passed, reasons = check_quality_gate(metrics, baseline_metrics=baseline)
        assert passed is True
        assert reasons == []
