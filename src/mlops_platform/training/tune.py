"""Hyperparameter search and optimization routines."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from mlops_platform.utils.logging import logger


def tune_logistic_regression(
    preprocessor: Any,
    X_train: Any,
    y_train: Any,
    cv_folds: int = 3,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune hyperparameters for Logistic Regression pipeline.

    Args:
        preprocessor: Preprocessing transformer pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv_folds: Number of stratified folds.
        random_state: Random state for reproducibility.

    Returns:
        tuple[Pipeline, dict[str, Any]]: (best_pipeline, best_params)
    """
    logger.info("Tuning Logistic Regression with %d-fold cross-validation", cv_folds)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 5.0],
        "classifier__solver": ["lbfgs"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info("Best Logistic Regression params: %s | Best F1: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest(
    preprocessor: Any,
    X_train: Any,
    y_train: Any,
    cv_folds: int = 3,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune hyperparameters for Random Forest pipeline.

    Args:
        preprocessor: Preprocessing transformer pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv_folds: Number of stratified folds.
        random_state: Random state for reproducibility.

    Returns:
        tuple[Pipeline, dict[str, Any]]: (best_pipeline, best_params)
    """
    logger.info("Tuning Random Forest with %d-fold cross-validation", cv_folds)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [6, 10],
        "classifier__min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info("Best Random Forest params: %s | Best F1: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, grid.best_params_
