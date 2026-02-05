"""Model training pipeline.

Supports four model types:
    baseline  — always predict the most frequent class
    logreg    — multinomial logistic regression with standard scaling
    rf        — random forest classifier
    xgb       — XGBoost (graceful fallback if not installed)

Every model is wrapped in a sklearn Pipeline, saved with joblib, and
accompanied by a JSON metrics report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc_predictor.config import (
    FEATURE_MATRIX_CSV,
    FEATURE_SCHEMA_JSON,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_CUTOFF_YEAR,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from wc_predictor.splits import (
    compute_metrics,
    print_metrics,
    save_report,
    temporal_split,
)

log = get_logger(__name__)


# ── Model factories ─────────────────────────────────────────────────────────


def _make_baseline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )


def _make_logreg() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    C=1.0,
                ),
            ),
        ]
    )


def _make_rf() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _make_xgb() -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.warning(
            "xgboost is not installed. Skipping XGB model. "
            "Install with: pip install xgboost"
        )
        return None

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    num_class=3,
                    random_state=RANDOM_SEED,
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    verbosity=0,
                ),
            ),
        ]
    )


MODEL_FACTORIES = {
    "baseline": _make_baseline,
    "logreg": _make_logreg,
    "rf": _make_rf,
    "xgb": _make_xgb,
}


# ── Training logic ──────────────────────────────────────────────────────────


def train_model(
    model_name: str,
    feature_csv: Path = FEATURE_MATRIX_CSV,
    schema_path: Path = FEATURE_SCHEMA_JSON,
    cutoff_year: int = TRAIN_CUTOFF_YEAR,
) -> Path | None:
    """Train a single model, evaluate on the temporal test set, and save.

    Returns the path to the saved model, or None if skipped (e.g. missing xgb).
    """
    seed_everything()
    ensure_dirs()

    if model_name not in MODEL_FACTORIES:
        log.error("Unknown model '%s'. Choose from: %s", model_name, list(MODEL_FACTORIES))
        sys.exit(1)

    # Load data
    df = pd.read_csv(feature_csv, parse_dates=["date"])
    schema = json.loads(schema_path.read_text())
    feature_cols = schema["feature_columns"]
    label_col = schema["label_column"]

    train_df, test_df = temporal_split(df, cutoff_year=cutoff_year)

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df[label_col].values.astype(int)

    log.info("Training '%s' on %d samples, testing on %d …", model_name, len(X_train), len(X_test))

    # Build pipeline
    pipeline = MODEL_FACTORIES[model_name]()
    if pipeline is None:
        return None

    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = (
        pipeline.predict_proba(X_test)
        if hasattr(pipeline, "predict_proba")
        else None
    )

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, header=f"{model_name} — test set")

    # Save model
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)
    log.info("Model saved -> %s", model_path)

    # Save report
    extra = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_columns": feature_cols,
    }
    save_report(metrics, model_name, extra=extra)

    return model_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a WC prediction model.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_FACTORIES),
        default="logreg",
        help="Model type to train (default: logreg).",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=TRAIN_CUTOFF_YEAR,
        help=f"Cutoff year for temporal split (default: {TRAIN_CUTOFF_YEAR}).",
    )
    args = parser.parse_args()
    train_model(args.model, cutoff_year=args.cutoff)


if __name__ == "__main__":
    main()
