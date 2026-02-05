"""Evaluate all trained models side-by-side and generate comparison plots.

Reads every .joblib model from artifacts/models/, evaluates on the test split,
and produces:
  - artifacts/reports/comparison.json
  - artifacts/figures/model_comparison.png
  - artifacts/figures/calibration.png  (reliability diagram)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wc_predictor.config import (
    FEATURE_MATRIX_CSV,
    FEATURE_SCHEMA_JSON,
    FIGURES_DIR,
    LABEL_NAMES,
    MODELS_DIR,
    REPORTS_DIR,
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


def _load_test_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load feature matrix, split, return (X_test, y_test, feature_cols)."""
    df = pd.read_csv(FEATURE_MATRIX_CSV, parse_dates=["date"])
    schema = json.loads(FEATURE_SCHEMA_JSON.read_text())
    feature_cols = schema["feature_columns"]
    _, test_df = temporal_split(df)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df[schema["label_column"]].values.astype(int)
    return X_test, y_test, feature_cols


def evaluate_all() -> dict[str, dict[str, float]]:
    """Evaluate every saved model and return {model_name: metrics}."""
    ensure_dirs()
    seed_everything()

    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        log.error("No trained models found in %s. Run `make train` first.", MODELS_DIR)
        return {}

    X_test, y_test, feature_cols = _load_test_data()

    results: dict[str, dict] = {}
    prob_cache: dict[str, np.ndarray] = {}

    for mf in model_files:
        name = mf.stem
        log.info("Evaluating model: %s", name)
        pipeline = joblib.load(mf)
        y_pred = pipeline.predict(X_test)
        y_prob = (
            pipeline.predict_proba(X_test)
            if hasattr(pipeline, "predict_proba")
            else None
        )
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(metrics, header=name)
        results[name] = metrics
        if y_prob is not None:
            prob_cache[name] = y_prob

    # Save comparison report
    report_path = REPORTS_DIR / "comparison.json"
    report_path.write_text(json.dumps(results, indent=2))
    log.info("Comparison report -> %s", report_path)

    # ── Plots ────────────────────────────────────────────────────────────
    _plot_comparison_bar(results)
    _plot_calibration(y_test, prob_cache)

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────


def _plot_comparison_bar(results: dict[str, dict[str, float]]) -> None:
    """Bar chart comparing models across metrics."""
    if not results:
        return

    metrics_to_plot = ["accuracy", "macro_f1", "log_loss", "brier_score"]
    models = list(results.keys())
    n_metrics = len(metrics_to_plot)
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics_to_plot):
        vals = [results[m].get(metric, 0.0) for m in models]
        ax.bar(x + i * width, vals, width, label=metric)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — World Cup Test Set")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Comparison plot -> %s", path)


def _plot_calibration(
    y_test: np.ndarray,
    prob_cache: dict[str, np.ndarray],
    n_bins: int = 10,
) -> None:
    """Reliability diagram (calibration plot) for each model.

    For multi-class, we plot the "one-vs-rest" calibration for each class
    on the same figure, one subplot per model.
    """
    if not prob_cache:
        return

    n_models = len(prob_cache)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)

    for col, (model_name, y_prob) in enumerate(prob_cache.items()):
        ax = axes[0, col]
        for cls_idx, cls_name in enumerate(LABEL_NAMES):
            binary_true = (y_test == cls_idx).astype(float)
            predicted_prob = y_prob[:, cls_idx]

            # Bin predictions
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_means_pred = []
            bin_means_true = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (predicted_prob >= lo) & (predicted_prob < hi)
                if mask.sum() > 0:
                    bin_means_pred.append(predicted_prob[mask].mean())
                    bin_means_true.append(binary_true[mask].mean())

            ax.plot(bin_means_pred, bin_means_true, "o-", label=cls_name, markersize=4)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration — {model_name}")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Calibration plot -> %s", path)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    evaluate_all()


if __name__ == "__main__":
    main()
