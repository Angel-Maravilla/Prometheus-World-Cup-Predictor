"""Time-aware train/test splitting and evaluation utilities.

Design
------
* **Temporal split**: train on matches up to (and including) a cutoff year,
  test on everything after.  This mirrors how a model would actually be used
  — you never train on future matches.
* **Time-series cross-validation**: wraps sklearn's ``TimeSeriesSplit`` but
  operates on the pre-sorted feature matrix so folds respect chronology.

Metrics
-------
accuracy, macro-F1, log-loss, multi-class Brier score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit

from wc_predictor.config import (
    LABEL_NAMES,
    REPORTS_DIR,
    TRAIN_CUTOFF_YEAR,
    get_logger,
)

log = get_logger(__name__)


# ── Splitting ────────────────────────────────────────────────────────────────


def temporal_split(
    df: pd.DataFrame,
    cutoff_year: int = TRAIN_CUTOFF_YEAR,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train (≤ cutoff year) and test (> cutoff year).

    Parameters
    ----------
    df : feature matrix with a *date* column.
    cutoff_year : last year included in training.

    Returns
    -------
    (train_df, test_df)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    years = df[date_col].dt.year

    train = df[years <= cutoff_year].reset_index(drop=True)
    test = df[years > cutoff_year].reset_index(drop=True)
    log.info(
        "Temporal split (cutoff %d): train=%d, test=%d",
        cutoff_year,
        len(train),
        len(test),
    )
    return train, test


def time_series_cv_indices(
    n_samples: int, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test index arrays respecting temporal order."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n_samples)))


# ── Metrics ──────────────────────────────────────────────────────────────────


def multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    """Multi-class Brier score (lower is better).

    BS = (1/N) Σ_i Σ_k (p_ik − y_ik)²
    where y_ik is 1 if sample i belongs to class k.
    """
    n = len(y_true)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the standard metric suite.

    Parameters
    ----------
    y_true : integer labels (0, 1, 2)
    y_pred : predicted integer labels
    y_prob : (n, 3) probability matrix; if None, skip prob-based metrics.

    Returns
    -------
    dict with metric_name → value.
    """
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        # Clip for numerical stability in log_loss
        eps = 1e-15
        y_prob_clip = np.clip(y_prob, eps, 1 - eps)
        # Renormalise rows after clipping
        y_prob_clip = y_prob_clip / y_prob_clip.sum(axis=1, keepdims=True)
        metrics["log_loss"] = float(log_loss(y_true, y_prob_clip, labels=[0, 1, 2]))
        metrics["brier_score"] = multiclass_brier(y_true, y_prob_clip)

    return metrics


def save_report(
    metrics: dict[str, float],
    model_name: str,
    extra: dict | None = None,
    dest_dir: Path = REPORTS_DIR,
) -> Path:
    """Persist a JSON report under artifacts/reports/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": model_name, "metrics": metrics}
    if extra:
        report.update(extra)
    path = dest_dir / f"{model_name}_report.json"
    path.write_text(json.dumps(report, indent=2))
    log.info("Report saved -> %s", path)
    return path


def print_metrics(metrics: dict[str, float], header: str = "Metrics") -> None:
    """Pretty-print a metrics dict."""
    log.info("=== %s ===", header)
    for k, v in metrics.items():
        log.info("  %-15s  %.4f", k, v)
