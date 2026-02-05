"""Tests for temporal splitting and metric computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc_predictor.splits import (
    compute_metrics,
    multiclass_brier,
    temporal_split,
    time_series_cv_indices,
)


# ── Temporal split ───────────────────────────────────────────────────────────


class TestTemporalSplit:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2010-06-01", "2012-06-01", "2014-06-01", "2018-06-01", "2022-06-01"]
                ),
                "feature": [1, 2, 3, 4, 5],
                "y": [0, 1, 2, 0, 1],
            }
        )

    def test_split_sizes(self, sample_df: pd.DataFrame) -> None:
        train, test = temporal_split(sample_df, cutoff_year=2014)
        assert len(train) == 3  # 2010, 2012, 2014
        assert len(test) == 2  # 2018, 2022

    def test_no_future_in_train(self, sample_df: pd.DataFrame) -> None:
        train, _ = temporal_split(sample_df, cutoff_year=2014)
        assert train["date"].max().year <= 2014

    def test_no_past_in_test(self, sample_df: pd.DataFrame) -> None:
        _, test = temporal_split(sample_df, cutoff_year=2014)
        assert test["date"].min().year > 2014

    def test_extreme_cutoff_all_train(self, sample_df: pd.DataFrame) -> None:
        train, test = temporal_split(sample_df, cutoff_year=2030)
        assert len(train) == 5
        assert len(test) == 0

    def test_extreme_cutoff_all_test(self, sample_df: pd.DataFrame) -> None:
        train, test = temporal_split(sample_df, cutoff_year=2000)
        assert len(train) == 0
        assert len(test) == 5


# ── Time-series CV ───────────────────────────────────────────────────────────


class TestTimeSeriesCV:
    def test_cv_respects_order(self) -> None:
        indices = time_series_cv_indices(10, n_splits=3)
        assert len(indices) == 3
        for train_idx, test_idx in indices:
            # All train indices should be < all test indices
            assert train_idx.max() < test_idx.min()

    def test_cv_folds_grow(self) -> None:
        indices = time_series_cv_indices(10, n_splits=3)
        sizes = [len(train) for train, _ in indices]
        assert sizes == sorted(sizes)  # training set grows each fold


# ── Metrics ──────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_perfect_accuracy(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_zero_accuracy(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_perfect_probabilities_low_logloss(self) -> None:
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
        ])
        m = compute_metrics(y_true, y_true, y_prob)
        assert m["log_loss"] < 0.05
        assert m["brier_score"] < 0.01

    def test_uniform_probabilities_higher_logloss(self) -> None:
        y_true = np.array([0, 1, 2])
        y_prob_good = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        y_prob_uniform = np.array([
            [0.33, 0.34, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.34, 0.33],
        ])
        m_good = compute_metrics(y_true, y_true, y_prob_good)
        m_uniform = compute_metrics(y_true, y_true, y_prob_uniform)
        assert m_good["log_loss"] < m_uniform["log_loss"]


class TestBrierScore:
    def test_perfect_brier_is_zero(self) -> None:
        y_true = np.array([0, 1, 2])
        y_prob = np.eye(3)  # perfect one-hot
        assert multiclass_brier(y_true, y_prob) == pytest.approx(0.0)

    def test_worst_brier(self) -> None:
        """Confidently wrong predictions should have high Brier score."""
        y_true = np.array([0, 1, 2])
        # Predict the opposite class with high confidence
        y_prob = np.array([
            [0.0, 0.0, 1.0],  # true=0, predict 2
            [1.0, 0.0, 0.0],  # true=1, predict 0
            [0.0, 1.0, 0.0],  # true=2, predict 1
        ])
        bs = multiclass_brier(y_true, y_prob)
        assert bs > 1.0

    def test_uniform_brier(self) -> None:
        """Uniform predictions: Brier = (1/N) * N * ((1/3-1)^2 + 2*(1/3)^2)
        = (2/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 ≈ 0.667"""
        y_true = np.array([0, 1, 2])
        y_prob = np.full((3, 3), 1 / 3)
        bs = multiclass_brier(y_true, y_prob)
        assert bs == pytest.approx(2 / 3, abs=0.01)
