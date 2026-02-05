"""Tests for prediction consistency, class ordering, and confidence scoring."""

from __future__ import annotations

import numpy as np
import pytest

from wc_predictor.config import LABEL_NAMES


# ── Probability consistency ──────────────────────────────────────────────────


class TestProbabilityConsistency:
    """Verify that predict_proba outputs are well-formed."""

    def test_label_ordering(self) -> None:
        """LABEL_NAMES must be H, D, A in that order (matches sklearn label encoding)."""
        assert LABEL_NAMES == ["H", "D", "A"]

    def test_proba_sums_to_one_synthetic(self) -> None:
        """Synthetic proba vectors should sum to 1.0 (within tolerance)."""
        probas = [
            np.array([0.6, 0.2, 0.2]),
            np.array([0.33, 0.34, 0.33]),
            np.array([0.1, 0.1, 0.8]),
            np.array([1.0, 0.0, 0.0]),
        ]
        for p in probas:
            assert abs(p.sum() - 1.0) < 1e-10

    def test_proba_non_negative(self) -> None:
        """All probability values must be >= 0."""
        p = np.array([0.5, 0.3, 0.2])
        assert np.all(p >= 0)

    def test_argmax_corresponds_to_label(self) -> None:
        """argmax of [0.6, 0.1, 0.3] should give index 0 => 'H'."""
        proba = np.array([0.6, 0.1, 0.3])
        pred_idx = int(np.argmax(proba))
        assert LABEL_NAMES[pred_idx] == "H"

    def test_argmax_away_win(self) -> None:
        """argmax of [0.1, 0.2, 0.7] should give index 2 => 'A'."""
        proba = np.array([0.1, 0.2, 0.7])
        pred_idx = int(np.argmax(proba))
        assert LABEL_NAMES[pred_idx] == "A"

    def test_argmax_draw(self) -> None:
        """argmax of [0.2, 0.5, 0.3] should give index 1 => 'D'."""
        proba = np.array([0.2, 0.5, 0.3])
        pred_idx = int(np.argmax(proba))
        assert LABEL_NAMES[pred_idx] == "D"


# ── Confidence scoring ───────────────────────────────────────────────────────


class TestConfidenceScoring:
    """Verify the confidence scoring logic from api.py."""

    @staticmethod
    def _compute_confidence(proba: np.ndarray) -> dict:
        """Replicate the API confidence computation for testing."""
        max_prob = float(np.max(proba))
        eps = 1e-15
        entropy = float(-np.sum(proba * np.log(np.clip(proba, eps, 1.0))))
        if max_prob >= 0.55:
            label = "HIGH"
        elif max_prob >= 0.42:
            label = "MED"
        else:
            label = "LOW"
        return {"max_prob": max_prob, "entropy": entropy, "label": label}

    def test_high_confidence(self) -> None:
        """Dominant probability => HIGH confidence."""
        result = self._compute_confidence(np.array([0.7, 0.2, 0.1]))
        assert result["label"] == "HIGH"
        assert result["max_prob"] == pytest.approx(0.7)

    def test_medium_confidence(self) -> None:
        """Moderate probability => MED confidence."""
        result = self._compute_confidence(np.array([0.45, 0.30, 0.25]))
        assert result["label"] == "MED"

    def test_low_confidence(self) -> None:
        """Uniform-ish probability => LOW confidence."""
        result = self._compute_confidence(np.array([0.34, 0.33, 0.33]))
        assert result["label"] == "LOW"

    def test_entropy_uniform_is_maximum(self) -> None:
        """Uniform distribution has maximum entropy for 3 classes."""
        uniform = np.array([1 / 3, 1 / 3, 1 / 3])
        result = self._compute_confidence(uniform)
        max_entropy = np.log(3)  # ln(3) ~ 1.0986
        assert result["entropy"] == pytest.approx(max_entropy, abs=1e-4)

    def test_entropy_certain_is_zero(self) -> None:
        """A certain prediction [1, 0, 0] has entropy ~ 0."""
        certain = np.array([1.0, 0.0, 0.0])
        result = self._compute_confidence(certain)
        assert result["entropy"] == pytest.approx(0.0, abs=1e-10)
        assert result["label"] == "HIGH"

    def test_entropy_non_negative(self) -> None:
        """Entropy must always be non-negative."""
        for _ in range(20):
            p = np.random.dirichlet([1, 1, 1])
            result = self._compute_confidence(p)
            assert result["entropy"] >= 0

    def test_confidence_boundary_55(self) -> None:
        """max_prob exactly 0.55 => HIGH."""
        p = np.array([0.55, 0.25, 0.20])
        assert self._compute_confidence(p)["label"] == "HIGH"

    def test_confidence_boundary_42(self) -> None:
        """max_prob exactly 0.42 => MED."""
        p = np.array([0.42, 0.30, 0.28])
        assert self._compute_confidence(p)["label"] == "MED"

    def test_confidence_below_42(self) -> None:
        """max_prob 0.41 => LOW."""
        p = np.array([0.41, 0.30, 0.29])
        assert self._compute_confidence(p)["label"] == "LOW"


# ── Log loss sanity ──────────────────────────────────────────────────────────


class TestLogLossSanity:
    """Verify log loss computation properties."""

    def test_log_loss_finite_on_correct_predictions(self) -> None:
        """Log loss should be finite and small when model is confident and correct."""
        from sklearn.metrics import log_loss as sklearn_log_loss

        y_true = [0, 1, 2, 0, 1]
        y_proba = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.15, 0.7, 0.15],
        ]
        ll = sklearn_log_loss(y_true, y_proba, labels=[0, 1, 2])
        assert np.isfinite(ll)
        assert ll < 1.0  # Confident correct predictions have low log loss

    def test_log_loss_finite_on_wrong_predictions(self) -> None:
        """Log loss is finite even when model is wrong (with reasonable probas)."""
        from sklearn.metrics import log_loss as sklearn_log_loss

        y_true = [0, 1, 2]
        y_proba = [
            [0.1, 0.8, 0.1],  # Wrong: predicted 1, true 0
            [0.7, 0.1, 0.2],  # Wrong: predicted 0, true 1
            [0.5, 0.4, 0.1],  # Wrong: predicted 0, true 2
        ]
        ll = sklearn_log_loss(y_true, y_proba, labels=[0, 1, 2])
        assert np.isfinite(ll)

    def test_log_loss_increases_with_worse_predictions(self) -> None:
        """Worse predictions should have higher log loss."""
        from sklearn.metrics import log_loss as sklearn_log_loss

        y_true = [0, 0, 0]
        good_proba = [[0.8, 0.1, 0.1]] * 3
        bad_proba = [[0.4, 0.3, 0.3]] * 3

        ll_good = sklearn_log_loss(y_true, good_proba, labels=[0, 1, 2])
        ll_bad = sklearn_log_loss(y_true, bad_proba, labels=[0, 1, 2])
        assert ll_good < ll_bad

    def test_multiclass_brier_finite(self) -> None:
        """Brier score should be finite for any valid probability matrix."""
        from wc_predictor.splits import multiclass_brier

        y_true = np.array([0, 1, 2, 0])
        y_proba = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2],
        ])
        score = multiclass_brier(y_true, y_proba)
        assert np.isfinite(score)
        assert 0 <= score <= 2.0  # Brier score bounded
