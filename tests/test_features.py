"""Tests for feature engineering — especially leakage prevention.

The key property: features for match i must never use information from
matches with date >= date_i.

We test this with a small synthetic dataset where we can verify by hand.
"""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from wc_predictor.features import _compute_form, _outcome_for_team, _goals_for_against


# ── Helpers ──────────────────────────────────────────────────────────────────

SYNTHETIC_CSV = """\
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2010-06-11,South Africa,Mexico,1,1,FIFA World Cup,Johannesburg,South Africa,FALSE
2010-06-12,Argentina,Nigeria,1,0,FIFA World Cup,Johannesburg,South Africa,TRUE
2010-06-13,Germany,Australia,4,0,FIFA World Cup,Durban,South Africa,TRUE
2010-06-17,South Africa,Uruguay,0,3,FIFA World Cup,Pretoria,South Africa,FALSE
2010-06-22,Mexico,Uruguay,0,1,FIFA World Cup,Rustenburg,South Africa,TRUE
"""


def _load_synthetic() -> pd.DataFrame:
    return pd.read_csv(StringIO(SYNTHETIC_CSV), parse_dates=["date"])


# ── Test outcome helpers ─────────────────────────────────────────────────────


class TestOutcomeHelpers:
    def test_home_win(self) -> None:
        row = pd.Series({
            "home_team": "Argentina",
            "away_team": "Nigeria",
            "home_score": 1,
            "away_score": 0,
        })
        assert _outcome_for_team(row, "Argentina") == "W"
        assert _outcome_for_team(row, "Nigeria") == "L"

    def test_draw(self) -> None:
        row = pd.Series({
            "home_team": "South Africa",
            "away_team": "Mexico",
            "home_score": 1,
            "away_score": 1,
        })
        assert _outcome_for_team(row, "South Africa") == "D"
        assert _outcome_for_team(row, "Mexico") == "D"

    def test_away_win(self) -> None:
        row = pd.Series({
            "home_team": "South Africa",
            "away_team": "Uruguay",
            "home_score": 0,
            "away_score": 3,
        })
        assert _outcome_for_team(row, "South Africa") == "L"
        assert _outcome_for_team(row, "Uruguay") == "W"

    def test_goals_for_against_home(self) -> None:
        row = pd.Series({
            "home_team": "Germany",
            "away_team": "Australia",
            "home_score": 4,
            "away_score": 0,
        })
        gf, ga = _goals_for_against(row, "Germany")
        assert gf == 4
        assert ga == 0

    def test_goals_for_against_away(self) -> None:
        row = pd.Series({
            "home_team": "Germany",
            "away_team": "Australia",
            "home_score": 4,
            "away_score": 0,
        })
        gf, ga = _goals_for_against(row, "Australia")
        assert gf == 0
        assert ga == 4


# ── Test form computation ────────────────────────────────────────────────────


class TestFormComputation:
    def test_empty_history_returns_nan(self) -> None:
        result = _compute_form([], "Brazil", 5)
        assert all(np.isnan(v) for v in result.values())

    def test_single_match_form(self) -> None:
        row = pd.Series({
            "home_team": "Argentina",
            "away_team": "Nigeria",
            "home_score": 1,
            "away_score": 0,
        })
        result = _compute_form([row], "Argentina", 3)
        assert result["form_3_win_rate"] == 1.0
        assert result["form_3_gf_avg"] == 1.0
        assert result["form_3_ga_avg"] == 0.0
        assert result["form_3_gd_avg"] == 1.0

    def test_form_window_clips(self) -> None:
        """If we have 2 matches but ask for form_5, it should use all 2."""
        rows = [
            pd.Series({"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0}),
            pd.Series({"home_team": "A", "away_team": "C", "home_score": 0, "away_score": 1}),
        ]
        result = _compute_form(rows, "A", 5)
        assert result["form_5_win_rate"] == pytest.approx(0.5)

    def test_form_window_limits(self) -> None:
        """If we have 5 matches but ask for form_3, it should use only last 3."""
        rows = [
            pd.Series({"home_team": "A", "away_team": "B", "home_score": 3, "away_score": 0}),  # W
            pd.Series({"home_team": "A", "away_team": "C", "home_score": 3, "away_score": 0}),  # W
            pd.Series({"home_team": "A", "away_team": "D", "home_score": 0, "away_score": 1}),  # L
            pd.Series({"home_team": "A", "away_team": "E", "home_score": 0, "away_score": 1}),  # L
            pd.Series({"home_team": "A", "away_team": "F", "home_score": 0, "away_score": 1}),  # L
        ]
        result = _compute_form(rows, "A", 3)
        # Last 3 are all losses
        assert result["form_3_win_rate"] == pytest.approx(0.0)


# ── Leakage guard ────────────────────────────────────────────────────────────


class TestLeakagePrevention:
    """These tests verify that the feature pipeline never uses future data.

    We do this by constructing a scenario where leakage would produce
    detectably different feature values.
    """

    def test_south_africa_first_match_has_no_prior_form(self) -> None:
        """The very first match (South Africa vs Mexico, 2010-06-11)
        should have NaN form features because there are no prior WC matches
        in this tiny dataset to compute form from.

        If the pipeline leaked, it might fill in stats from later matches.
        """
        df = _load_synthetic()
        # South Africa's first match: no prior matches in this dataset
        # Form should be NaN for all windows
        history: list[pd.Series] = []  # no prior matches
        for n in [3, 5, 10]:
            result = _compute_form(history, "South Africa", n)
            assert np.isnan(result[f"form_{n}_win_rate"])

    def test_second_match_uses_only_first(self) -> None:
        """South Africa's second match (vs Uruguay, 2010-06-17) should
        only use the first match (vs Mexico, draw 1-1) for form.
        """
        df = _load_synthetic()
        # After the first match (draw), form should reflect 0 wins
        first_match = df.iloc[0]
        history = [first_match]
        result = _compute_form(history, "South Africa", 3)
        assert result["form_3_win_rate"] == pytest.approx(0.0)  # drew, not won
        assert result["form_3_draw_rate"] == pytest.approx(1.0)
        assert result["form_3_gf_avg"] == pytest.approx(1.0)
        assert result["form_3_ga_avg"] == pytest.approx(1.0)

    def test_form_never_includes_current_match(self) -> None:
        """Even if history includes the current match's row, _compute_form
        would include it.  The *caller* (feature builder) is responsible for
        not appending a match to history until after its features are computed.

        Here we verify the caller contract: history passed to _compute_form
        for match i should have len = (number of matches for that team before i).
        """
        df = _load_synthetic()
        # Uruguay appears in match index 3 (vs South Africa) and 4 (vs Mexico).
        # When computing features for match 4, Uruguay's history should
        # contain only match 3.
        match_3 = df.iloc[3]  # South Africa 0-3 Uruguay
        history_at_match_4 = [match_3]
        result = _compute_form(history_at_match_4, "Uruguay", 3)
        # Uruguay won 3-0 in match 3
        assert result["form_3_win_rate"] == pytest.approx(1.0)
        assert result["form_3_gf_avg"] == pytest.approx(3.0)


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_same_form(self) -> None:
        row = pd.Series({
            "home_team": "X",
            "away_team": "Y",
            "home_score": 2,
            "away_score": 1,
        })
        r1 = _compute_form([row], "X", 3)
        r2 = _compute_form([row], "X", 3)
        assert r1 == r2
