"""Tests for the ELO rating system.

Covers:
    - Initial ratings
    - Update correctness (win/loss/draw)
    - Expected score symmetry
    - Home advantage effect
    - Neutral venue behaviour
    - Deterministic outputs
"""

from __future__ import annotations

import pytest
import numpy as np

from wc_predictor.elo import EloRatings


@pytest.fixture
def elo() -> EloRatings:
    """Fresh ELO instance with default parameters."""
    return EloRatings(k=32, initial=1500, home_advantage=100)


class TestEloInitial:
    def test_new_team_gets_initial_rating(self, elo: EloRatings) -> None:
        assert elo.get("Brazil") == 1500.0

    def test_different_teams_same_initial(self, elo: EloRatings) -> None:
        assert elo.get("Brazil") == elo.get("Germany")


class TestExpected:
    def test_equal_ratings_gives_half(self, elo: EloRatings) -> None:
        assert elo.expected(1500, 1500) == pytest.approx(0.5, abs=1e-10)

    def test_higher_rating_higher_expected(self, elo: EloRatings) -> None:
        assert elo.expected(1600, 1400) > 0.5
        assert elo.expected(1400, 1600) < 0.5

    def test_symmetry(self, elo: EloRatings) -> None:
        """E(A vs B) + E(B vs A) = 1."""
        e1 = elo.expected(1600, 1400)
        e2 = elo.expected(1400, 1600)
        assert e1 + e2 == pytest.approx(1.0, abs=1e-10)

    def test_400_point_difference(self, elo: EloRatings) -> None:
        """400 points difference → ~91% expected score."""
        e = elo.expected(1900, 1500)
        assert e == pytest.approx(0.9091, abs=0.01)


class TestUpdate:
    def test_home_win_increases_home_rating(self, elo: EloRatings) -> None:
        """Home win: home rating should go up, away down."""
        h_new, a_new = elo.update("Brazil", "Germany", 2, 0)
        assert h_new > 1500
        assert a_new < 1500

    def test_away_win_increases_away_rating(self, elo: EloRatings) -> None:
        h_new, a_new = elo.update("Brazil", "Germany", 0, 1)
        assert h_new < 1500
        assert a_new > 1500

    def test_draw_ratings_move_toward_each_other(self, elo: EloRatings) -> None:
        """When equal teams draw at home, home drops (because home advantage
        made them the favourite), away rises."""
        h_new, a_new = elo.update("Brazil", "Germany", 1, 1)
        # With home advantage, Brazil was expected to win → draw is underperformance
        assert h_new < 1500
        assert a_new > 1500

    def test_rating_changes_sum_to_zero(self, elo: EloRatings) -> None:
        """ELO is a zero-sum system: changes cancel out."""
        h_new, a_new = elo.update("Brazil", "Germany", 3, 1)
        delta_h = h_new - 1500
        delta_a = a_new - 1500
        assert delta_h + delta_a == pytest.approx(0.0, abs=1e-10)

    def test_upset_causes_larger_change(self, elo: EloRatings) -> None:
        """A big upset (low-rated team beats high-rated) should produce a
        larger rating change than a favoured win."""
        # Set up: make Brazil strong
        elo.ratings["Brazil"] = 1800
        elo.ratings["Andorra"] = 1200

        _, a_new_upset = elo.update("Brazil", "Andorra", 0, 1)  # Andorra wins
        upset_change = a_new_upset - 1200

        # Reset
        elo2 = EloRatings(k=32, initial=1500, home_advantage=100)
        elo2.ratings["Brazil"] = 1800
        elo2.ratings["Andorra"] = 1200
        h_new_fav, _ = elo2.update("Brazil", "Andorra", 3, 0)  # Brazil wins
        fav_change = h_new_fav - 1800

        assert abs(upset_change) > abs(fav_change)

    def test_sequential_updates_persist(self, elo: EloRatings) -> None:
        """Ratings should persist across multiple update calls."""
        elo.update("Brazil", "Germany", 2, 0)
        r1 = elo.get("Brazil")
        elo.update("Brazil", "France", 1, 1)
        r2 = elo.get("Brazil")
        assert r1 != r2  # second match should change the rating


class TestNeutralVenue:
    def test_neutral_no_home_advantage(self, elo: EloRatings) -> None:
        """At a neutral venue, equal-rated teams should have symmetric probs."""
        h_new, a_new = elo.update("Brazil", "Germany", 1, 1, neutral=True)
        # Both teams same initial → draw at neutral → no change
        delta_h = h_new - 1500
        delta_a = a_new - 1500
        assert delta_h == pytest.approx(0.0, abs=1e-10)
        assert delta_a == pytest.approx(0.0, abs=1e-10)

    def test_home_advantage_matters(self, elo: EloRatings) -> None:
        """Non-neutral draw should differ from neutral draw for equal teams."""
        elo1 = EloRatings(k=32, initial=1500, home_advantage=100)
        h1, _ = elo1.update("A", "B", 1, 1, neutral=False)

        elo2 = EloRatings(k=32, initial=1500, home_advantage=100)
        h2, _ = elo2.update("A", "B", 1, 1, neutral=True)

        assert h1 != h2


class TestDeterminism:
    def test_same_inputs_same_outputs(self) -> None:
        """Two independent ELO objects with the same inputs produce the same results."""
        matches = [
            ("Brazil", "Germany", 2, 1, False),
            ("Germany", "France", 0, 0, False),
            ("France", "Brazil", 1, 3, True),
        ]
        elo1 = EloRatings(k=32, initial=1500, home_advantage=100)
        elo2 = EloRatings(k=32, initial=1500, home_advantage=100)

        for h, a, hs, as_, n in matches:
            elo1.update(h, a, hs, as_, neutral=n)
            elo2.update(h, a, hs, as_, neutral=n)

        for team in ("Brazil", "Germany", "France"):
            assert elo1.get(team) == elo2.get(team)


class TestWinProbabilities:
    def test_probabilities_sum_to_one(self, elo: EloRatings) -> None:
        probs = elo.win_probabilities("Brazil", "Germany")
        total = probs["H"] + probs["D"] + probs["A"]
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_stronger_team_higher_win_prob(self, elo: EloRatings) -> None:
        elo.ratings["Brazil"] = 1800
        elo.ratings["Andorra"] = 1200
        probs = elo.win_probabilities("Brazil", "Andorra")
        assert probs["H"] > probs["A"]

    def test_all_probabilities_positive(self, elo: EloRatings) -> None:
        probs = elo.win_probabilities("Brazil", "Germany")
        assert all(p > 0 for p in probs.values())
