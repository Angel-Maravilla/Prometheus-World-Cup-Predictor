"""Sequential ELO rating system for international football teams.

Key design decisions
--------------------
* Ratings are computed **sequentially** over all international matches (not
  just World Cup) so that ELO reflects true team strength entering a
  tournament.  Only the *feature lookup* is restricted to World Cup matches.
* Home-field advantage is configurable (default +100 for the home team's
  expected score).  For neutral-venue matches the bonus is zeroed.
* The K-factor is fixed (default 32).  A tournament-weighted K (e.g. higher
  for World Cup) is a reasonable extension but adds complexity without
  guaranteed benefit, so we keep it simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wc_predictor.config import ELO_HOME_ADVANTAGE, ELO_INITIAL, ELO_K, get_logger

log = get_logger(__name__)


@dataclass
class EloRatings:
    """Maintains a mapping of team → current rating and exposes update logic."""

    k: float = ELO_K
    initial: float = ELO_INITIAL
    home_advantage: float = ELO_HOME_ADVANTAGE
    ratings: dict[str, float] = field(default_factory=dict)

    # ── public API ───────────────────────────────────────────────────────

    def get(self, team: str) -> float:
        """Return current rating for *team*, defaulting to initial."""
        return self.ratings.get(team, self.initial)

    def expected(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A given both ratings (logistic curve)."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Update ratings after a single match.  Returns (new_home, new_away).

        The actual score S is 1 for a win, 0.5 for a draw, 0 for a loss.
        """
        r_home = self.get(home_team)
        r_away = self.get(away_team)

        # Apply home advantage unless neutral venue
        ha = 0.0 if neutral else self.home_advantage
        e_home = self.expected(r_home + ha, r_away)
        e_away = 1.0 - e_home

        # Actual scores
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        new_home = r_home + self.k * (s_home - e_home)
        new_away = r_away + self.k * (s_away - e_away)
        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away
        return new_home, new_away

    def win_probabilities(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
    ) -> dict[str, float]:
        """Approximate H/D/A probabilities from ELO difference.

        Uses the logistic expected score as P(home win) proxy, then carves
        out a draw band proportional to how close the teams are.  This is a
        simple heuristic — *not* a calibrated model.
        """
        r_home = self.get(home_team)
        r_away = self.get(away_team)
        ha = 0.0 if neutral else self.home_advantage
        e_home = self.expected(r_home + ha, r_away)

        # Draw band: wider when teams are close.  Peaked at e_home=0.5.
        draw_base = 0.26  # base draw probability
        draw_boost = 0.12 * (1.0 - abs(2.0 * e_home - 1.0))
        p_draw = draw_base + draw_boost

        remainder = 1.0 - p_draw
        p_home = remainder * e_home
        p_away = remainder * (1.0 - e_home)
        return {"H": p_home, "D": p_draw, "A": p_away}


def compute_elo_history(
    all_matches: pd.DataFrame,
    k: float = ELO_K,
    initial: float = ELO_INITIAL,
    home_advantage: float = ELO_HOME_ADVANTAGE,
) -> pd.DataFrame:
    """Compute ELO ratings sequentially over *all* international matches.

    Parameters
    ----------
    all_matches : DataFrame with columns
        date, home_team, away_team, home_score, away_score, neutral
        — must be sorted by date ascending.

    Returns
    -------
    DataFrame with one row per match and columns:
        date, home_team, away_team,
        home_elo_before, away_elo_before,
        home_elo_after, away_elo_after
    """
    elo = EloRatings(k=k, initial=initial, home_advantage=home_advantage)
    records: list[dict] = []

    for row in all_matches.itertuples(index=False):
        h = row.home_team
        a = row.away_team
        h_before = elo.get(h)
        a_before = elo.get(a)

        neutral = bool(getattr(row, "neutral", False))
        h_after, a_after = elo.update(
            h, a, int(row.home_score), int(row.away_score), neutral=neutral,
        )
        records.append(
            {
                "date": row.date,
                "home_team": h,
                "away_team": a,
                "home_elo_before": h_before,
                "away_elo_before": a_before,
                "home_elo_after": h_after,
                "away_elo_after": a_after,
            }
        )

    df = pd.DataFrame(records)
    log.info(
        "Computed ELO for %d matches across %d teams.",
        len(df),
        len(elo.ratings),
    )
    # Sanity: ELO should stay in a reasonable range
    all_ratings = np.array(list(elo.ratings.values()))
    log.info(
        "Final ELO stats — min: %.0f  median: %.0f  max: %.0f",
        all_ratings.min(),
        np.median(all_ratings),
        all_ratings.max(),
    )
    return df
