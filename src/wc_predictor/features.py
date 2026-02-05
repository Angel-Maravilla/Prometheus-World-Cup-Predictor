"""Feature engineering — leak-free, computed from historical matches only.

Features
--------
1. **ELO ratings** (home_elo, away_elo, elo_diff) — computed sequentially
   over *all* international matches before the match date.
2. **Rolling form** for each team over the last N matches (N ∈ {3, 5, 10}):
   - win_rate, draw_rate, goals_for_avg, goals_against_avg, goal_diff_avg
3. **Head-to-head** record (last 5 meetings): home win rate.
4. **Neutral venue** indicator (binary).

Leakage guard
~~~~~~~~~~~~~
Every feature for match *i* is computed using only matches with date < date_i.
The ELO history is built over the full international calendar (not just WC)
but only *looked up* for World Cup feature rows.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from wc_predictor.config import (
    ARTIFACTS_DIR,
    FEATURE_MATRIX_CSV,
    FEATURE_SCHEMA_JSON,
    FORM_WINDOWS,
    LABEL_MAP,
    PROCESSED_CSV,
    RAW_CSV,
    ensure_dirs,
    get_confederation_code,
    get_logger,
    seed_everything,
)
from wc_predictor.elo import EloRatings, compute_elo_history

log = get_logger(__name__)


# ── Rolling form helpers ─────────────────────────────────────────────────────


def _outcome_for_team(row: pd.Series, team: str) -> str:
    """Return 'W', 'D', or 'L' from the perspective of *team*."""
    if row["home_team"] == team:
        if row["home_score"] > row["away_score"]:
            return "W"
        elif row["home_score"] < row["away_score"]:
            return "L"
        return "D"
    else:
        if row["away_score"] > row["home_score"]:
            return "W"
        elif row["away_score"] < row["home_score"]:
            return "L"
        return "D"


def _goals_for_against(row: pd.Series, team: str) -> tuple[int, int]:
    """Return (goals_for, goals_against) from the perspective of *team*."""
    if row["home_team"] == team:
        return int(row["home_score"]), int(row["away_score"])
    return int(row["away_score"]), int(row["home_score"])


def _compute_form(
    history: list[pd.Series], team: str, n: int
) -> dict[str, float]:
    """Compute rolling form stats from the last *n* matches in *history*."""
    recent = history[-n:] if len(history) >= n else history
    if not recent:
        return {
            f"form_{n}_win_rate": np.nan,
            f"form_{n}_draw_rate": np.nan,
            f"form_{n}_gf_avg": np.nan,
            f"form_{n}_ga_avg": np.nan,
            f"form_{n}_gd_avg": np.nan,
        }
    outcomes = [_outcome_for_team(r, team) for r in recent]
    gf_ga = [_goals_for_against(r, team) for r in recent]
    gf = [g[0] for g in gf_ga]
    ga = [g[1] for g in gf_ga]
    return {
        f"form_{n}_win_rate": sum(1 for o in outcomes if o == "W") / len(outcomes),
        f"form_{n}_draw_rate": sum(1 for o in outcomes if o == "D") / len(outcomes),
        f"form_{n}_gf_avg": np.mean(gf),
        f"form_{n}_ga_avg": np.mean(ga),
        f"form_{n}_gd_avg": np.mean(gf) - np.mean(ga),
    }


# ── Main feature builder ────────────────────────────────────────────────────


def build_feature_matrix(
    raw_path: Path = RAW_CSV,
    wc_path: Path = PROCESSED_CSV,
    out_path: Path = FEATURE_MATRIX_CSV,
    schema_path: Path = FEATURE_SCHEMA_JSON,
) -> pd.DataFrame:
    """Build the feature matrix for World Cup matches.

    Steps
    -----
    1. Load **all** international matches (raw) for ELO + form computation.
    2. Compute ELO history sequentially.
    3. Load **World Cup** matches (processed).
    4. For each WC match, look up ELO *before* the match and compute rolling
       form from prior matches only.
    5. Save the feature matrix and schema.

    Returns
    -------
    DataFrame with features X and label y.
    """
    ensure_dirs()
    seed_everything()

    # 1 — Load all matches
    log.info("Loading all international matches for ELO computation …")
    all_matches = pd.read_csv(raw_path, parse_dates=["date"])
    all_matches = all_matches.sort_values("date").reset_index(drop=True)
    # Fill missing neutral flag
    all_matches["neutral"] = all_matches["neutral"].fillna(False).astype(bool)

    # 2 — ELO history
    log.info("Computing ELO ratings across %d matches …", len(all_matches))
    elo_hist = compute_elo_history(all_matches)

    # Build a lookup: (date, home_team, away_team) → elo row.
    # Since there can be duplicates on the same day, use index alignment.
    elo_hist.index = all_matches.index  # same row ordering

    # 3 — Load WC matches
    wc = pd.read_csv(wc_path, parse_dates=["date"])
    log.info("Building features for %d World Cup matches …", len(wc))

    # Pre-index: for each match in all_matches, record it for the team histories
    # We iterate all_matches chronologically and, for each WC match we hit,
    # emit a feature row.
    team_histories: dict[str, list[pd.Series]] = defaultdict(list)
    wc_set = set(
        zip(
            wc["date"].astype(str),
            wc["home_team"],
            wc["away_team"],
        )
    )

    # We also need the ELO state *before* each match.  We'll carry an
    # EloRatings object in lockstep.
    elo = EloRatings()
    feature_rows: list[dict] = []

    for idx, row in all_matches.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10]
        neutral = bool(row["neutral"])

        # ── Snapshot features BEFORE update ──────────────────────────
        key = (date_str, h, a)
        if key in wc_set:
            feat: dict[str, object] = {
                "date": row["date"],
                "home_team": h,
                "away_team": a,
            }

            # ELO
            feat["home_elo"] = elo.get(h)
            feat["away_elo"] = elo.get(a)
            feat["elo_diff"] = feat["home_elo"] - feat["away_elo"]

            # ELO-based win probs (baseline model features)
            probs = elo.win_probabilities(h, a, neutral=neutral)
            feat["elo_prob_h"] = probs["H"]
            feat["elo_prob_d"] = probs["D"]
            feat["elo_prob_a"] = probs["A"]

            # Rolling form — home team
            for n in FORM_WINDOWS:
                form_h = _compute_form(team_histories[h], h, n)
                feat.update({f"home_{k}": v for k, v in form_h.items()})
                form_a = _compute_form(team_histories[a], a, n)
                feat.update({f"away_{k}": v for k, v in form_a.items()})

            # Head-to-head (last 5 meetings)
            h2h_matches = [
                m
                for m in team_histories.get(h, [])
                if m["home_team"] == a or m["away_team"] == a
            ]
            h2h_recent = h2h_matches[-5:] if h2h_matches else []
            if h2h_recent:
                h2h_wins = sum(
                    1 for m in h2h_recent if _outcome_for_team(m, h) == "W"
                )
                feat["h2h_home_win_rate"] = h2h_wins / len(h2h_recent)
            else:
                feat["h2h_home_win_rate"] = np.nan

            # Neutral venue
            feat["is_neutral"] = int(neutral)

            # Confederation features (static per team, no leakage concern)
            feat["home_confederation"] = get_confederation_code(h)
            feat["away_confederation"] = get_confederation_code(a)
            feat["same_confederation"] = int(
                get_confederation_code(h) == get_confederation_code(a)
            )

            # Label
            if row["home_score"] > row["away_score"]:
                feat["result"] = "H"
            elif row["home_score"] < row["away_score"]:
                feat["result"] = "A"
            else:
                feat["result"] = "D"

            feature_rows.append(feat)

        # ── Update ELO (always, for all matches) ────────────────────
        elo.update(h, a, int(row["home_score"]), int(row["away_score"]), neutral=neutral)

        # ── Update team histories (for form computation) ─────────────
        team_histories[h].append(row)
        team_histories[a].append(row)

    df = pd.DataFrame(feature_rows)
    log.info("Feature matrix: %d rows × %d columns.", *df.shape)

    # Encode label
    df["y"] = df["result"].map(LABEL_MAP)

    # Identify feature columns (everything numeric except y, date)
    meta_cols = {"date", "home_team", "away_team", "result", "y"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Save schema
    schema = {"feature_columns": feature_cols, "label_column": "y", "label_map": LABEL_MAP}
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2))
    log.info("Feature schema -> %s", schema_path)

    # Save matrix
    df.to_csv(out_path, index=False)
    log.info("Feature matrix -> %s", out_path)

    # Summary stats
    log.info("Missing values per feature:\n%s", df[feature_cols].isnull().sum().to_string())

    return df


def main() -> None:
    """CLI entry: build dataset then build features."""
    import argparse

    from wc_predictor.download_data import download, validate
    from wc_predictor.build_dataset import build

    parser = argparse.ArgumentParser(description="Build the feature matrix.")
    parser.add_argument(
        "--include-qualifiers",
        action="store_true",
        default=False,
        help="Include World Cup qualification matches in the target dataset.",
    )
    args = parser.parse_args()

    download()
    validate()
    build(include_qualifiers=args.include_qualifiers)
    build_feature_matrix()


if __name__ == "__main__":
    main()
