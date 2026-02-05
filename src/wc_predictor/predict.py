"""Single-match prediction interface.

Usage
-----
    python -m wc_predictor.predict --team_a Brazil --team_b Germany --date 2026-06-15

Loads the best available model and the full match history, computes pre-match
features for the given teams up to the given date, and outputs probabilities
for Home Win / Draw / Away Win.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wc_predictor.config import (
    FEATURE_SCHEMA_JSON,
    FORM_WINDOWS,
    LABEL_NAMES,
    MODELS_DIR,
    RAW_CSV,
    ensure_dirs,
    get_confederation_code,
    get_logger,
    seed_everything,
)
from wc_predictor.elo import EloRatings
from wc_predictor.features import _compute_form

log = get_logger(__name__)

# Preferred model order (best first)
MODEL_PREFERENCE = ["rf", "xgb", "logreg", "baseline"]


def _find_best_model() -> tuple[Path, str]:
    """Find the best available trained model."""
    for name in MODEL_PREFERENCE:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            return path, name
    log.error("No trained models found in %s. Run `make train` first.", MODELS_DIR)
    sys.exit(1)


def precompute_match_context(
    all_matches: pd.DataFrame,
) -> tuple[EloRatings, dict[str, list[pd.Series]]]:
    """Replay ELO and team histories over all matches in a single pass.

    Call once at startup and pass the results to ``compute_features_for_match``
    via its ``precomputed_elo`` / ``precomputed_histories`` keyword arguments
    to avoid replaying the full history on every prediction request.
    """
    sorted_matches = all_matches.sort_values("date").reset_index(drop=True)

    elo = EloRatings()
    team_histories: dict[str, list[pd.Series]] = defaultdict(list)

    for _, row in sorted_matches.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        n = bool(row.get("neutral", False))
        elo.update(h, a, int(row["home_score"]), int(row["away_score"]), neutral=n)
        team_histories[h].append(row)
        team_histories[a].append(row)

    log.info("Precomputed ELO for %d teams over %d matches.", len(elo.ratings), len(sorted_matches))
    return elo, dict(team_histories)


def compute_features_for_match(
    home_team: str,
    away_team: str,
    match_date: datetime,
    all_matches: pd.DataFrame,
    feature_cols: list[str],
    neutral: bool = True,
    return_explanation: bool = False,
    *,
    precomputed_elo: EloRatings | None = None,
    precomputed_histories: dict[str, list[pd.Series]] | None = None,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Compute the feature vector for a hypothetical match.

    When ``precomputed_elo`` and ``precomputed_histories`` are provided
    (the fast path used by the API), ELO/history replay is skipped entirely.
    Otherwise replays from scratch (the safe path for CLI/training where
    match_date may be in the middle of the dataset).

    If return_explanation is True, also returns a human-readable dict
    of the key pre-match signals (for the frontend explainability card).
    """
    from wc_predictor.features import _outcome_for_team

    if precomputed_elo is not None and precomputed_histories is not None:
        # Fast path: use pre-built state (safe when match_date > all data).
        elo = precomputed_elo
        team_histories = precomputed_histories
        prior_count = len(all_matches)
    else:
        # Slow path: replay from scratch (leak-free for any match_date).
        prior = all_matches[all_matches["date"] < pd.Timestamp(match_date)].copy()
        prior = prior.sort_values("date").reset_index(drop=True)

        if prior.empty:
            log.error("No historical matches found before %s.", match_date)
            sys.exit(1)

        elo = EloRatings()
        team_histories: dict[str, list[pd.Series]] = defaultdict(list)

        for _, row in prior.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            n = bool(row.get("neutral", False))
            elo.update(h, a, int(row["home_score"]), int(row["away_score"]), neutral=n)
            team_histories[h].append(row)
            team_histories[a].append(row)

        prior_count = len(prior)

    # Assemble features
    feat: dict[str, float] = {}
    feat["home_elo"] = elo.get(home_team)
    feat["away_elo"] = elo.get(away_team)
    feat["elo_diff"] = feat["home_elo"] - feat["away_elo"]

    probs = elo.win_probabilities(home_team, away_team, neutral=neutral)
    feat["elo_prob_h"] = probs["H"]
    feat["elo_prob_d"] = probs["D"]
    feat["elo_prob_a"] = probs["A"]

    for n in FORM_WINDOWS:
        form_h = _compute_form(team_histories.get(home_team, []), home_team, n)
        feat.update({f"home_{k}": v for k, v in form_h.items()})
        form_a = _compute_form(team_histories.get(away_team, []), away_team, n)
        feat.update({f"away_{k}": v for k, v in form_a.items()})

    # Head-to-head
    h2h = [
        m
        for m in team_histories.get(home_team, [])
        if m["home_team"] == away_team or m["away_team"] == away_team
    ]
    h2h_recent = h2h[-5:] if h2h else []
    if h2h_recent:
        h2h_wins = sum(1 for m in h2h_recent if _outcome_for_team(m, home_team) == "W")
        feat["h2h_home_win_rate"] = h2h_wins / len(h2h_recent)
    else:
        feat["h2h_home_win_rate"] = np.nan

    feat["is_neutral"] = int(neutral)

    # Confederation features
    feat["home_confederation"] = get_confederation_code(home_team)
    feat["away_confederation"] = get_confederation_code(away_team)
    feat["same_confederation"] = int(
        get_confederation_code(home_team) == get_confederation_code(away_team)
    )

    # Build array in the right column order
    vec = []
    for col in feature_cols:
        vec.append(feat.get(col, np.nan))

    X = np.array(vec, dtype=np.float64).reshape(1, -1)

    if not return_explanation:
        return X

    # ── Build human-readable explanation ─────────────────────────────
    def _form_summary(histories: list, team: str, n: int = 5) -> dict:
        """W/D/L counts and goal stats from last n matches."""
        recent = histories[-n:] if len(histories) >= n else histories
        if not recent:
            return {"matches": 0, "W": 0, "D": 0, "L": 0,
                    "goals_for_avg": None, "goals_against_avg": None}
        outcomes = [_outcome_for_team(r, team) for r in recent]
        from wc_predictor.features import _goals_for_against
        gf_ga = [_goals_for_against(r, team) for r in recent]
        return {
            "matches": len(recent),
            "W": sum(1 for o in outcomes if o == "W"),
            "D": sum(1 for o in outcomes if o == "D"),
            "L": sum(1 for o in outcomes if o == "L"),
            "goals_for_avg": round(float(np.mean([g[0] for g in gf_ga])), 2),
            "goals_against_avg": round(float(np.mean([g[1] for g in gf_ga])), 2),
        }

    # H2H summary
    h2h_summary = None
    if h2h_recent:
        h2h_home_w = sum(1 for m in h2h_recent if _outcome_for_team(m, home_team) == "W")
        h2h_draws = sum(1 for m in h2h_recent if _outcome_for_team(m, home_team) == "D")
        h2h_away_w = len(h2h_recent) - h2h_home_w - h2h_draws
        h2h_summary = {
            "last_n": len(h2h_recent),
            "home_wins": h2h_home_w,
            "draws": h2h_draws,
            "away_wins": h2h_away_w,
        }

    from wc_predictor.config import CONFEDERATION_MAP
    explanation = {
        "elo": {
            "home_elo": round(feat["home_elo"], 1),
            "away_elo": round(feat["away_elo"], 1),
            "elo_diff": round(feat["elo_diff"], 1),
            "elo_home_win_prob": round(feat["elo_prob_h"], 3),
            "elo_draw_prob": round(feat["elo_prob_d"], 3),
            "elo_away_win_prob": round(feat["elo_prob_a"], 3),
        },
        "home_form_last5": _form_summary(
            team_histories.get(home_team, []), home_team, 5
        ),
        "away_form_last5": _form_summary(
            team_histories.get(away_team, []), away_team, 5
        ),
        "head_to_head": h2h_summary,
        "neutral_venue": bool(neutral),
        "home_confederation": CONFEDERATION_MAP.get(home_team, "UNKNOWN"),
        "away_confederation": CONFEDERATION_MAP.get(away_team, "UNKNOWN"),
        "total_matches_in_history": prior_count,
    }

    return X, explanation


def predict_match(
    home_team: str,
    away_team: str,
    match_date: str,
    model_name: str | None = None,
    neutral: bool = True,
) -> dict:
    """Predict outcome probabilities for a single match.

    Returns a dict with probabilities and the top prediction.
    """
    seed_everything()
    ensure_dirs()

    # Parse date
    dt = datetime.strptime(match_date, "%Y-%m-%d")

    # Load model
    if model_name:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            log.error("Model '%s' not found at %s.", model_name, model_path)
            sys.exit(1)
        mname = model_name
    else:
        model_path, mname = _find_best_model()

    log.info("Using model: %s (%s)", mname, model_path)
    pipeline = joblib.load(model_path)

    # Load schema
    if not FEATURE_SCHEMA_JSON.exists():
        log.error("Feature schema not found. Run `python -m wc_predictor.features` first.")
        sys.exit(1)
    schema = json.loads(FEATURE_SCHEMA_JSON.read_text())
    feature_cols = schema["feature_columns"]

    # Load all international matches
    if not RAW_CSV.exists():
        log.error("Raw data not found. Run `python -m wc_predictor.download_data` first.")
        sys.exit(1)
    all_matches = pd.read_csv(RAW_CSV, parse_dates=["date"])
    all_matches["neutral"] = all_matches["neutral"].fillna(False).astype(bool)
    all_matches = all_matches.sort_values("date").reset_index(drop=True)

    # Compute features
    log.info(
        "Computing features for %s vs %s on %s (neutral=%s) …",
        home_team,
        away_team,
        match_date,
        neutral,
    )
    X = compute_features_for_match(
        home_team, away_team, dt, all_matches, feature_cols, neutral=neutral,
    )

    # Predict
    proba = pipeline.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = LABEL_NAMES[pred_idx]

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "date": match_date,
        "model": mname,
        "probabilities": {
            "H (Home Win)": round(float(proba[0]), 4),
            "D (Draw)": round(float(proba[1]), 4),
            "A (Away Win)": round(float(proba[2]), 4),
        },
        "prediction": pred_label,
    }

    log.info("==============================================")
    log.info("  %s vs %s  (%s)", home_team, away_team, match_date)
    log.info("  Model: %s", mname)
    log.info("----------------------------------------------")
    log.info("  P(Home Win)  = %.1f%%", proba[0] * 100)
    log.info("  P(Draw)      = %.1f%%", proba[1] * 100)
    log.info("  P(Away Win)  = %.1f%%", proba[2] * 100)
    log.info("----------------------------------------------")
    log.info("  >>> Prediction: %s <<<", pred_label)
    log.info("==============================================")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a World Cup match outcome.")
    parser.add_argument("--team_a", required=True, help="Home team name (e.g. Brazil).")
    parser.add_argument("--team_b", required=True, help="Away team name (e.g. Germany).")
    parser.add_argument(
        "--date",
        required=True,
        help="Match date YYYY-MM-DD (features computed from history before this date).",
    )
    parser.add_argument("--model", default=None, help="Model name (default: best available).")
    parser.add_argument(
        "--neutral",
        action="store_true",
        default=True,
        help="Treat as neutral venue (default for World Cup).",
    )
    args = parser.parse_args()

    predict_match(
        home_team=args.team_a,
        away_team=args.team_b,
        match_date=args.date,
        model_name=args.model,
        neutral=args.neutral,
    )


if __name__ == "__main__":
    main()
