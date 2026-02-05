"""Filter raw international results to World Cup matches and clean.

Produces data/processed/matches.csv with columns:
    date, home_team, away_team, home_score, away_score,
    tournament, neutral, result
where result ∈ {H, D, A} (Home win / Draw / Away win).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_predictor.config import (
    PROCESSED_CSV,
    RAW_CSV,
    ensure_dirs,
    get_logger,
)

log = get_logger(__name__)

WC_TOURNAMENT_NAME = "FIFA World Cup"


def build(
    raw_path: Path = RAW_CSV,
    out_path: Path = PROCESSED_CSV,
    include_qualifiers: bool = False,
) -> pd.DataFrame:
    """Read raw CSV, filter to World Cup, add result label, save processed CSV.

    Parameters
    ----------
    raw_path : path to results.csv
    out_path : destination for cleaned CSV
    include_qualifiers : if True, also keep "FIFA World Cup qualification" rows

    Returns
    -------
    Cleaned DataFrame.
    """
    ensure_dirs()
    log.info("Loading raw data from %s …", raw_path)
    df = pd.read_csv(raw_path, parse_dates=["date"])

    # Filter tournaments
    mask = df["tournament"] == WC_TOURNAMENT_NAME
    if include_qualifiers:
        mask |= df["tournament"].str.contains("FIFA World Cup", na=False)
    df = df.loc[mask].copy()
    log.info("Filtered to %d World Cup matches.", len(df))

    if df.empty:
        raise ValueError(
            f"No matches found with tournament == '{WC_TOURNAMENT_NAME}'. "
            "Check that the raw CSV is the expected dataset."
        )

    # Derive result label
    df["result"] = "D"
    df.loc[df["home_score"] > df["away_score"], "result"] = "H"
    df.loc[df["home_score"] < df["away_score"], "result"] = "A"

    # Keep useful columns, sort chronologically
    keep = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
        "result",
    ]
    df = df[keep].sort_values("date").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    log.info("Saved processed dataset -> %s  (%d rows)", out_path, len(df))

    # Quick class distribution
    dist = df["result"].value_counts()
    log.info("Class distribution:\n%s", dist.to_string())

    return df


if __name__ == "__main__":
    build()
