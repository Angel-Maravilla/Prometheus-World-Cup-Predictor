"""Download and cache the international football results dataset.

Data source: Mart Jürisoo's open dataset on GitHub.
https://github.com/martj42/international_results

Expected columns:
    date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from wc_predictor.config import (
    DATASET_URL,
    RAW_CSV,
    ensure_dirs,
    get_logger,
)

log = get_logger(__name__)

REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}


def download(url: str = DATASET_URL, dest: Path = RAW_CSV, force: bool = False) -> Path:
    """Download the raw CSV if it doesn't already exist.

    Returns the path to the local file.
    """
    ensure_dirs()

    if dest.exists() and not force:
        log.info("Raw data already cached at %s — skipping download.", dest)
        return dest

    log.info("Downloading dataset from %s …", url)
    try:
        import requests

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        log.info("Saved %d bytes -> %s", len(resp.content), dest)
    except Exception as exc:
        log.error(
            "Download failed: %s\n"
            "Please download the CSV manually and place it at:\n"
            "  %s\n"
            "Source: https://github.com/martj42/international_results",
            exc,
            dest,
        )
        sys.exit(1)

    return dest


def validate(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the CSV and validate that required columns exist."""
    log.info("Validating %s …", path)
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        log.error("Missing columns in dataset: %s", missing)
        sys.exit(1)
    log.info(
        "Validation OK — %d rows, columns: %s",
        len(df),
        list(df.columns),
    )
    return df


def main() -> None:
    """CLI entry point: download + validate."""
    path = download()
    validate(path)
    log.info("Data ready at %s", path)


if __name__ == "__main__":
    main()
