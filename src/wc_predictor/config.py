"""Centralised configuration: paths, constants, and deterministic seeding."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# ── Dataset ──────────────────────────────────────────────────────────────────
# Mart Jürisoo's "International football results from 1872 to 2024" on GitHub.
# Public domain / open data.  Columns: date, home_team, away_team,
# home_score, away_score, tournament, city, country, neutral.
DATASET_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
RAW_CSV = RAW_DIR / "results.csv"
PROCESSED_CSV = PROCESSED_DIR / "matches.csv"
FEATURE_MATRIX_CSV = PROCESSED_DIR / "feature_matrix.csv"
FEATURE_SCHEMA_JSON = ARTIFACTS_DIR / "feature_schema.json"

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
LABEL_NAMES = ["H", "D", "A"]

# ELO defaults
ELO_INITIAL = 1500.0
ELO_K = 32.0
ELO_HOME_ADVANTAGE = 100.0  # added to home team's rating before win-prob calc

# Feature engineering
FORM_WINDOWS = [3, 5, 10]  # last-N matches for rolling form

# Temporal split
TRAIN_CUTOFF_YEAR = 2014  # inclusive: train <= 2014, test > 2014

# Confederation mapping (FIFA member associations -> confederation code)
# Used as a categorical feature. Integer-encoded for modelling.
CONFEDERATION_MAP: dict[str, str] = {
    # UEFA (Europe)
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA", "Croatia": "UEFA",
    "Cyprus": "UEFA", "Czech Republic": "UEFA", "Czechoslovakia": "UEFA",
    "Denmark": "UEFA", "England": "UEFA", "Estonia": "UEFA",
    "Faroe Islands": "UEFA", "Finland": "UEFA", "France": "UEFA",
    "Georgia": "UEFA", "Germany": "UEFA", "Germany DR": "UEFA",
    "Gibraltar": "UEFA", "Greece": "UEFA", "Hungary": "UEFA",
    "Iceland": "UEFA", "Israel": "UEFA", "Italy": "UEFA",
    "Kazakhstan": "UEFA", "Kosovo": "UEFA", "Latvia": "UEFA",
    "Liechtenstein": "UEFA", "Lithuania": "UEFA", "Luxembourg": "UEFA",
    "Malta": "UEFA", "Moldova": "UEFA", "Montenegro": "UEFA",
    "Netherlands": "UEFA", "North Macedonia": "UEFA", "Northern Ireland": "UEFA",
    "Norway": "UEFA", "Poland": "UEFA", "Portugal": "UEFA",
    "Republic of Ireland": "UEFA", "Romania": "UEFA", "Russia": "UEFA",
    "San Marino": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Serbia and Montenegro": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Soviet Union": "UEFA", "Spain": "UEFA", "Sweden": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA",
    "Wales": "UEFA", "Yugoslavia": "UEFA",
    # CONMEBOL (South America)
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF (North/Central America & Caribbean)
    "Antigua and Barbuda": "CONCACAF", "Bahamas": "CONCACAF",
    "Barbados": "CONCACAF", "Belize": "CONCACAF", "Bermuda": "CONCACAF",
    "Canada": "CONCACAF", "Cayman Islands": "CONCACAF",
    "Costa Rica": "CONCACAF", "Cuba": "CONCACAF", "Curacao": "CONCACAF",
    "Dominica": "CONCACAF", "Dominican Republic": "CONCACAF",
    "El Salvador": "CONCACAF", "Grenada": "CONCACAF",
    "Guatemala": "CONCACAF", "Guyana": "CONCACAF", "Haiti": "CONCACAF",
    "Honduras": "CONCACAF", "Jamaica": "CONCACAF", "Mexico": "CONCACAF",
    "Montserrat": "CONCACAF", "Nicaragua": "CONCACAF", "Panama": "CONCACAF",
    "Puerto Rico": "CONCACAF", "Saint Kitts and Nevis": "CONCACAF",
    "Saint Lucia": "CONCACAF", "Saint Vincent and the Grenadines": "CONCACAF",
    "Suriname": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF",
    "United States": "CONCACAF", "US Virgin Islands": "CONCACAF",
    # CAF (Africa)
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF",
    "Botswana": "CAF", "Burkina Faso": "CAF", "Burundi": "CAF",
    "Cameroon": "CAF", "Cape Verde": "CAF",
    "Central African Republic": "CAF", "Chad": "CAF", "Comoros": "CAF",
    "Congo": "CAF", "Congo DR": "CAF", "Djibouti": "CAF",
    "Egypt": "CAF", "Equatorial Guinea": "CAF", "Eritrea": "CAF",
    "Eswatini": "CAF", "Ethiopia": "CAF", "Gabon": "CAF",
    "Gambia": "CAF", "Ghana": "CAF", "Guinea": "CAF",
    "Guinea-Bissau": "CAF", "Ivory Coast": "CAF", "Kenya": "CAF",
    "Lesotho": "CAF", "Liberia": "CAF", "Libya": "CAF",
    "Madagascar": "CAF", "Malawi": "CAF", "Mali": "CAF",
    "Mauritania": "CAF", "Mauritius": "CAF", "Morocco": "CAF",
    "Mozambique": "CAF", "Namibia": "CAF", "Niger": "CAF",
    "Nigeria": "CAF", "Rwanda": "CAF",
    "Sao Tome and Principe": "CAF", "Senegal": "CAF",
    "Seychelles": "CAF", "Sierra Leone": "CAF", "Somalia": "CAF",
    "South Africa": "CAF", "South Sudan": "CAF", "Sudan": "CAF",
    "Tanzania": "CAF", "Togo": "CAF", "Tunisia": "CAF",
    "Uganda": "CAF", "Zambia": "CAF", "Zimbabwe": "CAF",
    "Zaire": "CAF",
    # AFC (Asia)
    "Afghanistan": "AFC", "Australia": "AFC", "Bahrain": "AFC",
    "Bangladesh": "AFC", "Bhutan": "AFC", "Brunei": "AFC",
    "Cambodia": "AFC", "China PR": "AFC", "Chinese Taipei": "AFC",
    "Guam": "AFC", "Hong Kong": "AFC", "India": "AFC",
    "Indonesia": "AFC", "Iran": "AFC", "Iraq": "AFC",
    "Japan": "AFC", "Jordan": "AFC", "Kuwait": "AFC",
    "Kyrgyzstan": "AFC", "Laos": "AFC", "Lebanon": "AFC",
    "Macau": "AFC", "Malaysia": "AFC", "Maldives": "AFC",
    "Mongolia": "AFC", "Myanmar": "AFC", "Nepal": "AFC",
    "North Korea": "AFC", "Oman": "AFC", "Pakistan": "AFC",
    "Palestine": "AFC", "Philippines": "AFC", "Qatar": "AFC",
    "Saudi Arabia": "AFC", "Singapore": "AFC", "South Korea": "AFC",
    "Sri Lanka": "AFC", "Syria": "AFC", "Tajikistan": "AFC",
    "Thailand": "AFC", "Timor-Leste": "AFC", "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC", "Uzbekistan": "AFC",
    "Vietnam": "AFC", "Yemen": "AFC",
    "Dutch East Indies": "AFC",
    # OFC (Oceania)
    "American Samoa": "OFC", "Cook Islands": "OFC", "Fiji": "OFC",
    "New Caledonia": "OFC", "New Zealand": "OFC",
    "Papua New Guinea": "OFC", "Samoa": "OFC",
    "Solomon Islands": "OFC", "Tahiti": "OFC", "Tonga": "OFC",
    "Vanuatu": "OFC",
}

# Integer encoding for confederations (for use as numeric feature)
CONFEDERATION_CODES: dict[str, int] = {
    "UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "CAF": 3, "AFC": 4, "OFC": 5,
    "UNKNOWN": 6,
}


def get_confederation_code(team: str) -> int:
    """Return integer confederation code for a team, defaulting to UNKNOWN."""
    conf = CONFEDERATION_MAP.get(team, "UNKNOWN")
    return CONFEDERATION_CODES.get(conf, CONFEDERATION_CODES["UNKNOWN"])


# ── Helpers ──────────────────────────────────────────────────────────────────
def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across stdlib, numpy, sklearn."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
