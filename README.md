# World Cup Match Outcome Predictor

A production-quality ML pipeline that predicts FIFA World Cup match outcomes (Home Win / Draw / Away Win) using historical international football data, sequential ELO ratings, and rolling team form features.

## Problem Statement

Given two national teams and a match date, predict the probability of each outcome: home win, draw, or away win. The pipeline emphasises correct ML practice: no data leakage, time-aware evaluation splits, proper baselines, and honest calibration assessment.

## Data

**Source:** [International football results from 1872 to present](https://github.com/martj42/international_results) by Mart Jurisoo (public domain / open data).

| Column | Description |
|---|---|
| `date` | Match date |
| `home_team` / `away_team` | Team names |
| `home_score` / `away_score` | Final score |
| `tournament` | Competition name |
| `neutral` | Whether the venue is neutral |

By default the pipeline includes both World Cup finals and qualification matches (`--include-qualifiers`), giving ~9,700 target rows. ELO is computed over **all** ~49,000 international matches so that ratings reflect true team strength entering a tournament.

## Approach

### Feature Engineering (No Leakage)

Every feature for match *i* is computed using **only** matches with `date < date_i`:

1. **ELO ratings** — Sequential ELO computed over all international matches (K=32, home advantage=100). Features: `home_elo`, `away_elo`, `elo_diff`, plus ELO-derived win probabilities.

2. **Rolling form** (last 3, 5, 10 matches per team) — Win rate, draw rate, goals for/against averages, goal difference.

3. **Head-to-head** — Home team's win rate in the last 5 meetings.

4. **Neutral venue** — Binary indicator (most World Cup matches are at neutral venues).

5. **Confederation** — Integer-encoded FIFA confederation for each team (UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC) plus a same-confederation binary flag.

### Models

| Model | Description |
|---|---|
| **Baseline** | Always predicts the most frequent class |
| **Logistic Regression** | Multinomial, L2-regularised, with standard scaling |
| **Random Forest** | 300 trees, max depth 8 |
| **XGBoost** | Optional (graceful fallback if not installed) |

All models use a sklearn `Pipeline` with median imputation for missing features.

### Evaluation

- **Temporal split**: train on matches up to 2014, test on 2015+ (configurable via `--cutoff`)
- **Metrics**: accuracy, macro F1, log loss, multi-class Brier score
- **Baselines**: most-frequent-class and ELO-only probability baseline
- **Calibration**: reliability diagrams saved under `artifacts/figures/`

## Results

Results with qualifiers included and confederation features (train up to 2014, test on 2015+):

| Model | Accuracy | Macro F1 | Log Loss | Brier Score |
|---|---|---|---|---|
| Baseline (most frequent) | 0.477 | 0.215 | 18.08 | 1.047 |
| Logistic Regression | 0.639 | 0.503 | 0.808 | 0.469 |
| **Random Forest** | **0.644** | 0.481 | **0.805** | **0.468** |
| XGBoost | 0.626 | **0.519** | 0.881 | 0.500 |

Train: 6,989 matches (up to 2014). Test: 2,730 matches (2015+).

The cutoff is configurable: `python -m wc_predictor.train --model rf --cutoff 2010` trains on pre-2010 data and tests on 3,598 matches (accuracy: 0.640, similar generalization).

### Figures

After running the pipeline, comparison and calibration plots are saved to:
- `artifacts/figures/model_comparison.png`
- `artifacts/figures/calibration.png`

## How to Run

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Install the package and dependencies
make setup

# Or manually:
pip install -e ".[dev]"
```

### Full Pipeline

```bash
# 1. Download and cache the dataset
python -m wc_predictor.download_data

# 2. Build features (downloads data + processes + engineers features)
python -m wc_predictor.features --include-qualifiers
# (omit --include-qualifiers to use only World Cup finals matches)

# 3. Train all models
make train
# Or train individually (with optional cutoff year):
python -m wc_predictor.train --model baseline
python -m wc_predictor.train --model logreg
python -m wc_predictor.train --model rf
python -m wc_predictor.train --model xgb
python -m wc_predictor.train --model rf --cutoff 2010  # experiment with different splits

# 4. Evaluate and generate comparison plots
python -m wc_predictor.evaluate
# or: make evaluate

# 5. Predict a single match
python -m wc_predictor.predict --team_a Brazil --team_b Germany --date 2026-06-15
```

### Run Tests

```bash
make test
# or: python -m pytest tests/ -v
```

## Project Structure

```
.
├── pyproject.toml          # Package metadata and dependencies
├── Makefile                # Convenience targets
├── README.md
├── LICENSE                 # MIT
├── src/
│   └── wc_predictor/
│       ├── __init__.py
│       ├── __main__.py     # CLI dispatcher
│       ├── config.py       # Paths, constants, seeding
│       ├── download_data.py# Dataset downloader + validator
│       ├── build_dataset.py# Filter to WC matches, add labels
│       ├── elo.py          # Sequential ELO rating system
│       ├── features.py     # Leak-free feature engineering
│       ├── splits.py       # Temporal splits + metrics
│       ├── train.py        # Model training pipeline
│       ├── evaluate.py     # Multi-model comparison + plots
│       └── predict.py      # Single-match prediction CLI
├── tests/
│   ├── test_elo.py         # ELO correctness (20+ tests)
│   ├── test_features.py    # Leakage prevention + form logic
│   └── test_splits.py      # Splitting + metric computation
├── data/                   # Downloaded data (gitignored)
└── artifacts/
    ├── models/             # Saved .joblib models
    ├── reports/            # JSON metric reports
    └── figures/            # Comparison + calibration plots
```

## Security

- **No API keys required** -- the dataset is public domain; no secrets are used anywhere in the pipeline.
- **No user data collected** -- the API is stateless with no cookies, sessions, or personal data stored.
- **Rate limiting** -- `/api/predict` is limited to 30 requests/minute per IP to prevent abuse.
- **Same-origin deployment** -- the React frontend is served from the same FastAPI origin, eliminating CORS surface area.
- **Secret scanning** -- run `make scan-secrets` before committing to check for accidentally included credentials.

## Limitations and Ethical Notes

### Model Limitations

- **Moderate dataset size**: With qualifiers, the dataset has ~9,700 matches. Without qualifiers, only ~960 World Cup finals matches are available, which limits model capacity.
- **No player-level data**: Team strength is approximated via aggregate ELO and recent form. Injuries, suspensions, and squad composition are not captured.
- **Stationarity assumption**: The model assumes that patterns in historical World Cups transfer to future ones. Football evolves tactically and structurally over decades.
- **Home advantage in World Cup**: Most WC matches are at neutral venues, but the host nation does receive a genuine boost not fully captured by the binary neutral flag.
- **Class imbalance**: Draws are the least frequent outcome and hardest to predict. The model may underpredict draws.

### Ethical Notes

- This model is for **educational and analytical purposes only**. It should not be used for gambling or financial decisions.
- Prediction accuracy for sporting events is inherently limited. No model can reliably beat the market.
- Historical data reflects the geopolitics of FIFA membership: some regions are underrepresented in early data.

## What I Would Do Next

1. **Player-level features**: Aggregate squad market values, average age, key player availability from transfermarkt or similar open sources.
2. **Bookmaker odds as features**: Odds encode vast amounts of information (with the caveat that this is circular if the goal is to beat the market).
3. **Tournament structure features**: Round (group/R16/QF/SF/F), group-stage standings, goal difference entering knockout rounds.
4. **Hierarchical / mixed-effects models**: Account for team-level and confederation-level variance simultaneously.
5. **Bayesian calibration**: Replace the ad-hoc draw band in ELO win probabilities with a properly calibrated ordinal regression.
6. **Hyperparameter tuning**: Proper Bayesian optimisation over K-factor, form windows, model hyperparameters using time-series CV.
7. **Conformal prediction**: Produce prediction sets with guaranteed coverage instead of point probabilities.
8. **Venue distance features**: Compute travel distance / time-zone shift for each team to the match venue as a fatigue proxy.
