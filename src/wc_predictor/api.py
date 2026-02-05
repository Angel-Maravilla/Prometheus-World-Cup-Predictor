"""FastAPI backend for the World Cup predictor.

Endpoints
---------
GET  /api/health           - health check
GET  /api/teams            - list of all teams in the dataset
GET  /api/models           - list of trained models + their metrics
POST /api/predict          - predict a single match (with explainability)
GET  /api/comparison       - model comparison data with metadata
GET  /api/version          - build/dataset version info

Run with:
    uvicorn wc_predictor.api:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from wc_predictor.config import (
    FEATURE_MATRIX_CSV,
    FEATURE_SCHEMA_JSON,
    MODELS_DIR,
    RAW_CSV,
    REPORTS_DIR,
    TRAIN_CUTOFF_YEAR,
    get_logger,
    seed_everything,
    LABEL_NAMES,
)
from wc_predictor.predict import compute_features_for_match

log = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="World Cup Predictor API",
    description="Predict FIFA World Cup match outcomes using ML models.",
    version="1.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: configurable via CORS_ORIGINS env var (comma-separated), falls back to
# localhost for local development.  In production on Render the React app is
# served from the *same* origin so CORS is a no-op, but we keep it open for
# external API consumers.
_cors_env = os.environ.get("CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions: log server-side, return generic 500 to client."""
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# -- Cached state (loaded once at startup) ------------------------------------

_teams: list[str] = []
_all_matches: pd.DataFrame | None = None
_models: dict[str, object] = {}
_schema: dict | None = None
_dataset_hash: str = ""
_test_match_count: int = 0


def _load_data() -> None:
    """Load teams, models, and schema into memory."""
    global _teams, _all_matches, _models, _schema, _dataset_hash, _test_match_count

    seed_everything()

    # Load all international matches
    if RAW_CSV.exists():
        _all_matches = pd.read_csv(RAW_CSV, parse_dates=["date"])
        _all_matches["neutral"] = _all_matches["neutral"].fillna(False).astype(bool)
        _all_matches = _all_matches.sort_values("date").reset_index(drop=True)
        all_teams = set(_all_matches["home_team"].unique()) | set(_all_matches["away_team"].unique())
        _teams = sorted(all_teams)
        _dataset_hash = hashlib.md5(RAW_CSV.read_bytes()).hexdigest()[:8]
        log.info("Loaded %d teams from %d matches.", len(_teams), len(_all_matches))
    else:
        log.warning("Raw data not found at %s. Run download_data first.", RAW_CSV)

    # Compute test-set size for display
    if FEATURE_MATRIX_CSV.exists():
        fm = pd.read_csv(FEATURE_MATRIX_CSV, parse_dates=["date"])
        _test_match_count = int((fm["date"].dt.year > TRAIN_CUTOFF_YEAR).sum())

    # Load feature schema
    if FEATURE_SCHEMA_JSON.exists():
        _schema = json.loads(FEATURE_SCHEMA_JSON.read_text())
    else:
        log.warning("Feature schema not found. Run features first.")

    # Load all trained models
    for model_path in sorted(MODELS_DIR.glob("*.joblib")):
        name = model_path.stem
        _models[name] = joblib.load(model_path)
        log.info("Loaded model: %s", name)


@app.on_event("startup")
async def startup() -> None:
    _load_data()


# -- Pydantic schemas --------------------------------------------------------


class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Home team name", examples=["Brazil"])
    away_team: str = Field(..., description="Away team name", examples=["Germany"])
    match_date: date = Field(..., description="Match date (YYYY-MM-DD)", examples=["2026-06-15"])
    model: str | None = Field(None, description="Model name (default: best available)")
    neutral: bool = Field(True, description="Neutral venue (default: true for World Cup)")


class ConfidenceInfo(BaseModel):
    max_prob: float
    entropy: float
    label: str  # LOW / MED / HIGH


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    match_date: str
    model: str
    probabilities: dict[str, float]
    prediction: str
    prediction_label: str
    confidence: ConfidenceInfo
    explanation: dict


class ModelInfo(BaseModel):
    name: str
    metrics: dict[str, float]


# -- Helpers ------------------------------------------------------------------


def _compute_confidence(proba: np.ndarray) -> ConfidenceInfo:
    """Compute confidence metadata from a probability vector."""
    max_prob = float(np.max(proba))
    eps = 1e-15
    entropy = float(-np.sum(proba * np.log(np.clip(proba, eps, 1.0))))
    if max_prob >= 0.55:
        label = "HIGH"
    elif max_prob >= 0.42:
        label = "MED"
    else:
        label = "LOW"
    return ConfidenceInfo(
        max_prob=round(max_prob, 4),
        entropy=round(entropy, 4),
        label=label,
    )


# -- Endpoints ----------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "teams_loaded": len(_teams),
        "models_loaded": list(_models.keys()),
    }


@app.get("/api/teams", response_model=list[str])
async def get_teams() -> list[str]:
    if not _teams:
        raise HTTPException(503, "Team data not loaded. Run the pipeline first.")
    return _teams


@app.get("/api/models", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    results = []
    for name in sorted(_models.keys()):
        report_path = REPORTS_DIR / f"{name}_report.json"
        metrics = {}
        if report_path.exists():
            report = json.loads(report_path.read_text())
            metrics = report.get("metrics", {})
        results.append(ModelInfo(name=name, metrics=metrics))
    return results


@app.get("/api/comparison")
async def get_comparison() -> dict:
    comparison_path = REPORTS_DIR / "comparison.json"
    if not comparison_path.exists():
        raise HTTPException(404, "No comparison report found. Run `make evaluate` first.")
    data = json.loads(comparison_path.read_text())
    return {
        "models": data,
        "metadata": {
            "cutoff_year": TRAIN_CUTOFF_YEAR,
            "test_matches": _test_match_count,
            "note": (
                "Baseline log loss is intentionally high: DummyClassifier outputs "
                "100%% for one class, producing near-infinite log-loss on misclassified "
                "samples. This is the expected worst-case reference point."
            ),
        },
    }


@app.get("/api/version")
async def get_version() -> dict:
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass

    return {
        "api_version": "1.1.0",
        "dataset_hash": _dataset_hash,
        "models_available": list(_models.keys()),
        "cutoff_year": TRAIN_CUTOFF_YEAR,
        "git_commit": git_commit,
    }


@app.post("/api/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict(req: PredictionRequest, request: Request) -> PredictionResponse:
    if _all_matches is None or _schema is None:
        raise HTTPException(503, "Data not loaded. Run the pipeline first.")

    if req.home_team not in _teams:
        raise HTTPException(400, f"Unknown team: '{req.home_team}'. Use /api/teams for valid names.")
    if req.away_team not in _teams:
        raise HTTPException(400, f"Unknown team: '{req.away_team}'. Use /api/teams for valid names.")
    if req.home_team == req.away_team:
        raise HTTPException(400, "Home and away teams must be different.")

    model_preference = ["rf", "xgb", "logreg", "baseline"]
    if req.model:
        if req.model not in _models:
            raise HTTPException(400, f"Unknown model: '{req.model}'. Available: {list(_models.keys())}")
        model_name = req.model
    else:
        model_name = next((m for m in model_preference if m in _models), None)
        if model_name is None:
            raise HTTPException(503, "No trained models available.")

    pipeline = _models[model_name]
    feature_cols = _schema["feature_columns"]

    dt = datetime(req.match_date.year, req.match_date.month, req.match_date.day)
    X, explanation = compute_features_for_match(
        req.home_team, req.away_team, dt, _all_matches, feature_cols,
        neutral=req.neutral, return_explanation=True,
    )

    proba = pipeline.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = LABEL_NAMES[pred_idx]

    label_descriptions = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    confidence = _compute_confidence(proba)

    return PredictionResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        match_date=str(req.match_date),
        model=model_name,
        probabilities={
            "H": round(float(proba[0]), 4),
            "D": round(float(proba[1]), 4),
            "A": round(float(proba[2]), 4),
        },
        prediction=pred_label,
        prediction_label=label_descriptions[pred_label],
        confidence=confidence,
        explanation=explanation,
    )


# -- Static file serving (production) ----------------------------------------
# In production the built React app lives in frontend/dist/.  We serve it from
# FastAPI so the entire app runs on a single origin (no CORS issues, one
# Render service).  The guard ``if _frontend_dist.is_dir()`` keeps this inert
# during local development where Vite serves the frontend on port 5173.

from wc_predictor.config import PROJECT_ROOT  # noqa: E402

_frontend_dist = PROJECT_ROOT / "frontend" / "dist"

if _frontend_dist.is_dir():
    # Hashed JS/CSS bundles produced by ``vite build``
    app.mount(
        "/assets",
        StaticFiles(directory=_frontend_dist / "assets"),
        name="static-assets",
    )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> FileResponse:
        """Catch-all: serve static files or fall back to index.html for SPA routing."""
        file_path = _frontend_dist / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_frontend_dist / "index.html")
