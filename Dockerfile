# ── Stage 1: Build React frontend ──────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# XGBoost requires libgomp (OpenMP runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[all]"

# Copy data and model artifacts (committed to git)
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Tell the installed package where the project root is
ENV APP_ROOT=/app

# Render injects PORT env var (default 10000)
EXPOSE 10000

CMD ["sh", "-c", "uvicorn wc_predictor.api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 2 --log-level info"]
