.PHONY: setup test lint train evaluate download features predict clean api frontend dev scan-secrets

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[all,dev]"
	cd frontend && npm install

download:
	$(PYTHON) -m wc_predictor.download_data

features:
	$(PYTHON) -m wc_predictor.features --include-qualifiers

train: train-baseline train-logreg train-rf train-xgb

train-baseline:
	$(PYTHON) -m wc_predictor.train --model baseline

train-logreg:
	$(PYTHON) -m wc_predictor.train --model logreg

train-rf:
	$(PYTHON) -m wc_predictor.train --model rf

train-xgb:
	$(PYTHON) -m wc_predictor.train --model xgb

evaluate:
	$(PYTHON) -m wc_predictor.evaluate

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

# ── Web app ──────────────────────────────────────────────
api:
	$(PYTHON) -m uvicorn wc_predictor.api:app --reload --port 8000

frontend:
	cd frontend && npm run dev

dev:
	@echo "Run in two terminals:"
	@echo "  Terminal 1: make api"
	@echo "  Terminal 2: make frontend"
	@echo ""
	@echo "Then open http://localhost:5173"

clean:
	rm -rf artifacts/models/* artifacts/reports/* artifacts/figures/*
	rm -rf data/processed/*
	rm -rf frontend/dist

scan-secrets:
	@echo "Scanning for potential secrets..."
	@git grep -inE '(password|secret|api_key|token|private_key)\s*=' -- ':(exclude)Makefile' || echo "No secrets found."
	@echo "Done."
