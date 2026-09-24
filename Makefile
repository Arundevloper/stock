PY ?= .venv/bin/python
DEMO_DB = sqlite:///data/demo.db
# demo model/report live next to, never over, the real ones
DEMO_ENV = DATABASE_URL=$(DEMO_DB) MODEL_PATH=models/demo_model.pkl REPORT_PATH=models/demo_report.json

.PHONY: install test run reader login backfill train demo-data demo-train demo demo-replay

install:            ## create venv + install deps
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
	@test -f .env || cp .env.example .env

test:               ## run the test suite (leak tests included)
	$(PY) -m pytest -q tests

run:                ## backend + web UI on http://127.0.0.1:8000
	$(PY) -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

reader:             ## live Kite websocket reader
	$(PY) -m reader.kite_reader

login:              ## daily Kite login from the terminal
	$(PY) -m auth.login

backfill:           ## 180 days of minute history for universe + NIFTY
	$(PY) -m backfill.historical --days 180

train:              ## walk-forward train + threshold sweep + backtest
	$(PY) -m ml.train

# ---- demo without a Kite subscription (synthetic data in data/demo.db) ----
demo-data:
	$(PY) -m scripts.generate_sample_data --db $(DEMO_DB)

demo-train:
	$(DEMO_ENV) $(PY) -m ml.train

demo:               ## backend on the demo DB (scheduler off)
	$(DEMO_ENV) SCHEDULER_ENABLED=false $(PY) -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

demo-replay:        ## replay the last demo session into the running demo backend: make demo-replay DATE=2026-09-23
	$(DEMO_ENV) $(PY) -m reader.replay --date $(DATE) --speed 5 --reset
