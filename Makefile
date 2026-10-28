.PHONY: install bootstrap pipeline test lint format serve drift simulate-traffic retrain docker-build docker-up docker-down clean

PYTHON := python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

bootstrap:
	$(PYTHON) scripts/bootstrap.py

pipeline:
	$(PYTHON) scripts/run_pipeline.py --config configs/config.yaml

test:
	pytest tests -v --cov=src/mlops_platform

lint:
	ruff check .

format:
	ruff format .

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

drift:
	$(PYTHON) -c "import pandas as pd; from mlops_platform.monitoring.drift import detect_data_drift; from scripts.generate_sample_data import generate_drifted_data; ref = pd.read_parquet('data/reference/reference_features.parquet'); cur = generate_drifted_data(500); detect_data_drift(ref[['tenure', 'monthly_charges', 'total_charges']], cur[['tenure', 'monthly_charges', 'total_charges']])"

simulate-traffic:
	$(PYTHON) scripts/simulate_traffic.py --url http://localhost:8000 --requests 50

retrain:
	$(PYTHON) -c "from mlops_platform.monitoring.retraining import trigger_retraining; print(trigger_retraining())"

docker-build:
	docker build -t mlops-platform-api:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage dist build *.egg-info
