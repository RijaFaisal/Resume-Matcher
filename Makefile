.PHONY: help dev test lint docker clean install

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Start development environment
	pip install -r requirements.txt
	python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linting
	ruff check src/ tests/
	black --check src/ tests/

format: ## Format code
	black src/ tests/
	ruff --fix src/ tests/

docker: ## Build Docker image
	docker build -t resume-matcher:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 resume-matcher:latest

docker-dev: ## Run Docker in development mode
	docker-compose up --build

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

security: ## Run security audit
	pip-audit --desc

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

mlflow: ## Start MLflow server
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000

evidently: ## Start Evidently dashboard
	evidently ui --host 0.0.0.0 --port 7000
