# Makefile for the MLOps Resume Matching App

# Use .PHONY to ensure these targets run even if files with the same name exist.
.PHONY: help dev install test lint format docker run stop clean evidently audit security logs mlflow api test-guardrails test-monitoring rag monitoring-dashboard

# Default target: show the help message.
help:
	@echo "Available make targets for the Resume Matching App:"
	@echo "----------------------------------------------------"
	@echo "  Development & Setup:"
	@echo "    make dev        - Setup a new development virtual environment"
	@echo "    make install    - Install Python dependencies from requirements.txt"
	@echo "    make clean      - Clean up temporary files (__pycache__, .coverage, etc.)"
	@echo ""
	@echo "  Code Quality & Testing:"
	@echo "    make test       - Run pytest with coverage report"
	@echo "    make lint       - Run ruff and black linters to check for issues"
	@echo "    make format     - Automatically format code with black and ruff"
	@echo "    make test-guardrails - Run guardrails tests"
	@echo "    make test-monitoring - Run monitoring tests"
	@echo ""
	@echo "  Docker & Services:"
	@echo "    make docker     - Build the 'resume-matcher-api' Docker image"
	@echo "    make run        - Start all services (API, MLflow, Prometheus, Grafana) via Docker Compose"
	@echo "    make stop       - Stop all running Docker Compose services"
	@echo "    make logs       - Follow the logs from all running services"
	@echo ""
	@echo "  MLOps & Local Servers:"
	@echo "    make mlflow     - Run the MLflow server locally (without Docker)"
	@echo "    make api        - Run the FastAPI server locally with auto-reload (without Docker)"
	@echo "    make evidently  - Start the Evidently dashboard for data drift monitoring"
	@echo "    make rag        - Run full RAG pipeline end-to-end"
	@echo ""
	@echo "  Monitoring:"
	@echo "    make monitoring-dashboard - Open Grafana monitoring dashboard"
	@echo ""
	@echo "  Security:"
	@echo "    make audit      - Run pip-audit to scan dependencies for vulnerabilities"

# Setup the development environment
dev:
	@echo "🐍 Setting up Python virtual environment..."
	python -m venv venv
	@echo "✅ Virtual environment 'venv' created. Activate it with:"
	@echo "   Windows: .\\venv\\Scripts\\activate"
	@echo "   Linux/macOS: source venv/bin/activate"
	@echo "Then run 'make install' to install dependencies."

# Install dependencies
install:
	@echo "📦 Installing dependencies from requirements.txt..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed."

# Run tests with coverage
test:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term --cov-fail-under=80
	@echo "✅ Tests complete. Coverage report generated in: htmlcov/index.html"

# Run linters to check code quality
lint:
	@echo "🔍 Running linters (ruff, black)..."
	ruff check src/ tests/
	black --check src/ tests/
	@echo "✅ Linting checks passed."

# Automatically format the code
format:
	@echo "🎨 Formatting code with black and ruff..."
	black src/ tests/
	ruff check --fix src/ tests/
	@echo "✅ Code formatting complete."

# Build the Docker image for the API
docker:
	@echo "🐳 Building the 'resume-matcher-api' Docker image..."
	docker build -t resume-matcher-api:latest .
	@echo "✅ Docker image built successfully."

# Start all services defined in docker-compose.yml
run:
	@echo "🚀 Starting all MLOps services via Docker Compose..."
	docker-compose up -d
	@echo "✅ Services are starting in the background."
	@echo ""
	@echo "Access your services at:"
	@echo "  Resume Matcher API Docs: http://localhost:8000/docs"
	@echo "  MLflow UI:               http://localhost:5000"
	@echo "  Prometheus UI:           http://localhost:9090"
	@echo "  Grafana UI:              http://localhost:3000"
	@echo ""

# Stop all services
stop:
	@echo "🛑 Stopping all Docker Compose services..."
	docker-compose down
	@echo "✅ All services stopped."

# Start the Evidently dashboard for monitoring data drift
evidently:
	@echo "📊 Starting Evidently dashboard for monitoring resume/job text drift..."
	@echo "   (Run this after executing the '04_evidently_monitoring.ipynb' notebook)"
	evidently ui --workspace ./monitoring/evidently/workspace --port 7000

# Run a security audit on dependencies
audit security:
	@echo "🛡️ Running dependency vulnerability scan with pip-audit (failing on critical CVEs)..."
	pip install --upgrade pip > /dev/null 2>&1 || true
	pip install --disable-pip-version-check --quiet pip-audit
	pip-audit -r requirements.txt --fail-on critical --progress-spinner off
	@echo "✅ Security audit complete."

# Clean up temporary files and directories
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	@echo "✅ Cleanup complete."

# Follow logs from Docker Compose services
logs:
	@echo "📜 Following logs for all services (Press Ctrl+C to exit)..."
	docker-compose logs -f

# Run MLflow server locally for quick access
mlflow:
	@echo "📊 Starting local MLflow server..."
	mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000

# Run the FastAPI server locally with auto-reload for development
api:
	@echo "🚀 Starting local FastAPI server for the Resume Matcher..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run guardrails tests
test-guardrails:
	@echo "🛡️ Running guardrails tests..."
	pytest tests/test_guardrails.py -v --cov=src.guardrails --cov-report=term --cov-report=html
	@echo "✅ Guardrails tests complete."

# Run monitoring tests
test-monitoring:
	@echo "📊 Running monitoring tests..."
	pytest tests/test_monitoring.py -v --cov=src.monitoring --cov-report=term --cov-report=html
	@echo "✅ Monitoring tests complete."

# Run full RAG pipeline end-to-end
rag:
	@echo "🔄 Running RAG pipeline with guardrails and monitoring..."
	@echo "Step 1: Ingesting documents into vector store..."
	python -m src.api.ingest --data_dir ./data --index_dir ./vectorstore
	@echo "✅ Ingestion complete."
	@echo "Step 2: Starting RAG API server with guardrails and monitoring..."
	uvicorn src.api.app:app --host 0.0.0.0 --port 8001

# Open Grafana monitoring dashboard
monitoring-dashboard:
	@echo "📊 Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "API Metrics: http://localhost:8000/metrics"
	@open "http://localhost:3000" || xdg-open "http://localhost:3000" || echo "Please open http://localhost:3000"