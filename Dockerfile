#####################################################################
# Dockerfile for the MLOps Resume Matching API                      #
# Uses a multi-stage build for a smaller and more secure final image. #
#####################################################################

# ===== Stage 1: Builder =====
# This stage installs dependencies, including any that need compilation,
# into a virtual environment. The final image will only copy this venv,
# not the build tools like gcc.
FROM python:3.11-slim AS builder

# Set the virtual environment path
ENV VENV_PATH=/opt/venv

WORKDIR /app

# Install system dependencies required for building Python packages (e.g., numpy, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into the virtual environment
COPY requirements.txt ./requirements.txt
RUN python -m venv ${VENV_PATH} \
    && ${VENV_PATH}/bin/pip install --upgrade pip \
    && ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt

# ===== Stage 2: Runtime =====
# This is the final, lightweight image that will be deployed.
FROM python:3.11-slim AS runtime

# Set environment variables for the virtual environment and Python
ENV VENV_PATH=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the pre-built virtual environment from the 'builder' stage
COPY --from=builder ${VENV_PATH} ${VENV_PATH}

# Create a non-root user for security
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && chown -R app:app /app

# Copy the application source code
# This should contain your FastAPI app, matching logic, etc.
COPY src/ ./src/

# IMPORTANT: Do not bake secrets like .env into the image.
# These should be passed as environment variables at runtime, as configured in docker-compose.yml.
# COPY .env .env  <-- This line is intentionally disabled.

# Switch to the non-root user
USER app

# Expose the port the API will run on
EXPOSE 8000

# Healthcheck to ensure the API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# The command to run the Resume Matching API using Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]