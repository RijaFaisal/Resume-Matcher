# Dockerfile (frontend - dev version)
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements-dev.txt and install (lighter)
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

# ------------------------
# Runtime image
# ------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the same lighter deps in runtime
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY --from=builder /app /app

RUN useradd -m app
USER app

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
