
FROM python:3.11-slim AS builder

# Install system dependencies (you may need poppler-utils for PDF or docx support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# ------------------------
# Stage 2: Runtime
# ------------------------
FROM python:3.11-slim

# Create a non-root user
RUN useradd -m app
USER app
WORKDIR /app

# Copy everything from builder
COPY --from=builder /app /app

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Healthcheck to ensure Streamlit server is up
#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to start your app
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
