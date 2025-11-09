# tests/conftest.py
import pytest
import torch
import pandas as pd
from src.api.main import state, app
from prometheus_client import Counter, Histogram, Gauge, REGISTRY


@pytest.fixture(scope="session", autouse=True)
def setup_app_state():
    """Initialize dummy model, embeddings, DataFrame, and metrics for tests."""

    class DummyModel:
        def encode(self, texts, convert_to_tensor=True):
            # Return random embeddings
            return torch.rand(len(texts), 384)

    # Setup dummy state
    state["sbert_model"] = DummyModel()
    state["job_embeddings"] = torch.rand(5, 384)
    state["df_job_description"] = pd.DataFrame(
        {
            "Job Title": [
                "Data Scientist",
                "ML Engineer",
                "Data Analyst",
                "AI Specialist",
                "Python Developer",
            ],
            "Job Description": ["Job description"] * 5,
        }
    )


def get_or_create_metric(name, metric_type, *args, **kwargs):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_type(name, *args, **kwargs)


# Setup Prometheus metrics safely
app.state.METRICS = {
    "requests_total": get_or_create_metric(
        "matching_requests_total",
        Counter,
        "Total matching requests",
        ["model_version", "status"],
    ),
    "duration_seconds": get_or_create_metric(
        "matching_duration_seconds",
        Histogram,
        "Time spent processing a matching request",
        ["model_version"],
    ),
    "load_time_seconds": get_or_create_metric(
        "model_load_seconds", Gauge, "Time taken to load models"
    ),
    "errors_total": get_or_create_metric(
        "api_errors_total", Counter, "Total API errors", ["error_type"]
    ),
    "similarity_score": get_or_create_metric(
        "match_similarity_score",
        Histogram,
        "Distribution of similarity scores",
        ["model_version"],
    ),
    "http_requests_duration": get_or_create_metric(
        "http_request_duration_seconds",
        Histogram,
        "HTTP request latency",
        ["method", "endpoint", "status_code"],
    ),
}


@pytest.fixture
def client():
    """TestClient using FastAPI app with lifespan."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
