# tests/test_instrumentation.py
from src.api.main import app

SAMPLE_RESUME = "Experienced data scientist with Python, ML, and SQL expertise. Looking for new challenges."


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is True
    assert data["embeddings_loaded"] is True
    assert data["status"] in ["healthy", "degraded"]


def test_match_resume_endpoint_updates_metrics(client):
    resp = client.post("/match_resume", json={"resume_text": SAMPLE_RESUME, "top_n": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data
    assert len(data["matches"]) == 3

    # Check metrics exist
    metrics = app.state.METRICS
    assert "requests_total" in metrics
    assert "duration_seconds" in metrics
    assert "similarity_score" in metrics


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    content = resp.text
    # There should be Prometheus metric names in the response
    assert "matching_requests_total" in content
    assert "matching_duration_seconds" in content
    assert "match_similarity_score" in content
