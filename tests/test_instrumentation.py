from src.api.main import app

SAMPLE_RESUME = "Experienced data scientist with Python, ML, and SQL expertise. Looking for new challenges."

<<<<<<< HEAD
=======

>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a
def test_health_endpoint(client):
    resp = client.get("/health")
    data = resp.json()
    assert resp.status_code == 200
    assert data["model_loaded"] is True
    assert data["embeddings_loaded"] is True

<<<<<<< HEAD
=======

>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a
def test_match_resume_endpoint_updates_metrics(client):
    resp = client.post("/match_resume", json={"resume_text": SAMPLE_RESUME, "top_n": 3})
    data = resp.json()
    assert resp.status_code == 200
    assert len(data["matches"]) == 3
    metrics = app.state.METRICS
    assert "requests_total" in metrics
    assert "duration_seconds" in metrics
    assert "similarity_score" in metrics

<<<<<<< HEAD
=======

>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a
def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    content = resp.text
    assert "matching_requests_total" in content
