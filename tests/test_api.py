import pytest


@pytest.mark.parametrize("endpoint", ["/", "/health"])
def test_get_endpoints(client, endpoint):
    resp = client.get(endpoint)
    assert resp.status_code == 200


def test_match_resume_endpoint(client):
    payload = {
        "resume_text": "Experienced data scientist with Python, ML, and SQL expertise. Looking for new challenges.",
        "top_n": 2,
    }
    resp = client.post("/match_resume", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data
    assert len(data["matches"]) == 2
    assert "model_info" in data


def test_model_info_endpoint(client):
    resp = client.get("/model/info")
    assert resp.status_code in (200, 503)


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"matching_requests_total" in resp.content
