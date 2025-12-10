import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_state():
    with patch.dict("src.api.main.state", clear=True) as mock_st:
        mock_st["sbert_model"] = MagicMock()
        mock_st["job_embeddings"] = MagicMock()
        mock_st["df_job_description"] = MagicMock()
        mock_st["model_info"] = {"version": "v1"}
        mock_st["policy_engine"] = None # Optional
        yield mock_st

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Resume Matching API" in response.text

def test_health_check_healthy(mock_state):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_info(mock_state):
    mock_state["df_job_description"] = MagicMock()
    mock_state["df_job_description"].__len__.return_value = 10
    
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["loaded"] is True
    assert response.json()["total_jobs_indexed"] == 10

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200

def test_match_resume_success(mock_state):
    # Mock Tensor/List behavior
    import torch
    
    mock_state["sbert_model"].encode.return_value = torch.tensor([0.1, 0.2])
    # Mock job embeddings as tensor
    mock_state["job_embeddings"] = torch.tensor([[0.1, 0.2]])
    
    # Mock dataframe
    mock_df = MagicMock()
    mock_df.iloc.__getitem__.return_value = {"Job Title": "Dev"}
    mock_df.__len__.return_value = 1
    mock_state["df_job_description"] = mock_df
    
    response = client.post(
        "/match_resume",
        json={"resume_text": "Experienced Dev " * 5, "top_n": 1}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["matches"]) > 0
    assert data["matches"][0]["job_title"] == "Dev"
