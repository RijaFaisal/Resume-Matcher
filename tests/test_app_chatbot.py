import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.app import app

client = TestClient(app)

@pytest.fixture
def mock_dependencies():
    with patch("src.api.app.client") as mock_groq, \
         patch.dict("src.api.app.state", clear=True) as mock_state:
        
        # Mock Groq Response
        mock_message = MagicMock()
        mock_message.content = "Here is a resume advice."
        mock_groq.chat.completions.create.return_value.choices = [
            MagicMock(message=mock_message)
        ]
        
        # Mock State
        mock_state["policy_engine"] = None # Disable guardrails for logic simplification
        mock_state["sbert_model"] = MagicMock()
        import numpy as np
        # Ensure encode returns a numpy array for astype
        mock_state["sbert_model"].encode.return_value = np.array([0.1, 0.2])
        
        mock_state["faiss_index"] = MagicMock()
        # Ensure search returns (scores, ids) tuple
        # ids must be valid indices for documents list (0)
        mock_state["faiss_index"].search.return_value = (np.array([[0.9]]), np.array([[0]]))
        
        mock_state["documents"] = ["Doc 1"]
        
        yield mock_groq, mock_state

def test_ask_endpoint_success(mock_dependencies):
    mock_groq, _ = mock_dependencies
    
    response = client.post(
        "/ask",
        json={"question": "How to improve?", "user_context": "Junior Dev"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Here is a resume advice."
    assert "Doc 1" in data["context_used"][0]
    
    # Verify PDF generation logic hint
    mock_message_json = MagicMock()
    mock_message_json.content = '```json\n{"action": "generate_resume", "data": {"name": "Test"}}\n```'
    mock_groq.chat.completions.create.return_value.choices = [
        MagicMock(message=mock_message_json)
    ]
    
    response_pdf = client.post(
        "/ask",
        json={"question": "Make me a resume"}
    )
    assert response_pdf.status_code == 200
    assert response_pdf.json()["generated_pdf"] is not None

def test_ask_endpoint_no_client():
    # Simulate missing API key -> client is None
    with patch("src.api.app.client", None):
        response = client.post(
            "/ask",
            json={"question": "Hello"}
        )
        assert response.status_code == 200
        assert "unavailable" in response.json()["answer"]
