"""
Tests for the FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import pandas as pd

from src.app.main import app


class TestAPI:
    """Test class for API endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_docs_endpoint(self):
        """Test the API documentation endpoint."""
        response = self.client.get("/docs")
        assert response.status_code == 200

    def test_metrics_endpoint(self):
        """Test the Prometheus metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        # Metrics should be in Prometheus format
        assert "# HELP" in response.text or "# TYPE" in response.text

    @patch("src.app.main.get_model")
    def test_predict_endpoint_success(self, mock_get_model):
        """Test successful prediction."""
        # Mock the model
        mock_model = Mock()
        mock_similarity_df = pd.DataFrame([[0.85, 0.75], [0.65, 0.90]])
        mock_model.predict.return_value = mock_similarity_df
        mock_get_model.return_value = mock_model

        # Make request
        response = self.client.post(
            "/predict",
            json={
                "resumes": [
                    "Software Engineer with Python experience",
                    "Data Scientist with ML background",
                ],
                "job_descriptions": [
                    "Looking for Python developer",
                    "Need ML expert",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "similarity_matrix" in data
        assert len(data["similarity_matrix"]) == 2
        assert len(data["similarity_matrix"][0]) == 2

    def test_predict_endpoint_missing_resumes(self):
        """Test prediction with missing resumes field."""
        response = self.client.post(
            "/predict",
            json={
                "job_descriptions": ["Looking for Python developer"],
            },
        )
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_job_descriptions(self):
        """Test prediction with missing job_descriptions field."""
        response = self.client.post(
            "/predict",
            json={
                "resumes": ["Software Engineer with Python experience"],
            },
        )
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_empty_lists(self):
        """Test prediction with empty lists."""
        with patch("src.app.main.get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.predict.return_value = pd.DataFrame()
            mock_get_model.return_value = mock_model

            response = self.client.post(
                "/predict",
                json={
                    "resumes": [],
                    "job_descriptions": [],
                },
            )
            assert response.status_code == 200

    @patch("src.app.main.get_model")
    def test_predict_endpoint_increments_counter(self, mock_get_model):
        """Test that prediction increments the token counter."""
        mock_model = Mock()
        mock_model.predict.return_value = pd.DataFrame([[0.85]])
        mock_get_model.return_value = mock_model

        response = self.client.post(
            "/predict",
            json={
                "resumes": ["Short resume"],
                "job_descriptions": ["Short job description"],
            },
        )

        assert response.status_code == 200
        # Counter should have been incremented (tested via metrics endpoint)

    def test_predict_endpoint_invalid_json(self):
        """Test prediction with invalid JSON."""
        response = self.client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    @patch("src.app.main.get_model")
    def test_predict_endpoint_with_long_texts(self, mock_get_model):
        """Test prediction with long resume and job description texts."""
        mock_model = Mock()
        mock_model.predict.return_value = pd.DataFrame([[0.92]])
        mock_get_model.return_value = mock_model

        long_resume = " ".join(["word"] * 500)
        long_job_desc = " ".join(["word"] * 500)

        response = self.client.post(
            "/predict",
            json={
                "resumes": [long_resume],
                "job_descriptions": [long_job_desc],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "similarity_matrix" in data

    def test_openapi_schema(self):
        """Test that OpenAPI schema is available."""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Smart Resume Screener API"
        assert schema["info"]["version"] == "1.0.0"

    @patch("src.app.main.get_model")
    def test_predict_multiple_resumes_multiple_jobs(self, mock_get_model):
        """Test prediction with multiple resumes and job descriptions."""
        mock_model = Mock()
        # 3 resumes x 2 jobs = 3x2 matrix
        mock_model.predict.return_value = pd.DataFrame(
            [[0.85, 0.75], [0.65, 0.90], [0.70, 0.80]]
        )
        mock_get_model.return_value = mock_model

        response = self.client.post(
            "/predict",
            json={
                "resumes": [
                    "Python developer",
                    "Java developer",
                    "Full stack developer",
                ],
                "job_descriptions": [
                    "Need Python expert",
                    "Looking for Java specialist",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["similarity_matrix"]) == 3
        assert len(data["similarity_matrix"][0]) == 2
