"""
Tests for the FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import io

from src.main import app
from src.config import settings


class TestAPI:
    """Test class for API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.sample_resume_text = "Software Engineer with 5 years experience in Python and Machine Learning"
        self.sample_job_description = "Looking for a Python developer with ML experience"
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "Resume Matcher API"
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @patch('src.main.resume_matcher')
    @patch('src.main.document_processor')
    def test_match_resume_success(self, mock_processor, mock_matcher):
        """Test successful resume matching."""
        # Mock the document processor
        mock_processor.process_document = AsyncMock(
            return_value=self.sample_resume_text
        )
        
        # Mock the resume matcher
        mock_matcher.match = AsyncMock(return_value={
            "score": 0.85,
            "is_match": True,
            "matched_skills": ["python", "machine learning"],
            "missing_skills": ["sql"],
            "recommendations": ["Consider gaining experience in SQL"]
        })
        
        # Create a test file
        test_file = io.BytesIO(b"Sample resume content")
        test_file.name = "test_resume.pdf"
        
        response = self.client.post(
            "/match",
            files={"resume_file": ("test_resume.pdf", test_file, "application/pdf")},
            data={
                "job_description": self.sample_job_description,
                "threshold": "0.7"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "match_score" in data
        assert "is_match" in data
        assert "recommendations" in data
        assert data["match_score"] == 0.85
        assert data["is_match"] is True
    
    def test_match_resume_invalid_file_type(self):
        """Test resume matching with invalid file type."""
        test_file = io.BytesIO(b"Sample content")
        test_file.name = "test_resume.exe"
        
        response = self.client.post(
            "/match",
            files={"resume_file": ("test_resume.exe", test_file, "application/octet-stream")},
            data={
                "job_description": self.sample_job_description,
                "threshold": "0.7"
            }
        )
        
        assert response.status_code == 400
        assert "File type" in response.json()["detail"]
    
    def test_match_resume_file_too_large(self):
        """Test resume matching with file that's too large."""
        # Create a file larger than the limit
        large_content = b"x" * (settings.max_file_size + 1)
        test_file = io.BytesIO(large_content)
        test_file.name = "test_resume.pdf"
        
        response = self.client.post(
            "/match",
            files={"resume_file": ("test_resume.pdf", test_file, "application/pdf")},
            data={
                "job_description": self.sample_job_description,
                "threshold": "0.7"
            }
        )
        
        assert response.status_code == 400
        assert "File size exceeds" in response.json()["detail"]
    
    @patch('src.main.resume_matcher')
    @patch('src.main.document_processor')
    def test_batch_match_resumes(self, mock_processor, mock_matcher):
        """Test batch resume matching."""
        # Mock the document processor
        mock_processor.process_document = AsyncMock(
            return_value=self.sample_resume_text
        )
        
        # Mock the resume matcher
        mock_matcher.match = AsyncMock(return_value={
            "score": 0.85,
            "is_match": True,
            "matched_skills": ["python"],
            "missing_skills": [],
            "recommendations": []
        })
        
        # Create test files
        test_file1 = io.BytesIO(b"Sample resume 1")
        test_file1.name = "resume1.pdf"
        test_file2 = io.BytesIO(b"Sample resume 2")
        test_file2.name = "resume2.pdf"
        
        response = self.client.post(
            "/batch-match",
            files=[
                ("resume_files", ("resume1.pdf", test_file1, "application/pdf")),
                ("resume_files", ("resume2.pdf", test_file2, "application/pdf"))
            ],
            data={
                "job_description": self.sample_job_description,
                "threshold": "0.7"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
    
    @patch('src.main.resume_matcher')
    def test_get_model_info(self, mock_matcher):
        """Test getting model information."""
        mock_matcher.get_model_info = AsyncMock(return_value={
            "is_trained": True,
            "model_name": "test_model",
            "model_version": "1.0.0"
        })
        
        response = self.client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "is_trained" in data
        assert data["is_trained"] is True
    
    @patch('src.main.resume_matcher')
    def test_retrain_model(self, mock_matcher):
        """Test model retraining."""
        mock_matcher.retrain = AsyncMock()
        
        response = self.client.post("/model/retrain")
        assert response.status_code == 200
        assert "retraining initiated" in response.json()["message"]
    
    def test_match_resume_no_file(self):
        """Test resume matching without file."""
        response = self.client.post(
            "/match",
            data={
                "job_description": self.sample_job_description,
                "threshold": "0.7"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.main.resume_matcher')
    @patch('src.main.document_processor')
    def test_match_resume_processing_error(self, mock_processor, mock_matcher):
        """Test resume matching with processing error."""
        mock_processor.process_document = AsyncMock(
            side_effect=Exception("Processing error")
        )
        
        test_file = io.BytesIO(b"Sample resume content")
        test_file.name = "test_resume.pdf"
        
        response = self.client.post(
            "/match",
            files={"resume_file": ("test_resume.pdf", test_file, "application/pdf")},
            data={
                "job_description": self.sample_job_description,
                "threshold": "0.7"
            }
        )
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
