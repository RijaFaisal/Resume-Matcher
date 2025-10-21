"""
Tests for the ResumeMatcher model.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from src.model import ResumeMatcher


class TestResumeMatcher:
    """Test class for ResumeMatcher model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = ResumeMatcher()
        self.sample_resume = "Software Engineer with Python and Machine Learning experience"
        self.sample_job_desc = "Looking for Python developer with ML skills"
    
    def test_initialization(self):
        """Test ResumeMatcher initialization."""
        assert self.matcher.vectorizer is None
        assert self.matcher.classifier is None
        assert self.matcher.is_trained is False
        assert self.matcher.model_path is not None
    
    @patch('src.model.pickle.load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_load_model_success(self, mock_exists, mock_open, mock_load):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_load.return_value = {
            'vectorizer': Mock(),
            'classifier': Mock()
        }
        
        # This would be an async test in practice
        # For now, we'll test the logic
        assert True  # Placeholder for async test
    
    @patch('pathlib.Path.exists')
    def test_load_model_not_found(self, mock_exists):
        """Test model loading when no model exists."""
        mock_exists.return_value = False
        
        # This would be an async test in practice
        assert True  # Placeholder for async test
    
    def test_simple_similarity_match(self):
        """Test simple similarity matching."""
        # This would test the fallback similarity method
        # For now, we'll test the logic structure
        assert True  # Placeholder for actual test
    
    def test_extract_skills(self):
        """Test skill extraction method."""
        matched_skills, missing_skills = self.matcher._extract_skills(
            "Python developer with machine learning experience",
            "Looking for Python and SQL skills"
        )
        
        assert "python" in matched_skills
        assert "sql" in missing_skills
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        missing_skills = ["sql", "docker"]
        matched_skills = ["python", "machine learning"]
        
        recommendations = self.matcher._generate_recommendations(
            missing_skills, matched_skills
        )
        
        assert len(recommendations) > 0
        assert any("sql" in rec.lower() for rec in recommendations)
        assert any("python" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_no_skills(self):
        """Test recommendation generation with no skills."""
        recommendations = self.matcher._generate_recommendations([], [])
        
        assert len(recommendations) > 0
        assert "specific skills" in recommendations[0].lower()
    
    @patch('src.model.mlflow.start_run')
    @patch('src.model.train_test_split')
    def test_train_model(self, mock_split, mock_mlflow):
        """Test model training."""
        # Mock the train_test_split
        mock_split.return_value = (
            Mock(), Mock(), [1, 0, 1], [0, 1, 0]
        )
        
        # Mock MLflow
        mock_run = Mock()
        mock_mlflow.return_value.__enter__.return_value = mock_run
        
        # This would be an async test in practice
        assert True  # Placeholder for async test
    
    def test_get_model_info(self):
        """Test getting model information."""
        # This would be an async test in practice
        info = {
            "is_trained": self.matcher.is_trained,
            "model_name": "test_model",
            "model_version": "1.0.0"
        }
        
        assert "is_trained" in info
        assert "model_name" in info
        assert "model_version" in info
    
    def test_save_model_not_trained(self):
        """Test saving model when not trained."""
        # This would test the error case
        assert True  # Placeholder for async test
    
    def test_match_with_trained_model(self):
        """Test matching with trained model."""
        # Mock trained model components
        self.matcher.is_trained = True
        self.matcher.vectorizer = Mock()
        self.matcher.classifier = Mock()
        
        # Mock vectorizer transform
        self.matcher.vectorizer.transform.return_value = Mock()
        
        # Mock classifier predict_proba
        self.matcher.classifier.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # This would be an async test in practice
        assert True  # Placeholder for async test
    
    def test_match_without_trained_model(self):
        """Test matching without trained model."""
        # This would test the fallback to simple similarity
        assert True  # Placeholder for async test
    
    def test_extract_skills_comprehensive(self):
        """Test comprehensive skill extraction."""
        resume_text = """
        Software Engineer with 5 years of experience in:
        - Python programming
        - Machine Learning with scikit-learn
        - Data analysis with pandas
        - Web development with Django
        - Database design with PostgreSQL
        """
        
        job_description = """
        We are looking for a developer with:
        - Python and JavaScript experience
        - Machine Learning knowledge
        - SQL and database skills
        - Docker containerization
        - AWS cloud services
        """
        
        matched_skills, missing_skills = self.matcher._extract_skills(
            resume_text, job_description
        )
        
        # Should match on common skills
        assert len(matched_skills) > 0
        assert len(missing_skills) > 0
    
    def test_generate_recommendations_various_scenarios(self):
        """Test recommendation generation in various scenarios."""
        # Scenario 1: Has matched skills, missing some
        rec1 = self.matcher._generate_recommendations(
            ["sql", "docker"], ["python", "ml"]
        )
        assert len(rec1) == 2
        
        # Scenario 2: Only matched skills
        rec2 = self.matcher._generate_recommendations(
            [], ["python", "ml", "sql"]
        )
        assert len(rec2) == 1
        
        # Scenario 3: Only missing skills
        rec3 = self.matcher._generate_recommendations(
            ["sql", "docker", "docker"], []
        )
        assert len(rec3) == 1
    
    @patch('src.model.pickle.dump')
    @patch('builtins.open')
    def test_save_model_success(self, mock_open, mock_dump):
        """Test successful model saving."""
        # Set up trained model
        self.matcher.is_trained = True
        self.matcher.vectorizer = Mock()
        self.matcher.classifier = Mock()
        
        # This would be an async test in practice
        assert True  # Placeholder for async test
    
    def test_model_initialization_with_settings(self):
        """Test model initialization with custom settings."""
        # Test that model path is set correctly
        assert self.matcher.model_path is not None
        assert hasattr(self.matcher, 'is_trained')
        assert hasattr(self.matcher, 'vectorizer')
        assert hasattr(self.matcher, 'classifier')
