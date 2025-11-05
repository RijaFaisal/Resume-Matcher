"""
Tests for the ResumeScreener model.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import torch
import numpy as np

from src.app.model import ResumeScreener, get_model


class TestResumeScreener:
    """Test class for ResumeScreener model."""

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_initialization(self, mock_load_model):
        """Test ResumeScreener initialization."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        model_uri = "models:/test-model/Production"
        screener = ResumeScreener(model_uri=model_uri)

        assert screener.model == mock_model
        mock_load_model.assert_called_once_with(model_uri)

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_returns_dataframe(self, mock_load_model):
        """Test that predict returns a pandas DataFrame."""
        # Mock model predictions
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([[0.1, 0.2, 0.3]]),  # Resume embeddings
            np.array([[0.4, 0.5, 0.6]]),  # Job description embeddings
        ]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        result = screener.predict(
            resumes=["Software Engineer with Python"],
            job_descriptions=["Looking for Python developer"],
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 1)

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_multiple_resumes_and_jobs(self, mock_load_model):
        """Test prediction with multiple resumes and job descriptions."""
        mock_model = Mock()
        # 2 resumes, 3-dim embeddings
        mock_model.predict.side_effect = [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),  # 2 resumes
            np.array([[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]]),  # 2 jobs
        ]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        result = screener.predict(
            resumes=["Resume 1", "Resume 2"],
            job_descriptions=["Job 1", "Job 2"],
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)  # 2 resumes x 2 jobs

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_creates_proper_dataframes(self, mock_load_model):
        """Test that predict creates proper DataFrames for model input."""
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([[0.1, 0.2]]),
            np.array([[0.3, 0.4]]),
        ]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        screener.predict(
            resumes=["Test resume"],
            job_descriptions=["Test job"],
        )

        # Check that model.predict was called twice (once for resumes, once for jobs)
        assert mock_model.predict.call_count == 2

        # Check the structure of the input DataFrames
        first_call_df = mock_model.predict.call_args_list[0][0][0]
        second_call_df = mock_model.predict.call_args_list[1][0][0]

        assert isinstance(first_call_df, pd.DataFrame)
        assert isinstance(second_call_df, pd.DataFrame)
        assert "text" in first_call_df.columns
        assert "text" in second_call_df.columns

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_similarity_values_in_range(self, mock_load_model):
        """Test that cosine similarity values are in valid range [-1, 1]."""
        mock_model = Mock()
        # Normalized embeddings for realistic cosine similarity
        mock_model.predict.side_effect = [
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
        ]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        result = screener.predict(
            resumes=["Resume"],
            job_descriptions=["Job"],
        )

        # Cosine similarity should be in range [-1, 1]
        assert result.values.min() >= -1.0
        assert result.values.max() <= 1.0

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_empty_strings(self, mock_load_model):
        """Test prediction with empty strings."""
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
        ]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        result = screener.predict(
            resumes=[""],
            job_descriptions=[""],
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 1)

    def test_get_model_returns_singleton(self):
        """Test that get_model returns the same model instance."""
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2

    def test_get_model_returns_resume_screener(self):
        """Test that get_model returns a ResumeScreener instance."""
        model = get_model()
        assert isinstance(model, ResumeScreener)

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_with_special_characters(self, mock_load_model):
        """Test prediction with special characters in text."""
        mock_model = Mock()
        mock_model.predict.side_effect = [
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0.4, 0.5, 0.6]]),
        ]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        result = screener.predict(
            resumes=["C++ developer with @skills #coding!"],
            job_descriptions=["Need C++, Python & Java expert"],
        )

        assert isinstance(result, pd.DataFrame)

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_converts_to_tensors_correctly(self, mock_load_model):
        """Test that embeddings are converted to tensors correctly."""
        mock_model = Mock()
        embeddings = np.array([[0.5, 0.5, 0.5]])
        mock_model.predict.side_effect = [embeddings, embeddings]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        result = screener.predict(
            resumes=["Test"],
            job_descriptions=["Test"],
        )

        # Should complete without tensor conversion errors
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    @patch("src.app.model.mlflow.pyfunc.load_model")
    def test_predict_high_similarity_same_text(self, mock_load_model):
        """Test that identical texts have high similarity."""
        mock_model = Mock()
        # Same embedding for both
        same_embedding = np.array([[0.577, 0.577, 0.577]])  # Normalized
        mock_model.predict.side_effect = [same_embedding, same_embedding]
        mock_load_model.return_value = mock_model

        screener = ResumeScreener(model_uri="models:/test/Production")
        result = screener.predict(
            resumes=["Python developer"],
            job_descriptions=["Python developer"],
        )

        # Same text should have similarity close to 1
        assert result.values[0][0] > 0.9
