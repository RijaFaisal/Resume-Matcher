"""
Pytest configuration and fixtures for tests.
"""

import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing."""
    return "Software Engineer with 5 years of Python experience and ML expertise"


@pytest.fixture
def sample_job_description():
    """Sample job description for testing."""
    return "Looking for Python developer with ML experience"


@pytest.fixture
def sample_resumes_list():
    """Sample list of resumes for testing."""
    return [
        "Python developer with machine learning experience",
        "Java developer with backend experience",
        "Full stack developer with React and Node.js",
    ]


@pytest.fixture
def sample_job_descriptions_list():
    """Sample list of job descriptions for testing."""
    return [
        "Need Python ML engineer",
        "Looking for Java backend developer",
    ]


@pytest.fixture
def mock_model():
    """Mock ResumeScreener model for testing."""
    mock = Mock()
    mock.predict.return_value = pd.DataFrame([[0.85, 0.75], [0.65, 0.90]])
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.fixture
def sample_similarity_matrix():
    """Sample similarity matrix for testing."""
    return pd.DataFrame([[0.85, 0.72], [0.68, 0.91], [0.55, 0.78]])


@pytest.fixture
def api_prediction_request():
    """Sample API prediction request for testing."""
    return {
        "resumes": [
            "Software Engineer with Python experience",
            "Data Scientist with ML background",
        ],
        "job_descriptions": [
            "Looking for Python developer",
            "Need ML expert",
        ],
    }


@pytest.fixture
def api_prediction_response():
    """Sample API prediction response for testing."""
    return {"similarity_matrix": [[0.85, 0.72], [0.68, 0.91]]}


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def sample_mlflow_model_uri():
    """Sample MLflow model URI for testing."""
    return "models:/resume-screener-sbert-pretrained/Production"

