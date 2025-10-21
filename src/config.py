"""
Simple configuration management for the Resume Matcher application.
"""

import os
from typing import List


class Settings:
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "resume_matcher")
    
    # Evidently Configuration
    EVIDENTLY_WORKSPACE_PATH: str = os.getenv("EVIDENTLY_WORKSPACE_PATH", "./evidently_workspace")
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "resume_matcher_model")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: List[str] = os.getenv("ALLOWED_EXTENSIONS", "pdf,doc,docx,txt").split(",")


# Global settings instance
settings = Settings()