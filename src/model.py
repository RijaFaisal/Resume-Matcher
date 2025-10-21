"""
Resume matching model implementation.
"""

import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .config import settings

logger = logging.getLogger(__name__)


class ResumeMatcher:
    """
    Resume matching model using TF-IDF and machine learning techniques.
    """
    
    def __init__(self):
        """Initialize the ResumeMatcher."""
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[RandomForestClassifier] = None
        self.is_trained: bool = False
        self.model_path = Path(settings.model_path)
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
    
    async def load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            model_file = self.model_path / f"{settings.model_name}.pkl"
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.is_trained = True
                
                logger.info("Model loaded successfully")
            else:
                logger.warning("No trained model found, will train on first use")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def save_model(self) -> None:
        """Save the trained model to disk."""
        try:
            if not self.is_trained:
                raise ValueError("No trained model to save")
            
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }
            
            model_file = self.model_path / f"{settings.model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    async def train(
        self,
        resumes: List[str],
        job_descriptions: List[str],
        labels: List[int]
    ) -> Dict[str, Any]:
        """
        Train the resume matching model.
        
        Args:
            resumes: List of resume texts
            job_descriptions: List of job description texts
            labels: List of binary labels (1 for match, 0 for no match)
        
        Returns:
            Training metrics
        """
        try:
            with mlflow.start_run():
                # Combine resume and job description texts
                combined_texts = [
                    f"{resume} {job_desc}"
                    for resume, job_desc in zip(resumes, job_descriptions)
                ]
                
                # Vectorize texts
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                X = self.vectorizer.fit_transform(combined_texts)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=42, stratify=labels
                )
                
                # Train classifier
                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                )
                self.classifier.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = self.classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 10)
                
                # Log model
                mlflow.sklearn.log_model(
                    self.classifier,
                    "model",
                    registered_model_name=settings.model_name
                )
                
                self.is_trained = True
                
                # Save model
                await self.save_model()
                
                logger.info(f"Model trained with accuracy: {accuracy}")
                
                return {
                    "accuracy": accuracy,
                    "classification_report": classification_report(
                        y_test, y_pred, output_dict=True
                    )
                }
                
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    async def match(
        self,
        resume_text: str,
        job_description: str,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Match a resume against a job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            threshold: Matching threshold
        
        Returns:
            Matching results
        """
        try:
            if not self.is_trained:
                # Fallback to simple TF-IDF similarity
                return await self._simple_similarity_match(
                    resume_text, job_description, threshold
                )
            
            # Use trained model
            combined_text = f"{resume_text} {job_description}"
            X = self.vectorizer.transform([combined_text])
            
            # Get prediction probability
            proba = self.classifier.predict_proba(X)[0]
            match_probability = proba[1] if len(proba) > 1 else proba[0]
            
            is_match = match_probability >= threshold
            
            # Extract skills and generate recommendations
            matched_skills, missing_skills = self._extract_skills(
                resume_text, job_description
            )
            
            recommendations = self._generate_recommendations(
                missing_skills, matched_skills
            )
            
            return {
                "score": float(match_probability),
                "is_match": is_match,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in match: {e}")
            raise
    
    async def _simple_similarity_match(
        self,
        resume_text: str,
        job_description: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Fallback similarity matching using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            is_match = similarity >= threshold
            
            matched_skills, missing_skills = self._extract_skills(
                resume_text, job_description
            )
            
            recommendations = self._generate_recommendations(
                missing_skills, matched_skills
            )
            
            return {
                "score": float(similarity),
                "is_match": is_match,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in simple similarity match: {e}")
            raise
    
    def _extract_skills(
        self, resume_text: str, job_description: str
    ) -> Tuple[List[str], List[str]]:
        """Extract matched and missing skills."""
        # Simple keyword-based skill extraction
        # In a real implementation, you'd use NER or more sophisticated methods
        
        common_skills = [
            'python', 'java', 'javascript', 'sql', 'machine learning',
            'data analysis', 'project management', 'leadership',
            'communication', 'problem solving', 'teamwork'
        ]
        
        resume_lower = resume_text.lower()
        job_desc_lower = job_description.lower()
        
        matched_skills = []
        missing_skills = []
        
        for skill in common_skills:
            if skill in job_desc_lower:
                if skill in resume_lower:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
        
        return matched_skills, missing_skills
    
    def _generate_recommendations(
        self, missing_skills: List[str], matched_skills: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if missing_skills:
            recommendations.append(
                f"Consider gaining experience in: {', '.join(missing_skills)}"
            )
        
        if matched_skills:
            recommendations.append(
                f"Highlight your experience in: {', '.join(matched_skills)}"
            )
        
        if not matched_skills and not missing_skills:
            recommendations.append(
                "Consider adding more specific skills and technologies to your resume"
            )
        
        return recommendations
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "is_trained": self.is_trained,
            "model_name": settings.model_name,
            "model_version": settings.model_version,
            "model_path": str(self.model_path)
        }
    
    async def retrain(self) -> None:
        """Retrain the model with new data."""
        # This would typically load new training data and retrain
        logger.info("Model retraining initiated")
        # Implementation would depend on your data pipeline
