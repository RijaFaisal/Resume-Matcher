import pandas as pd
import torch
from sentence_transformers import util
import mlflow

class ResumeScreener:
    """
    A class to load the SBERT model (as pyfunc) FROM THE MLFLOW REGISTRY
    and compute similarity.
    """
    def __init__(self, model_uri: str):
        """
        Initializes by loading a pyfunc model from a given MLflow URI.
        """
        print(f"Loading pyfunc model from MLflow URI: {model_uri}")
        self.model = mlflow.pyfunc.load_model(model_uri)
        print("Pyfunc model loaded successfully from MLflow.")

    def predict(self, resumes: list[str], job_descriptions: list[str]) -> pd.DataFrame:
        """
        Encodes texts and computes the cosine similarity matrix.
        """
        resume_df = pd.DataFrame(resumes, columns=["text"])
        job_df = pd.DataFrame(job_descriptions, columns=["text"])

        resume_embeddings = self.model.predict(resume_df)
        job_embeddings = self.model.predict(job_df)

        resume_tensors = torch.from_numpy(resume_embeddings)
        job_tensors = torch.from_numpy(job_embeddings)

        cos_sim_matrix = util.cos_sim(resume_tensors, job_tensors).cpu().numpy()
        return pd.DataFrame(cos_sim_matrix)


# --- CRITICAL: Update this URI to point to the "normal" pre-trained model ---
production_model_uri = "models:/resume-screener-sbert-pretrained/Production"

# This instance is created only ONCE when the application starts.
model = ResumeScreener(model_uri=production_model_uri)

def get_model():
    """Function to get the singleton model instance."""
    return model