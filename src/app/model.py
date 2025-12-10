# src/app/model.py
import pandas as pd
import torch
from sentence_transformers import util
import mlflow
import numpy as np
from typing import List

class ResumeScreener:
    """
    Loads a pyfunc model from MLflow (if available) and computes similarity scores.
    The pyfunc model is expected (if present) to provide embeddings via .predict(df)
    or via .encode / .embed methods. If none are available, a deterministic dummy
    embedding is used as fallback.
    """
    def __init__(self, model_uri: str | None = None):
        self.model = None
        if model_uri:
            try:
                print(f"Loading pyfunc model from MLflow URI: {model_uri}")
                self.model = mlflow.pyfunc.load_model(model_uri)
                print("Pyfunc model loaded successfully from MLflow.")
            except Exception as e:
                print("Warning: failed to load pyfunc model from MLflow:", e)
                self.model = None

    def _get_embeddings_via_model(self, texts: List[str]) -> np.ndarray | None:
        """Try several ways to get embeddings from the loaded pyfunc model.
        Returns numpy ndarray shape (n_texts, embed_dim) or None if not possible."""
        if self.model is None:
            return None

        # 1) try pyfunc.predict on a DataFrame (common)
        try:
            df = pd.DataFrame({"text": texts})
            out = self.model.predict(df)
            # model might return numpy array, list, or DataFrame
            if isinstance(out, pd.DataFrame):
                arr = out.to_numpy()
            else:
                arr = np.asarray(out)
            if arr.size > 0:
                return arr.astype(np.float32)
        except Exception:
            pass

        # 2) try common embed/encode methods
        for attr in ("encode", "embed", "get_embeddings"):
            fn = getattr(self.model, attr, None)
            if callable(fn):
                try:
                    out = fn(texts)
                    arr = np.asarray(out)
                    if arr.size > 0:
                        return arr.astype(np.float32)
                except Exception:
                    continue

        return None

    def predict(self, resumes: list[str], job_descriptions: list[str]) -> pd.DataFrame:
        """
        Accepts:
      - resumes: list of resume text strings (e.g. [resume_text])
      - job_descriptions: list of job description strings
        Returns:
      - pandas.DataFrame with shape (n_resumes, n_jobs) of similarity scores.
    """
    # defensive defaults
    if resumes is None:
        resumes = []
    if job_descriptions is None:
        job_descriptions = []

    n_resumes = max(1, len(resumes))
    n_jobs = len(job_descriptions)

    # if no jobs, return empty dataframe
    if n_jobs == 0:
        return pd.DataFrame([[]])

    # Helper: ensure per-item embeddings (one row per input text)
    def _ensure_per_item_embeddings(texts, emb):
        # if no texts, return empty 2D array
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        # If model produced nothing, try to compute per-item via model
        if emb is None:
            per_item = []
            for t in texts:
                e = self._get_embeddings_via_model([t])
                if e is None:
                    per_item.append(None)
                else:
                    e_arr = np.asarray(e, dtype=np.float32)
                    if e_arr.ndim == 1:
                        e_arr = e_arr.reshape(1, -1)
                    per_item.append(e_arr)
            if any(x is None for x in per_item):
                return None
            return np.vstack(per_item)

        # convert to ndarray
        emb = np.asarray(emb, dtype=np.float32)

        # If emb is 1D but multiple texts were provided, compute per-item instead
        if emb.ndim == 1 and len(texts) > 1:
            return _ensure_per_item_embeddings(texts, None)

        # If emb has first dim mismatch, try per-item
        if emb.ndim >= 2 and emb.shape[0] != len(texts):
            per_item = []
            for t in texts:
                e = self._get_embeddings_via_model([t])
                if e is None:
                    per_item.append(None)
                else:
                    e_arr = np.asarray(e, dtype=np.float32)
                    if e_arr.ndim == 1:
                        e_arr = e_arr.reshape(1, -1)
                    per_item.append(e_arr)
            if any(x is None for x in per_item):
                return None
            return np.vstack(per_item)

        # Otherwise emb already matches inputs
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb

    # 1) Try to get embeddings from MLflow-loaded model
    resume_embeddings = self._get_embeddings_via_model(resumes)
    job_embeddings = self._get_embeddings_via_model(job_descriptions)

    # 2) If model didn't supply per-item embeddings, attempt direct predict on lists
    if resume_embeddings is None and self.model is not None:
        try:
            out = self.model.predict(resumes)
            resume_embeddings = np.asarray(out, dtype=np.float32)
        except Exception:
            resume_embeddings = None

    if job_embeddings is None and self.model is not None:
        try:
            out = self.model.predict(job_descriptions)
            job_embeddings = np.asarray(out, dtype=np.float32)
        except Exception:
            job_embeddings = None

    # 3) Ensure per-item shape (n_items, embed_dim) or fallback to per-item computation
    resume_embeddings = _ensure_per_item_embeddings(resumes, resume_embeddings)
    job_embeddings = _ensure_per_item_embeddings(job_descriptions, job_embeddings)

    # 4) Deterministic fallback if embeddings still None
    if resume_embeddings is None or job_embeddings is None:
        embed_dim = 384
        if resume_embeddings is None:
            resume_embeddings = np.tile(np.linspace(0.1, 0.9, embed_dim, dtype=np.float32), (n_resumes, 1))
        if job_embeddings is None:
            job_embeddings = np.tile(np.linspace(0.9, 0.1, embed_dim, dtype=np.float32), (n_jobs, 1))

    # Ensure numpy arrays and correct dtype
    resume_embeddings = np.asarray(resume_embeddings, dtype=np.float32)
    job_embeddings = np.asarray(job_embeddings, dtype=np.float32)

    # Ensure 2D shapes
    if resume_embeddings.ndim == 1:
        resume_embeddings = resume_embeddings.reshape(1, -1)
    if job_embeddings.ndim == 1:
        job_embeddings = job_embeddings.reshape(1, -1)

    # Convert to torch tensors
    resume_tensors = torch.from_numpy(resume_embeddings)
    job_tensors = torch.from_numpy(job_embeddings)

    # Normalize and compute cosine similarity matrix (n_resumes x n_jobs)
    resume_norm = resume_tensors / (resume_tensors.norm(dim=1, keepdim=True) + 1e-8)
    job_norm = job_tensors / (job_tensors.norm(dim=1, keepdim=True) + 1e-8)
    sim_matrix = (resume_norm @ job_norm.T).detach().cpu().numpy()

    # Return DataFrame shaped (n_resumes, n_jobs)
    # Use simple column names (ui only uses values), but keep job_descriptions length consistent
    col_names = [f"job_{i}" for i in range(n_jobs)]
    return pd.DataFrame(sim_matrix, columns=col_names)


# instantiate singleton using MLflow production URI if available
production_model_uri = "models:/resume-screener-sbert-pretrained/Production"
try:
    model = ResumeScreener(model_uri=production_model_uri)
except Exception:
    # fallback to a model instance without mlflow model loaded
    model = ResumeScreener(model_uri=None)

def get_model():
    """Return the singleton model instance."""
    return model
