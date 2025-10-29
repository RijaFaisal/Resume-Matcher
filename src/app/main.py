from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
from .model import ResumeScreener, get_model
from .instrumentation import get_instrumentator, TOKENS_PROCESSED_COUNTER

app = FastAPI(
    title="Smart Resume Screener API",
    description="API for calculating similarity between resumes and job descriptions.",
    version="1.0.0"
)


instrumentator = get_instrumentator()
instrumentator.instrument(app).expose(app)

class PredictionRequest(BaseModel):
    resumes: List[str]
    job_descriptions: List[str]

class PredictionResponse(BaseModel):
    similarity_matrix: List[List[float]]

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, model: ResumeScreener = Depends(get_model)):
    """
    Prediction endpoint to get cosine similarity.
    """
    total_tokens = sum(len(text.split()) for text in request.resumes + request.job_descriptions)
    TOKENS_PROCESSED_COUNTER.inc(total_tokens)
    similarity_df = model.predict(request.resumes, request.job_descriptions)
    return {"similarity_matrix": similarity_df.values.tolist()}