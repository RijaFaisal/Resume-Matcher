"""
FastAPI Application for the MLOps Resume Matching App
- Loads a SentenceTransformer model and pre-computed embeddings from S3.
- Provides an endpoint to match resumes against job descriptions.
- Exposes Prometheus metrics for monitoring.
"""
import os
import time
import logging
from io import BytesIO, StringIO
from typing import List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

import boto3
import torch
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from sentence_transformers import SentenceTransformer, util

# --- 1. Load Environment Variables & Basic Configuration ---
load_dotenv()

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 2. Centralized Configuration ---
class AppConfig:
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "resume-matcher-bucket-sahil")
    MODEL_S3_KEY = os.getenv("MODEL_S3_KEY", "models/sbert_model/")
    EMBEDDINGS_S3_KEY = os.getenv("EMBEDDINGS_S3_KEY", "models/job_embeddings.pt")
    JOB_DATA_S3_KEY = os.getenv("JOB_DATA_S3_KEY", "raw-data/job_title_des.csv")
    LOCAL_MODEL_PATH = "/tmp/sbert_model"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_VERSION = "1.0"

CONFIG = AppConfig()

# --- 3. Global State ---
state = {
    "sbert_model": None,
    "job_embeddings": None,
    "df_job_description": None,
    "model_info": {"model_name": CONFIG.MODEL_NAME, "version": CONFIG.MODEL_VERSION}
}

# --- 4. Lifespan Event for Model Loading & Metric Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup events:
    1. Initializes Prometheus metrics to avoid duplication with hot-reloading.
    2. Loads the ML model and data from S3.
    """
    # --- METRIC DEFINITIONS (MOVED HERE) ---
    # This prevents the "Duplicated timeseries" error with Uvicorn's reloader.
    app.state.METRICS = {
        "requests_total": Counter(
            "matching_requests_total", "Total number of matching requests.", ["model_version", "status"]
        ),
        "duration_seconds": Histogram(
            "matching_duration_seconds", "Time spent processing a matching request.", ["model_version"]
        ),
        "load_time_seconds": Gauge("model_load_seconds", "Time taken to load models and embeddings."),
        "errors_total": Counter("api_errors_total", "Total API errors.", ["error_type"]),
        "similarity_score": Histogram(
            "match_similarity_score", "Distribution of similarity scores for the top match.", ["model_version"]
        ),
        "http_requests_duration": Histogram(
            "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint", "status_code"]
        )
    }
    
    start_time = time.time()
    logger.info("🚀 Application startup initiated...")

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        s3_resource = boto3.resource('s3')

        # Load Job Descriptions DataFrame
        logger.info(f"Loading job descriptions from s3://{CONFIG.S3_BUCKET_NAME}/{CONFIG.JOB_DATA_S3_KEY}...")
        job_desc_obj = s3_client.get_object(Bucket=CONFIG.S3_BUCKET_NAME, Key=CONFIG.JOB_DATA_S3_KEY)
        state["df_job_description"] = pd.read_csv(StringIO(job_desc_obj["Body"].read().decode("utf-8")))
        logger.info(f"✅ Job descriptions DataFrame loaded ({len(state['df_job_description'])} rows).")

        # Download and Load SBERT Model
        logger.info(f"Downloading SBERT model from s3://{CONFIG.S3_BUCKET_NAME}/{CONFIG.MODEL_S3_KEY}...")
        if not os.path.exists(CONFIG.LOCAL_MODEL_PATH):
            os.makedirs(CONFIG.LOCAL_MODEL_PATH)
        
        bucket = s3_resource.Bucket(CONFIG.S3_BUCKET_NAME)
        for obj in bucket.objects.filter(Prefix=CONFIG.MODEL_S3_KEY):
            target = os.path.join(CONFIG.LOCAL_MODEL_PATH, os.path.relpath(obj.key, CONFIG.MODEL_S3_KEY))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] != '/':
                bucket.download_file(obj.key, target)
        
        state["sbert_model"] = SentenceTransformer(CONFIG.LOCAL_MODEL_PATH)
        logger.info(f"✅ SBERT model loaded successfully from {CONFIG.LOCAL_MODEL_PATH}.")

        # Load Pre-computed Job Embeddings
        logger.info(f"Loading job embeddings from s3://{CONFIG.S3_BUCKET_NAME}/{CONFIG.EMBEDDINGS_S3_KEY}...")
        embeddings_buffer = BytesIO()
        s3_client.download_fileobj(CONFIG.S3_BUCKET_NAME, CONFIG.EMBEDDINGS_S3_KEY, embeddings_buffer)
        embeddings_buffer.seek(0)
        state["job_embeddings"] = torch.load(embeddings_buffer)
        logger.info(f"✅ Job embeddings tensor loaded successfully with shape: {state['job_embeddings'].shape}.")

        load_duration = time.time() - start_time
        app.state.METRICS["load_time_seconds"].set(load_duration)
        logger.info(f"✅ Application startup complete in {load_duration:.2f}s. Model in use: {state['model_info']['model_name']}")

    except Exception as e:
        app.state.METRICS["errors_total"].labels(error_type="model_loading").inc()
        logger.exception(f"❌ CRITICAL ERROR during model loading: {e}")
        logger.warning("Model and embeddings not loaded. The /match_resume endpoint will fail.")

    yield

    logger.info("🔌 Shutting down application...")


# --- 5. FastAPI App Initialization ---
app = FastAPI(
    title="Resume Matching API",
    description="MLOps API for matching resumes to job descriptions using semantic similarity.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 6. Pydantic Models for API I/O ---
class MatchRequest(BaseModel):
    """Input schema for the /match_resume endpoint."""
    resume_text: str = Field(..., min_length=50, description="The full text of the resume to be matched.")
    top_n: int = Field(5, gt=0, le=50, description="The number of top matches to return.")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "resume_text": "Experienced data scientist with a background in machine learning, Python, and SQL...",
            "top_n": 3
        }
    })

class MatchResult(BaseModel):
    """Schema for a single job match result."""
    rank: int
    job_title: str
    similarity_score: float

class MatchResponse(BaseModel):
    """Output schema for the /match_resume endpoint."""
    matches: List[MatchResult]
    model_info: Dict[str, str]


# --- 7. API Endpoints ---
@app.get("/")
def root():
    """Root endpoint with basic API information."""
    return {"message": "Resume Matching API is running."}

@app.get("/health")
def health_check():
    """Health check endpoint to verify that the model and embeddings are loaded."""
    model_loaded = state["sbert_model"] is not None
    embeddings_loaded = state["job_embeddings"] is not None
    status = "healthy" if model_loaded and embeddings_loaded else "degraded"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "embeddings_loaded": embeddings_loaded,
        "model_info": state["model_info"],
    }

@app.post("/match_resume", response_model=MatchResponse)
def match_resume(request: MatchRequest, http_request: Request):
    """
    Accepts resume text and returns the top_n most similar job descriptions.
    """
    start_time = time.time()
    model_version_label = state["model_info"].get("version", "unknown")
    metrics = http_request.app.state.METRICS

    if not state["sbert_model"] or not isinstance(state["job_embeddings"], torch.Tensor):
        metrics["errors_total"].labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model or embeddings are not available. Service is degraded.")

    try:
        # Generate embedding for the new resume
        resume_embedding = state["sbert_model"].encode(request.resume_text, convert_to_tensor=True)

        # Compute cosine similarity
        cos_scores = util.cos_sim(resume_embedding, state["job_embeddings"])[0]

        # Get top N results
        k = min(request.top_n, len(state["df_job_description"]))
        top_results = torch.topk(cos_scores, k=k)

        # Format the response
        matches = [
            MatchResult(
                rank=i + 1,
                job_title=state["df_job_description"].iloc[idx.item()]['Job Title'],
                similarity_score=score.item()
            )
            for i, (score, idx) in enumerate(zip(top_results[0], top_results[1]))
        ]

        # Record metrics for a successful request
        duration = time.time() - start_time
        metrics["duration_seconds"].labels(model_version=model_version_label).observe(duration)
        metrics["requests_total"].labels(model_version=model_version_label, status="success").inc()
        if matches:
            metrics["similarity_score"].labels(model_version=model_version_label).observe(matches[0].similarity_score)

        logger.info(f"Successfully matched resume in {duration:.4f}s. Top score: {matches[0].similarity_score if matches else 'N/A'}")
        return MatchResponse(matches=matches, model_info=state["model_info"])

    except Exception as e:
        metrics["requests_total"].labels(model_version=model_version_label, status="error").inc()
        metrics["errors_total"].labels(error_type="matching_error").inc()
        logger.exception(f"An error occurred during matching: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during matching: {str(e)}")


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
def get_model_info():
    """Get detailed information about the currently loaded model and data."""
    if state["sbert_model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_info": state["model_info"],
        "loaded": True,
        "job_embeddings_shape": list(state["job_embeddings"].shape) if state["job_embeddings"] is not None else None,
        "total_jobs_indexed": len(state["df_job_description"]) if state["df_job_description"] is not None else 0,
    }

# --- 8. Middleware for HTTP Metrics ---
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    """Middleware to capture HTTP request duration for Prometheus."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Check if metrics have been initialized in the app state before using them
    if hasattr(request.app.state, 'METRICS'):
        request.app.state.METRICS["http_requests_duration"].labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).observe(duration)
    return response

# --- 9. Main entry point for running locally ---
if __name__ == "__main__":
    import uvicorn
    # This block is for local development and debugging.
    # In production, a process manager like Gunicorn with Uvicorn workers is recommended.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)