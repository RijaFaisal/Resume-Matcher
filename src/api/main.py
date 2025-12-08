"""
FastAPI Application for the MLOps Resume Matching App with Guardrails
"""
import os
import time
import logging
from io import BytesIO, StringIO
from typing import List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

import boto3
import torch
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, REGISTRY,  Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from sentence_transformers import SentenceTransformer, util

# Import guardrails
from src.guardrails import PolicyEngine, GuardrailsConfig, PolicyMode

# --- 1. Load environment ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 2. Configuration ---
class AppConfig:
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "resume-matcher-bucket-sahil")
    MODEL_S3_KEY = os.getenv("MODEL_S3_KEY", "models/sbert_model/")
    EMBEDDINGS_S3_KEY = os.getenv("EMBEDDINGS_S3_KEY", "models/job_embeddings.pt")
    JOB_DATA_S3_KEY = os.getenv("JOB_DATA_S3_KEY", "raw-data/job_title_des.csv")
    LOCAL_MODEL_PATH = "/tmp/sbert_model"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_VERSION = "1.0"

CONFIG = AppConfig()

# --- 3. Global state ---
state = {
    "sbert_model": None,
    "job_embeddings": None,
    "df_job_description": None,
    "model_info": {"model_name": CONFIG.MODEL_NAME, "version": CONFIG.MODEL_VERSION},
    "policy_engine": None,
}

# --- 3.5. Initialize Guardrails ---
def init_guardrails():
    """Initialize guardrails policy engine."""
    guardrails_config = GuardrailsConfig(
        mode=PolicyMode.BALANCED,
        enable_pii_detection=True,
        enable_injection_filter=True,
        enable_toxicity_filter=False,  # Resume content may contain various terms
        enable_hallucination_detector=False,  # Not applicable for resume matching
        mask_pii=True,
        strict_injection_filter=False,
        max_input_length=100000,  # Resumes can be long
        min_input_length=50,
        log_violations=True,
    )
    state["policy_engine"] = PolicyEngine(guardrails_config)
    logger.info("Guardrails policy engine initialized")



def get_metric(metric_type, name, description, labels=None):
    """Return existing metric if already registered, else create a new one."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    
    if metric_type == "counter":
        return Counter(name, description, labels or [])
    elif metric_type == "histogram":
        return Histogram(name, description, labels or [])
    elif metric_type == "gauge":
        return Gauge(name, description, labels or [])
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

# --- 4. Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prometheus metrics
    app.state.METRICS = {
    "requests_total": get_metric("counter", "matching_requests_total", "Total number of matching requests.", ["model_version", "status"]),
    "duration_seconds": get_metric("histogram", "matching_duration_seconds", "Time spent processing a matching request.", ["model_version"]),
    "load_time_seconds": get_metric("gauge", "model_load_seconds", "Time taken to load models and embeddings."),
    "errors_total": get_metric("counter", "api_errors_total", "Total API errors.", ["error_type"]),
    "similarity_score": get_metric("histogram", "match_similarity_score", "Distribution of similarity scores for the top match.", ["model_version"]),
    "http_requests_duration": get_metric("histogram", "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint", "status_code"])


    }

    start_time = time.time()
    logger.info("🚀 Application startup initiated...")

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        s3_resource = boto3.resource("s3")

        # Job descriptions
        job_desc_obj = s3_client.get_object(Bucket=CONFIG.S3_BUCKET_NAME, Key=CONFIG.JOB_DATA_S3_KEY)
        state["df_job_description"] = pd.read_csv(StringIO(job_desc_obj["Body"].read().decode("utf-8")))
        logger.info(f"✅ Job descriptions loaded ({len(state['df_job_description'])} rows).")

        # Download SBERT model
        if not os.path.exists(CONFIG.LOCAL_MODEL_PATH):
            os.makedirs(CONFIG.LOCAL_MODEL_PATH)

        bucket = s3_resource.Bucket(CONFIG.S3_BUCKET_NAME)
        for obj in bucket.objects.filter(Prefix=CONFIG.MODEL_S3_KEY):
            target = os.path.join(CONFIG.LOCAL_MODEL_PATH, os.path.relpath(obj.key, CONFIG.MODEL_S3_KEY))
            if obj.key[-1] != '/':
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                bucket.download_file(obj.key, target)

        state["sbert_model"] = SentenceTransformer(CONFIG.LOCAL_MODEL_PATH)
        logger.info(f"✅ SBERT model loaded from {CONFIG.LOCAL_MODEL_PATH}.")

        # Load embeddings
        embeddings_buffer = BytesIO()
        s3_client.download_fileobj(CONFIG.S3_BUCKET_NAME, CONFIG.EMBEDDINGS_S3_KEY, embeddings_buffer)
        embeddings_buffer.seek(0)
        state["job_embeddings"] = torch.load(embeddings_buffer)
        logger.info(f"✅ Job embeddings loaded: {state['job_embeddings'].shape}.")

        load_duration = time.time() - start_time
        app.state.METRICS["load_time_seconds"].set(load_duration)
        logger.info(f"✅ Startup complete in {load_duration:.2f}s.")
        
        # Initialize guardrails
        init_guardrails()
        logger.info("✅ Guardrails initialized.")

    except Exception as e:
        app.state.METRICS["errors_total"].labels(error_type="model_loading").inc()
        logger.exception(f"❌ ERROR during startup: {e}")

    yield
    logger.info("🔌 Shutting down application...")

# --- 5. FastAPI ---
app = FastAPI(
    title="Resume Matching API with Guardrails",
    description="MLOps API for matching resumes with comprehensive safety mechanisms.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 6. Pydantic Models ---
class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=50)
    top_n: int = Field(5, gt=0, le=50)

class MatchResult(BaseModel):
    rank: int
    job_title: str
    similarity_score: float

class MatchResponse(BaseModel):
    matches: List[MatchResult]
    model_info: Dict[str, str]

# --- 7. Endpoints ---
@app.get("/")
def root():
    return {"message": "Resume Matching API is running."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if state["sbert_model"] is not None and state["job_embeddings"] is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": state["sbert_model"] is not None,
        "embeddings_loaded": state["job_embeddings"] is not None,
        "model_info": state["model_info"],
    }

@app.post("/match_resume", response_model=MatchResponse)
def match_resume(request: MatchRequest, http_request: Request):
    """
    Resume matching endpoint with guardrails.
    
    Flow:
    1. Validate input (PII detection, length checks)
    2. Encode resume using SBERT
    3. Calculate similarity scores
    4. Return top N matches
    """
    start_time = time.time()
    metrics = http_request.app.state.METRICS
    version_label = state["model_info"].get("version", "unknown")

    if not state["sbert_model"] or not isinstance(state["job_embeddings"], torch.Tensor):
        metrics["errors_total"].labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model or embeddings not loaded")

    try:
        # ========== STEP 1: INPUT VALIDATION WITH GUARDRAILS ==========
        policy_engine = state.get("policy_engine")
        
        if policy_engine:
            logger.info(f"Validating resume text ({len(request.resume_text)} chars)...")
            input_validation = policy_engine.validate_input(request.resume_text)
            
            if not input_validation.allowed:
                logger.warning(f"Resume input blocked: {input_validation.input_validation.violations}")
                metrics["errors_total"].labels(error_type="input_validation_failed").inc()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Resume validation failed",
                        "violations": input_validation.input_validation.violations,
                        "risk_level": input_validation.input_validation.risk_level.value,
                        "message": "Please review your resume for sensitive information or formatting issues."
                    }
                )
            
            # Use sanitized input if PII was masked
            validated_resume = input_validation.sanitized_input or request.resume_text
            
            # Log warnings if any
            if input_validation.input_validation.violations:
                logger.warning(
                    f"Resume validation warnings: {input_validation.input_validation.violations}"
                )
        else:
            validated_resume = request.resume_text
            logger.warning("Guardrails not initialized - skipping validation")

        # ========== STEP 2: ENCODE RESUME ==========
        resume_embedding = state["sbert_model"].encode(validated_resume, convert_to_tensor=True)
        
        # ========== STEP 3: CALCULATE SIMILARITY ==========
        cos_scores = util.cos_sim(resume_embedding, state["job_embeddings"])[0]
        k = min(request.top_n, len(state["df_job_description"]))
        top_results = torch.topk(cos_scores, k=k)

        # ========== STEP 4: PREPARE MATCHES ==========
        matches = [
            MatchResult(
                rank=i+1,
                job_title=state["df_job_description"].iloc[idx.item()]["Job Title"],
                similarity_score=score.item()
            )
            for i, (score, idx) in enumerate(zip(top_results[0], top_results[1]))
        ]

        # Update metrics
        duration = time.time() - start_time
        metrics["duration_seconds"].labels(model_version=version_label).observe(duration)
        metrics["requests_total"].labels(model_version=version_label, status="success").inc()
        if matches:
            metrics["similarity_score"].labels(model_version=version_label).observe(matches[0].similarity_score)

        return MatchResponse(matches=matches, model_info=state["model_info"])

    except HTTPException:
        raise
    except Exception as e:
        metrics["requests_total"].labels(model_version=version_label, status="error").inc()
        metrics["errors_total"].labels(error_type="matching_error").inc()
        logger.exception(f"Error during matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
def get_model_info():
    if state["sbert_model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_info": state["model_info"],
        "loaded": True,
        "job_embeddings_shape": list(state["job_embeddings"].shape) if state["job_embeddings"] is not None else None,
        "total_jobs_indexed": len(state["df_job_description"]) if state["df_job_description"] is not None else 0,
        "guardrails_enabled": state["policy_engine"] is not None,
    }

@app.get("/guardrails/metrics")
def get_guardrails_metrics():
    """Get guardrails metrics."""
    if state["policy_engine"]:
        return state["policy_engine"].get_metrics()
    return {"error": "Guardrails not initialized"}

@app.post("/guardrails/metrics/reset")
def reset_guardrails_metrics():
    """Reset guardrails metrics."""
    if state["policy_engine"]:
        state["policy_engine"].reset_metrics()
        return {"message": "Guardrails metrics reset successfully"}
    return {"error": "Guardrails not initialized"}

# --- Middleware ---
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    if hasattr(request.app.state, "METRICS"):
        request.app.state.METRICS["http_requests_duration"].labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).observe(duration)
    return response

# --- Run locally ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
