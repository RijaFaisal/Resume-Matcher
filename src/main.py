"""
Main FastAPI application for the Resume Matcher system.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .model import ResumeMatcher
from .preprocessing import DocumentProcessor


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for model and processor
resume_matcher: ResumeMatcher = None
document_processor: DocumentProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global resume_matcher, document_processor
    
    # Startup
    logger.info("Starting Resume Matcher application...")
    
    try:
        # Initialize document processor
        document_processor = DocumentProcessor()
        logger.info("Document processor initialized")
        
        # Initialize resume matcher model
        resume_matcher = ResumeMatcher()
        await resume_matcher.load_model()
        logger.info("Resume matcher model loaded")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Resume Matcher application...")


# Create FastAPI application
app = FastAPI(
    title="Resume Matcher API",
    description="AI-powered resume matching system",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Resume Matcher API", "version": "0.1.0"}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/match")
async def match_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
    threshold: float = Form(0.7)
) -> Dict[str, Any]:
    """
    Match a resume against a job description.
    
    Args:
        resume_file: Uploaded resume file
        job_description: Job description text
        threshold: Matching threshold (0.0 to 1.0)
    
    Returns:
        Matching results with score and recommendations
    """
    try:
        # Validate file type
        if not resume_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = resume_file.filename.split('.')[-1].lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed"
            )
        
        # Validate file size
        content = await resume_file.read()
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds maximum allowed size"
            )
        
        # Process resume document
        resume_text = await document_processor.process_document(
            content, file_extension
        )
        
        # Perform matching
        match_result = await resume_matcher.match(
            resume_text=resume_text,
            job_description=job_description,
            threshold=threshold
        )
        
        return {
            "match_score": match_result["score"],
            "is_match": match_result["is_match"],
            "recommendations": match_result["recommendations"],
            "matched_skills": match_result["matched_skills"],
            "missing_skills": match_result["missing_skills"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in match_resume: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch-match")
async def batch_match_resumes(
    resume_files: list[UploadFile] = File(...),
    job_description: str = Form(...),
    threshold: float = Form(0.7)
) -> Dict[str, Any]:
    """
    Match multiple resumes against a job description.
    
    Args:
        resume_files: List of uploaded resume files
        job_description: Job description text
        threshold: Matching threshold (0.0 to 1.0)
    
    Returns:
        Batch matching results
    """
    try:
        results = []
        
        for resume_file in resume_files:
            try:
                # Process each resume
                content = await resume_file.read()
                file_extension = resume_file.filename.split('.')[-1].lower()
                
                resume_text = await document_processor.process_document(
                    content, file_extension
                )
                
                match_result = await resume_matcher.match(
                    resume_text=resume_text,
                    job_description=job_description,
                    threshold=threshold
                )
                
                results.append({
                    "filename": resume_file.filename,
                    "match_score": match_result["score"],
                    "is_match": match_result["is_match"],
                    "recommendations": match_result["recommendations"]
                })
                
            except Exception as e:
                logger.error(f"Error processing {resume_file.filename}: {e}")
                results.append({
                    "filename": resume_file.filename,
                    "error": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in batch_match_resumes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    try:
        return await resume_matcher.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/model/retrain")
async def retrain_model() -> Dict[str, str]:
    """Trigger model retraining."""
    try:
        # This would typically be an async task
        await resume_matcher.retrain()
        return {"message": "Model retraining initiated"}
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers if not settings.debug else 1
    )
