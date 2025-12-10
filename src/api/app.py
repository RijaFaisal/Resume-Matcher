import os
import pickle
import time
import logging
import numpy as np
import pandas as pd
import torch
import faiss
from groq import Groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import json
import base64
import re
import uuid
from fpdf import FPDF

# Import guardrails
from src.guardrails import PolicyEngine, GuardrailsConfig, PolicyMode

# Import monitoring
from src.monitoring import (
    LLMMetricsTracker,
    LLMProvider,
    get_prometheus_metrics,
    get_evidently_monitor
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================
#  CONFIGURATION
# ===============================
class AppConfig:
    # RAG Config
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # Matcher Config (Prefer local files)
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    VEC_DIR = os.path.join(ROOT_DIR, "vectorstore")
    NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")
    
    FAISS_INDEX_PATH = os.path.join(VEC_DIR, "faiss.index")
    METADATA_PATH = os.path.join(VEC_DIR, "metadata.pkl")
    
    JOB_EMBEDDINGS_PATH = os.path.join(NOTEBOOKS_DIR, "job_embeddings.pt")
    JOB_DATA_PATH = os.path.join(ROOT_DIR, "job_title_des.csv")
    
    MODEL_NAME = "all-MiniLM-L6-v2"

CONFIG = AppConfig()

# Initialize Groq Client
if not CONFIG.GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY is missing! Chatbot validation will fail.")
    client = None
else:
    client = Groq(api_key=CONFIG.GROQ_API_KEY)

# ===============================
#  GLOBAL STATE
# ===============================
state = {
    "sbert_model": None, # Shared model
    "job_embeddings": None,
    "df_job_description": None,
    "faiss_index": None,
    "documents": None,
    "policy_engine": None,
    "model_info": {"model_name": CONFIG.MODEL_NAME, "version": "2.0"}
}

# ===============================
#  INITIALIZATION
# ===============================
def init_guardrails():
    """Initialize guardrails policy engine."""
    guardrails_config = GuardrailsConfig(
        mode=PolicyMode.BALANCED,
        enable_pii_detection=True,
        enable_injection_filter=True,
        enable_toxicity_filter=True,
        enable_hallucination_detector=False, # Disabled for simpler local runs
        mask_pii=True,
        log_violations=True,
    )
    state["policy_engine"] = PolicyEngine(guardrails_config)
    logger.info("✅ Guardrails initialized")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Unified Backend...")
    start_time = time.time()
    
    # 1. Load SBERT Model (Shared)
    try:
        logger.info(f"Loading SBERT model: {CONFIG.MODEL_NAME}")
        state["sbert_model"] = SentenceTransformer(CONFIG.MODEL_NAME)
        logger.info("✅ SBERT model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load SBERT model: {e}")

    # 2. Load Matcher Data (Job Embeddings & Descriptions)
    try:
        if os.path.exists(CONFIG.JOB_EMBEDDINGS_PATH):
            state["job_embeddings"] = torch.load(CONFIG.JOB_EMBEDDINGS_PATH)
            logger.info(f"✅ Job embeddings loaded from {CONFIG.JOB_EMBEDDINGS_PATH}")
        else:
            logger.warning(f"⚠️ Job embeddings not found at {CONFIG.JOB_EMBEDDINGS_PATH}")

        if os.path.exists(CONFIG.JOB_DATA_PATH):
            state["df_job_description"] = pd.read_csv(CONFIG.JOB_DATA_PATH)
            logger.info(f"✅ Job descriptions loaded ({len(state['df_job_description'])} rows)")
        else:
            logger.warning(f"⚠️ Job descriptions not found at {CONFIG.JOB_DATA_PATH}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load Matcher data: {e}")

    # 3. Load RAG Data (FAISS & Metadata)
    try:
        if os.path.exists(CONFIG.FAISS_INDEX_PATH):
            state["faiss_index"] = faiss.read_index(CONFIG.FAISS_INDEX_PATH)
            logger.info("✅ FAISS index loaded")
        else:
            logger.error(f"❌ FAISS index not found at {CONFIG.FAISS_INDEX_PATH}")

        if os.path.exists(CONFIG.METADATA_PATH):
            with open(CONFIG.METADATA_PATH, "rb") as f:
                state["documents"] = pickle.load(f)
            logger.info("✅ Document metadata loaded")
        else:
            logger.error(f"❌ Metadata not found at {CONFIG.METADATA_PATH}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load RAG vectorstore: {e}")

    # 4. Init Guardrails
    init_guardrails()
    
    logger.info(f"✅ Startup complete in {time.time() - start_time:.2f}s")
    yield
    logger.info("🛑 Shutting down...")

# ===============================
#  FASTAPI APP
# ===============================
app = FastAPI(
    title="Resume Matcher & Chatbot API",
    description="Unified API for Resume Matching and RAG Chatbot",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
#  MODELS
# ===============================
# Chatbot Models
class Query(BaseModel):
    question: str
    user_context: Optional[str] = None

# Matcher Models
class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=10)
    top_n: int = Field(5, gt=0, le=50)

class MatchResult(BaseModel):
    rank: int
    job_title: str
    similarity_score: float

class MatchResponse(BaseModel):
    matches: List[MatchResult]
    model_info: Dict[str, str]

# ===============================
#  HELPER FUNCTIONS
# ===============================
def embed(text: str):
    """Embed text using shared SBERT model."""
    if not state["sbert_model"]:
        raise ValueError("SBERT model not initialized")
    return state["sbert_model"].encode(text, convert_to_numpy=True)

def retrieve(query, k=3):
    """Retrieve documents from FAISS."""
    if not state["faiss_index"] or not state["documents"]:
        return []
    try:
        q_emb = embed(query).astype("float32")
        scores, ids = state["faiss_index"].search(np.array([q_emb]), k)
        results = []
        for i in ids[0]:
            if i < len(state["documents"]):
                doc = state["documents"][i]
                if isinstance(doc, str): results.append(doc)
                elif isinstance(doc, dict): results.append(doc.get("text_snippet", ""))
        return results
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        logger.error(f"Retrieval error: {e}")
        return []

def generate_pdf_resume(data: Dict) -> str:
    """Generate a simple PDF resume and return base64 string."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 10, data.get("name", "Name Placeholder"), new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.set_font("Helvetica", "", 10)
        contact = f"{data.get('email', '')} | {data.get('phone', '')} | {data.get('location', '')}"
        pdf.cell(0, 10, contact, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)
        
        # Helper to add section
        def add_section(title, content):
            if not content: return
            
            # Ensure content is string (LLM might return list)
            if isinstance(content, list):
                content = "\n".join([str(item) for item in content])
            elif not isinstance(content, str):
                content = str(content)
                
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(0, 8, title.upper(), new_x="LMARGIN", new_y="NEXT", fill=False)
            pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2)
            
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 5, content)
            pdf.ln(5)

        add_section("Professional Summary", data.get("summary", ""))
        add_section("Skills", data.get("skills", ""))
        add_section("Experience", data.get("experience", ""))
        add_section("Education", data.get("education", ""))
        
        # Output to bytes
        pdf_bytes = pdf.output()
        return base64.b64encode(pdf_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"PDF Generation Error: {e}")
        return ""

from src.rag.prompts import SYSTEM_ROLE, PROMPT_INSTRUCTION, get_chat_prompt

def generate_answer(context, query, user_context=None):
    """Generate answer using Groq."""
    
    prompt = get_chat_prompt(SYSTEM_ROLE, PROMPT_INSTRUCTION, context, user_context, query)

    try:
        if not client:
             return "I cannot answer this question because the LLM service is currently unavailable (API Key missing)."

        response = client.chat.completions.create(
            model=CONFIG.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Groq API error occurred (check configuration/key)")
        # Optionally print type of e if needed, but keep it safe
        # logger.error(f"Error details: {str(e)}") 
        return "I encountered an error generating the answer (API Error)."

# ===============================
#  ENDPOINTS
# ===============================
@app.get("/")
def root():
    return {"status": "online", "services": ["chatbot", "matcher"]}

@app.get("/health")
def health():
    return {
        "status": "healthy" if state["sbert_model"] else "degraded",
        "ag_rag_ready": state["faiss_index"] is not None,
        "matcher_ready": state["job_embeddings"] is not None
    }

@app.post("/match_resume", response_model=MatchResponse)
def match_resume(request: MatchRequest):
    """Unified Resume Matching Endpoint."""
    if not state["sbert_model"] or state["job_embeddings"] is None:
        raise HTTPException(status_code=503, detail="Matcher services not ready (model/embeddings missing)")
    
    try:
        # validate input via guardrails
        if state["policy_engine"]:
             val_res = state["policy_engine"].validate_input(request.resume_text)
             # Log but don't strictly block for matcher unless critical?
             # Resume matching usually safe. Use sanitized.
             text_to_process = val_res.sanitized_input or request.resume_text
        else:
             text_to_process = request.resume_text

        # Embed resume
        resume_embedding = state["sbert_model"].encode(text_to_process, convert_to_tensor=True)
        
        # Calculate Cosine Sim
        cos_scores = util.cos_sim(resume_embedding, state["job_embeddings"])[0]
        
        # Get Top K
        k = min(request.top_n, len(state["df_job_description"]))
        top_results = torch.topk(cos_scores, k=k)
        
        matches = []
        for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            job_title = state["df_job_description"].iloc[idx.item()]["Job Title"]
            matches.append(MatchResult(
                rank=i+1,
                job_title=job_title,
                similarity_score=float(score.item())
            ))
            
        return MatchResponse(matches=matches, model_info=state["model_info"])

    except Exception as e:
        logger.exception(f"Matching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(payload: Query):
    """Unified RAG Chatbot Endpoint."""
    metrics_tracker = LLMMetricsTracker()
    metrics_tracker.start_request(provider=LLMProvider.GROQ, model=CONFIG.GROQ_MODEL)
    
    try:
        query = payload.question
        
        # 1. Validate
        if state["policy_engine"]:
            val = state["policy_engine"].validate_input(query)
            # Fix: Use PolicyEngine's decision and correct attribute access
            if not val.allowed:
                 violations = val.input_validation.violations if val.input_validation else []
                 return {
                     "answer": "I cannot answer that due to safety policies.",
                     "guardrails": {"blocked": True, "violations": violations}
                 }
            query = val.sanitized_input or query
            
        # 2. Retrieve & Generate
        docs = retrieve(query)
        context = "\n\n".join(docs) if docs else ""
        raw_answer = generate_answer(context, query, payload.user_context)
        
        # Check for JSON intent (PDF Generation)
        final_answer = raw_answer
        pdf_data = None
        
        try:
            # clean possible markdown wrappers
            clean_answer = raw_answer.strip()
            if clean_answer.startswith("```json"):
                clean_answer = clean_answer.replace("```json", "").replace("```", "")
            
            if "generate_resume" in clean_answer[:100]: # Hint check
                parsed = json.loads(clean_answer)
                if parsed.get("action") == "generate_resume":
                    final_answer = parsed.get("advice_text", "Here is your edited resume.")
                    resume_data = parsed.get("data", {})
                    pdf_data = generate_pdf_resume(resume_data)
        except json.JSONDecodeError:
            pass # Not JSON, treat as text
        
        # 3. Moderate Output
        if state["policy_engine"]:
            mod = state["policy_engine"].moderate_output(final_answer, context)
            if not mod.allowed:
                final_answer = "[Output blocked: content policy violation]"
        
        metrics_tracker.end_request(success=True)
        return {
            "question": payload.question,
            "answer": final_answer,
            "context_used": docs,
            "generated_pdf": pdf_data
        }
        
    except Exception as e:
        metrics_tracker.record_error(e)
        raise HTTPException(status_code=500, detail=str(e))
