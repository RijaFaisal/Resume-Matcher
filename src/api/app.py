import os
import pickle
import numpy as np
import faiss
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging

# Import guardrails
from src.guardrails import PolicyEngine, GuardrailsConfig, PolicyMode

# Import monitoring
from src.monitoring import (
    LLMMetricsTracker,
    LLMProvider,
    get_prometheus_metrics,
    get_evidently_monitor
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
#  SET YOUR GROQ API KEY HERE
# ===============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Print available models for debugging
try:
    models = client.models.list()
    print("Available Groq models:", models)
except Exception as e:
    logger.warning(f"Could not list Groq models: {e}")

# ===============================
#  GUARDRAILS CONFIGURATION
# ===============================
guardrails_config = GuardrailsConfig(
    mode=PolicyMode.BALANCED,
    enable_pii_detection=True,
    enable_injection_filter=True,
    enable_toxicity_filter=True,
    enable_hallucination_detector=True,
    mask_pii=True,
    toxicity_threshold=0.7,
    confidence_threshold=0.6,
    log_violations=True,
)
policy_engine = PolicyEngine(guardrails_config)

# ===============================
#  MONITORING CONFIGURATION
# ===============================
prometheus_metrics = get_prometheus_metrics()
evidently_monitor = get_evidently_monitor()

# ===============================
#  FASTAPI APP
# ===============================
app = FastAPI(
    title="RAG API with Guardrails",
    description="Retrieval-Augmented Generation API with comprehensive safety mechanisms",
    version="1.0.0"
)

# ===============================
#  LOAD VECTOR STORE
# ===============================
VEC_DIR = os.path.join(os.path.dirname(__file__), "../../vectorstore")

# Load FAISS index
faiss_index_path = os.path.join(VEC_DIR, "faiss.index")
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
index = faiss.read_index(faiss_index_path)

# Load documents metadata
metadata_path = os.path.join(VEC_DIR, "metadata.pkl")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
with open(metadata_path, "rb") as f:
    documents = pickle.load(f)  # list of strings or dicts corresponding to FAISS index

# ===============================
#  EMBEDDING FUNCTION
# ===============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

def embed(text: str):
    return embedder.encode(text, convert_to_numpy=True)

# ===============================
#  RAG RETRIEVAL
# ===============================
def retrieve(query, k=3):
    """Return top-k relevant documents from FAISS index as strings."""
    try:
        q_emb = embed(query).astype("float32")
        scores, ids = index.search(np.array([q_emb]), k)
        results = []
        for i in ids[0]:
            doc = documents[i]
            # Ensure returned result is always a string
            if isinstance(doc, str):
                results.append(doc)
            elif isinstance(doc, dict):
                # use text_snippet if text key doesn't exist
                results.append(doc.get("text_snippet", ""))
            else:
                results.append("")  # fallback for None or unknown type
        return results
    except Exception as e:
        print("Error in retrieval:", e)
        return [""] * k  # fallback empty context

# ===============================
#  LLM ANSWER GENERATION
# ===============================
def generate_answer(context, query):
    prompt = f"""
You are an AI assistant. Use ONLY the provided context to answer."

Context:
{context}

Question:
{query}

Answer:
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # updated model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        # Access content as attribute, not dict
        return response.choices[0].message.content
    except Exception as e:
        print("Error generating answer:", e)
        return "Error generating answer."

# ===============================
#  REQUEST BODY
# ===============================
class Query(BaseModel):
    question: str

# ===============================
#  API ROUTES
# ===============================
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "RAG API with Guardrails is running",
        "guardrails_enabled": True,
        "policy_mode": guardrails_config.mode.value
    }

@app.get("/guardrails/metrics")
def get_guardrails_metrics():
    """Get guardrails metrics."""
    return policy_engine.get_metrics()

@app.post("/guardrails/metrics/reset")
def reset_guardrails_metrics():
    """Reset guardrails metrics."""
    policy_engine.reset_metrics()
    return {"message": "Metrics reset successfully"}

@app.post("/ask")
def ask(payload: Query):
    """
    RAG endpoint with comprehensive guardrails and monitoring.
    
    Flow:
    1. Start metrics tracking
    2. Validate input (PII detection, prompt injection filter)
    3. Retrieve relevant documents from FAISS
    4. Generate answer using Groq LLM
    5. Moderate output (toxicity, hallucination detection)
    6. Record metrics and return safe, validated response
    """
    # Initialize metrics tracker
    metrics_tracker = LLMMetricsTracker()
    metrics_tracker.start_request(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
    
    try:
        query = payload.question
        metrics_tracker.record_input(query)
        
        # ========== STEP 1: INPUT VALIDATION ==========
        logger.info(f"Processing query: {query[:100]}...")
        
        input_validation = policy_engine.validate_input(query)
        
        if not input_validation.allowed:
            logger.warning(f"Input blocked: {input_validation.input_validation.violations}")
            metrics_tracker.record_guardrail_check(
                violations=len(input_validation.input_validation.violations),
                blocked=True
            )
            metrics_tracker.end_request(success=False)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Input validation failed",
                    "violations": input_validation.input_validation.violations,
                    "risk_level": input_validation.input_validation.risk_level.value,
                }
            )
        
        # Use sanitized input if PII was masked
        validated_query = input_validation.sanitized_input or query
        
        # Log warnings if any
        if input_validation.input_validation.violations:
            logger.warning(
                f"Input validation warnings: {input_validation.input_validation.violations}"
            )
            metrics_tracker.record_guardrail_check(
                violations=len(input_validation.input_validation.violations),
                blocked=False
            )

        # ========== STEP 2: RETRIEVE CONTEXT ==========
        docs = retrieve(validated_query)
        combined_context = "\n\n".join(docs)

        # ========== STEP 3: GENERATE ANSWER ==========
        answer = generate_answer(combined_context, validated_query)
        metrics_tracker.record_output(answer)

        # ========== STEP 4: OUTPUT MODERATION ==========
        output_moderation = policy_engine.moderate_output(answer, context=combined_context)
        
        if not output_moderation.allowed:
            logger.warning(f"Output blocked: {output_moderation.output_moderation.violations}")
            metrics_tracker.record_guardrail_check(
                violations=len(output_moderation.output_moderation.violations),
                blocked=True
            )
            
            # End tracking and record metrics
            llm_metrics = metrics_tracker.end_request(success=False)
            prometheus_metrics.record_request(llm_metrics)
            evidently_monitor.log_metrics(llm_metrics.to_dict())
            
            return {
                "question": query,
                "answer": output_moderation.filtered_output,
                "context_used": [],
                "moderation": {
                    "blocked": True,
                    "violations": output_moderation.output_moderation.violations,
                    "action": output_moderation.output_moderation.action.value,
                }
            }
        
        # Log warnings if any
        if output_moderation.output_moderation.violations:
            logger.warning(
                f"Output moderation warnings: {output_moderation.output_moderation.violations}"
            )
            metrics_tracker.record_guardrail_check(
                violations=len(output_moderation.output_moderation.violations),
                blocked=False
            )

        # ========== STEP 5: RECORD METRICS & RETURN ==========
        llm_metrics = metrics_tracker.end_request(success=True)
        prometheus_metrics.record_request(llm_metrics)
        evidently_monitor.log_metrics(llm_metrics.to_dict())
        
        return {
            "question": query,
            "context_used": docs,
            "answer": answer,
            "guardrails": {
                "input_validation": {
                    "passed": True,
                    "warnings": input_validation.input_validation.violations if input_validation.input_validation.violations else None,
                    "risk_level": input_validation.input_validation.risk_level.value,
                },
                "output_moderation": {
                    "passed": True,
                    "warnings": output_moderation.output_moderation.violations if output_moderation.output_moderation.violations else None,
                    "confidence_scores": output_moderation.output_moderation.confidence_scores,
                }
            },
            "metrics": {
                "latency_ms": llm_metrics.total_latency_ms,
                "tokens": llm_metrics.total_tokens,
                "cost": llm_metrics.total_cost,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Catch all errors to prevent 500 Internal Server Error
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        metrics_tracker.record_error(e)
        llm_metrics = metrics_tracker.end_request(success=False)
        prometheus_metrics.record_request(llm_metrics)
        evidently_monitor.log_metrics(llm_metrics.to_dict())
        
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        )
