import os
import pickle
import numpy as np
import faiss
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ===============================
#  SET YOUR GROQ API KEY HERE
# ===============================
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Print available models for debugging
models = client.models.list()
print("Available Groq models:", models)

# ===============================
#  FASTAPI APP
# ===============================
app = FastAPI()

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
#  API ROUTE
# ===============================
@app.post("/ask")
def ask(payload: Query):
    try:
        query = payload.question

        # 1. Retrieve top-k context documents
        docs = retrieve(query)
        combined_context = "\n\n".join(docs)

        # 2. Generate answer using Groq LLM
        answer = generate_answer(combined_context, query)

        return {
            "question": query,
            "context_used": docs,
            "answer": answer
        }
    except Exception as e:
        # Catch all errors to prevent 500 Internal Server Error
        print("Error in /ask endpoint:", e)
        return {"error": str(e)}
