import os
import pandas as pd
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR  # Assuming CSVs are in root or downloaded there
VEC_DIR = os.path.join(ROOT_DIR, "vectorstore")
MODEL_NAME = "all-MiniLM-L6-v2"

# File Paths
FILES = {
    "jobs": os.path.join(DATA_DIR, "job_title_des.csv"),
    "qa": os.path.join(DATA_DIR, "Career QA Dataset.csv"),
    "resumes": os.path.join(DATA_DIR, "Resume.csv")
}

def build_index():
    print(f"🚀 Starting RAG Index Builder...")
    print(f"📂 Root Directory: {ROOT_DIR}")
    
    # 1. Load Data
    documents = []
    
    # Load Jobs
    if os.path.exists(FILES["jobs"]):
        print("Processing Jobs...")
        df_jobs = pd.read_csv(FILES["jobs"])
        # Format: "Job: [Title] - [Description]"
        for _, row in df_jobs.iterrows():
            text = f"Job Opportunity: {row.get('Job Title', '')}\nDetails: {row.get('Job Description', '')}"
            documents.append(text)
    else:
        print(f"⚠️ Warning: {FILES['jobs']} not found.")
        
    # Load QA
    if os.path.exists(FILES["qa"]):
        print("Processing QA Dataset...")
        df_qa = pd.read_csv(FILES["qa"])
        # Format: "Q: [Question] A: [Answer]"
        for _, row in df_qa.iterrows():
            text = f"Career Advice Q&A:\nQuestion: {row.get('question', '')}\nAnswer: {row.get('answer', '')}"
            documents.append(text)
    else:
        print(f"⚠️ Warning: {FILES['qa']} not found.")

    # Load Resumes (Sample limit to avoid memory issues if massive)
    if os.path.exists(FILES["resumes"]):
        print("Processing Resumes (limiting to 1000 samples for demo speed)...")
        try:
            # We limit to 1000 for this exercise to prevent massive RAM usage/long wait times
            # In production, this would be a full batch job
            df_resumes = pd.read_csv(FILES["resumes"]).head(1000) 
            for _, row in df_resumes.iterrows():
                # Format: "Sample Resume (ID: ...): [Content]"
                # Adjust column names based on actual CSV
                resume_str = row.get('Resume_str') or row.get('Resume') or str(row)
                text = f"Sample Resume Strategy (ID: {row.get('ID', 'N/A')}):\n{resume_str[:1000]}..." # Truncate for RAG context window
                documents.append(text)
        except Exception as e:
            print(f"❌ Error processing resumes: {e}")
    else:
        print(f"⚠️ Warning: {FILES['resumes']} not found.")
        
    print(f"✅ Total Documents to Index: {len(documents)}")
    
    if not documents:
        print("❌ No documents found. Aborting.")
        return

    # 2. Embed
    print("🧠 Loading Embedding Model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("🔄 Generating Embeddings (this may take a moment)...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. Build Index
    print("🏗️ Building FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 4. Save
    os.makedirs(VEC_DIR, exist_ok=True)
    
    index_path = os.path.join(VEC_DIR, "faiss.index")
    metadata_path = os.path.join(VEC_DIR, "metadata.pkl")
    
    print(f"💾 Saving to {VEC_DIR}...")
    faiss.write_index(index, index_path)
    
    with open(metadata_path, "wb") as f:
        pickle.dump(documents, f)
        
    print("🎉 Index successfully created!")

if __name__ == "__main__":
    build_index()
