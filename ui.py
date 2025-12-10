# ui.py
"""
Streamlit frontend for Resume Matcher.

Behavior:
- By default, uses a local model (or DummyModel fallback) to produce similarity scores.
- If environment variable BACKEND_URL is set (e.g. "http://backend:8000")
  and USE_DUMMY is not "true", the UI will call BACKEND_URL + "/predict" with JSON:
    {"resume_text": "<text>"}
  and expects a JSON response containing a list of scores under key "scores":
    {"scores": [0.1, 0.2, ...]}

Adjust parsing if your backend returns a different schema.
"""

import os
import io
import json
import tempfile
import traceback
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

# Use requests to call backend API
import requests

# Try to import your project model. If not available, use DummyModel fallback.
try:
    # Adjust import path if your model class/function lives elsewhere.
    # Example: from src.app.model import Model  (update if different)
    from src.app.model import Model  # optional, will fail if not present
    HAS_LOCAL_MODEL = True
except Exception:
    HAS_LOCAL_MODEL = False

# -------------------------
# DummyModel (fallback)
# -------------------------
class DummyModel:
    """A minimal dummy model that returns random similarity scores.
    This ensures the UI works even if the real model is not available.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def predict(self, resumes: List[str], job_descriptions: List[str]):
        # returns a pandas DataFrame-like object with one row per resume
        # and one column per job description. Here we produce one sample row.
        # We return a DataFrame with shape (len(resumes), len(job_descriptions))
        sims = []
        for _ in resumes:
            row = self.rng.rand(len(job_descriptions))
            sims.append(row)
        df = pd.DataFrame(sims, columns=[f"job_{i}" for i in range(len(job_descriptions))])
        return df

# instantiate local model (either real or dummy)
if HAS_LOCAL_MODEL:
    try:
        # If your Model requires a different construction, modify here.
        local_model = Model()
    except Exception:
        local_model = DummyModel()
else:
    local_model = DummyModel()

# -------------------------
# Helpers
# -------------------------
def load_job_titles_and_descriptions(path="job_title_des.csv"):
    """Load job titles and descriptions CSV. Expects columns 'job_title' and 'job_description'
    or similar. Returns two lists: titles, descriptions.
    """
    if not os.path.exists(path):
        st.warning(f"Job titles/descriptions file not found at {path}. Using placeholders.")
        # fallback placeholders
        titles = [f"Job {i+1}" for i in range(10)]
        descs = [f"Description for job {i+1}" for i in range(10)]
        return titles, descs
    try:
        df = pd.read_csv(path)
        # try to infer columns
        if "job_title" in df.columns and "job_description" in df.columns:
            titles = df["job_title"].astype(str).tolist()
            descs = df["job_description"].astype(str).tolist()
        elif "title" in df.columns and "description" in df.columns:
            titles = df["title"].astype(str).tolist()
            descs = df["description"].astype(str).tolist()
        else:
            # fallback: use first two columns
            cols = df.columns.tolist()
            titles = df[cols[0]].astype(str).tolist()
            descs = df[cols[1]] .astype(str).tolist() if len(cols) > 1 else [""] * len(titles)
        return titles, descs
    except Exception as e:
        st.error(f"Failed to load job_title_des.csv: {e}")
        return [], []

def build_results_dataframe(job_titles, job_descriptions, scores):
    """Construct a DataFrame from job titles, descriptions and scores (1D or array-like)."""
    try:
        # Accept pandas Series, numpy array, or list
        arr = np.array(scores).astype(float)
        # If backend/model returns a single vector of length n_jobs
        if arr.ndim == 1:
            return pd.DataFrame({
                "Job Title": job_titles,
                "Job Description": job_descriptions,
                "Similarity Score": arr
            })
        # If model returns shape (1, n_jobs) or (n_resumes, n_jobs) and we want first resume
        if arr.ndim == 2:
            arr1 = arr[0]
            return pd.DataFrame({
                "Job Title": job_titles,
                "Job Description": job_descriptions,
                "Similarity Score": arr1
            })
    except Exception:
        st.error("Error constructing results dataframe.")
        traceback.print_exc()
    # fallback empty
    return pd.DataFrame({
        "Job Title": job_titles,
        "Job Description": job_descriptions,
        "Similarity Score": [0.0] * len(job_titles)
    })

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("Resume Matcher")
st.write("Upload your resume (PDF / TXT) or paste its text, then find best matching job descriptions.")

# left column: upload / paste
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload resume (pdf/txt)", type=["pdf", "txt", "docx"], accept_multiple_files=False)
    paste_text = st.text_area("Or paste resume text here", height=250)
    top_n = st.slider("How many top matches to show", min_value=1, max_value=20, value=5)

    # Options
    st.markdown("### Options")
    use_dummy_env = os.getenv("USE_DUMMY", "true").lower()
    use_dummy_flag = use_dummy_env == "true"
    st.write(f"Using DummyModel by default? **{use_dummy_flag}** (set `USE_DUMMY=false` to call backend when BACKEND_URL is provided)")

    backend_env = os.getenv("BACKEND_URL", "").strip()
    if backend_env:
        st.write(f"Backend URL (from env): `{backend_env}`")
    else:
        st.write("No BACKEND_URL provided — UI will use local model.")

with col2:
    st.header("Job Descriptions")
    job_titles, job_descriptions = load_job_titles_and_descriptions()
    # show a sample table (first few)
    sample_df = pd.DataFrame({
        "Job Title": job_titles[:10],
        "Job Description": job_descriptions[:10]
    })
    st.dataframe(sample_df, height=300)

# Prepare resume text
resume_text = ""
if uploaded_file is not None:
    # Try to extract text from supported types
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".txt"):
            raw = uploaded_file.read().decode(errors="ignore")
            resume_text = raw
        elif fname.endswith(".pdf"):
            # lightweight approach: try to use PyPDF2 if available
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                pages = [p.extract_text() or "" for p in reader.pages]
                resume_text = "\n".join(pages)
            except Exception:
                # fallback: save file to temp and skip extraction
                st.warning("PDF text extraction failed (PyPDF2 not available or failed). Please paste text instead.")
                resume_text = ""
        elif fname.endswith(".docx"):
            try:
                import docx2txt
                # docx2txt expects a file path, so write to temp
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp.close()
                resume_text = docx2txt.process(tmp.name) or ""
            except Exception:
                st.warning("DOCX extraction failed. Please paste text instead.")
                resume_text = ""
        else:
            resume_text = uploaded_file.read().decode(errors="ignore")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        resume_text = ""
else:
    resume_text = paste_text or ""

# Button and prediction
if st.button("Find Best Matching Jobs"):
    if not resume_text or not resume_text.strip():
        st.warning("Please upload a file or paste resume content.")
    else:
        with st.spinner("Analyzing resume..."):
            scores = None
            # Use backend if BACKEND_URL is set and USE_DUMMY is not true
            if backend_env and not use_dummy_flag:
                try:
                    api_url = backend_env.rstrip("/") + "/predict"
                    payload = {"resume_text": resume_text}
                    resp = requests.post(api_url, json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    # Expecting {"scores": [...]}. Adapt as necessary.
                    if isinstance(data, dict) and "scores" in data:
                        scores = data["scores"]
                    else:
                        # If backend returns DataFrame-like (e.g., dict of lists), try to coerce
                        # Try to find the first list-like value
                        if isinstance(data, dict):
                            for v in data.values():
                                if isinstance(v, list):
                                    scores = v
                                    break
                        # if still not found, try reading as list
                        if scores is None and isinstance(data, list):
                            scores = data
                        if scores is None:
                            st.error("Unexpected backend response format. Falling back to local model.")
                            scores = None
                except Exception as e:
                    st.error(f"Error calling backend API: {e}")
                    # optional: show more detail in expander
                    with st.expander("Backend error (trace)"):
                        st.text(traceback.format_exc())
                    scores = None

            # If backend not used or failed, run local model
            if scores is None:
                try:
                    # local_model.predict expects (list_of_resumes, job_descriptions) and returns DF-like
                    pred_df = local_model.predict([resume_text], job_descriptions)
                    # support pandas dataframe or array-like
                    if hasattr(pred_df, "iloc"):
                        scores = pred_df.iloc[0].values.tolist()
                    else:
                        scores = np.array(pred_df).flatten().tolist()
                except Exception as e:
                    st.error(f"Local model prediction failed: {e}")
                    with st.expander("Model error (trace)"):
                        st.text(traceback.format_exc())
                    # fallback to zeros
                    scores = [0.0] * len(job_titles)

            # Build results table
            results_df = build_results_dataframe(job_titles, job_descriptions, scores)
            top_matches = results_df.sort_values(by="Similarity Score", ascending=False).head(top_n)

        st.success("Analysis complete — top matches below:")

        for idx, row in top_matches.iterrows():
            st.markdown("---")
            st.markdown(f"### {row['Job Title']}")
            # progress expects 0..1; if score looks like 0..100, normalize
            score_val = float(row["Similarity Score"])
            if score_val > 1.0:
                display_val = max(0.0, min(1.0, score_val / 100.0))
            else:
                display_val = max(0.0, min(1.0, score_val))
            st.progress(display_val)
            st.write(f"Similarity: **{score_val:.2%}**")
            with st.expander("Job description"):
                st.write(row["Job Description"])

# footer / debug
st.markdown("---")
if st.checkbox("Show debug info"):
    st.subheader("Debug info")
    st.write("BACKEND_URL:", backend_env)
    st.write("USE_DUMMY:", use_dummy_flag)
    st.write("Has local model:", HAS_LOCAL_MODEL)
    st.write("Number of job titles loaded:", len(job_titles))
