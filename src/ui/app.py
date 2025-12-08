import os
import time

import docx
import fitz
import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MATCH_ENDPOINT = f"{API_BASE}/match_resume"



def analyze_resume_with_retry(payload, max_retries=5, delay=2, timeout=30):
    for attempt in range(max_retries):
        try:
            resp = requests.post(MATCH_ENDPOINT, json=payload, timeout=timeout)
            if resp.status_code == 422:
                return False, None, f"Validation error: {resp.text}"
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.RequestException as e:
            if attempt >= max_retries - 1:
                return False, None, f"Backend not responding ({e})"
            time.sleep(delay)
    return False, None, "Backend service not responding"


st.set_page_config(page_title="Smart Resume Screener UI", page_icon="📄", layout="wide")



def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            return "".join(page.get_text() for page in doc)
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif uploaded_file.name.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


st.title("📄 Smart Resume Screener")

# Try multiple possible paths for the job descriptions file
job_file_paths = [
    "job_title_des.csv",  # Root directory
    "notebooks/job_title_des.csv",  # Notebooks directory
    "src/ui/notebooks/job_title_des.csv",  # UI notebooks directory
]

df_jobs = None
for path in job_file_paths:
    try:
        if os.path.exists(path):
            df_jobs = pd.read_csv(path)
            break
    except Exception:
        continue

if df_jobs is None:
    st.error("job_title_des.csv not found. Please ensure the file exists in one of these locations:")
    st.code("\n".join(job_file_paths))
    st.stop()

try:
    job_descriptions_dict = dict(zip(df_jobs["Job Title"], df_jobs["Job Description"]))
    job_titles = df_jobs["Job Title"].tolist()
except KeyError as e:
    st.error(f"CSV file missing required column: {e}")
    st.stop()

st.subheader("Provide Your Resume")
uploaded_file = st.file_uploader("Upload resume", type=["txt", "pdf", "docx"])
resume_text_area = st.text_area("Or paste resume content here:", height=200)

resume_text = ""
if uploaded_file:
    resume_text = extract_text_from_file(uploaded_file)
else:
    resume_text = resume_text_area

st.subheader("Matching Results")
top_n = st.number_input("Top matches to display:", min_value=1, max_value=20, value=5)

if st.button("Find Best Matching Jobs"):
    if not resume_text.strip():
        st.warning("Upload a file or paste content.")
    else:
        status_placeholder = st.empty()
        results_placeholder = st.empty()
        with status_placeholder:
            st.info("🔄 Starting backend analysis...")
        with st.spinner("Analyzing..."):
            payload = {"resume_text": resume_text, "top_n": top_n}
            success, result, error_msg = analyze_resume_with_retry(payload)
            if success:
                matches = result["matches"]
                results_df = pd.DataFrame(
                    [
                        {
                            "Job Title": m["job_title"],
                            "Rank": m["rank"],
                            "Similarity Score": m["similarity_score"],
                        }
                        for m in matches
                    ]
                )
                status_placeholder.empty()
                with results_placeholder:
                    st.success("✅ Analysis complete!")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader("🎯 Top Matching Jobs")
                        for _, row in results_df.iterrows():
                            with st.container():
                                st.markdown("---")
                                st.markdown(f"### #{row['Rank']} - {row['Job Title']}")
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    st.progress(row["Similarity Score"])
                                with c2:
                                    st.markdown(
                                        f"**Match:** `{row['Similarity Score']:.2%}`"
                                    )
                                with st.expander("🔍 View Job Description"):
                                    st.markdown(
                                        job_descriptions_dict.get(
                                            row["Job Title"], "Not available"
                                        )
                                    )
            else:
                with status_placeholder:
                    st.error(f"❌ {error_msg}")
                results_placeholder.empty()
