import streamlit as st
import requests
import pandas as pd
import fitz
import docx
import os
import time

# Get the API URL from the environment variable set in docker-compose.yml
API_URL = os.getenv("API_URL", "http://resume_matcher_api:8000/match_resume")

def analyze_resume_with_retry(payload, max_retries=40, delay=5):
    """Try to analyze resume with retries, returns (success, result, error_msg)"""
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            return True, response.json(), None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(delay)
            continue
    return False, None, "Backend service is not responding. Please try again in a few minutes."

st.set_page_config(
    page_title="Smart Resume Screener UI",
    page_icon="📄",
    layout="wide"
)

def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            return "".join(page.get_text() for page in doc)
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif uploaded_file.name.endswith('.txt'):
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

st.title("📄 Smart Resume Screener")

try:
    df_jobs = pd.read_csv("data/raw/job_title_des.csv")
    # Create a dictionary mapping job titles to descriptions for easy lookup
    job_descriptions_dict = dict(zip(df_jobs["Job Title"], df_jobs["Job Description"]))
    job_titles = df_jobs["Job Title"].tolist()
except FileNotFoundError:
    st.error("Error: Make sure `job_title_des.csv` is in the `data/raw` directory.")
    st.stop()

st.subheader("Provide Your Resume")
uploaded_file = st.file_uploader("Upload your resume file", type=["txt", "pdf", "docx"])
resume_text_area = st.text_area("Or paste your resume content here:", height=200)

resume_text = ""
if uploaded_file:
    resume_text = extract_text_from_file(uploaded_file)
else:
    resume_text = resume_text_area

st.subheader("Matching Results")
top_n = st.number_input("Number of top matches to display:", min_value=1, max_value=20, value=5)

if st.button("Find Best Matching Jobs", type="primary"):
    if not resume_text or not resume_text.strip():
        st.warning("Please upload a file or paste resume content.")
    else:
        # Create placeholder for status messages and results
        status_placeholder = st.empty()
        results_placeholder = st.empty()
        
        with status_placeholder:
            st.info("🔄 Backend service is starting up... This may take a minute or two while the ML model loads...")
            
        # Show a spinner while we're trying
        with st.spinner("Running resume analysis..."):
            payload = {
                "resume_text": resume_text,
                "top_n": top_n
            }
            
            success, result, error_msg = analyze_resume_with_retry(payload)
            
            if success:
                matches = result["matches"]
                results_df = pd.DataFrame([{
                    'Job Title': match['job_title'],
                    'Rank': match['rank'],
                    'Similarity Score': match['similarity_score']
                } for match in matches])
                
                # Clear the status message
                status_placeholder.empty()
                
                with results_placeholder:
                    st.success("✅ Analysis complete!")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("🎯 Top Matching Jobs")
                        for index, row in results_df.iterrows():
                            with st.container():
                                st.markdown("---")
                                # Title and match score in the same line
                                st.markdown(f"### #{row['Rank']} - {row['Job Title']}")
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.progress(row['Similarity Score'])
                                with col_b:
                                    st.markdown(f"**Match:** `{row['Similarity Score']:.2%}`")
                                
                                # Expander with clear indication
                                with st.expander("� View Complete Job Description"):
                                    if row['Job Title'] in job_descriptions_dict:
                                        st.markdown(job_descriptions_dict[row['Job Title']])
                                    else:
                                        st.warning("Job description not available")
            else:
                # Update status message to show error
                with status_placeholder:
                    st.error(f"❌ {error_msg}")
                # Clear the results area if there was an error
                results_placeholder.empty()