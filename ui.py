import streamlit as st
import pandas as pd
import sys
import os
import fitz  
import docx  


sys.path.append(os.path.abspath('src'))


st.set_page_config(
    page_title="Smart Resume Screener",
    page_icon="📄",
    layout="wide"
)


@st.cache_resource
def load_model():
    """
    Retrieves the singleton ResumeScreener model instance.
    The model is loaded only once when the application starts.
    """
    from app.model import model
    return model

@st.cache_data
def load_job_data(path: str) -> pd.DataFrame:
    """Loads the job descriptions data once and caches it."""
    return pd.read_csv(path)

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


st.title("AI based Smart Resume Screener")


try:
    model = load_model()
    df_jobs = load_job_data("job_title_des.csv")
    job_descriptions_list = df_jobs["Job Description"].tolist()
    job_titles = df_jobs["Job Title"].tolist()
except FileNotFoundError:
    st.error("Error: Make sure `job_title_des.csv` is in the root directory.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model from the MLflow Registry. Is the MLflow server running? Have you promoted the model to 'Production'?")
    st.info(f"Details: {e}")
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
        with st.spinner("Analyzing your resume..."):
            scores = model.predict([resume_text], job_descriptions_list).iloc[0].values
            
           
            results_df = pd.DataFrame({
                'Job Title': job_titles,
                'Job Description': job_descriptions_list, 
                'Similarity Score': scores
            })
            top_matches = results_df.sort_values(by="Similarity Score", ascending=False).head(top_n)

        st.success("Analysis complete!")
        for index, row in top_matches.iterrows():
            st.markdown("---")
            st.markdown(f"### **{row['Job Title']}**")
            st.progress(row['Similarity Score'])
            st.markdown(f"**Similarity:** `{row['Similarity Score']:.2%}`")
            
            
            with st.expander("View Job Description"):
                st.write(row['Job Description']) 