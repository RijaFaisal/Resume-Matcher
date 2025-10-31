# 🧭 D7 API Documentation

## 📘 Overview
The **Resume Matcher Inference API** is built with **FastAPI** and serves as the interface between the trained NLP model and client applications (e.g., Streamlit UI or CI/CD pipelines).

This API allows users to send a resume and a job description, and returns a **match score (0–1)** that represents how closely the resume aligns with the job posting.

🔗 **Public Swagger Docs:** [http://16.16.197.220:8000/docs](http://16.16.197.220:8000/docs)  
📘 **ReDoc (Alternate UI):** [http://16.16.197.220:8000/redoc](http://16.16.197.220:8000/redoc)

---

## ⚙️ Example Endpoint

### **`POST /predict`**
Performs an inference using the trained NLP model and returns a similarity score.

**Description:**  
Accepts a candidate’s resume text and a job description, and returns a numerical match score between 0 (no match) and 1 (perfect match).

---

### 🧾 Example Request (cURL)
```bash
curl -X 'POST' \
  'http://16.16.197.220:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "resume_text": "Experienced data scientist skilled in Python, ML, and NLP.",
    "job_description": "Looking for an NLP engineer with strong Python experience."
}'
'''

### 🧾 Example Response 
```bash{
  "match_score": 0.82,
  "input": {
    "resume_text": "Experienced data scientist skilled in Python, ML, and NLP.",
    "job_description": "Looking for an NLP engineer with strong Python experience."
  }
}