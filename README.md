## ☁️ Cloud Deployment

### Services Used
- **Amazon EC2:** Hosts both backend (FastAPI API) and frontend (Streamlit UI) containers.
- **Amazon ECR:** Stores Docker images for frontend and backend built by GitHub Actions.
- **Amazon S3:** Stores raw dataset (`job_title_des.csv`) accessed by backend during inference.
- **IAM Role:** `mlflow-s3-access-role` attached to EC2 grants secure S3 access.

### Architecture Overview
The workflow integrates AWS services as follows:

1. **Model & Data Storage:** Training data (`job_title_des.csv`) resides in an S3 bucket.
2. **Continuous Deployment:**  
   - On each `git push`, GitHub Actions builds and pushes Docker images to Amazon ECR.  
   - The EC2 instance then pulls these images and runs them via Docker.
3. **Serving:**  
   - Backend (FastAPI) runs on port 8000 and serves `/match_resume` API.  
   - Frontend (Streamlit) runs on port 8501, sends resume text to backend API, and displays job matches.

### How to Reproduce
1. Launch an EC2 instance with Ubuntu or Amazon Linux 2023.  
2. Install Docker and AWS CLI, then authenticate to ECR:  
   ```bash
   aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.eu-north-1.amazonaws.com

Pull and run images:

docker pull <ACCOUNT_ID>.dkr.ecr.eu-north-1.amazonaws.com/resume-frontend:<tag>
docker pull <ACCOUNT_ID>.dkr.ecr.eu-north-1.amazonaws.com/resume-backend:<tag>
docker run -d -p 8000:8000 resume-backend
docker run -d -p 8501:8501 resume-frontend

Interaction Between Components

The frontend sends resume text to http://<EC2_PUBLIC_IP>:8000/match_resume.
The backend loads job descriptions from S3, computes similarity scores, and returns results.
All logs and health checks are viewable via EC2 and Docker.