from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
name = "resume-screener-sbert-pretrained"

try:
    versions = client.get_latest_versions(name)
    print("Found versions:", versions)
except Exception as e:
    print("ERROR checking versions:", e)
