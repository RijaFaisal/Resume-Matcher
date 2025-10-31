import mlflow
from mlflow import pyfunc
from mlflow.tracking import MlflowClient
import os

MLFLOW_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

# Define a tiny pyfunc model that returns 0.5 for each job (deterministic)
class SimpleModel(pyfunc.PythonModel):
    def predict(self, context, model_input):
        # model_input is a DataFrame or list-like; return fixed scores
        n = len(model_input) if hasattr(model_input, "__len__") else 1
        # return a 2D-like list where each row is a list of job scores;
        # but your UI expects model.predict([resume_text], job_descriptions) -> DataFrame with one row per resume
        # We'll just return one row of fixed scores; MLflow will save it as a pyfunc model.
        import numpy as np
        # we'll create a placeholder single-row output; the UI expects the shape later
        return [0.5] * 10  # 10 placeholder job scores

# Save the pyfunc model to a local folder
save_path = "tmp_pyfunc_model"
if os.path.exists(save_path):
    import shutil
    shutil.rmtree(save_path)

pyfunc.save_model(path=save_path, python_model=SimpleModel())

# Register the model in the MLflow Model Registry
model_uri = "file://" + os.path.abspath(save_path)
model_name = "resume-screener-sbert-pretrained"

print("Registering model from:", model_uri)
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print("Registered model name:", registered_model.name, "version:", registered_model.version)

# Transition the new version to Production
client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=False
)
print(f"Model {model_name} version {registered_model.version} moved to Production.")
