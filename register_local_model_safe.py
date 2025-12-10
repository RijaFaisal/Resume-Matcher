import mlflow
from mlflow import pyfunc
from mlflow.tracking import MlflowClient
import os, time, shutil

MLFLOW_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

class SimpleModel(pyfunc.PythonModel):
    def predict(self, context, model_input):
        # returns a fixed-length vector per input (10 placeholder jobs)
        return [0.5] * 10

save_path = "tmp_pyfunc_model"
if os.path.exists(save_path):
    shutil.rmtree(save_path)

pyfunc.save_model(path=save_path, python_model=SimpleModel())

model_uri = "file://" + os.path.abspath(save_path)
model_name = "resume-screener-sbert-pretrained"

print("Registering model from:", model_uri)
registered = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Requested registration: name={registered.name} version={registered.version}")

version = registered.version
print(f"Waiting for version {version} to become READY...")
for i in range(60):  # wait up to 60s
    try:
        mv = client.get_model_version(name=model_name, version=version)
        status = mv.status
        print(f" check {i+1}: status={status}")
        if status == "READY":
            print("Model version is READY.")
            break
    except Exception as e:
        print(" check error:", e)
    time.sleep(1)
else:
    raise RuntimeError("Timed out waiting for model version to be READY. Inspect MLflow server logs.")

print("Transitioning version to Production...")
client.transition_model_version_stage(name=model_name, version=version, stage="Production", archive_existing_versions=False)
print(f"Model {model_name} version {version} moved to Production.")
