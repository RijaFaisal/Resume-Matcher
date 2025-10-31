import mlflow
from mlflow import pyfunc
from mlflow.tracking import MlflowClient
import time, shutil, os

MLFLOW_URI = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

class SimpleModel(pyfunc.PythonModel):
    def predict(self, context, model_input):
        return [0.5] * 10

# Log model as a run artifact and register it
with mlflow.start_run() as run:
    artifact_path = 'model_art'
    mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=SimpleModel())
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    print('Logged model to run:', run_id)
    print('Registering model from uri:', model_uri)
    registered = mlflow.register_model(model_uri=model_uri, name='resume-screener-sbert-pretrained')
    version = registered.version
    print('Requested registration: version', version)

# Poll until READY
for i in range(60):
    try:
        mv = client.get_model_version(name='resume-screener-sbert-pretrained', version=version)
        status = getattr(mv, 'status', None)
        print(f'check {i+1}: status={status}')
        if status == 'READY':
            print('Model version is READY.')
            break
    except Exception as e:
        print('check error:', e)
    time.sleep(1)
else:
    raise RuntimeError('Timed out waiting for model version to be READY.')

# Promote to Production
client.transition_model_version_stage(name='resume-screener-sbert-pretrained', version=version, stage='Production', archive_existing_versions=False)
print(f'Model resume-screener-sbert-pretrained version {version} moved to Production.')
