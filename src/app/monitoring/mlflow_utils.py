import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os
import tempfile


class SBERTWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """This method is called when the model is loaded for inference."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = SentenceTransformer(
            context.artifacts["sbert_model_path"],
            device=device
        )
        print("Pre-trained SBERT model loaded successfully from artifacts.")

    def predict(self, context, model_input):
        """This method is called for predictions."""
        sentences = model_input["text"].tolist()
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings.cpu().numpy()


def log_pretrained_sbert_model():
    """
    Downloads a pre-trained SBERT model from Hugging Face and logs it to MLflow.
    """
    registered_model_name = "resume-screener-sbert-pretrained"

    mlflow.set_experiment("Resume Screener Pre-trained")
    with mlflow.start_run(run_name="Log_all-MiniLM-L6-v2") as run:

        model_name_hf = "sentence-transformers/all-MiniLM-L6-v2"
        mlflow.log_param("model_name", model_name_hf)

        print(f"Logging and registering pre-trained model as '{registered_model_name}'...")

        
        with tempfile.TemporaryDirectory() as tmpdir:
            
            model_to_save = SentenceTransformer(model_name_hf)
            
            local_model_path = os.path.join(tmpdir, "sbert_model_local")
           
            model_to_save.save(local_model_path)
            print(f"Model downloaded and saved temporarily to: {local_model_path}")

            
            artifacts = {
                "sbert_model_path": local_model_path
            }

            
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SBERTWrapper(),
                artifacts=artifacts,
                registered_model_name=registered_model_name
            )

        print("Pre-trained model registration complete.")

if __name__ == "__main__":
    log_pretrained_sbert_model()