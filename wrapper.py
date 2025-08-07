# Wrapper.py
import mlflow
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def get_signature():
    """Defines the input and output schema for the model."""
    input_schema = mlflow.types.Schema([
        mlflow.types.ColSpec("string", "text")
    ])
    output_schema = mlflow.types.Schema([
        mlflow.types.ColSpec("string", "predictions")
    ])
    return mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)

class LlmWrapper(mlflow.pyfunc.PythonModel):
    """A wrapper for the Hugging Face model to conform to MLflow's format."""
    def __init__(self, context_window, device):
        self.context_window = int(context_window) if context_window else 512
        self.device = device or "cpu"
        self.model = None
        self.tokenizer = None

    def load_context(self, context):
        """This method is called when the model is loaded for inference."""
        model_path = context.artifacts["hf_model"]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, context, model_input):
        """This method is called for generating predictions."""
        texts = model_input[model_input.columns[0]].tolist()
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_window
        ).to(self.device)
        
        outputs = self.model.generate(**inputs)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return pd.DataFrame(decoded_outputs, columns=["predictions"])
