import os, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
import pandas as pd
from mlflow.models import infer_signature
import mlflow
import torch
import numpy as np

class Llama3bWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, context_window : int, device : str):
        self.context_window = context_window
        self.device = device

    def load_context(self, context):
        """
        Load the tokenizer and LLaMA model from the artifact directory.
        """
        model_dir = context.artifacts["hf_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True)

        # Decide device based on env variable
        inference_mode = os.getenv("INFERENCE", "false").lower() == "true"
        if inference_mode:
            print("Running in inference mode: loading model to CUDA")
            self.model.to(self.device)
        else:
            print("Not in inference mode: using CPU for model logging")
            self.model.to("cpu")

        self.model.eval()

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        prompts = model_input["input_text"].tolist()

        # Default generation parameters (can be overridden per-row)
        default_kwargs = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.0,
        }

        outputs = []
        for _, row in model_input.iterrows():
            prompt = row["input_text"]

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.context_window)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Merge default generation kwargs with row-specific overrides
            row_kwargs = {
                key: row[key] if key in row and pd.notna(row[key]) else default
                for key, default in default_kwargs.items()
            }

            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **row_kwargs)

            # Decode output tokens
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(generated_text)

        return pd.DataFrame({"output_text": outputs})


def get_signature():
    """
    Defines a flexible input schema allowing for optional generation-time parameters.
    """
    example_input = pd.DataFrame({
        "input_text": ["Translate English to German: Hello"],
        "max_length": [None],
        "temperature": [None],
        "top_p": [None],
        "top_k": [None],
        "do_sample": [None],
    })

    example_output = pd.DataFrame({"output_text": ["Hallo"]})
    signature = infer_signature(example_input, example_output)
    return signature

def sanitize_tokenizer_config(tokenizer):
    """
    Clean the tokenizer config so it's JSON serializable.
    Replaces all non-serializable values like `dtype` with string equivalents.
    """
    def sanitize(obj):
        if isinstance(obj, (torch.dtype, np.dtype)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(i) for i in obj]
        else:
            return obj

    tokenizer.init_kwargs = sanitize(tokenizer.init_kwargs)
    return tokenizer
