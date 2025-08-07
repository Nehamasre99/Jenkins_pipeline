# mlops_sdk/src/config_schema.py

from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal


class BaseMLflowConfig(BaseModel):
    mode: Literal["remote", "local", ""] = ""
    model_name: str
    experiment_name: str
    run_name: Optional[str]
    mlflow_registry_name: str
    tracking_url: HttpUrl
    trained_model_dir: str
    log_system_metrics: bool
    force_cpu: bool
    context_window: int
    inference_device: Optional[str] = "cpu"


class RemoteMLflowConfig(BaseMLflowConfig):
    artifact_uri: str
    mlflow_s3_endpoint_url: HttpUrl
    aws_access_key_id: str
    aws_secret_access_key: str
    inference_device: str

class LocalMLflowConfig(BaseMLflowConfig):
    pass  # No extra fields required
