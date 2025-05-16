# mlops_sdk/src/mlflow_registry_manager.py
"""
Defines the class and provides APIs for registering ML models to the MLflow model registry
"""
import mlflow
import device_config

class MLflowRegistryManager:

    def __init__(self, mlflow_registry_name : str = None):
        self.cfg, self.mode = device_config.load_config()
        self.mlflow_registry_name = mlflow_registry_name or self.cfg.mlflow_registry_name

    def register(self, run_id : str, artifact_path : str = "model") -> str:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        registered_model = mlflow.register_model(model_uri=model_uri, name=self.mlflow_registry_name)
        return registered_model.version
    
    def set_model_version_tags(self, model_version: int, tags: dict):
        for key, value in tags.items():
            mlflow.set_model_version_tag(name = self.mlflow_registry_name, version = model_version, key = key, value = value)

    def set_registered_model_tags(self, tags : dict):
        for key, value in tags.items():
            mlflow.set_registered_model_tag(name=self.mlflow_registry_name, key=key, value=value)

    def get_model_uri(self, model_version : int):
        model_uri = f"models:/{self.mlflow_registry_name}"


def get_registry_manager(mlflow_registry_name : str):
    return MLflowRegistryManager(mlflow_registry_name = mlflow_registry_name)



