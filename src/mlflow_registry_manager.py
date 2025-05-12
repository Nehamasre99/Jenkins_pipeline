# mlops_sdk/src/mlflow_registry_manager.py
"""
Defines the class and provides APIs for registering ML models to the MLflow model registry
"""
import mlflow


class MLflowRegistryManager:

    def __init__(self, registry_name : str):
        self.registry_name = registry_name

    def register(self, run_id : str, artifact_path : str = "model") -> str:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        registered_model = mlflow.register_model(model_uri=model_uri, name=self.registry_name)
        return registered_model.version
    
    def set_model_version_tags(self, model_name: str, model_version: int, tags: dict):
        for key, value in tags.items():
            mlflow.set_model_version_tag(model_name, model_version, key, value)

def get_registry_manager(registry_name : str):
    return MLflowRegistryManager(registry_name = registry_name)






