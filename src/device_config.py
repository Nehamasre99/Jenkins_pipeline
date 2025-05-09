# mlops_sdk/src/device.config.py

import os
import torch
import yaml
from .config_schema import LocalMLflowConfig, RemoteMLflowConfig

# Define the absolute path to the config file
current_dir = os.path.dirname(__file__)          # This is mlops_sdk/
parent_dir = os.path.abspath(os.path.join(current_dir, "..",".."))  # Go up two level
config_path = os.path.join(parent_dir, 'mlflow_config.yaml')   # Now look for config in project root

# Load the config file
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)
    mode = full_config.get("mode", "local")  # Default to 'local'
    selected_config = full_config.get(f"mlflow_{mode}")  # Select based on mode

    if mode == "remote":
        cfg = RemoteMLflowConfig(**selected_config)  # Validate and type-check
    elif mode == "local":
        cfg = LocalMLflowConfig(**selected_config)  # Validate and type-check

def get_device():
    """Returns 'cuda' if available and force_cpu is False, else 'cpu'."""
    return "cpu" if cfg.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
