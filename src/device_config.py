# mlops_sdk/src/device.config.py

import os
import torch
import yaml
from typing import Optional, Union
from .config_schema import LocalMLflowConfig, RemoteMLflowConfig


def load_config(config_path: Optional[str] = None) -> tuple[Union[LocalMLflowConfig, RemoteMLflowConfig], str]:
    # Check if config path is provided in the environment variable and set it
    if config_path is None:
        config_path = os.environ.get("MLOPS_SDK_CONFIG_PATH") 

    # If config_path is still None
    if config_path is None:
        current_dir = os.path.dirname(__file__) # get current path of device.config.py
        # get a relative path two levels up and find mlflow_cnfig.yaml
        default_path = os.path.join(current_dir, "..", "mlflow_config.yaml")
        # Convert the above relative path to an absolute path
        config_path = os.path.abspath(default_path)
    # If config path is available from env variable or through argument
    # Convert it into absolute path as user could have passed a relative path
    else:
        config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
        mode = full_config.get("mode", "local")
        selected_config = full_config.get(f"mlflow_{mode}")

        if not selected_config:
            raise ValueError(f"No configuration found for mode '{mode}'")

        if mode == "remote":
            cfg =  RemoteMLflowConfig(**selected_config)
        elif mode == "local":
            cfg = LocalMLflowConfig(**selected_config)
        else:
            raise ValueError(f"Unsupported mode '{mode}' in config")

        return cfg, mode


def get_device(force_cpu: Optional[bool] = None) -> str:

    if force_cpu is not None:
        return "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    # Lazy-load config to read force_cpu
    cfg, mode = load_config()
    return "cpu" if cfg.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

def get_mode() -> str:

    # Lazy-load config to read mode
    cfg, mode = load_config()
    return mode

