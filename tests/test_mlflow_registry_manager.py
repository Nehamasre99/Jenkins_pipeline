# mlops_sdk/tests/test_registry_manager.py

import sys
import os
import importlib
from types import SimpleNamespace
import pytest

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Global test constants
MOCK_ARTIFACT_PATH = "model"
MOCK_RUN_ID = "run_xyz"
MOCK_VERSION = "1"
MOCK_REGISTRY_NAME = "test_registry"

@pytest.fixture
def patched_mlflow_registry_manager(mocker):
    dummy_cfg = SimpleNamespace(
        mlflow_registry_name = MOCK_REGISTRY_NAME
    )
    dummy_mode = "remote"

    mocker.patch("mlflow_registry_manager.device_config.load_config", return_value=(dummy_cfg, dummy_mode))

    mlflow_registry_module = importlib.import_module("mlflow_registry_manager")

    return mlflow_registry_module.MLflowRegistryManager


def test_register(patched_mlflow_registry_manager, mocker):

	mock_model = mocker.Mock()
	mock_register_model = mocker.patch("mlflow_registry_manager.mlflow.register_model", return_value = mock_model)
	mock_model.version = MOCK_VERSION
	mock_model_uri = model_uri = f"runs:/{MOCK_RUN_ID}/{MOCK_ARTIFACT_PATH}"

	registry_manager = patched_mlflow_registry_manager()

	version = registry_manager.register(run_id = MOCK_RUN_ID, artifact_path = MOCK_ARTIFACT_PATH)

	assert registry_manager.mlflow_registry_name == MOCK_REGISTRY_NAME
	assert version == MOCK_VERSION

	mock_register_model.assert_called_once_with(model_uri=model_uri, name = MOCK_REGISTRY_NAME)