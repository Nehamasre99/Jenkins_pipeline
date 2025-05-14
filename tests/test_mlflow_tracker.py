# mlops_sdk/tests/test_mlflow_tracker.py
"""
Module containing unit tests for mlflow_tracker. Usage: pytest mlflow_tracker.py
"""

import sys
import os
import importlib
from unittest.mock import patch
from types import SimpleNamespace
import pytest

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Global test constants
MOCK_EXPERIMENT_ID = "1234"
MOCK_RUN_ID = "run_xyz"

# A pytest fixture is a function that sets up preconditions for tests and returns objects needed by the tests.
# This one creates a mocked version of the MLflowTracker class with dependencies stubbed out.
# It takes mocker as an argument, which is provided by the pytest-mock plugin. mocker is a wrapper around Python’s unittest.mock library.

# During test discovery:
# When pytest scans your test file, it finds a function with @pytest.fixture.
# It registers this function (patched_mlflow_tracker) as a fixture provider — a reusable "test dependency" that can be injected into any test function.
# when patched_mlflow_tracker it can be called using patched_mlflow_tracker to give the return value
# which is the MlflowTracker object with mocked load_config

@pytest.fixture
def patched_mlflow_tracker(mocker):
    """
    Mimicks load_config() to return a dummy config, mode and imports MLflowTracker using it

    Returns:
    MLFlowTracker Class as a parameter which can be used in functions as an agument
    """

    # SimpleNamespace: a lightweight class used for mock config objects.
    # Creates a fake config object and mode, mimicking what load_config() normally returns
    dummy_cfg = SimpleNamespace(
        tracking_url="http://localhost:5000",
        experiment_name="test-experiment-1",
        log_system_metrics=True,
        run_name="unit-test-run",
        mlflow_s3_endpoint_url="http://1.1.1.1:9000",
        artifact_uri="s3://mlflow-artifacts/",
        aws_access_key_id="admin",
        aws_secret_access_key="pass",
        context_window = "512",
        inference_device = "cpu"
    )
    dummy_mode = "remote"

    # This replaces the actual function load_config() with a fake one that returns a known tuple.
    # mocker.patch() works like unittest.mock.patch():
    # It replaces the function at import time with a Mock or MagicMock
    mocker.patch("mlflow_tracker.device_config.load_config", return_value=(dummy_cfg, dummy_mode))

    # Imports mlflow_tracker only after mocking load_config() to ensure the class uses the mocked config.
    mlflow_tracker_module = importlib.import_module("mlflow_tracker")

    # Returns the MlFlowTracker class as a parameter which can passed in functions as an argument
    # so the variable patched_mlflow_tracker contains the MLflowTrackerClass
    # and calling patched_mlflow_tracker() creates an object of MLflowTrackerClass()
    # This behavior is because of the pytest.fixture decorator making it into a parameter that can be passed in fucntions
    # If it was a regular function, patched_mlflow_tracker() would return the class and patched_mlflow_config()() would be the object
    return mlflow_tracker_module.MLflowTracker

# Prevents actual communication with an MLflow server.
# become no ops stubs
# patch where a function is used, not defined (hence patching inside mlflow_trakcer.py file)

# mocker.patch("mlflow_tracker.mlflow.set_tracking_uri") replaces the actual mlflow.set_tracking_uri
# function inside mlflow_tracker.py with a Mock object.
# That mock function does nothing (i.e., it doesn't try to write to a  server, etc.), but:
# It tracks if it was called
# It records how it was called (arguments, how many times, etc.)


@pytest.fixture
def setup_start_run(mocker, patched_mlflow_tracker):

    # _setup is a factory fixture which allows the setup_start_run pyest.fixture to return a function.
    # This allows us to execute conditional logic at run time instead of writing two different versions of
    # setup_start_run(). _setup is the reference and can be called using setup_start_run(experiment_exists = True)
    # using var1, var2 .. = setup_start_run(experiment_exists = True) evaluates to 
    # var1, var2 ... = _setup(experiment_exists = True)
    def _setup(experiment_exists=True):
        mocker.patch("mlflow_tracker.weakref.finalize")
        mocker.patch("mlflow_tracker.mlflow.set_tracking_uri")
        mocker.patch("mlflow_tracker.mlflow.set_experiment")

        mock_create_experiment = mocker.patch("mlflow_tracker.mlflow.create_experiment")
        mock_experiment = mocker.Mock()
        mock_experiment.experiment_id = MOCK_EXPERIMENT_ID

        if experiment_exists:
            # Simulate experiment exists
            mock_get_experiment = mocker.patch(
                "mlflow_tracker.mlflow.get_experiment_by_name",
                return_value = mock_experiment
            )
        else:
            # Simulate experiment does not exist
            mock_get_experiment = mocker.patch(
                "mlflow_tracker.mlflow.get_experiment_by_name",
                side_effect = [None, mock_experiment, mock_experiment]
            )

        mock_start_run = mocker.patch("mlflow_tracker.mlflow.start_run")

        mock_active_run = mocker.Mock()
        mock_active_run.info.run_id = MOCK_RUN_ID

        # active_run is called 3 times and returns None first time, and a mock_active_run object 2 times
        # use side_effect list when the same function returns different objects on different calls
        # if it returns the same object on every call, use return_value = <object_name>
        mocker.patch("mlflow_tracker.mlflow.active_run", side_effect=[None, mock_active_run])

        tracker = patched_mlflow_tracker()

        return tracker, mock_create_experiment, mock_get_experiment, mock_start_run, mock_active_run

    return _setup


def check_remote_env_set(tracker):
    if tracker.mode == "remote":
        assert os.environ["MLFLOW_S3_ENDPOINT_URL"] == tracker.cfg.mlflow_s3_endpoint_url
        assert os.environ["AWS_ACCESS_KEY_ID"] == tracker.cfg.aws_access_key_id
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == tracker.cfg.aws_secret_access_key

    elif tracker.mode == "local":
        assert "MLFLOW_S3_ENDPOINT_URL" not in os.environ
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ


def generate_run_kwargs(tracker):
    run_kwargs = {
    "experiment_id": tracker.experiment.experiment_id,
    "log_system_metrics": tracker.cfg.log_system_metrics 
    }
    if tracker.cfg.run_name:
        run_kwargs["run_name"] = tracker.cfg.run_name
    return run_kwargs


def test_start_run_sunny_day_remote(mocker, setup_start_run):

    # use setup_start_run fixture to get the mock objects for testing
    tracker, mock_create_experiment, mock_get_experiment, mock_start_run, mock_active_run = setup_start_run(experiment_exists = False)

    tracker.call_create_experiment()
    run_kwargs = generate_run_kwargs(tracker)
    
    with patch.dict(os.environ, {}, clear=True):
    # start run using tracker.start_run()
        tracker.start_run()

        # Verifies that the start_run() correctly updates the internal state.
        mock_get_experiment.assert_called_with(tracker.cfg.experiment_name)
        mock_create_experiment.assert_called_once_with(tracker.cfg.experiment_name, artifact_location=tracker.cfg.artifact_uri)

        assert tracker.experiment.experiment_id == MOCK_EXPERIMENT_ID

        check_remote_env_set(tracker)

        mock_start_run.assert_called_once_with(**run_kwargs)

        assert tracker.run_id == MOCK_RUN_ID
        assert tracker.started
        assert not tracker.ended


def test_start_run_sunny_day_local(mocker, setup_start_run):

    # use setup_start_run fixture to get the mock objects for testing
    tracker, mock_create_experiment, mock_get_experiment, mock_start_run, mock_active_run = setup_start_run(experiment_exists = False)
    
    # override mode to local
    tracker.mode = "local"
    tracker.call_create_experiment()
    run_kwargs = generate_run_kwargs(tracker)

    with patch.dict(os.environ, {}, clear=True):
    # start run using tracker.start_run()
        tracker.start_run()

        # Verifies that the start_run() correctly updates the internal state.
        mock_get_experiment.assert_called_with(tracker.cfg.experiment_name)
        mock_create_experiment.assert_called_once_with(tracker.cfg.experiment_name)

        assert tracker.experiment.experiment_id == MOCK_EXPERIMENT_ID

        check_remote_env_set(tracker)

        mock_start_run.assert_called_once_with(**run_kwargs)

        assert tracker.run_id == MOCK_RUN_ID
        assert tracker.started
        assert not tracker.ended


def test_start_run_already_started(mocker, setup_start_run):

    # use setup_start_run fixture to get the mock objects for testing
    tracker, mock_create_experiment, mock_get_experiment, mock_start_run, mock_active_run = setup_start_run(experiment_exists = True)
    
    # set started to true to mimick an active run
    tracker.started = True

    # Check for runtime error because start_run() has been called for the second time
    with pytest.raises(RuntimeError, match="start_run\\(\\) has already been called. use resume\\(\\) instead"):
        tracker.start_run()


def test_start_run_experiment_exists(mocker, setup_start_run):

    # use setup_start_run fixture to get the mock objects for testing
    tracker, mock_create_experiment, mock_get_experiment, mock_start_run, mock_active_run = setup_start_run(experiment_exists = True)
    
    tracker.call_create_experiment()
    run_kwargs = generate_run_kwargs(tracker)
    
    with patch.dict(os.environ, {}, clear=True):
    # start run using tracker.start_run()
        tracker.start_run()

        # Verifies that the start_run() correctly updates the internal state.
        mock_get_experiment.assert_called_with(tracker.cfg.experiment_name)
        mock_create_experiment.assert_not_called()

        assert tracker.experiment.experiment_id == MOCK_EXPERIMENT_ID

        check_remote_env_set(tracker)

        mock_start_run.assert_called_once_with(**run_kwargs)

        assert tracker.run_id == MOCK_RUN_ID
        assert tracker.started
        assert not tracker.ended


def test_log_hparams(patched_mlflow_tracker, mocker):

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    # mock mlflow.log_params() to create a no ops stub
    mock_log_hparams = mocker.patch("mlflow_tracker.mlflow.log_params")
    # get patched tracker object
    tracker = patched_mlflow_tracker()
    # set started attribute to True to simulate an active run
    tracker.started = True  
    # Dummy dict constaining parameters to be logged
    params = {"lr": 0.001, "batch_size": 32}

    # Call the function under test with mocked mlflow.log_params()
    tracker.log_hparams(params)

    # Assert
    mock_log_hparams.assert_called_once_with(params)


def test_log_metrics_with_step(patched_mlflow_tracker, mocker):

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mock_log_metrics = mocker.patch("mlflow_tracker.mlflow.log_metrics")
    # get patched tracker object
    tracker = patched_mlflow_tracker()
    # set started attribute to True to simulate an active run
    tracker.started = True 

    metrics = {"accuracy": 0.97, "loss": 0.05}

    tracker.log_metrics(metrics, step=5)
    mock_log_metrics.assert_called_once_with(metrics, step = 5)


def test_log_metrics_without_step(patched_mlflow_tracker, mocker):

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mock_log_metrics = mocker.patch("mlflow_tracker.mlflow.log_metrics")
    # get patched tracker object
    tracker = patched_mlflow_tracker()
    # set started attribute to True to simulate an active run
    tracker.started = True 

    metrics = {"accuracy": 0.97, "loss": 0.05}

    tracker.log_metrics(metrics)
    mock_log_metrics.assert_called_once_with(metrics, step = None)


def test_log_model(patched_mlflow_tracker, mocker):

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mock_model = mocker.Mock()
    mock_tokenizer = mocker.Mock()

    mocker.patch("mlflow_tracker.os.path.exists", return_value = True)
    mocker.patch("mlflow_tracker.shutil.rmtree")

    # Mock save pretrained functions
    # Reason we are creating mock objects and not mocker.patch is because mock_model and 
    # mock_tokenizer are already mock objects. mocker.patch() is only for real fucntions
    # for ex mlflow.pyfunc.log_model()
    mock_model.save_pretrained = mocker.Mock()
    mock_tokenizer.save_pretrained = mocker.Mock()


    # Patch get_signature
    dummy_signature = mocker.Mock()
    mocker.patch("mlflow_tracker.get_signature", return_value=dummy_signature)

    mock_llm_wrapper = mocker.Mock()
    mocker.patch("mlflow_tracker.LlmWrapper", return_value=mock_llm_wrapper)

    mock_log_model = mocker.patch("mlflow_tracker.mlflow.pyfunc.log_model")

    # creater MLflowTracker() instance
    tracker = patched_mlflow_tracker()

    # Call the method under test
    tracker.log_model(mock_model, mock_tokenizer, safe_serialization=True)

    # Assert save_pretrained was called correctly
    mock_model.save_pretrained.assert_called_once_with("hf_model", safe_serialization=True)
    mock_tokenizer.save_pretrained.assert_called_once_with("hf_model")

    # Assert log_model was called with correct arguments
    mock_log_model.assert_called_once_with(
        artifact_path="model",
        python_model=mock_llm_wrapper,
        artifacts={"hf_model": "hf_model"},
        code_path=["./Wrapper.py"],
        signature=dummy_signature
    )


def test_end_active_run(patched_mlflow_tracker, mocker):
    # Create a mock active run
    mock_active_run = mocker.Mock()
    mock_active_run.info.run_id = "run_xyz"

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mocker.patch("mlflow_tracker.mlflow.active_run", side_effect = [None, mock_active_run])
    mock_end_run = mocker.patch("mlflow_tracker.mlflow.end_run")

    # Create the tracker
    tracker = patched_mlflow_tracker()
    tracker.started = True

    # Call end and assert
    tracker.end()

    # Check assertions
    assert tracker.ended
    mock_end_run.assert_called_once()


def test_end_no_active_run(patched_mlflow_tracker, mocker):

    
    # using return value = None makes it so that any number of calls tp mlflow.active_run() return None
    # Here active_run() is called thrice - in __init__ , end() and __del__ and should be returning None in all cases
    # side_effect = [] is used when the return values are different for different calls to the function

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mocker.patch("mlflow_tracker.mlflow.active_run", return_value = None)
    mock_end_run = mocker.patch("mlflow_tracker.mlflow.end_run")

    # Create the tracker
    tracker = patched_mlflow_tracker()
    
    #Escape ( and ) in regex: resume\(\) and start_run\(\)
    #Then escape each \ again in the Python string: \\( and \\)

    with pytest.raises(RuntimeError, match="Cannot call end\\(\\) before start_run\\(\\) or resume\\(\\) has been called."):
        tracker.end()


    mock_end_run.assert_not_called()


def test_resume_before_start_run(patched_mlflow_tracker, mocker):

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mocker.patch("mlflow_tracker.mlflow.active_run", return_value = None)
    mock_resume = mocker.patch("mlflow_tracker.mlflow.start_run")

    # Create the tracker instance
    tracker = patched_mlflow_tracker()
    tracker.run_id = "xyz"

    with pytest.raises(RuntimeError, match="Cannot call resume\\(\\) before start_run\\(\\) has been called."):
        tracker.resume()

    mock_resume.assert_not_called()


def test_resume_with_active_run(patched_mlflow_tracker, mocker):

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mock_active_run = mocker.Mock()

    # Third none in side_effect is to 
    mocker.patch("mlflow_tracker.mlflow.active_run", side_effect = [None, mock_active_run])

    mock_resume = mocker.patch("mlflow_tracker.mlflow.start_run")

    # Create the tracker instance
    tracker = patched_mlflow_tracker()
    tracker.started = True
    tracker.run_id = "xyz"

    with pytest.raises(RuntimeError, match="Cannot call resume\\(\\) while there is an active mlflow run."):
        tracker.resume()

    mock_resume.assert_not_called()


def test_resume_without_active_run(patched_mlflow_tracker, mocker):

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mock_active_run = mocker.Mock()
    mocker.patch("mlflow_tracker.mlflow.active_run", return_value = None)

    mock_resume = mocker.patch("mlflow_tracker.mlflow.start_run")

    # Create the tracker instance
    tracker = patched_mlflow_tracker()
    tracker.started = True
    tracker.run_id = "xyz"
    tracker.ended = True

    tracker.resume()

    mock_resume.assert_called_once()
    assert not tracker.ended

