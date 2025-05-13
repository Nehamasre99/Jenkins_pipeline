# mlops_sdk/tests/test_mlflow_tracker.py
"""
Module containing unit tests for mlflow_tracker. Usage: pytest mlflow_tracker.py
"""


import sys
import os
import pytest
import importlib
from types import SimpleNamespace

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
    # SimpleNamespace: a lightweight class used for mock config objects.
    # Creates a fake config object and mode, mimicking what load_config() normally returns
    dummy_cfg = SimpleNamespace(
        tracking_url="http://localhost:5000",
        experiment_name="test-experiment",
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




def test_start_run(patched_mlflow_tracker, mocker):

    # Prevents actual communication with an MLflow server.
    # become no ops stubs
    # patch where a function is used, not defined (hence patching inside mlflow_trakcer.py file)

    #  mocker.patch("mlflow_tracker.mlflow.set_tracking_uri") replaces the actual mlflow.set_tracking_uri
    # function inside mlflow_tracker.py with a Mock object.
    # That mock function does nothing (i.e., it doesn't try to write to a  server, etc.), but:
    # It tracks if it was called
    # It records how it was called (arguments, how many times, etc.)

    mocker.patch("mlflow_tracker.weakref.finalize")  # disables finalization
    mocker.patch("mlflow_tracker.mlflow.set_tracking_uri")
    mocker.patch("mlflow_tracker.mlflow.set_experiment")
    mocker.patch("mlflow_tracker.mlflow.create_experiment")

    # patches the function get_experiment_by_name to get a fake experiment object
    # mocker.Mock() creates a dummy object that pretends to be a real object and supports attribute access like experiment_id.
    # Used to simulate MLflow returning an experiment object.

    mocker.patch("mlflow_tracker.mlflow.get_experiment_by_name", return_value=mocker.Mock(experiment_id="123"))
    mocker.patch("mlflow_tracker.mlflow.start_run")

    # create a Mock object for run and set it's run_id attribute with a dummy object
    mock_active_run = mocker.Mock()
    mock_active_run.info.run_id = "run_xyz"

    # Since mlflow.active_run() is called multiple times in the code and expects different run_ids each time
    # need to add side_effect and set what the run instances will be each time

    # First call to mlflow.active_run() returns None → simulates “no run is active.”
    # Second call to mlflow.active_run() returns a run with a fake run_id
    # Third call to mlflow.active_run() returns a run with a fake run_id 

    # Need to add more entries in the side_effect list for testing end_run(), and resume()
    mocker.patch("mlflow_tracker.mlflow.active_run", side_effect=[None, mock_active_run, mock_active_run])

    # Instantiates MLflowTracker with mocked dependencies and starts the run.
    tracker = patched_mlflow_tracker()
    tracker.start_run()

    # Verifies that the start_run() correctly updates the internal state.
    assert tracker.run_id == "run_xyz"
    assert tracker.started


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
    mocker.patch("mlflow_tracker.mlflow.active_run", return_value    = None)

    mock_resume = mocker.patch("mlflow_tracker.mlflow.start_run")

    # Create the tracker instance
    tracker = patched_mlflow_tracker()
    tracker.started = True
    tracker.run_id = "xyz"
    tracker.ended = True

    tracker.resume()

    mock_resume.assert_called_once()
    assert not tracker.ended

