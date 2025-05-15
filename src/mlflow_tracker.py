# mlops_sdk/src/mlflow_tracker.py
"""
Defines the class and provides APIs for MLFlow experiment tracking and logging
"""

import os
import sys
import weakref
import shutil
import mlflow
import device_config

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Wrapper import LlmWrapper, get_signature

class MLflowTracker:
    """
    MLFlowTracker APIS:
    - start_run()
    - log_hparams()
    - log_metrics()
    - end()
    - resume()
    """

    def __init__(self):
        # initialize run_id variable to None. Will be updated after calling start_run()
        self.experiment = None
        self.run_id = None
        self.started = False
        self.ended = False
        self.cfg, self.mode = device_config.load_config()
        if mlflow.active_run():
            raise RuntimeError(
                "Another MLflow run is already active. "
                "Please call end() on the existing tracker instance before creating a new one."
            )
        self.finalizer = weakref.finalize(self, self._finalize_run)

    def _finalize_run(self):
        if not self.ended and mlflow.active_run():
            print("Auto-ending active MLflow run.") 
            mlflow.end_run()

    def _create_experiment(self):
        """
        Internal function to create a new experiment using the name provided in config file
        """

        # retrieve tracking_uri from mlflow_config.yaml and set it for the current run
        mlflow.set_tracking_uri(str(self.cfg.tracking_url))

        # check if an experiment already exists with experiment_name
        experiment = mlflow.get_experiment_by_name(self.cfg.experiment_name)

        # if no experiment exists with experiment_name, create a new one
        if experiment is None:
            if self.mode == "remote": # if mode is remote
                mlflow.create_experiment(self.cfg.experiment_name, artifact_location=self.cfg.artifact_uri)
            elif self.mode == "local": # if mode is local
                mlflow.create_experiment(self.cfg.experiment_name)

            print(f"Experiment '{self.cfg.experiment_name}' created.")
            # update experiment variable with newly created experiment
            experiment = mlflow.get_experiment_by_name(self.cfg.experiment_name)

        # if experiment already exists with experiment_name, no action is needed
        else:
            print(f"Experiment '{self.cfg.experiment_name}' already exists with ID: {experiment.experiment_id}")
        
        # set the experiment attribute at the end of the fucntion to avoid any intermediate
        # states where the object would return None
        self.experiment = experiment
        # retrieve experiment_name from mlflow_config.yaml and set it for the current run
        mlflow.set_experiment(self.cfg.experiment_name)

    def _set_remote_env_vars(self):
        """
        Internal function to set remote environment variables for 
        MLflow Tracking and minIO servers from mlflow_config.yaml
        """
        # set remote only environment variables
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = str(self.cfg.mlflow_s3_endpoint_url)
        os.environ["AWS_ACCESS_KEY_ID"] = self.cfg.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.cfg.aws_secret_access_key

    def _create_run(self):
        """
        Starts a run and uses the name from mlflow_config.yaml if provided
        """
        # define kwargs for creating a new run
        run_kwargs = {
            "experiment_id": self.experiment.experiment_id,
            "log_system_metrics": self.cfg.log_system_metrics 
        }
        if self.cfg.run_name:
            # if run_name is provided by user in mlflow_config.yaml, add it to kwargs
            run_kwargs["run_name"] = self.cfg.run_name

        # start a new run with the generated kwargs
        mlflow.start_run(**run_kwargs)

        self.run_id = mlflow.active_run().info.run_id # retrieve current run_id and store it
        self.started = True
        self.ended = False


    def call_create_experiment(self):
        return self._create_experiment()


    def start_run(self):
        """
        Creates MLFLow experiment with given experiment_name and starts a run in that experiment
        """
        # Check if start_run() has already been called. 
        if self.started:
            raise RuntimeError("start_run() has already been called. use resume() instead")

        # set environemnt vars for Mlflow tracking and minIO servers
        if self.mode =="remote":
            self._set_remote_env_vars()

        # check if experiment exists and create a new one if needed    
        self._create_experiment()

        # Check for active runs and start a new run
        self._create_run()


    def log_hparams(self, params: dict):
        """
        Logs one or more hyperparameters to MLflow

        Args:
            params (dict): A dictionary of parameter names and their values.
        """

        if self.ended:
            raise RuntimeError("No active run. Call this function after start_run() or resume()")

        mlflow.log_params(params)


    def log_metrics(self, metrics: dict, step: int = None):
        """
        Logs one or more scalar metrics to MLflow.

        Args:
            metrics (dict): A dictionary of metric names and their float values.
            step (int, optional): An integer step index. Default is None.
        """
        if self.ended:
            raise RuntimeError("No active run. Call this function after start_run() or resume()")
        mlflow.log_metrics(metrics, step=step)


    def set_tags(self, tags: dict):
        """
        Sets one or more tags for the current MLflow run.

        Args:
            tags (dict): A dictionary of tag keys and their values.
        """
        if self.ended:
            raise RuntimeError("No active run. Call this function after start_run() or resume()")
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    def log_model(self, model, tokenizer, safe_serialization : bool = True):
        """
        Logs the model as a pyfunc model along with wrapper class and signature for inference
        """
        # Save locally to log as artifacts
        if self.ended:
            raise RuntimeError("No active run. Call this function after start_run() or resume()")
            
        save_path = "hf_model"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)  # Deletes the directory and its contents
        model.save_pretrained(save_path, safe_serialization = safe_serialization)
        tokenizer.save_pretrained(save_path)

        artifacts = {
            "hf_model": save_path,  # your HuggingFace model dir
        }

        # Log as PyFunc model with wrapped tokenizer + model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LlmWrapper(context_window = self.cfg.context_window,
            device = self.cfg.inference_device),
            artifacts=artifacts,
            code_path=["./Wrapper.py"],
            signature=get_signature()
        )

    def resume(self):
        """
        Uses the input run_id to restart it.
        Use it to continue logging to the same run_id after it has ended
        Call end() after logging to end the resumed run
        """
        if not self.started:
            raise RuntimeError("Cannot call resume() before start_run() has been called.")
        if mlflow.active_run():
            raise RuntimeError("Cannot call resume() while there is an active mlflow run.")            

        mlflow.start_run(run_id=self.run_id)
        self.ended = False

    def end(self):
        """
        Checks for an active run and ends it if one is available
        Must call after either init() or resume() to end the active run
        """
        if mlflow.active_run():
            mlflow.end_run()
            self.ended = True
        else:
            raise RuntimeError("Cannot call end() before start_run() or resume() has been called.")

def get_mlflow_tracker():
    """
    Import only this function to create an instance of MLflowTracker
    """
    return MLflowTracker()
