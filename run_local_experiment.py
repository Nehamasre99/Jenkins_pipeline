# run_local_experiment.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Import the factory functions from your SDK in the 'src' directory
from src import get_mlflow_tracker, get_registry_manager

def main():
    print("Initializing MLflow Tracker...")
    # The SDK automatically reads mlflow_config.yaml
    tracker = get_mlflow_tracker()

    print("Starting MLflow run...")
    tracker.start_run()
    run_id = tracker.run_id
    print(f"MLflow run started with ID: {run_id}")

    # 1. Log some example hyperparameters
    hparams = {"learning_rate": 0.01, "epochs": 5, "batch_size": 16}
    tracker.log_hparams(hparams)
    print(f"Logged Hyperparameters: {hparams}")

    # 2. Log some example metrics
    metrics = {"final_loss": 0.54, "accuracy": 0.88}
    tracker.log_metrics(metrics)
    print(f"Logged Metrics: {metrics}")

    # 3. Load the model specified in the config and log it
    model_name = tracker.cfg.model_name
    print(f"Loading model '{model_name}' from Hugging Face Hub...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully.")

    print("Logging model to MLflow...")
    # The log_model function packages the model using Wrapper.py
    tracker.log_model(model=model, tokenizer=tokenizer)
    print("Model logged successfully.")

    # 4. End the MLflow run
    tracker.end()
    print("MLflow run ended.")

    # 5. Register the logged model
    print("Initializing MLflow Registry Manager...")
    # Let it use the model name from the config file
    registry_manager = get_registry_manager(mlflow_registry_name=None)

    print(f"Registering model from run ID: {run_id}")
    # The 'artifact_path' must be 'model' to match what log_model uses internally
    version = registry_manager.register(run_id=run_id, artifact_path="model")
    print(f"Model registered successfully as '{registry_manager.mlflow_registry_name}' version {version}!")

if __name__ == "__main__":
    main()
