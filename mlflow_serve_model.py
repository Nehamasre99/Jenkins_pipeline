# mlflow_serve_model.py
import subprocess
import requests
import time


class MLflowModelServe:

    def __init__(self, model_uri : str, gpu_flag : bool = False, ssh_user : str = "shrivatsan"):
        self.model_uri = model_uri
        self.gpu_flag = gpu_flag
        self.ssh_user = ssh_user
        self.server_ip = "10.10.1.45"

    def serve_model(self, port: int = 5001, container_name: str = "mlflow_model_server"):
        cmd_parts = [
        "docker run -d --rm",
        "--network=host",
        f"-e MLFLOW_TRACKING_URI=http://{self.server_ip}:5000",
        "-e AWS_ACCESS_KEY_ID=minioadmin",
        "-e AWS_SECRET_ACCESS_KEY=minioadmin",
        f"-e MLFLOW_S3_ENDPOINT_URL=http://{self.server_ip}:9000",
        f"-e INFERENCE=true",  # Set the INFERENCE environment variable here
        f"--name {container_name}",
    ]
    
        if self.gpu_flag:
            cmd_parts.append("--gpus all")

        cmd_parts += [
        "mlflow-custom",
        "mlflow models serve",
        f"-m {self.model_uri}",
        f"-p {port}",
        "--no-conda",
        "--host 0.0.0.0",
        "--env-manager=local"
        ]

        docker_cmd = " \\\n  ".join(cmd_parts)

        ssh_cmd = f'ssh {self.ssh_user}@{self.server_ip} "{docker_cmd}"'

        try:
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            print("Model server started successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start model server: {e}")

    def stop_model_server(self, container_name: str):
        name = container_name
        if not name:
            print("Container name not set. Please provide it or call serve_model() first.")
            return

        ssh_cmd = f'ssh {self.ssh_user}@{self.server_ip} "docker stop {name}"'

        try:
            subprocess.run(ssh_cmd, shell=True, check=True, text=True)
            print(f"Stopped model server container '{name}' on remote server.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop model server: {e}")


    def wait_for_server(self, port=5001, timeout=350):
        url = f"http://localhost:{port}/invocations"
        headers = {"Content-Type": "application/json"}

        # Dummy, intentionally invalid input
        dummy_payload = {"foo": "bar"}

        start = time.time()
        while time.time() - start < timeout:
            try:
                res = requests.post(url, json=dummy_payload, headers=headers)
                if res.status_code != 404 and res.status_code != 500:
                    print(f"Server responded (status {res.status_code}) — it's ready.")
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(5)

        raise TimeoutError("Model server did not become responsive in time.")


def get_mlflow_model_serve(model_uri : str, gpu_flag : bool = False):
    return MLflowModelServe(model_uri = model_uri, gpu_flag = gpu_flag)
