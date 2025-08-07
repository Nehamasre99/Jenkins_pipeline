// pipeline {
//     agent any

//     environment {
//         VENV_DIR = 'venv'
//         GIT_SSL_NO_VERIFY = 'true'
//     }

//     stages {
//         stage('Clone Repo') {
//             steps {
//                 checkout scm
//             }
//         }

//         stage('Set Up Python Environment') {
//             steps {
//                 sh '''
//                     python3 -m venv ${VENV_DIR}
//                     . ${VENV_DIR}/bin/activate
//                     pip install --upgrade pip
//                     pip install -r requirements.txt pytest
//                 '''
//             }
//         }

//         stage('Run Tests') {
//             steps {
//                 sh '''
//                     . ${VENV_DIR}/bin/activate
//                     pytest tests/
//                 '''
//             }
//         }
//     }

//     post {
//         always {
//             echo 'Cleaning up...'
//         }
//         success {
//             echo 'Tests passed!'
//         }
//         failure {
//             echo 'Tests failed.'
//         }
//     }
// }




// /////////////////////////////////////////////////


// Jenkinsfile (Simplified Version)
pipeline {
    agent any // Specifies that Jenkins can use any available agent to run this pipeline

    environment {
        // Define the MLflow Tracking URI as an environment variable
        // IMPORTANT: Replace this with the actual URL of your MLflow server.
        // If Jenkins and MLflow are on the same machine, use your machine's IP, not 'localhost'.
        // Example: 'http://192.168.1.10:5000'
        MLFLOW_TRACKING_URI = 'http://localhost:5000'
    }

    stages {
        stage('Checkout Code') {
            steps {
                // This step clones your project from the Git repository
                echo 'Cloning the repository...'
                git 'https://github.com/your-username/your-mlops-repo.git' // <-- CHANGE THIS to your repository URL
            }
        }

        stage('Run MLflow Experiment') {
            steps {
                // This step executes your main script to run the experiment
                echo "Running the ML experiment script..."
                echo "MLflow tracking server is set to: ${env.MLFLOW_TRACKING_URI}"
                sh 'python3 run_local_experiment.py'
            }
        }
    }

    post {
        always {
            // This block runs after all stages, regardless of the outcome
            echo 'Pipeline finished.'
        }
    }
}
