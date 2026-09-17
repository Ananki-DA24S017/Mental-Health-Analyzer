# src/model_training/register_model.py
import mlflow
import json
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/model_registry_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_registry")

def register_best_model(model_info_file):
    """Register the best model in MLflow model registry"""
    try:
        # Load model info
        with open(model_info_file, 'r') as f:
            model_info = json.load(f)
        
        best_model_name = model_info['model_name']
        best_accuracy = model_info['accuracy']
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Get all runs for the experiment
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Find the experiment ID
        experiment = mlflow.get_experiment_by_name("mental-health-analysis")
        if experiment is None:
            logger.error("Experiment 'mental-health-analysis' not found")
            return
        
        experiment_id = experiment.experiment_id
        
        # Get all runs for the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        # Filter to find the run for the best model
        best_run = runs[runs['tags.mlflow.runName'] == best_model_name]
        
        if len(best_run) == 0:
            logger.error(f"Run for model {best_model_name} not found")
            return
        
        best_run_id = best_run.iloc[0]['run_id']
        
        # Register the model
        model_uri = f"runs:/{best_run_id}/{best_model_name}"
        registered_model_name = "mental-health-analyzer"
        
        model_details = mlflow.register_model(model_uri, registered_model_name)
        
        logger.info(f"Model {best_model_name} registered as {registered_model_name} version {model_details.version}")
        
        # Create a file with registration details
        registration_info = {
            'model_name': best_model_name,
            'registered_name': registered_model_name,
            'version': model_details.version,
            'accuracy': best_accuracy,
            'run_id': best_run_id
        }
        
        with open("models/registration_info.json", 'w') as f:
            json.dump(registration_info, f, indent=2)
        
        logger.info(f"Registration info saved to models/registration_info.json")
        
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")

if __name__ == "__main__":
    model_info_file = "models/model_info.json"
    register_best_model(model_info_file)