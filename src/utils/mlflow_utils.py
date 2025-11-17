"""
MLflow utilities for experiment tracking.
"""
import os
import mlflow
import torch
from typing import Dict, Any, Optional
import yaml


class MLflowTracker:
    """
    Wrapper for MLflow experiment tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary containing MLflow settings
        """
        self.config = config
        mlflow_config = config.get('mlflow', {})
        
        # Set up MLflow
        tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = mlflow_config.get('experiment_name', 'few-shot-learning')
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=mlflow_config.get('artifact_location', './mlartifacts')
            )
        
        mlflow.set_experiment(experiment_name)
        
        self.run = None
        self.run_id = None
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        print(f"Started MLflow run: {self.run_id}")
        
        # Log configuration
        self._log_config()
    
    def _log_config(self):
        """Log the full configuration."""
        # Flatten config for logging
        flat_params = self._flatten_dict(self.config)
        mlflow.log_params(flat_params)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        mlflow.log_params(params)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file)."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: If provided, register the model with this name
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
    
    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_path: str
    ):
        """
        Save model checkpoint and log to MLflow.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            checkpoint_path: Path to save the checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Log to MLflow
        mlflow.log_artifact(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def log_figure(self, figure, artifact_file: str):
        """
        Log matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            artifact_file: Name of the artifact file
        """
        mlflow.log_figure(figure, artifact_file)
    
    def end_run(self):
        """End the current MLflow run."""
        if self.run:
            mlflow.end_run()
            print(f"Ended MLflow run: {self.run_id}")
            self.run = None
            self.run_id = None
    
    def set_tag(self, key: str, value: Any):
        """Set a tag for the current run."""
        mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, Any]):
        """Set multiple tags for the current run."""
        mlflow.set_tags(tags)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
