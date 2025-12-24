"""
Weights & Biases integration for experiment tracking.

Provides logging, artifact management, and experiment comparison.
"""

from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class WandBLogger:
    """
    Weights & Biases logger for experiment tracking.
    
    Features:
    - Experiment logging
    - Model artifact storage
    - Hyperparameter tracking
    - Visualization and comparison
    """
    
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity/team
            name: Run name
            config: Hyperparameters to log
            tags: Run tags
            notes: Run notes
            mode: W&B mode (online, offline, disabled)
        """
        self.project = project or settings.wandb.project
        self.entity = entity or settings.wandb.entity
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.mode = mode or settings.wandb.mode
        
        self._run = None
        self._initialized = False
    
    def init(self) -> None:
        """Initialize W&B run."""
        if self._initialized:
            return
        
        try:
            import wandb
            
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                mode=self.mode,
            )
            
            self._initialized = True
            
            logger.info(
                "W&B initialized",
                project=self.project,
                run_name=self._run.name,
                run_id=self._run.id,
            )
            
        except Exception as e:
            logger.warning("Failed to initialize W&B", error=str(e))
            self.mode = "disabled"
    
    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if self.mode == "disabled":
            return
        
        if not self._initialized:
            self.init()
        
        if self._run:
            import wandb
            wandb.log(metrics, step=step)
    
    def log_config(self, config: dict) -> None:
        """
        Log configuration/hyperparameters.
        
        Args:
            config: Configuration dictionary
        """
        if self.mode == "disabled":
            return
        
        if not self._initialized:
            self.init()
        
        if self._run:
            import wandb
            wandb.config.update(config)
    
    def log_model(
        self,
        model_path: Path,
        name: str = "model",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a model artifact.
        
        Args:
            model_path: Path to model checkpoint
            name: Artifact name
            metadata: Optional metadata
        """
        if self.mode == "disabled":
            return
        
        if not self._initialized:
            self.init()
        
        if self._run:
            import wandb
            
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {},
            )
            artifact.add_dir(str(model_path))
            self._run.log_artifact(artifact)
            
            logger.info("Logged model artifact", name=name)
    
    def log_dataset(
        self,
        dataset_path: Path,
        name: str = "dataset",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a dataset artifact.
        
        Args:
            dataset_path: Path to dataset
            name: Artifact name
            metadata: Optional metadata
        """
        if self.mode == "disabled":
            return
        
        if not self._initialized:
            self.init()
        
        if self._run:
            import wandb
            
            artifact = wandb.Artifact(
                name=name,
                type="dataset",
                metadata=metadata or {},
            )
            
            if dataset_path.is_dir():
                artifact.add_dir(str(dataset_path))
            else:
                artifact.add_file(str(dataset_path))
            
            self._run.log_artifact(artifact)
            
            logger.info("Logged dataset artifact", name=name)
    
    def log_table(
        self,
        table_name: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """
        Log a table for visualization.
        
        Args:
            table_name: Name of the table
            columns: Column names
            data: Table data
        """
        if self.mode == "disabled":
            return
        
        if not self._initialized:
            self.init()
        
        if self._run:
            import wandb
            
            table = wandb.Table(columns=columns, data=data)
            wandb.log({table_name: table})
    
    def watch(self, model: Any, log_freq: int = 100) -> None:
        """
        Watch a model for gradient logging.
        
        Args:
            model: PyTorch model
            log_freq: Logging frequency
        """
        if self.mode == "disabled":
            return
        
        if not self._initialized:
            self.init()
        
        if self._run:
            import wandb
            wandb.watch(model, log_freq=log_freq)
    
    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ) -> None:
        """
        Send an alert.
        
        Args:
            title: Alert title
            text: Alert text
            level: Alert level (INFO, WARN, ERROR)
        """
        if self.mode == "disabled":
            return
        
        if not self._initialized:
            self.init()
        
        if self._run:
            import wandb
            
            level_map = {
                "INFO": wandb.AlertLevel.INFO,
                "WARN": wandb.AlertLevel.WARN,
                "ERROR": wandb.AlertLevel.ERROR,
            }
            
            wandb.alert(
                title=title,
                text=text,
                level=level_map.get(level, wandb.AlertLevel.INFO),
            )
    
    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run:
            import wandb
            wandb.finish()
            self._run = None
            self._initialized = False
            logger.info("W&B run finished")
    
    @property
    def run(self):
        """Get the W&B run object."""
        return self._run
    
    @property
    def run_url(self) -> Optional[str]:
        """Get the W&B run URL."""
        if self._run:
            return self._run.url
        return None
