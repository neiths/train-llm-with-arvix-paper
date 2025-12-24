"""
Processing pipeline orchestrator.

Manages the execution of processing jobs in sequence.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession

from config.settings import settings
from src.logging_config import get_logger
from src.processing.base import BaseSparkJob
from src.processing.content_moderation import ContentModerationJob
from src.processing.deduplication import DeduplicationJob
from src.processing.normalization import NormalizationJob

logger = get_logger(__name__)


class ProcessingPipeline:
    """
    Orchestrates the data processing pipeline.
    
    Manages:
    - Job sequencing
    - Intermediate data storage
    - Checkpoint management
    - Error recovery
    """
    
    STAGES = ["deduplication", "normalization", "content_moderation"]
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize the processing pipeline.
        
        Args:
            spark: Shared Spark session
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = checkpoint_dir or settings.data_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create shared Spark session
        self.spark = spark or self._create_spark_session()
        
        # Initialize jobs
        self.jobs: dict[str, BaseSparkJob] = {
            "deduplication": DeduplicationJob(spark=self.spark),
            "normalization": NormalizationJob(spark=self.spark),
            "content_moderation": ContentModerationJob(spark=self.spark),
        }
        
        logger.info(
            "ProcessingPipeline initialized",
            stages=self.STAGES,
            checkpoint_dir=str(self.checkpoint_dir),
        )
    
    def _create_spark_session(self) -> SparkSession:
        """Create a shared Spark session."""
        return SparkSession.builder \
            .appName("arxiv-processing-pipeline") \
            .master(settings.spark.master) \
            .config("spark.driver.memory", settings.spark.driver_memory) \
            .config("spark.executor.memory", settings.spark.executor_memory) \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def run(
        self,
        input_path: str,
        output_path: str,
        stages: Optional[list[str]] = None,
        resume_from: Optional[str] = None,
    ) -> dict:
        """
        Run the processing pipeline.
        
        Args:
            input_path: Path to raw input data
            output_path: Path for final processed output
            stages: Specific stages to run (default: all)
            resume_from: Stage to resume from (uses checkpoint)
            
        Returns:
            dict: Pipeline execution statistics
        """
        stages = stages or self.STAGES
        
        # Validate stages
        for stage in stages:
            if stage not in self.STAGES:
                raise ValueError(f"Unknown stage: {stage}")
        
        logger.info(
            "Starting processing pipeline",
            input_path=input_path,
            output_path=output_path,
            stages=stages,
        )
        
        # Determine starting point
        if resume_from:
            if resume_from not in stages:
                raise ValueError(f"Resume stage not in pipeline: {resume_from}")
            start_idx = stages.index(resume_from)
            stages = stages[start_idx:]
            current_input = self._get_checkpoint_path(resume_from)
            logger.info("Resuming from checkpoint", stage=resume_from)
        else:
            current_input = input_path
        
        pipeline_stats = {
            "started_at": datetime.utcnow().isoformat(),
            "input_path": input_path,
            "output_path": output_path,
            "stages": {},
        }
        
        # Run each stage
        for i, stage in enumerate(stages):
            is_last_stage = i == len(stages) - 1
            stage_output = output_path if is_last_stage else self._get_checkpoint_path(stage)
            
            try:
                logger.info(
                    "Running stage",
                    stage=stage,
                    input_path=current_input,
                    output_path=stage_output,
                )
                
                job = self.jobs[stage]
                stage_stats = job.run(current_input, stage_output)
                
                pipeline_stats["stages"][stage] = {
                    "status": "success",
                    "stats": stage_stats,
                }
                
                # Update checkpoint
                self._save_checkpoint(stage, stage_output)
                current_input = stage_output
                
            except Exception as e:
                logger.error(
                    "Stage failed",
                    stage=stage,
                    error=str(e),
                )
                pipeline_stats["stages"][stage] = {
                    "status": "failed",
                    "error": str(e),
                }
                raise RuntimeError(f"Pipeline failed at stage {stage}: {e}") from e
        
        pipeline_stats["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info("Pipeline complete", **pipeline_stats)
        return pipeline_stats
    
    def run_stage(
        self,
        stage: str,
        input_path: str,
        output_path: str,
    ) -> dict:
        """
        Run a single processing stage.
        
        Args:
            stage: Stage name
            input_path: Input data path
            output_path: Output data path
            
        Returns:
            dict: Stage statistics
        """
        if stage not in self.jobs:
            raise ValueError(f"Unknown stage: {stage}")
        
        job = self.jobs[stage]
        return job.run(input_path, output_path)
    
    def _get_checkpoint_path(self, stage: str) -> str:
        """Get checkpoint path for a stage."""
        return str(self.checkpoint_dir / f"{stage}_checkpoint")
    
    def _save_checkpoint(self, stage: str, output_path: str) -> None:
        """Save checkpoint metadata."""
        import json
        
        checkpoint_meta = {
            "stage": stage,
            "output_path": output_path,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        meta_path = self.checkpoint_dir / f"{stage}_checkpoint.json"
        with open(meta_path, "w") as f:
            json.dump(checkpoint_meta, f)
    
    def _load_checkpoint(self, stage: str) -> Optional[dict]:
        """Load checkpoint metadata."""
        import json
        
        meta_path = self.checkpoint_dir / f"{stage}_checkpoint.json"
        if not meta_path.exists():
            return None
        
        with open(meta_path, "r") as f:
            return json.load(f)
    
    def get_pipeline_status(self) -> dict:
        """Get status of all pipeline stages."""
        status = {}
        
        for stage in self.STAGES:
            checkpoint = self._load_checkpoint(stage)
            if checkpoint:
                status[stage] = {
                    "completed": True,
                    "output_path": checkpoint["output_path"],
                    "timestamp": checkpoint["timestamp"],
                }
            else:
                status[stage] = {
                    "completed": False,
                }
        
        return status
    
    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        import shutil
        
        for stage in self.STAGES:
            checkpoint_path = Path(self._get_checkpoint_path(stage))
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            
            meta_path = self.checkpoint_dir / f"{stage}_checkpoint.json"
            if meta_path.exists():
                meta_path.unlink()
        
        logger.info("Cleared all checkpoints")
    
    def stop(self) -> None:
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Stopped Spark session")
