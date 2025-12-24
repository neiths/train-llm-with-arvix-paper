"""
Base class for Spark processing jobs.
"""

from abc import ABC, abstractmethod
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class BaseSparkJob(ABC):
    """
    Abstract base class for Spark processing jobs.
    
    Provides common functionality for Spark job execution.
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        app_name: Optional[str] = None,
    ):
        """
        Initialize the Spark job.
        
        Args:
            spark: Existing Spark session (creates new if not provided)
            app_name: Application name
        """
        self.app_name = app_name or self.job_name
        self._spark = spark
        self._owns_session = spark is None
        
    @property
    def job_name(self) -> str:
        """Get job name from class name."""
        return self.__class__.__name__
    
    @property
    def spark(self) -> SparkSession:
        """Get or create Spark session."""
        if self._spark is None:
            self._spark = self._create_session()
        return self._spark
    
    def _create_session(self) -> SparkSession:
        """Create a new Spark session."""
        builder = SparkSession.builder \
            .appName(self.app_name) \
            .master(settings.spark.master) \
            .config("spark.driver.memory", settings.spark.driver_memory) \
            .config("spark.executor.memory", settings.spark.executor_memory) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        # Add S3/MinIO configuration
        builder = builder \
            .config("spark.hadoop.fs.s3a.endpoint", f"http://{settings.minio.endpoint}") \
            .config("spark.hadoop.fs.s3a.access.key", settings.minio.access_key) \
            .config("spark.hadoop.fs.s3a.secret.key", settings.minio.secret_key) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        
        session = builder.getOrCreate()
        logger.info("Created Spark session", app_name=self.app_name)
        return session
    
    def read_parquet(self, path: str) -> DataFrame:
        """
        Read Parquet data from storage.
        
        Args:
            path: S3/local path to read from
            
        Returns:
            DataFrame: Spark DataFrame
        """
        return self.spark.read.parquet(path)
    
    def write_parquet(
        self,
        df: DataFrame,
        path: str,
        mode: str = "overwrite",
        partition_by: Optional[list[str]] = None,
    ) -> None:
        """
        Write DataFrame to Parquet.
        
        Args:
            df: DataFrame to write
            path: Output path
            mode: Write mode (overwrite, append, etc.)
            partition_by: Partition columns
        """
        writer = df.write.mode(mode)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        writer.parquet(path)
        logger.info("Wrote parquet", path=path, mode=mode)
    
    @abstractmethod
    def run(self, input_path: str, output_path: str) -> dict:
        """
        Run the processing job.
        
        Args:
            input_path: Input data path
            output_path: Output data path
            
        Returns:
            dict: Job statistics
        """
        pass
    
    def stop(self) -> None:
        """Stop the Spark session if owned."""
        if self._owns_session and self._spark is not None:
            self._spark.stop()
            logger.info("Stopped Spark session")
