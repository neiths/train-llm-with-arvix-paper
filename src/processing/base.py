"""Base class for Spark processing jobs."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, spark: Optional[SparkSession] = None, app_name: str = "DataProcessor"):
        """Initialize processor with Spark session."""
        self.spark = spark or self._create_spark_session(app_name)
    
    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Get or create a Spark session."""
        # Don't specify master here - let spark-submit handle it
        return SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
    
    @abstractmethod
    def process(self, input_df: DataFrame) -> DataFrame:
        """Process input DataFrame."""
        pass
    
    def read_text_files(self, input_path: Path) -> DataFrame:
        """Read text files into a DataFrame."""
        return self.spark.read.text(str(input_path / "*.txt"))
    
    def write_output(self, df: DataFrame, output_path: Path, format: str = "parquet"):
        """Write DataFrame to output path."""
        df.write.mode("overwrite").format(format).save(str(output_path))