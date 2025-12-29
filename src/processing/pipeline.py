"""Data processing pipeline orchestration."""
import logging
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession

from .normalization import NormalizationProcessor
from .deduplication import DeduplicationProcessor

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrate the data processing pipeline."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize pipeline with shared Spark session."""
        # Don't specify master here - let spark-submit handle it
        self.spark = spark or SparkSession.builder \
            .appName("ProcessingPipeline") \
            .getOrCreate()
    
    def run(self, input_path: Path, output_path: Path) -> bool:
        """Run the complete processing pipeline."""
        try:
            logger.info(f"Starting pipeline: {input_path} -> {output_path}")
            
            # Read raw text files - use string path with glob
            input_glob = str(input_path) + "/*.txt"
            logger.info(f"Reading from: {input_glob}")
            df = self.spark.read.text(input_glob)
            logger.info(f"Loaded {df.count()} documents")
            
            # Step 1: Normalization
            normalizer = NormalizationProcessor(spark=self.spark)
            df = normalizer.process(df)
            logger.info(f"After normalization: {df.count()} documents")
            
            # Step 2: Deduplication
            deduplicator = DeduplicationProcessor(spark=self.spark)
            df = deduplicator.process(df)
            logger.info(f"After deduplication: {df.count()} documents")
            
            # Save output
            output_path.mkdir(parents=True, exist_ok=True)
            df.write.mode("overwrite").text(str(output_path))
            
            logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False