#!/usr/bin/env python3
"""
Spark job runner - entry point for spark-submit.

Usage:
    spark-submit --master spark://spark-master:7077 run_pipeline.py /data/raw /data/processed
"""
import sys
import argparse
import logging

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Spark Processing Pipeline')
    parser.add_argument('input_path', help='Input directory containing .txt files')
    parser.add_argument('output_path', help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Import PySpark and create session FIRST, before importing our modules
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("ArxivDataProcessing") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark session created: {spark.sparkContext.appName}")
    logger.info(f"Master: {spark.sparkContext.master}")
    
    # Now add src to path and import our modules
    sys.path.insert(0, '/app/src')
    
    from pathlib import Path
    from processing.pipeline import ProcessingPipeline
    
    try:
        logger.info(f"Starting pipeline: {args.input_path} -> {args.output_path}")
        
        pipeline = ProcessingPipeline(spark=spark)
        success = pipeline.run(Path(args.input_path), Path(args.output_path))
        
        if success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
