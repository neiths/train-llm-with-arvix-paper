"""
Deduplication job using MinHash LSH for near-duplicate detection.

Uses Locality-Sensitive Hashing to efficiently find similar documents.
"""

from typing import Optional

from pyspark.ml.feature import HashingTF, MinHashLSH, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType

from config.settings import settings
from src.logging_config import get_logger
from src.processing.base import BaseSparkJob

logger = get_logger(__name__)


class DeduplicationJob(BaseSparkJob):
    """
    Near-duplicate detection using MinHash LSH.
    
    Features:
    - Configurable similarity threshold
    - Efficient LSH-based similarity search
    - Preserves the "best" document from duplicate groups
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        threshold: Optional[float] = None,
        num_hash_tables: int = 5,
    ):
        """
        Initialize deduplication job.
        
        Args:
            spark: Spark session
            threshold: Jaccard similarity threshold (0-1)
            num_hash_tables: Number of hash tables for LSH
        """
        super().__init__(spark, "arxiv-deduplication")
        self.threshold = threshold or settings.processing.dedup_threshold
        self.num_hash_tables = num_hash_tables
    
    def run(self, input_path: str, output_path: str) -> dict:
        """
        Run deduplication on the input data.
        
        Args:
            input_path: Path to input Parquet data
            output_path: Path for deduplicated output
            
        Returns:
            dict: Job statistics including duplicates found
        """
        logger.info(
            "Starting deduplication",
            input_path=input_path,
            threshold=self.threshold,
        )
        
        # Read input data
        df = self.read_parquet(input_path)
        initial_count = df.count()
        
        # Prepare text for hashing
        df_prepared = self._prepare_text(df)
        
        # Create feature vectors
        df_features = self._create_features(df_prepared)
        
        # Find duplicates using LSH
        df_deduplicated = self._find_and_remove_duplicates(df_features)
        
        # Select original columns plus dedup metadata
        result_df = df_deduplicated.select(
            df.columns + ["is_duplicate", "duplicate_of"]
        )
        
        # Filter to keep only non-duplicates
        result_df = result_df.filter(~F.col("is_duplicate"))
        
        final_count = result_df.count()
        duplicates_removed = initial_count - final_count
        
        # Write output
        self.write_parquet(result_df, output_path)
        
        stats = {
            "initial_count": initial_count,
            "final_count": final_count,
            "duplicates_removed": duplicates_removed,
            "dedup_ratio": duplicates_removed / initial_count if initial_count > 0 else 0,
        }
        
        logger.info("Deduplication complete", **stats)
        return stats
    
    def _prepare_text(self, df: DataFrame) -> DataFrame:
        """Prepare text for duplicate detection."""
        # Combine title and abstract for comparison
        # Normalize whitespace
        df = df.withColumn(
            "combined_text",
            F.lower(
                F.concat_ws(
                    " ",
                    F.col("title"),
                    F.col("abstract"),
                )
            ),
        )
        
        # Remove punctuation and extra whitespace
        df = df.withColumn(
            "combined_text",
            F.regexp_replace(F.col("combined_text"), r"[^\w\s]", " "),
        )
        df = df.withColumn(
            "combined_text",
            F.regexp_replace(F.col("combined_text"), r"\s+", " "),
        )
        
        return df
    
    def _create_features(self, df: DataFrame) -> DataFrame:
        """Create feature vectors for LSH."""
        # Tokenize text
        tokenizer = Tokenizer(inputCol="combined_text", outputCol="tokens")
        df_tokenized = tokenizer.transform(df)
        
        # Create hash vectors
        hashing_tf = HashingTF(
            inputCol="tokens",
            outputCol="features",
            numFeatures=10000,
        )
        df_features = hashing_tf.transform(df_tokenized)
        
        return df_features
    
    def _find_and_remove_duplicates(self, df: DataFrame) -> DataFrame:
        """Find duplicates using MinHash LSH."""
        # Fit MinHash LSH model
        mh = MinHashLSH(
            inputCol="features",
            outputCol="hashes",
            numHashTables=self.num_hash_tables,
        )
        model = mh.fit(df)
        
        # Transform to add hashes
        df_hashed = model.transform(df)
        
        # Find similar pairs
        similar_pairs = model.approxSimilarityJoin(
            df_hashed.alias("a"),
            df_hashed.alias("b"),
            self.threshold,
            distCol="jaccard_distance",
        )
        
        # Filter to pairs where a.arxiv_id < b.arxiv_id (avoid duplicates)
        similar_pairs = similar_pairs.filter(
            F.col("a.arxiv_id") < F.col("b.arxiv_id")
        )
        
        # Collect IDs to mark as duplicates (keep the one with earlier publication)
        # For simplicity, we'll keep the first one alphabetically
        duplicate_ids = similar_pairs.select(
            F.col("b.arxiv_id").alias("duplicate_id"),
            F.col("a.arxiv_id").alias("original_id"),
        ).distinct()
        
        # Mark duplicates in original dataframe
        df_marked = df_hashed.join(
            duplicate_ids,
            df_hashed["arxiv_id"] == duplicate_ids["duplicate_id"],
            "left",
        ).withColumn(
            "is_duplicate",
            F.col("duplicate_id").isNotNull(),
        ).withColumn(
            "duplicate_of",
            F.col("original_id"),
        ).drop("duplicate_id", "original_id", "hashes", "features", "tokens", "combined_text")
        
        return df_marked
    
    def find_duplicates_batch(
        self,
        new_df: DataFrame,
        existing_path: str,
    ) -> DataFrame:
        """
        Find duplicates of new documents against existing corpus.
        
        Args:
            new_df: New documents to check
            existing_path: Path to existing corpus
            
        Returns:
            DataFrame: New documents with duplicate flags
        """
        existing_df = self.read_parquet(existing_path)
        
        # Prepare both datasets
        new_prepared = self._prepare_text(new_df)
        new_features = self._create_features(new_prepared)
        
        existing_prepared = self._prepare_text(existing_df)
        existing_features = self._create_features(existing_prepared)
        
        # Fit model on existing data
        mh = MinHashLSH(
            inputCol="features",
            outputCol="hashes",
            numHashTables=self.num_hash_tables,
        )
        model = mh.fit(existing_features)
        
        # Transform both datasets
        new_hashed = model.transform(new_features)
        existing_hashed = model.transform(existing_features)
        
        # Find similar pairs
        similar_pairs = model.approxSimilarityJoin(
            new_hashed.alias("new"),
            existing_hashed.alias("existing"),
            self.threshold,
            distCol="jaccard_distance",
        )
        
        # Extract duplicate IDs
        duplicate_ids = similar_pairs.select(
            F.col("new.arxiv_id").alias("new_id"),
            F.col("existing.arxiv_id").alias("existing_id"),
        ).distinct()
        
        # Mark duplicates
        result = new_df.join(
            duplicate_ids,
            new_df["arxiv_id"] == duplicate_ids["new_id"],
            "left",
        ).withColumn(
            "is_duplicate",
            F.col("new_id").isNotNull(),
        ).withColumn(
            "duplicate_of",
            F.col("existing_id"),
        ).drop("new_id", "existing_id")
        
        return result
