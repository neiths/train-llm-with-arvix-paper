"""Deduplication processor using MinHash LSH."""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, xxhash64
from pyspark.sql.types import ArrayType, LongType
from pyspark.ml.feature import HashingTF, MinHashLSH

from .base import BaseProcessor


class DeduplicationProcessor(BaseProcessor):
    """Remove duplicate documents using MinHash LSH."""
    
    def __init__(self, *args, similarity_threshold: float = 0.8, **kwargs):
        super().__init__(*args, app_name="Deduplication", **kwargs)
        self.similarity_threshold = similarity_threshold
    
    def process(self, input_df: DataFrame) -> DataFrame:
        """Remove near-duplicate documents."""
        
        # Create shingles (n-grams)
        @udf(ArrayType(LongType()))
        def create_shingles(text: str, n: int = 5) -> list:
            if not text or len(text) < n:
                return []
            words = text.split()
            shingles = []
            for i in range(len(words) - n + 1):
                shingle = " ".join(words[i:i+n])
                shingles.append(hash(shingle))
            return shingles
        
        # Add document ID
        df = input_df.withColumn("doc_id", xxhash64(col("normalized_text")))
        
        # Create shingles
        df = df.withColumn("shingles", create_shingles(col("normalized_text")))
        
        # Use HashingTF for feature vectors
        hashing_tf = HashingTF(inputCol="shingles", outputCol="features", numFeatures=1000)
        df = hashing_tf.transform(df)
        
        # Apply MinHash LSH
        mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
        model = mh.fit(df)
        
        # Find similar pairs and remove duplicates (keep first occurrence)
        similar_pairs = model.approxSimilarityJoin(
            df, df, 
            threshold=1 - self.similarity_threshold,
            distCol="distance"
        ).filter(col("datasetA.doc_id") < col("datasetB.doc_id"))
        
        # Get IDs to remove
        duplicates_to_remove = similar_pairs.select(col("datasetB.doc_id").alias("remove_id"))
        
        # Filter out duplicates
        result = df.join(
            duplicates_to_remove,
            df.doc_id == duplicates_to_remove.remove_id,
            "left_anti"
        )
        
        return result.select("normalized_text")