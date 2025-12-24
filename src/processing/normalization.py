"""
Text normalization job using NLTK and spaCy.

Cleans and standardizes text for ML training.
"""

import re
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType

from config.settings import settings
from src.logging_config import get_logger
from src.processing.base import BaseSparkJob

logger = get_logger(__name__)


# UDF functions for text normalization
def normalize_text(text: str) -> str:
    """
    Normalize text content.
    
    Operations:
    - Remove HTML tags
    - Remove LaTeX commands
    - Normalize unicode
    - Fix encoding issues
    - Normalize whitespace
    """
    if not text:
        return ""
    
    import ftfy
    import regex
    
    # Fix encoding issues
    text = ftfy.fix_text(text)
    
    # Remove HTML tags
    text = regex.sub(r"<[^>]+>", "", text)
    
    # Remove LaTeX commands (basic)
    text = regex.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = regex.sub(r"\\[a-zA-Z]+", "", text)
    text = regex.sub(r"\$[^$]+\$", "[MATH]", text)
    text = regex.sub(r"\\\[.*?\\\]", "[MATH]", text, flags=regex.DOTALL)
    text = regex.sub(r"\\\(.*?\\\)", "[MATH]", text, flags=regex.DOTALL)
    
    # Remove URLs
    text = regex.sub(r"https?://\S+", "[URL]", text)
    
    # Remove email addresses
    text = regex.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
    
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    
    # Normalize whitespace
    text = regex.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def detect_language(text: str) -> str:
    """Detect language of text."""
    if not text or len(text) < 20:
        return "unknown"
    
    try:
        from langdetect import detect
        return detect(text[:1000])  # Use first 1000 chars
    except Exception:
        return "unknown"


def extract_sentences(text: str) -> list[str]:
    """Extract sentences using NLTK."""
    if not text:
        return []
    
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            sentences = nltk.sent_tokenize(text)
        return sentences
    except Exception:
        # Fallback to simple splitting
        return re.split(r"[.!?]+", text)


class NormalizationJob(BaseSparkJob):
    """
    Text normalization job for cleaning and standardizing text.
    
    Features:
    - HTML and LaTeX removal
    - Unicode normalization
    - Language detection
    - Sentence tokenization
    - Word counting
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        """
        Initialize normalization job.
        
        Args:
            spark: Spark session
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
        """
        super().__init__(spark, "arxiv-normalization")
        self.min_length = min_length or settings.processing.min_text_length
        self.max_length = max_length or settings.processing.max_text_length
        
        # Register UDFs
        self._register_udfs()
    
    def _register_udfs(self) -> None:
        """Register UDFs with Spark."""
        self.normalize_udf = F.udf(normalize_text, StringType())
        self.word_count_udf = F.udf(count_words, IntegerType())
        self.language_udf = F.udf(detect_language, StringType())
        self.sentences_udf = F.udf(extract_sentences, ArrayType(StringType()))
    
    def run(self, input_path: str, output_path: str) -> dict:
        """
        Run normalization on input data.
        
        Args:
            input_path: Path to input Parquet data
            output_path: Path for normalized output
            
        Returns:
            dict: Job statistics
        """
        logger.info(
            "Starting normalization",
            input_path=input_path,
            min_length=self.min_length,
            max_length=self.max_length,
        )
        
        # Read input data
        df = self.read_parquet(input_path)
        initial_count = df.count()
        
        # Apply normalization
        df = self._normalize_fields(df)
        
        # Add computed fields
        df = self._add_computed_fields(df)
        
        # Filter by length
        df = self._filter_by_length(df)
        
        final_count = df.count()
        filtered_count = initial_count - final_count
        
        # Write output
        self.write_parquet(df, output_path)
        
        stats = {
            "initial_count": initial_count,
            "final_count": final_count,
            "filtered_count": filtered_count,
            "filter_ratio": filtered_count / initial_count if initial_count > 0 else 0,
        }
        
        logger.info("Normalization complete", **stats)
        return stats
    
    def _normalize_fields(self, df: DataFrame) -> DataFrame:
        """Normalize text fields."""
        # Normalize title
        df = df.withColumn("title", self.normalize_udf(F.col("title")))
        
        # Normalize abstract
        df = df.withColumn("abstract", self.normalize_udf(F.col("abstract")))
        
        # Normalize full_text if present
        if "full_text" in df.columns:
            df = df.withColumn(
                "full_text",
                F.when(
                    F.col("full_text").isNotNull(),
                    self.normalize_udf(F.col("full_text")),
                ).otherwise(None),
            )
        
        return df
    
    def _add_computed_fields(self, df: DataFrame) -> DataFrame:
        """Add computed metadata fields."""
        # Combine text for processing
        df = df.withColumn(
            "text",
            F.coalesce(
                F.col("full_text"),
                F.concat_ws("\n\n", F.col("title"), F.col("abstract")),
            ),
        )
        
        # Word count
        df = df.withColumn("word_count", self.word_count_udf(F.col("text")))
        
        # Character count
        df = df.withColumn("char_count", F.length(F.col("text")))
        
        # Language detection
        df = df.withColumn("language", self.language_udf(F.col("abstract")))
        
        return df
    
    def _filter_by_length(self, df: DataFrame) -> DataFrame:
        """Filter by text length."""
        df = df.filter(
            (F.col("char_count") >= self.min_length) &
            (F.col("char_count") <= self.max_length)
        )
        
        # Filter to English only
        df = df.filter(F.col("language") == "en")
        
        return df
    
    def normalize_batch(self, df: DataFrame) -> DataFrame:
        """
        Normalize a batch of documents without filtering.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Normalized DataFrame
        """
        df = self._normalize_fields(df)
        df = self._add_computed_fields(df)
        return df
