"""
Content moderation job for quality filtering and PII detection.

Ensures data quality and privacy compliance.
"""

import re
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, BooleanType, FloatType, StringType

from config.settings import settings
from src.logging_config import get_logger
from src.processing.base import BaseSparkJob

logger = get_logger(__name__)


# PII patterns
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


def detect_pii(text: str) -> list[str]:
    """
    Detect PII types present in text.
    
    Returns list of PII types found.
    """
    if not text:
        return []
    
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pii_type)
    
    return found


def redact_pii(text: str) -> str:
    """
    Redact PII from text.
    
    Replaces PII with type-specific placeholders.
    """
    if not text:
        return ""
    
    for pii_type, pattern in PII_PATTERNS.items():
        placeholder = f"[{pii_type.upper()}]"
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    
    return text


def calculate_quality_score(
    text: str,
    word_count: int,
    char_count: int,
) -> float:
    """
    Calculate a quality score for the text.
    
    Factors:
    - Length (normalized)
    - Vocabulary diversity
    - Sentence structure
    - Special character ratio
    """
    if not text or char_count == 0:
        return 0.0
    
    score = 0.0
    
    # Length score (0-0.3)
    # Optimal length is 1000-10000 characters
    if char_count < 100:
        length_score = 0.0
    elif char_count < 1000:
        length_score = char_count / 1000 * 0.3
    elif char_count <= 10000:
        length_score = 0.3
    else:
        length_score = max(0.1, 0.3 - (char_count - 10000) / 100000)
    score += length_score
    
    # Vocabulary diversity (0-0.3)
    words = text.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        vocab_score = min(unique_ratio * 0.5, 0.3)  # Cap at 0.3
        score += vocab_score
    
    # Sentence structure (0-0.2)
    # Check for proper sentence endings
    sentences = re.split(r"[.!?]+", text)
    valid_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
    if len(sentences) > 0:
        sentence_score = min(valid_sentences / len(sentences), 1.0) * 0.2
        score += sentence_score
    
    # Special character ratio (0-0.2)
    # Lower is better (scholarly text shouldn't have too many special chars)
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / char_count
    if special_ratio < 0.1:
        special_score = 0.2
    elif special_ratio < 0.2:
        special_score = 0.1
    else:
        special_score = 0.0
    score += special_score
    
    return min(score, 1.0)


def is_low_quality(
    text: str,
    word_count: int,
    min_words: int = 50,
) -> bool:
    """
    Check if text is low quality.
    
    Criteria:
    - Too short
    - Too many repeated words
    - Too few unique words
    - Contains spam patterns
    """
    if not text or word_count < min_words:
        return True
    
    words = text.lower().split()
    
    # Check unique word ratio
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.2:  # Less than 20% unique words
        return True
    
    # Check for spam patterns
    spam_patterns = [
        r"click here",
        r"buy now",
        r"free money",
        r"congratulations you won",
    ]
    
    text_lower = text.lower()
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


class ContentModerationJob(BaseSparkJob):
    """
    Content moderation job for quality filtering and PII handling.
    
    Features:
    - PII detection and redaction
    - Quality scoring
    - Low-quality content filtering
    - Content classification
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        redact_pii: bool = True,
        min_quality_score: float = 0.3,
        filter_low_quality: bool = True,
    ):
        """
        Initialize content moderation job.
        
        Args:
            spark: Spark session
            redact_pii: Whether to redact PII
            min_quality_score: Minimum quality score to keep
            filter_low_quality: Whether to filter low-quality content
        """
        super().__init__(spark, "arxiv-content-moderation")
        self.redact_pii = redact_pii
        self.min_quality_score = min_quality_score
        self.filter_low_quality = filter_low_quality
        
        # Register UDFs
        self._register_udfs()
    
    def _register_udfs(self) -> None:
        """Register UDFs with Spark."""
        self.detect_pii_udf = F.udf(detect_pii, ArrayType(StringType()))
        self.redact_pii_udf = F.udf(redact_pii, StringType())
        self.quality_score_udf = F.udf(calculate_quality_score, FloatType())
        self.low_quality_udf = F.udf(is_low_quality, BooleanType())
    
    def run(self, input_path: str, output_path: str) -> dict:
        """
        Run content moderation on input data.
        
        Args:
            input_path: Path to input Parquet data
            output_path: Path for moderated output
            
        Returns:
            dict: Job statistics
        """
        logger.info(
            "Starting content moderation",
            input_path=input_path,
            redact_pii=self.redact_pii,
            min_quality_score=self.min_quality_score,
        )
        
        # Read input data
        df = self.read_parquet(input_path)
        initial_count = df.count()
        
        # Detect PII
        df = self._detect_pii(df)
        
        # Redact PII if enabled
        if self.redact_pii:
            df = self._redact_pii(df)
        
        # Calculate quality scores
        df = self._calculate_quality_scores(df)
        
        # Filter by quality
        if self.filter_low_quality:
            df = self._filter_quality(df)
        
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
        
        logger.info("Content moderation complete", **stats)
        return stats
    
    def _detect_pii(self, df: DataFrame) -> DataFrame:
        """Detect PII in text fields."""
        # Detect PII in combined text
        df = df.withColumn(
            "pii_types",
            self.detect_pii_udf(F.col("text")),
        )
        
        # Flag if any PII found
        df = df.withColumn(
            "has_pii",
            F.size(F.col("pii_types")) > 0,
        )
        
        return df
    
    def _redact_pii(self, df: DataFrame) -> DataFrame:
        """Redact PII from text fields."""
        df = df.withColumn("text", self.redact_pii_udf(F.col("text")))
        df = df.withColumn("title", self.redact_pii_udf(F.col("title")))
        df = df.withColumn("abstract", self.redact_pii_udf(F.col("abstract")))
        
        if "full_text" in df.columns:
            df = df.withColumn(
                "full_text",
                F.when(
                    F.col("full_text").isNotNull(),
                    self.redact_pii_udf(F.col("full_text")),
                ).otherwise(None),
            )
        
        return df
    
    def _calculate_quality_scores(self, df: DataFrame) -> DataFrame:
        """Calculate quality scores."""
        df = df.withColumn(
            "quality_score",
            self.quality_score_udf(
                F.col("text"),
                F.col("word_count"),
                F.col("char_count"),
            ),
        )
        
        df = df.withColumn(
            "is_low_quality",
            self.low_quality_udf(F.col("text"), F.col("word_count")),
        )
        
        return df
    
    def _filter_quality(self, df: DataFrame) -> DataFrame:
        """Filter by quality score and low-quality flag."""
        df = df.filter(
            (F.col("quality_score") >= self.min_quality_score) &
            (~F.col("is_low_quality"))
        )
        return df
    
    def moderate_batch(self, df: DataFrame) -> DataFrame:
        """
        Moderate a batch of documents without filtering.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Moderated DataFrame with scores
        """
        df = self._detect_pii(df)
        if self.redact_pii:
            df = self._redact_pii(df)
        df = self._calculate_quality_scores(df)
        return df
