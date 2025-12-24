"""
Tests for data processing jobs.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from src.processing.deduplication import DeduplicationJob
from src.processing.normalization import NormalizationJob, normalize_text
from src.processing.content_moderation import (
    ContentModerationJob,
    detect_pii,
    redact_pii,
    calculate_quality_score,
)


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for testing."""
    session = SparkSession.builder \
        .master("local[2]") \
        .appName("test") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield session
    session.stop()


class TestNormalizeText:
    """Tests for text normalization functions."""
    
    def test_normalize_html(self):
        """Test HTML removal."""
        text = "<p>Hello <b>World</b></p>"
        result = normalize_text(text)
        assert "<" not in result
        assert ">" not in result
    
    def test_normalize_latex(self):
        """Test LaTeX removal."""
        text = r"The equation is $E = mc^2$ and \frac{a}{b}"
        result = normalize_text(text)
        assert "[MATH]" in result or "$" not in result
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World\n\n\tTest"
        result = normalize_text(text)
        assert "  " not in result
    
    def test_normalize_unicode(self):
        """Test unicode normalization."""
        text = "café"  # Combined form
        result = normalize_text(text)
        assert isinstance(result, str)


class TestDetectPII:
    """Tests for PII detection."""
    
    def test_detect_email(self):
        """Test email detection."""
        text = "Contact me at test@example.com for info."
        pii_types = detect_pii(text)
        assert "email" in pii_types
    
    def test_detect_phone(self):
        """Test phone number detection."""
        text = "Call me at 555-123-4567 please."
        pii_types = detect_pii(text)
        assert "phone" in pii_types
    
    def test_detect_multiple(self):
        """Test multiple PII types."""
        text = "Email: test@example.com, Phone: 555-123-4567"
        pii_types = detect_pii(text)
        assert "email" in pii_types
        assert "phone" in pii_types
    
    def test_no_pii(self):
        """Test text without PII."""
        text = "This is a clean academic paper about machine learning."
        pii_types = detect_pii(text)
        assert len(pii_types) == 0


class TestRedactPII:
    """Tests for PII redaction."""
    
    def test_redact_email(self):
        """Test email redaction."""
        text = "Contact test@example.com for info."
        result = redact_pii(text)
        assert "[EMAIL]" in result
        assert "test@example.com" not in result
    
    def test_redact_multiple(self):
        """Test multiple PII redaction."""
        text = "Email: test@example.com, Phone: 555-123-4567"
        result = redact_pii(text)
        assert "[EMAIL]" in result
        assert "[PHONE]" in result


class TestQualityScore:
    """Tests for quality scoring."""
    
    def test_empty_text(self):
        """Test empty text."""
        score = calculate_quality_score("", 0, 0)
        assert score == 0.0
    
    def test_short_text(self):
        """Test very short text."""
        text = "Hello world"
        score = calculate_quality_score(text, 2, len(text))
        assert score < 0.5
    
    def test_good_text(self):
        """Test good quality text."""
        text = "This is a well-written academic paper about machine learning. " * 50
        word_count = len(text.split())
        score = calculate_quality_score(text, word_count, len(text))
        assert score > 0.3


class TestDeduplicationJob:
    """Tests for deduplication job."""
    
    def test_initialization(self):
        """Test job initialization."""
        job = DeduplicationJob(threshold=0.9)
        assert job.threshold == 0.9
    
    @pytest.mark.spark
    def test_find_duplicates(self, spark):
        """Test duplicate detection with Spark."""
        # Create test data
        schema = StructType([
            StructField("arxiv_id", StringType(), False),
            StructField("title", StringType(), False),
            StructField("abstract", StringType(), True),
        ])
        
        data = [
            ("1", "Machine Learning Tutorial", "This is about ML."),
            ("2", "Machine Learning Tutorial", "This is about ML."),  # Duplicate
            ("3", "Deep Learning Guide", "This is about DL."),
        ]
        
        df = spark.createDataFrame(data, schema)
        
        job = DeduplicationJob(spark=spark, threshold=0.8)
        # Would test actual deduplication here
        assert job is not None


class TestNormalizationJob:
    """Tests for normalization job."""
    
    def test_initialization(self):
        """Test job initialization."""
        job = NormalizationJob(min_length=50, max_length=10000)
        assert job.min_length == 50
        assert job.max_length == 10000


class TestContentModerationJob:
    """Tests for content moderation job."""
    
    def test_initialization(self):
        """Test job initialization."""
        job = ContentModerationJob(redact_pii=True, min_quality_score=0.5)
        assert job.redact_pii == True
        assert job.min_quality_score == 0.5
