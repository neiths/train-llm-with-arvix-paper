"""Text normalization processor."""
import re
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, length
from pyspark.sql.types import StringType

from .base import BaseProcessor


class NormalizationProcessor(BaseProcessor):
    """Normalize and clean text content."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, app_name="TextNormalization", **kwargs)
    
    def process(self, input_df: DataFrame) -> DataFrame:
        """Normalize text content."""
        
        @udf(StringType())
        def normalize_text(text: str) -> str:
            if not text:
                return ""
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove control characters
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
            
            # Normalize unicode
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            return text.strip()
        
        return input_df \
            .withColumn("normalized_text", normalize_text(col("value"))) \
            .filter(length(col("normalized_text")) > 100)  # Filter very short texts