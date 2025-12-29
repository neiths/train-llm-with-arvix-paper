"""Processing module for Spark data pipeline."""
from .base import BaseProcessor
from .normalization import NormalizationProcessor
from .deduplication import DeduplicationProcessor
from .pipeline import ProcessingPipeline

__all__ = [
    "BaseProcessor",
    "NormalizationProcessor", 
    "DeduplicationProcessor",
    "ProcessingPipeline",
]
