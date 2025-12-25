"""Storage module."""
from .minio_client import MinIOClient
from .schemas import PaperMetadata, ProcessedDocument

__all__ = ["MinIOClient", "PaperMetadata", "ProcessedDocument"]