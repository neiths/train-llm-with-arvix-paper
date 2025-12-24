"""
Configuration management for the arXiv LLM training pipeline.

Uses Pydantic Settings for environment-based configuration with validation.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class MinIOSettings(BaseSettings):
    """MinIO/S3 configuration."""
    
    model_config = SettingsConfigDict(env_prefix="MINIO_")
    
    endpoint: str = Field(default="localhost:9000", description="MinIO server endpoint")
    access_key: str = Field(default="minioadmin", description="Access key")
    secret_key: str = Field(default="minioadmin", description="Secret key")
    secure: bool = Field(default=False, description="Use HTTPS")
    region: str = Field(default="us-east-1", description="S3 region")


class BucketSettings(BaseSettings):
    """Storage bucket configuration."""
    
    model_config = SettingsConfigDict(env_prefix="BUCKET_")
    
    raw: str = Field(default="raw-data", description="Raw data bucket")
    processed: str = Field(default="processed-data", description="Processed data bucket")
    curated: str = Field(default="curated-data", description="Curated data bucket")
    models: str = Field(default="models", description="Models bucket")


class ArxivSettings(BaseSettings):
    """arXiv API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="ARXIV_")
    
    rate_limit: float = Field(default=3.0, description="Requests per second limit")
    max_results: int = Field(default=1000, description="Maximum results per query")
    categories: str = Field(
        default="cs.LG,cs.CL,cs.AI",
        description="Comma-separated arXiv categories"
    )
    
    @property
    def category_list(self) -> list[str]:
        """Parse categories into a list."""
        return [cat.strip() for cat in self.categories.split(",")]


class SparkSettings(BaseSettings):
    """Apache Spark configuration."""
    
    model_config = SettingsConfigDict(env_prefix="SPARK_")
    
    master: str = Field(default="local[*]", description="Spark master URL")
    driver_memory: str = Field(default="4g", description="Driver memory")
    executor_memory: str = Field(default="4g", description="Executor memory")
    app_name: str = Field(default="arxiv-llm-pipeline", description="Spark app name")


class ProcessingSettings(BaseSettings):
    """Data processing configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    dedup_threshold: float = Field(
        default=0.85, 
        ge=0.0, 
        le=1.0,
        description="Similarity threshold for deduplication"
    )
    min_text_length: int = Field(
        default=100,
        ge=0,
        description="Minimum text length to keep"
    )
    max_text_length: int = Field(
        default=1000000,
        description="Maximum text length to keep"
    )


class TokenizationSettings(BaseSettings):
    """Tokenization configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    vocab_size: int = Field(default=32000, description="Vocabulary size")
    tokenizer_type: str = Field(
        default="sentencepiece",
        description="Tokenizer type: sentencepiece or bpe"
    )
    
    @field_validator("tokenizer_type")
    @classmethod
    def validate_tokenizer_type(cls, v: str) -> str:
        valid_types = {"sentencepiece", "bpe"}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid tokenizer type: {v}. Must be one of {valid_types}")
        return v.lower()


class TrainingSettings(BaseSettings):
    """ML training configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    model_name: str = Field(default="arxiv-gpt", description="Model name")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    batch_size: int = Field(default=8, description="Training batch size")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    gradient_accumulation_steps: int = Field(
        default=4,
        description="Gradient accumulation steps"
    )
    warmup_ratio: float = Field(default=0.1, description="Warmup ratio")
    weight_decay: float = Field(default=0.01, description="Weight decay")


class WandBSettings(BaseSettings):
    """Weights & Biases configuration."""
    
    model_config = SettingsConfigDict(env_prefix="WANDB_")
    
    project: str = Field(default="arxiv-llm-training", description="W&B project name")
    entity: Optional[str] = Field(default=None, description="W&B entity/team")
    api_key: Optional[str] = Field(default=None, description="W&B API key")
    mode: str = Field(default="online", description="W&B mode: online, offline, disabled")


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format: json or text")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    
    # Base paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Base directory"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data",
        description="Data directory"
    )
    
    # Nested settings
    minio: MinIOSettings = Field(default_factory=MinIOSettings)
    buckets: BucketSettings = Field(default_factory=BucketSettings)
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    spark: SparkSettings = Field(default_factory=SparkSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    tokenization: TokenizationSettings = Field(default_factory=TokenizationSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    wandb: WandBSettings = Field(default_factory=WandBSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Convenience exports
settings = get_settings()
