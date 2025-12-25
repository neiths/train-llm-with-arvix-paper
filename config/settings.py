"""Application configuration using pydantic-settings."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    
    # MinIO configuration
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    
    # arXiv configuration
    arxiv_categories: list[str] = ["cs.LG", "cs.CL", "cs.AI"]
    
    # Tokenization
    vocab_size: int = 32000
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_seq_length: int = 512
    
    # W&B
    wandb_project: str = "arxiv-llm-training"
    wandb_enabled: bool = False


settings = Settings()