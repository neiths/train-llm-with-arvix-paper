"""Application configuration using pydantic-settings."""
from functools import cached_property
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
    
    # arXiv configuration (comma-separated string from env)
    arxiv_categories_str: str = "cs.LG,cs.CL,cs.AI"
    
    @cached_property
    def arxiv_categories(self) -> list[str]:
        """Parse comma-separated categories string to list."""
        return [cat.strip() for cat in self.arxiv_categories_str.split(",") if cat.strip()]
    
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