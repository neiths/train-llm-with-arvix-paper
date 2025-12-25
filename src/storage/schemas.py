"""Data models for the storage layer."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata for an arXiv paper."""
    
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2301.00001')")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(..., description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    categories: list[str] = Field(..., description="List of categories")
    published: datetime = Field(..., description="Publication date")
    updated: Optional[datetime] = None
    pdf_url: str
    
    # Processing status
    pdf_downloaded: bool = False
    text_extracted: bool = False
    processed: bool = False


class ProcessedDocument(BaseModel):
    """A processed document ready for tokenization."""
    
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2301.00001')")
    title: str = Field(..., description="Paper title")
    content: str = Field(..., description="Cleaned text content")
    word_count: int = Field(..., description="Word count")
    created_at: datetime = Field(default_factory=datetime.utcnow)
