"""
Data schemas for the arXiv LLM pipeline.

Defines Pydantic models and Arrow schemas for data validation.
"""

from datetime import datetime
from typing import Optional

import pyarrow as pa
from pydantic import BaseModel, Field, field_validator


class PaperMetadata(BaseModel):
    """Pydantic model for paper metadata validation."""
    
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., min_length=1, description="Paper title")
    abstract: str = Field(default="", description="Paper abstract")
    authors: list[str] = Field(default_factory=list, description="Author names")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    primary_category: str = Field(default="", description="Primary category")
    published: datetime = Field(..., description="Publication date")
    updated: datetime = Field(..., description="Last update date")
    pdf_url: str = Field(..., description="PDF download URL")
    doi: Optional[str] = Field(default=None, description="DOI")
    journal_ref: Optional[str] = Field(default=None, description="Journal reference")
    comment: Optional[str] = Field(default=None, description="Author comment")
    full_text: Optional[str] = Field(default=None, description="Extracted full text")
    pdf_path: Optional[str] = Field(default=None, description="Local PDF path")
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="Collection timestamp")
    content_hash: Optional[str] = Field(default=None, description="Content hash")
    
    @field_validator("arxiv_id")
    @classmethod
    def validate_arxiv_id(cls, v: str) -> str:
        """Validate arXiv ID format."""
        v = v.strip()
        if not v:
            raise ValueError("arXiv ID cannot be empty")
        return v
    
    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        """Clean and normalize title."""
        return " ".join(v.split())


class ProcessedPaper(BaseModel):
    """Model for processed paper data."""
    
    arxiv_id: str
    title: str
    text: str = Field(..., description="Processed text content")
    word_count: int = Field(default=0, description="Word count")
    char_count: int = Field(default=0, description="Character count")
    language: str = Field(default="en", description="Detected language")
    
    # Quality metrics
    is_duplicate: bool = Field(default=False, description="Duplicate flag")
    duplicate_of: Optional[str] = Field(default=None, description="Original paper ID if duplicate")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality score")
    
    # Processing metadata
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_version: str = Field(default="1.0", description="Processing pipeline version")


class TokenizedPaper(BaseModel):
    """Model for tokenized paper data."""
    
    arxiv_id: str
    token_ids: list[int] = Field(..., description="Token IDs")
    attention_mask: list[int] = Field(default_factory=list, description="Attention mask")
    num_tokens: int = Field(default=0, description="Number of tokens")
    
    # Tokenization metadata
    tokenizer_name: str = Field(default="sentencepiece", description="Tokenizer used")
    vocab_size: int = Field(default=32000, description="Vocabulary size")
    tokenized_at: datetime = Field(default_factory=datetime.utcnow)


class PaperSchema:
    """
    Arrow schema definitions for Parquet storage.
    
    Provides consistent schemas for data serialization.
    """
    
    @property
    def arrow_schema(self) -> pa.Schema:
        """Get Arrow schema for paper metadata."""
        return pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("abstract", pa.string()),
            pa.field("authors", pa.list_(pa.string())),
            pa.field("categories", pa.list_(pa.string())),
            pa.field("primary_category", pa.string()),
            pa.field("published", pa.timestamp("us")),
            pa.field("updated", pa.timestamp("us")),
            pa.field("pdf_url", pa.string()),
            pa.field("doi", pa.string()),
            pa.field("journal_ref", pa.string()),
            pa.field("comment", pa.string()),
            pa.field("full_text", pa.large_string()),
            pa.field("pdf_path", pa.string()),
            pa.field("collected_at", pa.timestamp("us")),
            pa.field("content_hash", pa.string()),
        ])
    
    @property
    def processed_schema(self) -> pa.Schema:
        """Get Arrow schema for processed papers."""
        return pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("text", pa.large_string(), nullable=False),
            pa.field("word_count", pa.int32()),
            pa.field("char_count", pa.int32()),
            pa.field("language", pa.string()),
            pa.field("is_duplicate", pa.bool_()),
            pa.field("duplicate_of", pa.string()),
            pa.field("quality_score", pa.float32()),
            pa.field("processed_at", pa.timestamp("us")),
            pa.field("processing_version", pa.string()),
        ])
    
    @property
    def tokenized_schema(self) -> pa.Schema:
        """Get Arrow schema for tokenized papers."""
        return pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("attention_mask", pa.list_(pa.int32())),
            pa.field("num_tokens", pa.int32()),
            pa.field("tokenizer_name", pa.string()),
            pa.field("vocab_size", pa.int32()),
            pa.field("tokenized_at", pa.timestamp("us")),
        ])


class TrainingExample(BaseModel):
    """Model for a training example."""
    
    input_ids: list[int]
    attention_mask: list[int]
    labels: Optional[list[int]] = None
    
    # Metadata
    source_id: str
    sequence_index: int = 0
    total_sequences: int = 1


class TrainingBatch(BaseModel):
    """Model for a training batch."""
    
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    labels: Optional[list[list[int]]] = None
    batch_size: int
    max_length: int
