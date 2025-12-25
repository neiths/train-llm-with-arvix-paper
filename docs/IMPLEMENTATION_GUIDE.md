# LLM Training Pipeline - Step-by-Step Implementation Guide

This guide breaks down the entire project into **6 phases** with clear milestones, allowing you to implement the pipeline incrementally from scratch.

---

## 📋 Overview

| Phase | Component | Estimated Time | Dependencies |
|-------|-----------|----------------|--------------|
| 1 | Project Setup & Configuration | 1-2 hours | None |
| 2 | Storage Layer (MinIO) | 2-3 hours | Phase 1 |
| 3 | Data Collection (arXiv) | 3-4 hours | Phase 1, 2 |
| 4 | Data Processing (PySpark) | 4-6 hours | Phase 2, 3 |
| 5 | Tokenization | 2-3 hours | Phase 4 |
| 6 | Model Training (PyTorch) | 4-6 hours | Phase 5 |

---

## Phase 1: Project Setup & Configuration

### 1.1 Create Project Structure

```bash
mkdir train-llm-with-arxiv-data
cd train-llm-with-arxiv-data

# Create directory structure
mkdir -p src/{collectors,storage,processing,tokenization,training}
mkdir -p data/{raw,processed,tokenized,pdfs}
mkdir -p config notebooks tests
```

### 1.2 Initialize with `pyproject.toml`

Create `pyproject.toml`:

```toml
[project]
name = "arxiv-llm-pipeline"
version = "0.1.0"
description = "LLM Training Pipeline with arXiv Data"
requires-python = ">=3.10"
dependencies = [
    # Core
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    
    # CLI
    "click>=8.0",
    "rich>=13.0",
    
    # arXiv collection
    "arxiv>=2.0",
    "requests>=2.28",
    "aiohttp>=3.8",
    
    # PDF parsing
    "PyMuPDF>=1.23",
    
    # Storage
    "minio>=7.2",
    
    # Processing (add later in Phase 4)
    # "pyspark>=3.5",
    
    # Tokenization
    "sentencepiece>=0.1.99",
    
    # Training
    "torch>=2.0",
    "wandb>=0.16",
]

[project.scripts]
arxiv-llm = "src.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 1.3 Create Configuration

Create `config/settings.py`:

```python
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
```

### 1.4 Create `.env.example`

```env
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# arXiv Categories
ARXIV_CATEGORIES=cs.LG,cs.CL,cs.AI

# Tokenization
VOCAB_SIZE=32000

# Training
BATCH_SIZE=8
LEARNING_RATE=1e-4

# W&B (optional)
WANDB_PROJECT=arxiv-llm-training
WANDB_ENABLED=false
```

### ✅ Phase 1 Checkpoint

- [ ] Directory structure created
- [ ] `pyproject.toml` with dependencies
- [ ] Configuration system with pydantic-settings
- [ ] Can run `uv sync` successfully

---

## Phase 2: Storage Layer (MinIO)

### 2.1 Docker Compose for MinIO

Create `docker-compose.yml`:

```yaml
version: "3.9"

services:
  minio:
    image: minio/minio:latest
    container_name: arxiv-minio
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  minio_data:
```

### 2.2 Create Storage Schemas

Create `src/storage/schemas.py`:

```python
"""Data models for the storage layer."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata for an arXiv paper."""
    
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2301.00001')")
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    updated: Optional[datetime] = None
    pdf_url: str
    
    # Processing status
    pdf_downloaded: bool = False
    text_extracted: bool = False
    processed: bool = False


class ProcessedDocument(BaseModel):
    """A processed document ready for tokenization."""
    
    arxiv_id: str
    title: str
    content: str  # Cleaned text content
    word_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### 2.3 Create MinIO Client

Create `src/storage/minio_client.py`:

```python
"""MinIO client for object storage operations."""
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Generator, Optional

from minio import Minio
from minio.error import S3Error

from config.settings import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """Client for interacting with MinIO object storage."""
    
    # Bucket names
    RAW_BUCKET = "raw-data"
    PROCESSED_BUCKET = "processed-data"
    TOKENIZED_BUCKET = "tokenized-data"
    
    def __init__(self):
        """Initialize MinIO client."""
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self._ensure_buckets()
    
    def _ensure_buckets(self) -> None:
        """Create required buckets if they don't exist."""
        for bucket in [self.RAW_BUCKET, self.PROCESSED_BUCKET, self.TOKENIZED_BUCKET]:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Error creating bucket {bucket}: {e}")
                raise
    
    def upload_file(self, bucket: str, object_name: str, file_path: Path) -> bool:
        """Upload a file to MinIO."""
        try:
            self.client.fput_object(bucket, object_name, str(file_path))
            logger.debug(f"Uploaded {file_path} to {bucket}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def upload_json(self, bucket: str, object_name: str, data: dict) -> bool:
        """Upload JSON data to MinIO."""
        try:
            json_bytes = json.dumps(data, default=str).encode("utf-8")
            self.client.put_object(
                bucket,
                object_name,
                BytesIO(json_bytes),
                len(json_bytes),
                content_type="application/json"
            )
            return True
        except S3Error as e:
            logger.error(f"JSON upload failed: {e}")
            return False
    
    def download_file(self, bucket: str, object_name: str, file_path: Path) -> bool:
        """Download a file from MinIO."""
        try:
            self.client.fget_object(bucket, object_name, str(file_path))
            return True
        except S3Error as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def list_objects(
        self, bucket: str, prefix: str = ""
    ) -> Generator[str, None, None]:
        """List objects in a bucket."""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            for obj in objects:
                yield obj.object_name
        except S3Error as e:
            logger.error(f"List objects failed: {e}")
```

### 2.4 Create `__init__.py`

Create `src/storage/__init__.py`:

```python
"""Storage module."""
from .minio_client import MinIOClient
from .schemas import PaperMetadata, ProcessedDocument

__all__ = ["MinIOClient", "PaperMetadata", "ProcessedDocument"]
```

### ✅ Phase 2 Checkpoint

- [ ] Docker Compose runs MinIO successfully
- [ ] MinIO Console accessible at http://localhost:9001
- [ ] `MinIOClient` can create buckets and upload/download files
- [ ] Test with: `docker-compose up -d && python -c "from src.storage import MinIOClient; c = MinIOClient()"`

---

## Phase 3: Data Collection (arXiv)

### 3.1 Create arXiv API Client

Create `src/collectors/arxiv_client.py`:

```python
"""arXiv API client for paper collection."""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import arxiv

from src.storage.schemas import PaperMetadata

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for querying arXiv API and downloading papers."""
    
    def __init__(self, data_dir: Path):
        """Initialize arXiv client."""
        self.data_dir = data_dir
        self.pdf_dir = data_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
    
    def search_papers(
        self,
        categories: list[str],
        limit: int = 100,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate
    ) -> list[PaperMetadata]:
        """Search for papers in specified categories."""
        # Build query
        query = " OR ".join(f"cat:{cat}" for cat in categories)
        
        logger.info(f"Searching arXiv: {query} (limit={limit})")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=sort_by
        )
        
        papers = []
        for result in client.results(search):
            paper = PaperMetadata(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                categories=result.categories,
                published=result.published,
                updated=result.updated,
                pdf_url=result.pdf_url
            )
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers")
        return papers
    
    async def download_pdf(
        self,
        paper: PaperMetadata,
        session: aiohttp.ClientSession
    ) -> Optional[Path]:
        """Download PDF for a paper."""
        pdf_path = self.pdf_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        
        if pdf_path.exists():
            logger.debug(f"PDF already exists: {pdf_path}")
            return pdf_path
        
        try:
            async with session.get(paper.pdf_url) as response:
                if response.status == 200:
                    content = await response.read()
                    pdf_path.write_bytes(content)
                    logger.info(f"Downloaded: {paper.arxiv_id}")
                    return pdf_path
                else:
                    logger.warning(f"Failed to download {paper.arxiv_id}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading {paper.arxiv_id}: {e}")
            return None
    
    async def download_papers(
        self,
        papers: list[PaperMetadata],
        concurrency: int = 5
    ) -> list[Path]:
        """Download PDFs for multiple papers with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)
        downloaded = []
        
        async def download_with_semaphore(paper: PaperMetadata, session: aiohttp.ClientSession):
            async with semaphore:
                return await self.download_pdf(paper, session)
        
        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [download_with_semaphore(p, session) for p in papers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Path):
                    downloaded.append(result)
        
        logger.info(f"Downloaded {len(downloaded)}/{len(papers)} PDFs")
        return downloaded
```

### 3.2 Create PDF Text Extractor

Create `src/collectors/ingestion.py`:

```python
"""PDF ingestion and text extraction."""
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text content from PDF files."""
    
    def __init__(self, output_dir: Path):
        """Initialize PDF extractor."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                text_parts.append(text)
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            logger.debug(f"Extracted {len(full_text)} characters from {pdf_path.name}")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def process_pdf(self, pdf_path: Path) -> Optional[Path]:
        """Process a PDF and save extracted text."""
        text = self.extract_text(pdf_path)
        
        if text is None:
            return None
        
        # Save to output directory
        output_path = self.output_dir / f"{pdf_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        
        return output_path
    
    def process_directory(self, pdf_dir: Path) -> list[Path]:
        """Process all PDFs in a directory."""
        pdf_files = list(pdf_dir.glob("*.pdf"))
        processed = []
        
        logger.info(f"Processing {len(pdf_files)} PDFs...")
        
        for pdf_path in pdf_files:
            result = self.process_pdf(pdf_path)
            if result:
                processed.append(result)
        
        logger.info(f"Processed {len(processed)}/{len(pdf_files)} PDFs")
        return processed
```

### 3.3 Create Collectors `__init__.py`

Create `src/collectors/__init__.py`:

```python
"""Data collectors module."""
from .arxiv_client import ArxivClient
from .ingestion import PDFExtractor

__all__ = ["ArxivClient", "PDFExtractor"]
```

### ✅ Phase 3 Checkpoint

- [ ] Can search arXiv API for papers
- [ ] Can download PDFs asynchronously
- [ ] Can extract text from PDFs using PyMuPDF
- [ ] Test with: `python -c "from src.collectors import ArxivClient; c = ArxivClient(Path('data')); papers = c.search_papers(['cs.LG'], limit=5); print(papers)"`

---

## Phase 4: Data Processing (PySpark)

> ⚠️ **Note**: This phase requires WSL/Linux for PySpark. Skip if on Windows without WSL.

### 4.1 Add PySpark Dependencies

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "pyspark>=3.5",
]
```

### 4.2 Create Base Processor

Create `src/processing/base.py`:

```python
"""Base class for Spark processing jobs."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, spark: Optional[SparkSession] = None, app_name: str = "DataProcessor"):
        """Initialize processor with Spark session."""
        self.spark = spark or self._create_spark_session(app_name)
    
    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Create a local Spark session."""
        return SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
    
    @abstractmethod
    def process(self, input_df: DataFrame) -> DataFrame:
        """Process input DataFrame."""
        pass
    
    def read_text_files(self, input_path: Path) -> DataFrame:
        """Read text files into a DataFrame."""
        return self.spark.read.text(str(input_path / "*.txt"))
    
    def write_output(self, df: DataFrame, output_path: Path, format: str = "parquet"):
        """Write DataFrame to output path."""
        df.write.mode("overwrite").format(format).save(str(output_path))
```

### 4.3 Create Normalization Processor

Create `src/processing/normalization.py`:

```python
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
```

### 4.4 Create Deduplication Processor

Create `src/processing/deduplication.py`:

```python
"""Deduplication processor using MinHash LSH."""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, xxhash64
from pyspark.sql.types import ArrayType, LongType
from pyspark.ml.feature import HashingTF, MinHashLSH

from .base import BaseProcessor


class DeduplicationProcessor(BaseProcessor):
    """Remove duplicate documents using MinHash LSH."""
    
    def __init__(self, *args, similarity_threshold: float = 0.8, **kwargs):
        super().__init__(*args, app_name="Deduplication", **kwargs)
        self.similarity_threshold = similarity_threshold
    
    def process(self, input_df: DataFrame) -> DataFrame:
        """Remove near-duplicate documents."""
        
        # Create shingles (n-grams)
        @udf(ArrayType(LongType()))
        def create_shingles(text: str, n: int = 5) -> list:
            if not text or len(text) < n:
                return []
            words = text.split()
            shingles = []
            for i in range(len(words) - n + 1):
                shingle = " ".join(words[i:i+n])
                shingles.append(hash(shingle))
            return shingles
        
        # Add document ID
        df = input_df.withColumn("doc_id", xxhash64(col("normalized_text")))
        
        # Create shingles
        df = df.withColumn("shingles", create_shingles(col("normalized_text")))
        
        # Use HashingTF for feature vectors
        hashing_tf = HashingTF(inputCol="shingles", outputCol="features", numFeatures=1000)
        df = hashing_tf.transform(df)
        
        # Apply MinHash LSH
        mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
        model = mh.fit(df)
        
        # Find similar pairs and remove duplicates (keep first occurrence)
        similar_pairs = model.approxSimilarityJoin(
            df, df, 
            threshold=1 - self.similarity_threshold,
            distCol="distance"
        ).filter(col("datasetA.doc_id") < col("datasetB.doc_id"))
        
        # Get IDs to remove
        duplicates_to_remove = similar_pairs.select(col("datasetB.doc_id").alias("remove_id"))
        
        # Filter out duplicates
        result = df.join(
            duplicates_to_remove,
            df.doc_id == duplicates_to_remove.remove_id,
            "left_anti"
        )
        
        return result.select("normalized_text")
```

### 4.5 Create Pipeline Orchestrator

Create `src/processing/pipeline.py`:

```python
"""Data processing pipeline orchestration."""
import logging
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession

from .normalization import NormalizationProcessor
from .deduplication import DeduplicationProcessor

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrate the data processing pipeline."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize pipeline with shared Spark session."""
        self.spark = spark or SparkSession.builder \
            .appName("ProcessingPipeline") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
    
    def run(self, input_path: Path, output_path: Path) -> bool:
        """Run the complete processing pipeline."""
        try:
            logger.info(f"Starting pipeline: {input_path} -> {output_path}")
            
            # Read raw text files
            df = self.spark.read.text(str(input_path / "*.txt"))
            logger.info(f"Loaded {df.count()} documents")
            
            # Step 1: Normalization
            normalizer = NormalizationProcessor(spark=self.spark)
            df = normalizer.process(df)
            logger.info(f"After normalization: {df.count()} documents")
            
            # Step 2: Deduplication
            deduplicator = DeduplicationProcessor(spark=self.spark)
            df = deduplicator.process(df)
            logger.info(f"After deduplication: {df.count()} documents")
            
            # Save output
            output_path.mkdir(parents=True, exist_ok=True)
            df.write.mode("overwrite").text(str(output_path))
            
            logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
```

### ✅ Phase 4 Checkpoint

- [ ] Normalization processor cleans text
- [ ] Deduplication processor removes near-duplicates
- [ ] Pipeline orchestrator runs all steps
- [ ] Test (in WSL): `arxiv-llm process pipeline -i data/raw -o data/processed`

---

## Phase 5: Tokenization

### 5.1 Create SentencePiece Trainer

Create `src/tokenization/trainer.py`:

```python
"""Tokenizer training module."""
import logging
from pathlib import Path
from typing import Optional

import sentencepiece as spm

logger = logging.getLogger(__name__)


class TokenizerTrainer:
    """Train SentencePiece tokenizer on text data."""
    
    def __init__(self, vocab_size: int = 32000, model_type: str = "bpe"):
        """Initialize tokenizer trainer."""
        self.vocab_size = vocab_size
        self.model_type = model_type
    
    def train(
        self,
        input_file: Path,
        output_prefix: Path,
        character_coverage: float = 0.9995
    ) -> Path:
        """Train tokenizer on input text file."""
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training {self.model_type} tokenizer with vocab_size={self.vocab_size}")
        
        spm.SentencePieceTrainer.train(
            input=str(input_file),
            model_prefix=str(output_prefix),
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        
        model_path = output_prefix.with_suffix(".model")
        logger.info(f"Tokenizer saved to: {model_path}")
        return model_path


class Tokenizer:
    """Wrapper for SentencePiece tokenizer."""
    
    def __init__(self, model_path: Path):
        """Load tokenizer from model file."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
    
    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.sp.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(ids)
    
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]
```

### 5.2 Create Tokenization `__init__.py`

Create `src/tokenization/__init__.py`:

```python
"""Tokenization module."""
from .trainer import TokenizerTrainer, Tokenizer

__all__ = ["TokenizerTrainer", "Tokenizer"]
```

### ✅ Phase 5 Checkpoint

- [ ] Can train SentencePiece tokenizer
- [ ] Can encode/decode text
- [ ] Test: `arxiv-llm tokenize train -i data/processed/text.txt --vocab-size 32000`

---

## Phase 6: Model Training (PyTorch)

### 6.1 Create Transformer Model

Create `src/training/model.py`:

```python
"""Transformer model for language modeling."""
import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    """Decoder-only Transformer for language modeling."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Causal mask
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # If no memory, use self-attention only (decoder-only style)
        if memory is None:
            memory = x
        
        x = self.transformer(x, memory, tgt_mask=mask)
        logits = self.output_projection(x)
        
        return logits
```

### 6.2 Create Trainer

Create `src/training/trainer.py`:

```python
"""Training loop and utilities."""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import TransformerLM

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for tokenized text sequences."""
    
    def __init__(self, data_path: Path, seq_length: int = 512):
        """Load tokenized data."""
        self.seq_length = seq_length
        
        # Load all token IDs
        self.tokens = []
        for file in data_path.glob("*.txt"):
            with open(file) as f:
                for line in f:
                    ids = [int(x) for x in line.strip().split()]
                    self.tokens.extend(ids)
        
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        logger.info(f"Loaded {len(self.tokens)} tokens")
    
    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_length - 1)
    
    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.seq_length + 1]
        return chunk[:-1], chunk[1:]  # input, target


class Trainer:
    """Trainer for Transformer language model."""
    
    def __init__(
        self,
        model: TransformerLM,
        train_dataset: TextDataset,
        val_dataset: Optional[TextDataset] = None,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        device: str = "auto"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)
    
    def train(self, epochs: int, checkpoint_dir: Path) -> None:
        """Train for multiple epochs."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
```

### 6.3 Create Training `__init__.py`

Create `src/training/__init__.py`:

```python
"""Training module."""
from .model import TransformerLM
from .trainer import Trainer, TextDataset

__all__ = ["TransformerLM", "Trainer", "TextDataset"]
```

### ✅ Phase 6 Checkpoint

- [ ] TransformerLM model compiles and runs
- [ ] Trainer can load data and train
- [ ] Test: `arxiv-llm train start --data data/tokenized --epochs 1 --batch-size 4`

---

## Phase 7: CLI (Tying It All Together)

### 7.1 Create CLI Entry Point

Create `src/cli.py`:

```python
"""Command-line interface for the LLM training pipeline."""
import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console

from config.settings import settings

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool):
    """arXiv LLM Training Pipeline CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@cli.group()
def collect():
    """Data collection commands."""
    pass


@collect.command("arxiv")
@click.option("--limit", "-l", default=100, help="Number of papers to collect")
@click.option("--category", "-c", multiple=True, default=["cs.LG"], help="arXiv categories")
@click.option("--download", is_flag=True, help="Download PDFs after metadata collection")
def collect_arxiv(limit: int, category: tuple, download: bool):
    """Collect papers from arXiv API."""
    from src.collectors import ArxivClient, PDFExtractor
    
    console.print(f"[bold blue]Collecting {limit} papers from arXiv...[/]")
    
    client = ArxivClient(settings.data_dir)
    papers = client.search_papers(list(category), limit=limit)
    
    console.print(f"[green]Found {len(papers)} papers[/]")
    
    if download:
        console.print("[bold blue]Downloading PDFs...[/]")
        asyncio.run(client.download_papers(papers))
        
        # Extract text
        extractor = PDFExtractor(settings.data_dir / "raw")
        extractor.process_directory(settings.data_dir / "pdfs")


@cli.group()
def process():
    """Data processing commands."""
    pass


@process.command("pipeline")
@click.option("--input", "-i", "input_path", type=Path, required=True)
@click.option("--output", "-o", "output_path", type=Path, required=True)
def run_pipeline(input_path: Path, output_path: Path):
    """Run the data processing pipeline."""
    from src.processing.pipeline import ProcessingPipeline
    
    console.print(f"[bold blue]Running processing pipeline...[/]")
    pipeline = ProcessingPipeline()
    success = pipeline.run(input_path, output_path)
    
    if success:
        console.print("[green]Pipeline completed successfully![/]")
    else:
        console.print("[red]Pipeline failed.[/]")


@cli.group()
def tokenize():
    """Tokenization commands."""
    pass


@tokenize.command("train")
@click.option("--input", "-i", "input_file", type=Path, required=True)
@click.option("--vocab-size", default=32000, help="Vocabulary size")
@click.option("--output", "-o", "output_dir", type=Path, default=Path("models"))
def train_tokenizer(input_file: Path, vocab_size: int, output_dir: Path):
    """Train a SentencePiece tokenizer."""
    from src.tokenization import TokenizerTrainer
    
    console.print(f"[bold blue]Training tokenizer (vocab_size={vocab_size})...[/]")
    
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    model_path = trainer.train(input_file, output_dir / "tokenizer")
    
    console.print(f"[green]Tokenizer saved to: {model_path}[/]")


@cli.group()
def train():
    """Model training commands."""
    pass


@train.command("start")
@click.option("--data", "-d", "data_path", type=Path, required=True)
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--batch-size", default=8, help="Batch size")
@click.option("--checkpoint-dir", type=Path, default=Path("checkpoints"))
def start_training(data_path: Path, epochs: int, batch_size: int, checkpoint_dir: Path):
    """Start model training."""
    from src.training import TransformerLM, Trainer, TextDataset
    
    console.print(f"[bold blue]Starting training...[/]")
    
    # Load dataset
    dataset = TextDataset(data_path)
    
    # Create model
    model = TransformerLM(vocab_size=settings.vocab_size)
    
    # Train
    trainer = Trainer(model, dataset, batch_size=batch_size)
    trainer.train(epochs, checkpoint_dir)
    
    console.print(f"[green]Training completed![/]")


if __name__ == "__main__":
    cli()
```

### 7.2 Create Main Package `__init__.py`

Create `src/__init__.py`:

```python
"""arXiv LLM Training Pipeline."""
__version__ = "0.1.0"
```

---

## ✅ Final Checklist

- [ ] **Phase 1**: Project setup complete
- [ ] **Phase 2**: MinIO storage layer working
- [ ] **Phase 3**: arXiv collection and PDF extraction working
- [ ] **Phase 4**: PySpark processing pipeline running (WSL/Linux)
- [ ] **Phase 5**: Tokenizer training and encoding working
- [ ] **Phase 6**: Model training with PyTorch working
- [ ] **Phase 7**: CLI commands all functional

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Collect papers | `arxiv-llm collect arxiv --limit 100 --download` |
| Process data | `arxiv-llm process pipeline -i data/raw -o data/processed` |
| Train tokenizer | `arxiv-llm tokenize train -i data/processed/text.txt` |
| Start training | `arxiv-llm train start --data data/tokenized --epochs 3` |

---

**Happy building! 🚀**
