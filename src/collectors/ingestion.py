"""
Data ingestion pipeline for uploading collected data to storage.

Handles validation, transformation, and upload to MinIO/S3.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings
from src.collectors.arxiv_client import ArxivPaper
from src.logging_config import get_logger
from src.storage.minio_client import MinioStorage
from src.storage.schemas import PaperSchema

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Pipeline for ingesting collected data into storage.
    
    Features:
    - Validates data against schemas
    - Converts to Parquet format
    - Uploads to MinIO/S3 with proper partitioning
    - Tracks ingestion metadata
    """
    
    def __init__(
        self,
        storage: Optional[MinioStorage] = None,
        batch_size: int = 100,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            storage: Storage client (creates new if not provided)
            batch_size: Number of papers per batch
        """
        self.storage = storage or MinioStorage()
        self.batch_size = batch_size
        self.schema = PaperSchema()
        
        logger.info("IngestionPipeline initialized", batch_size=batch_size)
    
    async def ingest_papers(
        self,
        papers: list[ArxivPaper],
        partition_by_date: bool = True,
    ) -> dict:
        """
        Ingest papers into storage.
        
        Args:
            papers: List of papers to ingest
            partition_by_date: Whether to partition by date
            
        Returns:
            dict: Ingestion statistics
        """
        if not papers:
            logger.warning("No papers to ingest")
            return {"ingested": 0, "failed": 0}
        
        stats = {
            "ingested": 0,
            "failed": 0,
            "start_time": datetime.utcnow().isoformat(),
            "batches": [],
        }
        
        # Process in batches
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i:i + self.batch_size]
            batch_id = i // self.batch_size + 1
            
            try:
                batch_stats = await self._process_batch(
                    batch, 
                    batch_id, 
                    partition_by_date
                )
                stats["batches"].append(batch_stats)
                stats["ingested"] += batch_stats["count"]
                
            except Exception as e:
                logger.error(
                    "Batch processing failed",
                    batch_id=batch_id,
                    error=str(e),
                )
                stats["failed"] += len(batch)
        
        stats["end_time"] = datetime.utcnow().isoformat()
        
        logger.info(
            "Ingestion complete",
            ingested=stats["ingested"],
            failed=stats["failed"],
        )
        
        return stats
    
    async def _process_batch(
        self,
        papers: list[ArxivPaper],
        batch_id: int,
        partition_by_date: bool,
    ) -> dict:
        """Process a batch of papers."""
        logger.debug("Processing batch", batch_id=batch_id, count=len(papers))
        
        # Convert to Arrow table
        data = [self._paper_to_row(paper) for paper in papers]
        table = pa.Table.from_pylist(data, schema=self.schema.arrow_schema)
        
        # Determine partition path
        if partition_by_date:
            # Use current date for partition
            now = datetime.utcnow()
            partition_path = f"year={now.year}/month={now.month:02d}/day={now.day:02d}"
        else:
            partition_path = "unpartitioned"
        
        # Write to temporary file
        temp_path = Path(f"/tmp/papers_batch_{batch_id}.parquet")
        pq.write_table(table, temp_path)
        
        # Upload to storage
        object_name = f"{partition_path}/papers_{batch_id}_{now.strftime('%H%M%S')}.parquet"
        
        await self.storage.upload_file(
            bucket_name=settings.buckets.raw,
            object_name=object_name,
            file_path=temp_path,
        )
        
        # Cleanup
        temp_path.unlink(missing_ok=True)
        
        return {
            "batch_id": batch_id,
            "count": len(papers),
            "path": object_name,
        }
    
    def _paper_to_row(self, paper: ArxivPaper) -> dict:
        """Convert paper to row dict for Arrow."""
        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "categories": paper.categories,
            "primary_category": paper.primary_category,
            "published": paper.published,
            "updated": paper.updated,
            "pdf_url": paper.pdf_url,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comment": paper.comment,
            "full_text": paper.full_text,
            "pdf_path": paper.pdf_path,
            "collected_at": paper.collected_at,
            "content_hash": paper.content_hash,
        }
    
    async def ingest_from_json(
        self,
        json_path: Path,
        partition_by_date: bool = True,
    ) -> dict:
        """
        Ingest papers from a JSON file.
        
        Args:
            json_path: Path to JSON file
            partition_by_date: Whether to partition by date
            
        Returns:
            dict: Ingestion statistics
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        papers = []
        for item in data:
            paper = ArxivPaper(
                arxiv_id=item["arxiv_id"],
                title=item["title"],
                abstract=item["abstract"],
                authors=item["authors"],
                categories=item["categories"],
                primary_category=item["primary_category"],
                published=datetime.fromisoformat(item["published"]),
                updated=datetime.fromisoformat(item["updated"]),
                pdf_url=item["pdf_url"],
                doi=item.get("doi"),
                journal_ref=item.get("journal_ref"),
                comment=item.get("comment"),
                full_text=item.get("full_text"),
                pdf_path=item.get("pdf_path"),
            )
            papers.append(paper)
        
        return await self.ingest_papers(papers, partition_by_date)
    
    async def create_manifest(
        self,
        prefix: str = "",
    ) -> dict:
        """
        Create a manifest of ingested data.
        
        Args:
            prefix: Path prefix to scan
            
        Returns:
            dict: Manifest with file information
        """
        files = await self.storage.list_objects(
            bucket_name=settings.buckets.raw,
            prefix=prefix,
        )
        
        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "bucket": settings.buckets.raw,
            "prefix": prefix,
            "files": files,
            "total_files": len(files),
        }
        
        # Upload manifest
        manifest_path = Path("/tmp/manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        await self.storage.upload_file(
            bucket_name=settings.buckets.raw,
            object_name="_manifest.json",
            file_path=manifest_path,
        )
        
        manifest_path.unlink(missing_ok=True)
        
        logger.info("Manifest created", total_files=len(files))
        return manifest
