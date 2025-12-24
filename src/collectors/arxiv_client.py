"""
arXiv API client for fetching paper metadata and PDFs.

Implements rate limiting and retry logic for robust data collection.
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import aiofiles
import aiohttp
import arxiv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ArxivPaper:
    """Represents an arXiv paper with metadata and content."""
    
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    pdf_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comment: Optional[str] = None
    
    # Content fields (populated during processing)
    full_text: Optional[str] = None
    pdf_path: Optional[str] = None
    
    # Metadata
    collected_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comment": self.comment,
            "full_text": self.full_text,
            "pdf_path": self.pdf_path,
            "collected_at": self.collected_at.isoformat(),
            "content_hash": self.content_hash,
        }
    
    @classmethod
    def from_arxiv_result(cls, result: arxiv.Result) -> "ArxivPaper":
        """Create from arxiv library Result object."""
        return cls(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            abstract=result.summary,
            authors=[author.name for author in result.authors],
            categories=result.categories,
            primary_category=result.primary_category,
            published=result.published,
            updated=result.updated,
            pdf_url=result.pdf_url,
            doi=result.doi,
            journal_ref=result.journal_ref,
            comment=result.comment,
        )


class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, rate: float = 3.0):
        """
        Initialize rate limiter.
        
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_passed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + time_passed * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class ArxivClient:
    """
    Client for fetching papers from arXiv.
    
    Features:
    - Rate limiting to respect arXiv API limits
    - Async PDF downloads
    - Retry logic for transient failures
    - Streaming results for memory efficiency
    """
    
    def __init__(
        self,
        rate_limit: Optional[float] = None,
        download_dir: Optional[Path] = None,
    ):
        """
        Initialize the arXiv client.
        
        Args:
            rate_limit: Requests per second (default from settings)
            download_dir: Directory for PDF downloads (default from settings)
        """
        self.rate_limit = rate_limit or settings.arxiv.rate_limit
        self.download_dir = download_dir or settings.data_dir / "pdfs"
        self.rate_limiter = RateLimiter(self.rate_limit)
        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=1.0 / self.rate_limit,
            num_retries=3,
        )
        
        logger.info(
            "ArxivClient initialized",
            rate_limit=self.rate_limit,
            download_dir=str(self.download_dir),
        )
    
    def search(
        self,
        query: Optional[str] = None,
        categories: Optional[list[str]] = None,
        max_results: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ) -> AsyncIterator[ArxivPaper]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query string
            categories: List of arXiv categories (e.g., ["cs.LG", "cs.CL"])
            max_results: Maximum number of results
            start_date: Filter papers after this date
            end_date: Filter papers before this date
            sort_by: Sort criterion
            sort_order: Sort order
            
        Yields:
            ArxivPaper: Paper metadata and content
        """
        # Build query
        query_parts = []
        
        if query:
            query_parts.append(query)
        
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query_parts.append(f"({cat_query})")
        
        if start_date:
            query_parts.append(f"submittedDate:[{start_date.strftime('%Y%m%d')}* TO *]")
        
        if end_date:
            query_parts.append(f"submittedDate:[* TO {end_date.strftime('%Y%m%d')}*]")
        
        final_query = " AND ".join(query_parts) if query_parts else "all:*"
        max_results = max_results or settings.arxiv.max_results
        
        logger.info(
            "Starting arXiv search",
            query=final_query,
            max_results=max_results,
        )
        
        search = arxiv.Search(
            query=final_query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        
        return self._fetch_results(search)
    
    async def _fetch_results(self, search: arxiv.Search) -> AsyncIterator[ArxivPaper]:
        """Fetch results with rate limiting."""
        count = 0
        
        for result in self._client.results(search):
            await self.rate_limiter.acquire()
            
            paper = ArxivPaper.from_arxiv_result(result)
            count += 1
            
            if count % 100 == 0:
                logger.info("Fetched papers", count=count)
            
            yield paper
        
        logger.info("Search complete", total_papers=count)
    
    @retry(
        retry=retry_if_exception_type(aiohttp.ClientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def download_pdf(
        self,
        paper: ArxivPaper,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Download PDF for a paper.
        
        Args:
            paper: Paper to download
            output_dir: Output directory (default from settings)
            
        Returns:
            Path: Path to downloaded PDF
        """
        output_dir = output_dir or self.download_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_id = re.sub(r"[^\w\-.]", "_", paper.arxiv_id)
        pdf_path = output_dir / f"{safe_id}.pdf"
        
        if pdf_path.exists():
            logger.debug("PDF already exists", arxiv_id=paper.arxiv_id)
            return pdf_path
        
        await self.rate_limiter.acquire()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(paper.pdf_url) as response:
                response.raise_for_status()
                
                async with aiofiles.open(pdf_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
        
        # Calculate content hash
        paper.pdf_path = str(pdf_path)
        paper.content_hash = await self._calculate_hash(pdf_path)
        
        logger.debug(
            "Downloaded PDF",
            arxiv_id=paper.arxiv_id,
            path=str(pdf_path),
        )
        
        return pdf_path
    
    async def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    async def collect_papers(
        self,
        categories: Optional[list[str]] = None,
        limit: int = 100,
        download_pdfs: bool = True,
    ) -> list[ArxivPaper]:
        """
        Convenience method to collect papers.
        
        Args:
            categories: arXiv categories to collect
            limit: Maximum number of papers
            download_pdfs: Whether to download PDFs
            
        Returns:
            list[ArxivPaper]: Collected papers
        """
        categories = categories or settings.arxiv.category_list
        papers = []
        
        async for paper in self.search(categories=categories, max_results=limit):
            if download_pdfs:
                try:
                    await self.download_pdf(paper)
                except Exception as e:
                    logger.warning(
                        "Failed to download PDF",
                        arxiv_id=paper.arxiv_id,
                        error=str(e),
                    )
            
            papers.append(paper)
        
        logger.info("Collection complete", total_papers=len(papers))
        return papers
