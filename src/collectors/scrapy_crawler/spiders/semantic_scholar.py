"""
Semantic Scholar spider for supplementary paper data.

Collects additional metadata like citations and references.
"""

from datetime import datetime
from typing import Any, Iterator

import scrapy
from scrapy.http import Response

from src.collectors.scrapy_crawler.items import ScrapyPaperItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class SemanticScholarSpider(scrapy.Spider):
    """
    Spider for collecting paper data from Semantic Scholar.
    
    This spider is used to supplement arXiv data with citation
    information and related papers.
    """
    
    name = "semantic_scholar"
    allowed_domains = ["api.semanticscholar.org"]
    
    # API configuration
    API_BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "paperId,title,abstract,authors,year,citationCount,fieldsOfStudy,externalIds"
    
    def __init__(
        self,
        arxiv_ids: list[str] | None = None,
        query: str | None = None,
        limit: int = 100,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the spider.
        
        Args:
            arxiv_ids: List of arXiv IDs to fetch
            query: Search query string
            limit: Maximum results for search
        """
        super().__init__(*args, **kwargs)
        self.arxiv_ids = arxiv_ids or []
        self.query = query
        self.limit = limit
    
    def start_requests(self) -> Iterator[scrapy.Request]:
        """Generate initial requests."""
        if self.arxiv_ids:
            # Fetch by arXiv IDs
            for arxiv_id in self.arxiv_ids:
                url = f"{self.API_BASE}/paper/ARXIV:{arxiv_id}?fields={self.FIELDS}"
                yield scrapy.Request(
                    url,
                    callback=self.parse_paper,
                    meta={"arxiv_id": arxiv_id},
                    headers={"Accept": "application/json"},
                )
        elif self.query:
            # Search by query
            url = f"{self.API_BASE}/paper/search?query={self.query}&limit={self.limit}&fields={self.FIELDS}"
            yield scrapy.Request(
                url,
                callback=self.parse_search,
                headers={"Accept": "application/json"},
            )
    
    def parse_search(self, response: Response) -> Iterator[ScrapyPaperItem]:
        """Parse search results."""
        data = response.json()
        papers = data.get("data", [])
        
        for paper in papers:
            yield self._extract_paper(paper)
        
        # Handle pagination
        next_offset = data.get("next")
        if next_offset and len(papers) > 0:
            next_url = f"{self.API_BASE}/paper/search?query={self.query}&limit={self.limit}&offset={next_offset}&fields={self.FIELDS}"
            yield scrapy.Request(
                next_url,
                callback=self.parse_search,
                headers={"Accept": "application/json"},
            )
    
    def parse_paper(self, response: Response) -> ScrapyPaperItem:
        """Parse a single paper response."""
        data = response.json()
        return self._extract_paper(data, arxiv_id=response.meta.get("arxiv_id"))
    
    def _extract_paper(
        self,
        data: dict,
        arxiv_id: str | None = None,
    ) -> ScrapyPaperItem:
        """Extract paper data from API response."""
        # Extract arXiv ID from external IDs if not provided
        if not arxiv_id:
            external_ids = data.get("externalIds", {})
            arxiv_id = external_ids.get("ArXiv", "")
        
        # Extract authors
        authors = [
            author.get("name", "")
            for author in data.get("authors", [])
        ]
        
        item = ScrapyPaperItem(
            source="semantic_scholar",
            source_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=authors,
            published_date=str(data.get("year", "")),
            url=f"https://www.semanticscholar.org/paper/{data.get('paperId', '')}",
            categories=data.get("fieldsOfStudy", []),
            citations=data.get("citationCount", 0),
            collected_at=datetime.utcnow().isoformat(),
            raw_data=data,
        )
        
        return item
