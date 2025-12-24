"""
Tests for the arXiv API client.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.collectors.arxiv_client import ArxivClient, ArxivPaper, RateLimiter


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter(rate=10.0)  # 10 requests per second
    
    @pytest.mark.asyncio
    async def test_acquire_immediate(self, rate_limiter):
        """Test that first requests go through immediately."""
        start = datetime.now()
        await rate_limiter.acquire()
        elapsed = (datetime.now() - start).total_seconds()
        assert elapsed < 0.1
    
    @pytest.mark.asyncio
    async def test_acquire_rate_limit(self, rate_limiter):
        """Test that rate limiting kicks in after tokens exhausted."""
        # Exhaust all tokens
        for _ in range(10):
            await rate_limiter.acquire()
        
        # This should wait
        start = datetime.now()
        await rate_limiter.acquire()
        elapsed = (datetime.now() - start).total_seconds()
        assert elapsed >= 0.05  # Should have some delay


class TestArxivPaper:
    """Tests for the ArxivPaper class."""
    
    def test_to_dict(self):
        """Test paper serialization."""
        paper = ArxivPaper(
            arxiv_id="2312.12345",
            title="Test Paper",
            abstract="This is a test abstract.",
            authors=["Author A", "Author B"],
            categories=["cs.LG", "cs.AI"],
            primary_category="cs.LG",
            published=datetime(2023, 12, 1),
            updated=datetime(2023, 12, 15),
            pdf_url="https://arxiv.org/pdf/2312.12345.pdf",
        )
        
        data = paper.to_dict()
        
        assert data["arxiv_id"] == "2312.12345"
        assert data["title"] == "Test Paper"
        assert len(data["authors"]) == 2
        assert "collected_at" in data


class TestArxivClient:
    """Tests for the ArxivClient class."""
    
    @pytest.fixture
    def client(self):
        return ArxivClient(rate_limit=10.0)
    
    def test_initialization(self, client):
        """Test client initialization."""
        assert client.rate_limit == 10.0
        assert client.rate_limiter is not None
    
    @pytest.mark.asyncio
    async def test_collect_papers_empty(self, client):
        """Test collecting papers with mocked empty response."""
        with patch.object(client._client, 'results', return_value=iter([])):
            papers = await client.collect_papers(
                categories=["cs.LG"],
                limit=10,
                download_pdfs=False,
            )
            assert papers == []


class TestArxivClientIntegration:
    """Integration tests for ArxivClient (requires network)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_real(self):
        """Test actual search against arXiv API."""
        client = ArxivClient(rate_limit=1.0)
        
        papers = []
        async for paper in client.search(
            categories=["cs.LG"],
            max_results=5,
        ):
            papers.append(paper)
        
        assert len(papers) <= 5
        if papers:
            assert papers[0].arxiv_id
            assert papers[0].title
