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