"""
Scrapy items for data collection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import scrapy


@dataclass
class PaperItem:
    """Represents a collected paper from any source."""
    
    source: str = ""
    source_id: str = ""
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    url: str = ""
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    citations: int = 0
    
    # Metadata
    collected_at: datetime = field(default_factory=datetime.utcnow)
    raw_data: Optional[dict] = None


class ScrapyPaperItem(scrapy.Item):
    """Scrapy item for papers."""
    
    source = scrapy.Field()
    source_id = scrapy.Field()
    title = scrapy.Field()
    abstract = scrapy.Field()
    authors = scrapy.Field()
    published_date = scrapy.Field()
    url = scrapy.Field()
    pdf_url = scrapy.Field()
    full_text = scrapy.Field()
    categories = scrapy.Field()
    keywords = scrapy.Field()
    citations = scrapy.Field()
    collected_at = scrapy.Field()
    raw_data = scrapy.Field()
