"""
Scrapy pipelines for data validation and storage.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from itemadapter import ItemAdapter

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class ValidationPipeline:
    """
    Pipeline for validating scraped items.
    
    Ensures all required fields are present and valid.
    """
    
    required_fields = ["source", "source_id", "title"]
    
    def process_item(self, item: Any, spider: Any) -> Any:
        """Validate item fields."""
        adapter = ItemAdapter(item)
        
        # Check required fields
        for field in self.required_fields:
            if not adapter.get(field):
                logger.warning(
                    "Missing required field",
                    field=field,
                    source_id=adapter.get("source_id"),
                )
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure collected_at is set
        if not adapter.get("collected_at"):
            adapter["collected_at"] = datetime.utcnow().isoformat()
        
        # Ensure lists are properly formatted
        for list_field in ["authors", "categories", "keywords"]:
            value = adapter.get(list_field)
            if value is None:
                adapter[list_field] = []
            elif isinstance(value, str):
                adapter[list_field] = [value]
        
        return item


class StoragePipeline:
    """
    Pipeline for storing scraped items.
    
    Stores items as JSON files for later processing.
    """
    
    def __init__(self):
        self.items: list[dict] = []
        self.output_dir = Path(settings.data_dir) / "scraped"
        
    def open_spider(self, spider: Any) -> None:
        """Initialize storage on spider open."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.items = []
        logger.info("StoragePipeline opened", spider=spider.name)
    
    def process_item(self, item: Any, spider: Any) -> Any:
        """Store item for batch processing."""
        adapter = ItemAdapter(item)
        self.items.append(dict(adapter))
        return item
    
    def close_spider(self, spider: Any) -> None:
        """Save all items on spider close."""
        if not self.items:
            logger.info("No items to save", spider=spider.name)
            return
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{spider.name}_{timestamp}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=2, default=str)
        
        logger.info(
            "Saved scraped items",
            spider=spider.name,
            count=len(self.items),
            path=str(output_path),
        )


class DuplicatesPipeline:
    """
    Pipeline for filtering duplicate items.
    
    Tracks seen source IDs to avoid duplicates.
    """
    
    def __init__(self):
        self.seen_ids: set[str] = set()
    
    def open_spider(self, spider: Any) -> None:
        """Reset seen IDs on spider open."""
        self.seen_ids = set()
    
    def process_item(self, item: Any, spider: Any) -> Any:
        """Filter duplicate items."""
        adapter = ItemAdapter(item)
        item_id = f"{adapter['source']}:{adapter['source_id']}"
        
        if item_id in self.seen_ids:
            logger.debug("Duplicate item filtered", item_id=item_id)
            raise ValueError(f"Duplicate item: {item_id}")
        
        self.seen_ids.add(item_id)
        return item
