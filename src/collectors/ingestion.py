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