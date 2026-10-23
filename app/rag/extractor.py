import os
from typing import List, Dict, Any
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

class ExtractedPage:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text

def extract_text_from_file(file_path: str, file_type: str) -> List[ExtractedPage]:
    """
    Extracts text page-by-page or block-by-block with metadata.
    """
    pages: List[ExtractedPage] = []
    ext = file_type.lower().replace(".", "")

    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                pages.append(ExtractedPage(page_number=idx + 1, text=extracted.strip()))
        elif ext in ["txt", "md", "markdown", "json", "csv"]:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # If plain text is large, we can treat every ~3000 chars as a virtual page
            chunk_size = 3000
            if len(content) <= chunk_size:
                pages.append(ExtractedPage(page_number=1, text=content.strip()))
            else:
                for i in range(0, len(content), chunk_size):
                    page_num = (i // chunk_size) + 1
                    pages.append(ExtractedPage(page_number=page_num, text=content[i:i+chunk_size].strip()))
        elif ext in ["png", "jpg", "jpeg", "webp"]:
            # Image OCR placeholder / fallback
            pages.append(ExtractedPage(page_number=1, text=f"[Image File: {os.path.basename(file_path)}]"))
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            pages.append(ExtractedPage(page_number=1, text=content.strip()))
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        pages.append(ExtractedPage(page_number=1, text=""))

    return pages
