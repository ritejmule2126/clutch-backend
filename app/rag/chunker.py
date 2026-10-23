from typing import List, Dict, Any
from app.rag.extractor import ExtractedPage

class Chunk:
    def __init__(self, index: int, page_number: int, content: str, token_count: int):
        self.index = index
        self.page_number = page_number
        self.content = content
        self.token_count = token_count

def chunk_pages(pages: List[ExtractedPage], chunk_size: int = 650, chunk_overlap: int = 120) -> List[Chunk]:
    """
    Chunks extracted pages with overlap and preserves exact page attribution.
    """
    chunks: List[Chunk] = []
    chunk_index = 0

    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        # If page text is within chunk size, keep as single chunk
        if len(text) <= chunk_size:
            chunks.append(Chunk(
                index=chunk_index,
                page_number=page.page_number,
                content=text,
                token_count=max(1, len(text) // 4)
            ))
            chunk_index += 1
            continue

        # Split into overlapping windows
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Find closest sentence or space boundary if not end of string
            if end < len(text):
                last_period = text.rfind(". ", start, end)
                last_space = text.rfind(" ", start, end)
                if last_period != -1 and last_period > start + (chunk_size // 2):
                    end = last_period + 1
                elif last_space != -1 and last_space > start + (chunk_size // 2):
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    index=chunk_index,
                    page_number=page.page_number,
                    content=chunk_text,
                    token_count=max(1, len(chunk_text) // 4)
                ))
                chunk_index += 1

            if end >= len(text):
                break
            start = end - chunk_overlap

    return chunks
