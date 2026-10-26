from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

class DocumentChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    chunk_index: int
    page_number: int
    content: str
    token_count: int

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    original_filename: str
    file_type: str
    file_size: int
    checksum: str
    status: str
    chunk_count: int
    summary: Optional[str] = ""
    created_at: datetime
    updated_at: datetime

class DocumentDetailResponse(DocumentResponse):
    chunks: List[DocumentChunkResponse] = []

class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int
    content: str
    similarity_score: float

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    document_id: Optional[str] = None
