from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from datetime import datetime

class CitationItem(BaseModel):
    document_id: str
    document_title: str
    page_number: int
    snippet: str

class ChatMessageCreate(BaseModel):
    content: str
    session_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    use_rag: bool = True

class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    role: str
    content: str
    citations: List[CitationItem] = []
    created_at: datetime

class ChatSessionCreate(BaseModel):
    title: Optional[str] = "New Chat"
    provider: Optional[str] = None
    model: Optional[str] = None

class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    provider: str
    model: str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0

class ChatSessionDetailResponse(ChatSessionResponse):
    messages: List[ChatMessageResponse] = []
