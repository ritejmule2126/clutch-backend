from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class MemoryCreate(BaseModel):
    category: str = "General"
    content: str
    importance: int = 1

class MemoryUpdate(BaseModel):
    category: Optional[str] = None
    content: Optional[str] = None
    importance: Optional[int] = None

class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    category: str
    content: str
    importance: int
    created_at: datetime
    updated_at: datetime
