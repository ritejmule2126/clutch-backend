from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel

class PromptMessage(BaseModel):
    role: str  # system, user, assistant
    content: str

class LLMResponse(BaseModel):
    content: str
    token_count: int = 0
    model: str
    provider: str

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """Yields string token deltas."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> LLMResponse:
        """Returns completed response."""
        pass
