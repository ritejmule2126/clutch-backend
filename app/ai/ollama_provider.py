from __future__ import annotations
import json
from typing import AsyncGenerator, List, Optional
import httpx
from app.ai.base import BaseLLMProvider, PromptMessage, LLMResponse
from app.config import settings

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.default_model = "llama3.2"

    async def generate_stream(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        selected_model = model or self.default_model
        payload = {
            "model": selected_model,
            "messages": [m.model_dump() for m in messages],
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": True
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status_code != 200:
                    err = await response.aread()
                    raise RuntimeError(f"Ollama error {response.status_code}: {err.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    async def generate(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> LLMResponse:
        full_content = []
        async for chunk in self.generate_stream(messages, model, temperature, max_tokens):
            full_content.append(chunk)
        content = "".join(full_content)
        return LLMResponse(
            content=content,
            token_count=len(content) // 4,
            model=model or self.default_model,
            provider="ollama"
        )
