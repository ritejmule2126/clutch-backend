from __future__ import annotations
import json
from typing import AsyncGenerator, List, Optional
import httpx
from app.ai.base import BaseLLMProvider, PromptMessage, LLMResponse
from app.config import settings

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI Provider supporting GPT-4o, GPT-4o-mini, and o3-mini.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1"
        self.default_model = "gpt-4o-mini"

    async def generate_stream(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        selected_model = model or self.default_model
        payload = {
            "model": selected_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise RuntimeError(f"OpenAI API error {response.status_code}: {err_text.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue

    async def generate(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> LLMResponse:
        tokens: List[str] = []
        async for token in self.generate_stream(messages, model, temperature, max_tokens):
            tokens.append(token)
        full_content = "".join(tokens)
        return LLMResponse(
            content=full_content,
            token_count=len(full_content) // 4,
            model=model or self.default_model,
            provider="openai"
        )
