from __future__ import annotations
import json
from typing import AsyncGenerator, List, Optional
import httpx
from app.ai.base import BaseLLMProvider, PromptMessage, LLMResponse
from app.config import settings

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini Provider supporting Gemini 2.0 Flash & Gemini 1.5 Pro.
    """
    CANDIDATE_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-lite-latest"]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.default_model = "gemini-3.5-flash"

    async def generate_stream(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        # Extract system prompt and contents
        system_instruction = None
        contents = []

        for m in messages:
            if m.role == "system":
                system_instruction = {"parts": [{"text": m.content}]}
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": m.content}]
                })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        models_to_try = [model] if model else self.CANDIDATE_MODELS

        for candidate in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:streamGenerateContent?alt=sse&key={self.api_key}"
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    async with client.stream(
                        "POST",
                        url,
                        headers={"Content-Type": "application/json"},
                        json=payload
                    ) as response:
                        if response.status_code in [404, 429, 503]:
                            continue  # Try next model in pool

                        if response.status_code != 200:
                            err_bytes = await response.aread()
                            raise RuntimeError(f"Gemini API error {response.status_code}: {err_bytes.decode('utf-8')}")

                        streamed_any = False
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            try:
                                data = json.loads(data_str)
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for p in parts:
                                        text = p.get("text", "")
                                        if text:
                                            streamed_any = True
                                            yield text
                            except Exception:
                                continue

                        if streamed_any:
                            return
            except Exception:
                continue

        # If all candidates exhausted, yield fallback message
        yield "I encountered temporary high demand from the upstream provider. Please retry your message in a few seconds."

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
        full_text = "".join(tokens)
        return LLMResponse(
            content=full_text,
            token_count=len(full_text) // 4,
            model=model or self.default_model,
            provider="gemini"
        )
