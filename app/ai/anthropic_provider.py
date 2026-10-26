from __future__ import annotations
import json
from typing import AsyncGenerator, List, Optional
import httpx
from app.ai.base import BaseLLMProvider, PromptMessage, LLMResponse
from app.config import settings

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude Provider supporting Claude 3.7 Sonnet (with Extended Thinking)
    and Claude 3.5 Sonnet.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.base_url = "https://api.anthropic.com/v1"
        self.default_model = "claude-3-7-sonnet-20250219"

    async def generate_stream(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")

        selected_model = model or self.default_model
        
        # Extract system prompt
        system_content = ""
        user_and_asst_messages = []
        for m in messages:
            if m.role == "system":
                system_content += m.content + "\n\n"
            else:
                user_and_asst_messages.append({
                    "role": m.role,
                    "content": m.content
                })

        # Anthropic payload
        payload = {
            "model": selected_model,
            "max_tokens": max_tokens,
            "messages": user_and_asst_messages,
            "stream": True
        }

        if system_content.strip():
            payload["system"] = system_content.strip()

        # If Claude 3.7, configure extended thinking
        is_claude_37 = "claude-3-7" in selected_model
        if is_claude_37:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(2048, max_tokens - 1024)
            }
        else:
            payload["temperature"] = temperature

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        in_thinking_block = False

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    raise RuntimeError(f"Anthropic API error {response.status_code}: {err_bytes.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type")

                            # Handle thinking block start/delta/stop
                            if event_type == "content_block_start":
                                block = data.get("content_block", {})
                                if block.get("type") == "thinking":
                                    in_thinking_block = True
                                    yield "<thinking>\n"
                            
                            elif event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                delta_type = delta.get("type")

                                if delta_type == "thinking_delta":
                                    thinking_text = delta.get("thinking", "")
                                    if thinking_text:
                                        yield thinking_text

                                elif delta_type == "text_delta":
                                    # If transitioning out of thinking
                                    if in_thinking_block:
                                        yield "\n</thinking>\n\n"
                                        in_thinking_block = False
                                    text = delta.get("text", "")
                                    if text:
                                        yield text

                            elif event_type == "content_block_stop":
                                if in_thinking_block:
                                    yield "\n</thinking>\n\n"
                                    in_thinking_block = False

                        except Exception:
                            continue

                if in_thinking_block:
                    yield "\n</thinking>\n\n"

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
            provider="anthropic"
        )
