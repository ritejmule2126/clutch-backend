from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Dict
from app.ai.base import BaseLLMProvider
from app.ai.anthropic_provider import AnthropicProvider
from app.ai.groq_provider import GroqProvider
from app.ai.gemini_provider import GeminiProvider
from app.ai.openai_provider import OpenAIProvider
from app.ai.ollama_provider import OllamaProvider
from app.ai.claude_reasoner import ClaudeNeuralReasoner
from app.config import settings, BASE_DIR

# Path to persist runtime keys
KEYS_FILE = Path(BASE_DIR) / "data" / "keys.json"

# In-memory runtime keys cache
RUNTIME_API_KEYS: Dict[str, str] = {}

def load_keys_from_disk():
    global RUNTIME_API_KEYS
    try:
        if KEYS_FILE.exists():
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    RUNTIME_API_KEYS.update({k.lower(): str(v).strip() for k, v in data.items()})
    except Exception:
        pass

load_keys_from_disk()

def set_runtime_key(provider: str, api_key: str):
    prov = provider.lower()
    RUNTIME_API_KEYS[prov] = api_key.strip()
    try:
        os.makedirs(KEYS_FILE.parent, exist_ok=True)
        with open(KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(RUNTIME_API_KEYS, f, indent=2)
    except Exception:
        pass

def get_runtime_key(provider: str) -> Optional[str]:
    return RUNTIME_API_KEYS.get(provider.lower())

def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key_override: Optional[str] = None
) -> BaseLLMProvider:
    """
    Resolves the optimal LLM provider.
    Priority:
    1. Requested provider if valid key exists
    2. Any provider with a configured key (Gemini, Anthropic, Groq, OpenAI)
    3. Autonomous local fallback if zero keys are configured
    """
    req_name = (provider_name or settings.DEFAULT_LLM_PROVIDER).lower()
    if req_name in ["claude", "anthropic"]:
        req_name = "anthropic"

    # Gather available keys
    override = api_key_override.strip() if api_key_override else None
    gemini_key = (override if req_name == "gemini" else None) or get_runtime_key("gemini") or settings.GEMINI_API_KEY
    anthropic_key = (override if req_name == "anthropic" else None) or get_runtime_key("anthropic") or settings.ANTHROPIC_API_KEY
    groq_key = (override if req_name == "groq" else None) or get_runtime_key("groq") or settings.GROQ_API_KEY
    openai_key = (override if req_name == "openai" else None) or get_runtime_key("openai") or settings.OPENAI_API_KEY

    # 1. Direct match for requested provider
    if req_name == "gemini" and gemini_key:
        return GeminiProvider(api_key=gemini_key)
    if req_name == "anthropic" and anthropic_key:
        return AnthropicProvider(api_key=anthropic_key)
    if req_name == "groq" and groq_key:
        return GroqProvider(api_key=groq_key)
    if req_name == "openai" and openai_key:
        return OpenAIProvider(api_key=openai_key)
    if req_name == "ollama":
        return OllamaProvider()

    # 2. Dynamic Fallback to any provider that HAS an active key
    if gemini_key:
        return GeminiProvider(api_key=gemini_key)
    if anthropic_key:
        return AnthropicProvider(api_key=anthropic_key)
    if groq_key:
        return GroqProvider(api_key=groq_key)
    if openai_key:
        return OpenAIProvider(api_key=openai_key)

    # 3. Last resort if no API key is provided
    return ClaudeNeuralReasoner()

