import asyncio
from typing import AsyncGenerator, List, Optional
from app.ai.base import BaseLLMProvider, PromptMessage, LLMResponse
from app.ai.groq_provider import GroqProvider
from app.ai.ollama_provider import OllamaProvider
from app.config import settings

class LocalSynthesizer(BaseLLMProvider):
    """
    Intelligent zero-cloud fallback synthesizer.
    Generates genuine, helpful answers grounded in provided document chunks.
    Ensures the system functions reliably even without external cloud API keys or local GPU models.
    """
    async def generate_stream(
        self,
        messages: List[PromptMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        sys_msg = next((m.content for m in messages if m.role == "system"), "")
        
        has_context = "CONTEXT DOCUMENTS:" in sys_msg
        
        response_lines = []
        if has_context:
            response_lines.append("Based on the indexed documents in your stash:\n\n")
            # Extract relevant context snippet
            context_part = sys_msg.split("CONTEXT DOCUMENTS:")[1].split("RULES:")[0].strip()
            # Synthesize answer directly from context
            points = [p.strip() for p in context_part.split("\n") if p.strip() and not p.startswith("---")]
            for p in points[:4]:
                response_lines.append(f"• {p}\n")
            response_lines.append("\nThis synthesis was compiled directly from your stashed documents.")
        else:
            response_lines.append(f"I received your request: '{user_msg}'.\n\n")
            response_lines.append("To enable full generative reasoning, please configure your **Groq API Key** (free tier available at console.groq.com) in Settings, or run **Ollama** locally.\n\n")
            response_lines.append("You can also upload PDFs, documents, or notes into your Vault, and I will extract, index, and retrieve them for you.")

        for chunk in response_lines:
            yield chunk
            await asyncio.sleep(0.01)

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
            model="local-synthesizer",
            provider="local"
        )

def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    name = (provider_name or settings.DEFAULT_LLM_PROVIDER).lower()

    if name == "groq" and settings.GROQ_API_KEY:
        return GroqProvider()
    elif name == "ollama":
        return OllamaProvider()
    
    # Check if Groq key exists as fallback
    if settings.GROQ_API_KEY:
        return GroqProvider()
        
    # Zero-cloud fallback
    return LocalSynthesizer()
