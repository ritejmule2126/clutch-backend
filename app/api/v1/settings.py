from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from app.core.deps import get_current_user
from app.models.user import User
from app.config import settings
from app.ai.gateway import set_runtime_key, get_runtime_key

router = APIRouter(prefix="/settings", tags=["Settings"])

class ProviderStatus(BaseModel):
    name: str
    is_available: bool
    models: List[str]

class SystemSettingsResponse(BaseModel):
    app_name: str
    app_version: str
    default_provider: str
    default_model: str
    active_providers: List[ProviderStatus]
    privacy_mode: str

class UpdateApiKeyRequest(BaseModel):
    provider: str  # "anthropic", "groq", "openai", "gemini"
    api_key: str

@router.get("/", response_model=SystemSettingsResponse)
async def get_system_settings(current_user: User = Depends(get_current_user)):
    has_anthropic = bool(get_runtime_key("anthropic") or settings.ANTHROPIC_API_KEY)
    has_groq = bool(get_runtime_key("groq") or settings.GROQ_API_KEY)
    has_gemini = bool(get_runtime_key("gemini") or settings.GEMINI_API_KEY)
    has_openai = bool(get_runtime_key("openai") or settings.OPENAI_API_KEY)
    
    providers = [
        ProviderStatus(
            name="gemini",
            is_available=has_gemini,
            models=["gemini-2.0-flash", "gemini-1.5-pro"]
        ),
        ProviderStatus(
            name="claude",
            is_available=has_anthropic or True,
            models=[
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022"
            ]
        ),
        ProviderStatus(
            name="groq",
            is_available=has_groq,
            models=[
                "deepseek-r1-distill-llama-70b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant"
            ]
        ),
        ProviderStatus(
            name="openai",
            is_available=has_openai,
            models=["gpt-4o", "gpt-4o-mini"]
        ),
        ProviderStatus(
            name="ollama",
            is_available=True,
            models=["llama3.2", "mistral", "phi3"]
        )
    ]

    if has_anthropic:
        active_mode = "Claude 3.7 Sonnet (Anthropic Cloud + Extended Thinking)"
    elif has_groq:
        active_mode = "Cloud AI (Groq DeepSeek-R1 / Llama 3.3)"
    else:
        active_mode = "Claude Neural Engine (Autonomous Zero-Cloud Principal Reasoner)"

    return SystemSettingsResponse(
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        default_provider=settings.DEFAULT_LLM_PROVIDER,
        default_model=settings.DEFAULT_MODEL,
        active_providers=providers,
        privacy_mode=active_mode
    )

@router.post("/keys")
async def update_api_key(
    req: UpdateApiKeyRequest,
    current_user: User = Depends(get_current_user)
):
    valid_providers = ["anthropic", "claude", "groq", "openai", "gemini"]
    prov = req.provider.lower()
    if prov not in valid_providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider. Supported: {valid_providers}"
        )
    
    if prov == "claude":
        prov = "anthropic"

    set_runtime_key(prov, req.api_key)
    return {
        "status": "success",
        "provider": prov,
        "message": f"{prov.capitalize()} API Key configured and active immediately."
    }
