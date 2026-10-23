from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.core.deps import get_current_user
from app.models.user import User
from app.config import settings

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

@router.get("/", response_model=SystemSettingsResponse)
async def get_system_settings(current_user: User = Depends(get_current_user)):
    providers = [
        ProviderStatus(
            name="groq",
            is_available=bool(settings.GROQ_API_KEY),
            models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        ),
        ProviderStatus(
            name="ollama",
            is_available=True,  # local offline
            models=["llama3.2", "mistral", "phi3"]
        ),
        ProviderStatus(
            name="local-synthesizer",
            is_available=True,
            models=["built-in-rag"]
        )
    ]

    active_mode = "Cloud AI (Groq)" if settings.GROQ_API_KEY else "Local Hybrid (Zero Cloud)"

    return SystemSettingsResponse(
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        default_provider=settings.DEFAULT_LLM_PROVIDER,
        default_model=settings.DEFAULT_MODEL,
        active_providers=providers,
        privacy_mode=active_mode
    )
