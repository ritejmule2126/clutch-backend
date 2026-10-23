from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.config import settings
from app.database import init_db, AsyncSessionLocal
from app.models.user import User
from app.core.security import get_password_hash
from app.api.v1.auth import router as auth_router
from app.api.v1.documents import router as documents_router
from app.api.v1.chat import router as chat_router
from app.api.v1.memories import router as memories_router
from app.api.v1.settings import router as settings_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize DB tables
    await init_db()
    
    # 2. Seed default demo user if not present
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).filter(User.email == "demo@clutch.ai"))
        if not result.scalars().first():
            demo_user = User(
                email="demo@clutch.ai",
                hashed_password=get_password_hash("clutchdemo123"),
                full_name="Clutch User"
            )
            session.add(demo_user)
            await session.commit()
            
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade backend for Clutch AI with RAG, LLM streaming, and cross-platform sync.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(memories_router, prefix="/api/v1")
app.include_router(settings_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Clutch AI v2.0 API is operational.",
        "docs_url": "/docs",
        "health_url": "/health"
    }
