from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.user import User
from app.models.memory import UserMemory
from app.schemas.memory import MemoryCreate, MemoryUpdate, MemoryResponse
from app.core.deps import get_current_user

router = APIRouter(prefix="/memories", tags=["Memories"])

@router.get("/", response_model=List[MemoryResponse])
async def list_memories(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(UserMemory)
        .filter(UserMemory.user_id == current_user.id)
        .order_by(desc(UserMemory.created_at))
    )
    return result.scalars().all()

@router.post("/", response_model=MemoryResponse)
async def create_memory(
    req: MemoryCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    memory = UserMemory(
        user_id=current_user.id,
        category=req.category,
        content=req.content,
        importance=req.importance
    )
    db.add(memory)
    await db.commit()
    await db.refresh(memory)
    return memory

@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(UserMemory).filter(
            UserMemory.id == memory_id,
            UserMemory.user_id == current_user.id
        )
    )
    memory = result.scalars().first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found.")

    await db.delete(memory)
    await db.commit()
    return {"status": "deleted", "id": memory_id}
