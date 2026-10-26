from __future__ import annotations
import json
import asyncio
from datetime import datetime, timezone
import zoneinfo
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db, AsyncSessionLocal
from app.models.user import User
from app.models.chat import ChatSession, ChatMessage
from app.schemas.chat import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionDetailResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    CitationItem
)
from app.core.deps import get_current_user
from app.rag.retriever import search_relevant_chunks
from app.ai.gateway import get_llm_provider
from app.ai.base import PromptMessage
from app.ai.prompts import CLAUDE_PRINCIPAL_ENGINEER_SYSTEM_PROMPT
from app.ai.stream_parser import ThinkingStreamParser
from app.config import settings

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(desc(ChatSession.updated_at))
    )
    sessions = result.scalars().all()
    return sessions

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(
    req: ChatSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    session = ChatSession(
        user_id=current_user.id,
        title=req.title or "New Chat",
        provider=req.provider or settings.DEFAULT_LLM_PROVIDER,
        model=req.model or settings.DEFAULT_MODEL
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session

@router.get("/sessions/{session_id}", response_model=ChatSessionDetailResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    session = result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    msgs_res = await db.execute(
        select(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = msgs_res.scalars().all()

    formatted_msgs = []
    for m in messages:
        cits = []
        if m.citations_json:
            try:
                cits = [CitationItem(**c) for c in json.loads(m.citations_json)]
            except Exception:
                pass
        formatted_msgs.append(ChatMessageResponse(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            citations=cits,
            created_at=m.created_at
        ))

    return ChatSessionDetailResponse(
        id=session.id,
        title=session.title,
        provider=session.provider,
        model=session.model,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=formatted_msgs
    )

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    session = result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    await db.delete(session)
    await db.commit()
    return {"status": "deleted", "id": session_id}

@router.post("/stream")
async def stream_chat_message(
    request: Request,
    req: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # 1. Resolve or create session
    session_id = req.session_id
    if not session_id:
        new_session = ChatSession(
            user_id=current_user.id,
            title=req.content[:36] + ("..." if len(req.content) > 36 else ""),
            provider=req.provider or settings.DEFAULT_LLM_PROVIDER,
            model=req.model or settings.DEFAULT_MODEL
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        session_id = new_session.id
    else:
        # Verify ownership or adopt client-generated session ID
        sess_check = await db.execute(
            select(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        )
        if not sess_check.scalars().first():
            adopted_session = ChatSession(
                id=session_id,
                user_id=current_user.id,
                title=req.content[:36] + ("..." if len(req.content) > 36 else ""),
                provider=req.provider or settings.DEFAULT_LLM_PROVIDER,
                model=req.model or settings.DEFAULT_MODEL
            )
            db.add(adopted_session)
            await db.commit()

    # 2. Save user message to database
    user_msg_rec = ChatMessage(
        session_id=session_id,
        user_id=current_user.id,
        role="user",
        content=req.content
    )
    db.add(user_msg_rec)
    await db.commit()

    # 3. Retrieve relevant chunks if RAG is requested
    citations: List[CitationItem] = []
    context_text = ""
    if req.use_rag:
        retrieved_chunks = await search_relevant_chunks(
            user_id=current_user.id,
            query=req.content,
            db=db,
            limit=4
        )
        if retrieved_chunks:
            context_blocks = []
            for idx, ch in enumerate(retrieved_chunks):
                cit = CitationItem(
                    document_id=ch.document_id,
                    document_title=ch.document_title,
                    page_number=ch.page_number,
                    snippet=ch.content[:160] + "..."
                )
                citations.append(cit)
                context_blocks.append(f"[{idx+1}] Source: {ch.document_title} (Page {ch.page_number}):\n{ch.content}")
            context_text = "\n\n".join(context_blocks)

    # 4. Fetch recent conversation history
    history_res = await db.execute(
        select(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    all_prev = history_res.scalars().all()
    # Keep last 8 messages for context window management
    recent_history = all_prev[-8:] if len(all_prev) > 8 else all_prev

    # 5. Dynamically resolve runtime client timezone and clock (zero hardcoded values)
    client_tz_name = (
        req.timezone
        or request.headers.get("x-timezone")
        or request.headers.get("cf-timezone")
        or None
    )

    utc_now = datetime.now(timezone.utc)
    user_tz = None
    if client_tz_name:
        try:
            user_tz = zoneinfo.ZoneInfo(client_tz_name)
        except Exception:
            user_tz = None

    # Fallback to dynamic host environment timezone if not specified by client
    if not user_tz:
        user_tz = utc_now.astimezone().tzinfo or timezone.utc
        client_tz_name = getattr(user_tz, "key", str(user_tz))

    local_now = utc_now.astimezone(user_tz)
    tz_abbrev = local_now.strftime("%Z")
    tz_offset = local_now.strftime("%z")
    formatted_local = local_now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
    formatted_utc = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")

    temporal_block = f"""CLIENT ENVIRONMENT & TEMPORAL CONTEXT:
- Local Client Date & Time: {formatted_local} ({tz_abbrev}, UTC{tz_offset})
- Local Client Timezone: {client_tz_name}
- UTC Timestamp: {formatted_utc}

OPERATIONAL TEMPORAL DIRECTIVES:
- You have verified access to the client's current runtime system clock above.
- When asked for the current time, date, day, or year, state their exact local time ({formatted_local} {tz_abbrev}) directly, politely, and accurately.
- Never state that you lack a real-time clock or do not know the user's timezone.
- For simple everyday questions (time, greetings, short facts), answer naturally and directly without generating a <thinking> block."""

    sys_instruction = f"{CLAUDE_PRINCIPAL_ENGINEER_SYSTEM_PROMPT}\n\n{temporal_block}"
    if context_text:
        sys_instruction += f"\n\nCONTEXT DOCUMENTS:\n{context_text}\n\nRULES: Base your response on the verified evidence above."

    prompt_messages = [PromptMessage(role="system", content=sys_instruction)]
    for m in recent_history:
        prompt_messages.append(PromptMessage(role=m.role, content=m.content))

    # 6. Stream tokens via SSE with Thinking stream separation
    llm_provider = get_llm_provider(req.provider, api_key_override=req.api_key)

    async def event_generator():
        parser = ThinkingStreamParser()
        try:
            # Yield session ID first so client knows which session this belongs to
            yield f"data: {json.dumps({'type': 'init', 'session_id': session_id})}\n\n"

            async for token in llm_provider.generate_stream(prompt_messages, model=req.model):
                for event_type, chunk in parser.feed(token):
                    payload = {"type": event_type}
                    if event_type in ["token", "thinking"]:
                        payload["delta"] = chunk
                    yield f"data: {json.dumps(payload)}\n\n"

            for event_type, chunk in parser.flush():
                payload = {"type": event_type}
                if event_type in ["token", "thinking"]:
                    payload["delta"] = chunk
                yield f"data: {json.dumps(payload)}\n\n"

            clean_answer = parser.clean_text

            # Persist assistant response with clean answer (no raw thinking tags)
            async with AsyncSessionLocal() as save_db:
                asst_rec = ChatMessage(
                    session_id=session_id,
                    user_id=current_user.id,
                    role="assistant",
                    content=clean_answer,
                    citations_json=json.dumps([c.model_dump() for c in citations]) if citations else None
                )
                save_db.add(asst_rec)
                await save_db.commit()

            yield f"data: {json.dumps({'type': 'done', 'citations': [c.model_dump() for c in citations]})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
