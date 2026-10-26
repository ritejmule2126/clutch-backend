from __future__ import annotations
import os
import hashlib
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.user import User
from app.models.document import Document, DocumentChunk
from app.schemas.document import DocumentResponse, DocumentDetailResponse, SearchRequest, SearchResultItem
from app.core.deps import get_current_user
from app.config import settings
from app.rag.extractor import extract_text_from_file
from app.rag.chunker import chunk_pages
from app.rag.embeddings import get_embedding
from app.rag.retriever import search_relevant_chunks

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Read file contents and compute checksum
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
        )

    checksum = hashlib.sha256(contents).hexdigest()
    
    # Save file to user-isolated folder
    user_dir = os.path.join(settings.UPLOAD_DIR, current_user.id)
    os.makedirs(user_dir, exist_ok=True)
    
    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower().replace(".", "")
    save_path = os.path.join(user_dir, f"{checksum[:16]}_{filename}")
    
    with open(save_path, "wb") as f:
        f.write(contents)

    doc_title = title.strip() if title and title.strip() else filename

    # Create Document record
    doc = Document(
        user_id=current_user.id,
        title=doc_title,
        original_filename=filename,
        file_type=ext or "bin",
        file_size=len(contents),
        checksum=checksum,
        storage_path=save_path,
        status="processing"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        # 1. Extract text
        pages = extract_text_from_file(save_path, ext)
        # 2. Chunk text
        chunks = chunk_pages(pages)
        
        # 3. Compute embeddings & create DocumentChunk records
        for c in chunks:
            emb = await get_embedding(c.content)
            chunk_rec = DocumentChunk(
                document_id=doc.id,
                user_id=current_user.id,
                chunk_index=c.index,
                page_number=c.page_number,
                content=c.content,
                embedding_json=json.dumps(emb),
                token_count=c.token_count
            )
            db.add(chunk_rec)

        # Generate brief summary from first chunk
        summary_preview = chunks[0].content[:280] + "..." if chunks else "No text extracted."
        doc.chunk_count = len(chunks)
        doc.summary = summary_preview
        doc.status = "ready"
        await db.commit()
        await db.refresh(doc)
    except Exception as e:
        doc.status = "failed"
        doc.summary = f"Error during processing: {str(e)}"
        await db.commit()
        await db.refresh(doc)

    return doc

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(desc(Document.created_at))
    )
    return result.scalars().all()

@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        )
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks_res = await db.execute(
        select(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    doc_chunks = chunks_res.scalars().all()

    return DocumentDetailResponse(
        id=doc.id,
        title=doc.title,
        original_filename=doc.original_filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        checksum=doc.checksum,
        status=doc.status,
        chunk_count=doc.chunk_count,
        summary=doc.summary,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        chunks=doc_chunks
    )

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        )
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove file on disk if exists
    if os.path.exists(doc.storage_path):
        try:
            os.remove(doc.storage_path)
        except OSError:
            pass

    await db.delete(doc)
    await db.commit()
    return {"status": "deleted", "id": document_id}

@router.post("/search", response_model=List[SearchResultItem])
async def search_documents(
    req: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    return await search_relevant_chunks(
        user_id=current_user.id,
        query=req.query,
        db=db,
        limit=req.limit or 5,
        document_id=req.document_id
    )
