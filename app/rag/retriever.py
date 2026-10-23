import json
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import Document, DocumentChunk
from app.rag.embeddings import get_embedding, cosine_similarity
from app.schemas.document import SearchResultItem

async def search_relevant_chunks(
    user_id: str,
    query: str,
    db: AsyncSession,
    limit: int = 5,
    document_id: Optional[str] = None
) -> List[SearchResultItem]:
    """
    Retrieves the most semantically relevant document chunks for the user.
    """
    if not query.strip():
        return []

    query_embedding = await get_embedding(query)

    # Build query
    stmt = select(DocumentChunk, Document.title).join(
        Document, DocumentChunk.document_id == Document.id
    ).filter(
        DocumentChunk.user_id == user_id
    )

    if document_id:
        stmt = stmt.filter(DocumentChunk.document_id == document_id)

    result = await db.execute(stmt)
    rows = result.all()

    scored_items: List[SearchResultItem] = []
    for chunk, doc_title in rows:
        if not chunk.embedding_json:
            continue
        try:
            chunk_embedding = json.loads(chunk.embedding_json)
            score = cosine_similarity(query_embedding, chunk_embedding)
            
            # Keyword bonus for exact word matches in content
            query_words = set(query.lower().split())
            content_lower = chunk.content.lower()
            matching_words = sum(1 for w in query_words if len(w) > 3 and w in content_lower)
            if matching_words > 0:
                score += min(0.3, matching_words * 0.1)

            scored_items.append(SearchResultItem(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=doc_title,
                page_number=chunk.page_number,
                content=chunk.content,
                similarity_score=round(score, 4)
            ))
        except Exception:
            continue

    # Sort descending by similarity score
    scored_items.sort(key=lambda x: x.similarity_score, reverse=True)
    return scored_items[:limit]
