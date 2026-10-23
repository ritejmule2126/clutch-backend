import pytest
import os
import io
import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import init_db
from app.rag.extractor import extract_text_from_file
from app.rag.chunker import chunk_pages, ExtractedPage
from app.rag.embeddings import generate_local_embedding, cosine_similarity

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    asyncio.run(init_db())

@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

@pytest.mark.asyncio
async def test_auth_registration_and_login():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register new user
        unique_email = f"test_{os.urandom(4).hex()}@example.com"
        reg_res = await client.post("/api/v1/auth/register", json={
            "email": unique_email,
            "password": "Password123!",
            "full_name": "Test User"
        })
        assert reg_res.status_code == 200
        token_data = reg_res.json()
        assert "access_token" in token_data
        assert token_data["email"] == unique_email

        # Login with same user
        login_res = await client.post("/api/v1/auth/login", json={
            "email": unique_email,
            "password": "Password123!"
        })
        assert login_res.status_code == 200
        login_token = login_res.json()["access_token"]

        # Fetch profile
        me_res = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {login_token}"})
        assert me_res.status_code == 200
        assert me_res.json()["email"] == unique_email

@pytest.mark.asyncio
async def test_rag_chunking_and_embedding():
    pages = [
        ExtractedPage(page_number=1, text="Clutch AI is a high-speed personal assistant designed for engineers."),
        ExtractedPage(page_number=2, text="The system supports offline vector embeddings and RAG citation grounding.")
    ]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2

    # Test local deterministic embedding
    emb1 = generate_local_embedding("Clutch AI assistant")
    emb2 = generate_local_embedding("Clutch AI personal assistant")
    emb3 = generate_local_embedding("Completely unrelated cooking recipe for pasta")

    sim_close = cosine_similarity(emb1, emb2)
    sim_far = cosine_similarity(emb1, emb3)

    assert sim_close > sim_far
    assert len(emb1) == 384

@pytest.mark.asyncio
async def test_document_upload_and_search():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register user
        unique_email = f"doc_{os.urandom(4).hex()}@example.com"
        reg = await client.post("/api/v1/auth/register", json={
            "email": unique_email,
            "password": "Password123!"
        })
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Upload text document
        sample_doc = b"Project Clutch architecture uses FastAPI and React. It operates within a 23000 rupee budget."
        files = {"file": ("architecture_spec.txt", io.BytesIO(sample_doc), "text/plain")}
        upload_res = await client.post("/api/v1/documents/upload", files=files, headers=headers)
        assert upload_res.status_code == 200
        doc_data = upload_res.json()
        assert doc_data["chunk_count"] > 0
        assert doc_data["status"] == "ready"

        # Search document
        search_res = await client.post("/api/v1/documents/search", json={
            "query": "budget and architecture"
        }, headers=headers)
        assert search_res.status_code == 200
        results = search_res.json()
        assert len(results) > 0
        assert "FastAPI" in results[0]["content"]

@pytest.mark.asyncio
async def test_chat_streaming():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register user
        unique_email = f"chat_{os.urandom(4).hex()}@example.com"
        reg = await client.post("/api/v1/auth/register", json={
            "email": unique_email,
            "password": "Password123!"
        })
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Stream chat message
        stream_res = await client.post("/api/v1/chat/stream", json={
            "content": "Hello Clutch AI, what are your core capabilities?",
            "use_rag": False
        }, headers=headers)
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        
        # Verify content has tokens and done signal
        body = stream_res.text
        assert "data: " in body
        assert '"type": "init"' in body
        assert '"type": "done"' in body
