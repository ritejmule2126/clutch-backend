import hashlib
import json
import numpy as np
from typing import List, Optional
import httpx
from app.config import settings

def generate_local_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Deterministic subword and n-gram feature hashing vector generator.
    Produces a normalized 384-dimensional dense vector representing semantic keyword density.
    Works 100% offline, requires 0 MB downloads, and supports fast cosine similarity.
    """
    vec = np.zeros(dim, dtype=np.float32)
    cleaned = text.lower()
    words = cleaned.split()
    
    # Unigram and character tri-gram hashing
    for word in words:
        h = int(hashlib.sha256(word.encode('utf-8')).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
        
        # Subword tri-grams
        if len(word) >= 3:
            for i in range(len(word) - 2):
                tri = word[i:i+3]
                h_tri = int(hashlib.md5(tri.encode('utf-8')).hexdigest()[:8], 16)
                vec[h_tri % dim] += 0.5

    # L2 Normalization
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

async def get_embedding(text: str) -> List[float]:
    """
    Returns an embedding vector for the text using the configured provider or local fallback.
    """
    # 1. Try Ollama embeddings if configured and available
    if settings.OLLAMA_BASE_URL and settings.DEFAULT_LLM_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": "all-minilm", "prompt": text}
                )
                if res.status_code == 200:
                    emb = res.json().get("embedding")
                    if emb and len(emb) == settings.EMBEDDING_DIMENSION:
                        return emb
        except Exception:
            pass

    # 2. Local fallback
    return generate_local_embedding(text, dim=settings.EMBEDDING_DIMENSION)

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
