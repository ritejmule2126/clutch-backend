from __future__ import annotations
import os
import base64
from typing import List, Dict, Any
from pypdf import PdfReader
import httpx
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class ExtractedPage:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text

def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from images (JPG, PNG, WEBP) using Gemini Vision OCR.
    """
    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        mime = f"image/{ext}" if ext in ["png", "webp"] else "image/jpeg"
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        models = ["gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": mime, "data": b64}},
                    {"text": "Transcribe and extract all readable text, numbers, labels, tables, and details from this image verbatim. If this is an invoice, document, screenshot, or receipt, transcribe all fields completely."}
                ]
            }]
        }
        for m in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={settings.GEMINI_API_KEY}"
            try:
                with httpx.Client(timeout=25.0) as client:
                    r = client.post(url, json=payload)
                    if r.status_code == 200:
                        candidates = r.json().get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            res_text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
                            if res_text.strip():
                                return res_text.strip()
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Error in extract_text_from_image for {file_path}: {e}")

    return f"[Image Document: {os.path.basename(file_path)}]"

def extract_text_from_file(file_path: str, file_type: str) -> List[ExtractedPage]:
    """
    Extracts text page-by-page or block-by-block with metadata.
    """
    pages: List[ExtractedPage] = []
    ext = file_type.lower().replace(".", "")

    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                pages.append(ExtractedPage(page_number=idx + 1, text=extracted.strip()))
        elif ext in ["txt", "md", "markdown", "json", "csv"]:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            chunk_size = 3000
            if len(content) <= chunk_size:
                pages.append(ExtractedPage(page_number=1, text=content.strip()))
            else:
                for i in range(0, len(content), chunk_size):
                    page_num = (i // chunk_size) + 1
                    pages.append(ExtractedPage(page_number=page_num, text=content[i:i+chunk_size].strip()))
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]:
            ocr_text = extract_text_from_image(file_path)
            pages.append(ExtractedPage(page_number=1, text=ocr_text))
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            pages.append(ExtractedPage(page_number=1, text=content.strip()))
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        pages.append(ExtractedPage(page_number=1, text=""))

    return pages
