from __future__ import annotations
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
from typing import Optional, List, Dict
import asyncio
from datetime import datetime

LIVE_QUERY_PATTERNS = [
    r"\bipo\b", r"\blisting\b", r"\ballotment\b", r"\bgmp\b",
    r"\bnews\b", r"\blatest\b", r"\btoday\b", r"\byesterday\b", r"\bcurrent\b",
    r"\bnow\b", r"\bupdate\b", r"\brecent\b", r"\bstock\b", r"\bshare price\b",
    r"\bmarket\b", r"\brelease date\b", r"\bwhen will\b", r"\bwhen is\b",
    r"\bwho is\b", r"\bwho won\b", r"\bscore\b", r"\bmatch\b", r"\belection\b",
    r"\bannouncement\b", r"\bweather\b", r"\bhappened\b", r"\bhappening\b"
]

def should_fetch_live_web(query: str) -> bool:
    q_lower = query.lower().strip()
    return any(re.search(pattern, q_lower) for pattern in LIVE_QUERY_PATTERNS)

async def search_live_news(query: str, max_results: int = 5) -> Optional[str]:
    """
    Fetches real-time, live news articles via Google News RSS for up-to-the-minute events.
    """
    clean_query = query.strip()
    # Strip common conversational prefix
    clean_query = re.sub(r"^(can you tell me|what is|tell me|explain|search for|find)\s+", "", clean_query, flags=re.IGNORECASE)

    def _sync_fetch() -> Optional[str]:
        encoded_q = urllib.parse.quote(clean_query)
        # 1. Primary: India/Regional edition
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                items = root.findall(".//item")
                if not items:
                    return None

                results = []
                for idx, item in enumerate(items[:max_results]):
                    title = item.find("title").text if item.find("title") is not None else "News Item"
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    desc = item.find("description").text if item.find("description") is not None else ""
                    # Clean any raw HTML inside description
                    clean_desc = re.sub(r"<[^>]+>", " ", desc).strip()
                    clean_desc = " ".join(clean_desc.split())[:220]

                    results.append(
                        f"[{idx+1}] Headline: {title}\n"
                        f"    Date: {pub_date}\n"
                        f"    Context: {clean_desc}"
                    )

                if results:
                    today_str = datetime.now().strftime("%A, %d %B %Y")
                    header = f"[LIVE REAL-TIME NEWS & WEB CONTEXT (As of {today_str})]:\n"
                    return header + "\n\n".join(results)
        except Exception:
            return None
        return None

    return await asyncio.to_thread(_sync_fetch)
