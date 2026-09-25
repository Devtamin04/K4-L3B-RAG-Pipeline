"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from .task1_collect_legal_docs import LEGAL_DOCUMENTS


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"
PAGEINDEX_BASE_URL = "https://api.pageindex.ai"
PAGEINDEX_TIMEOUT = float(os.getenv("PAGEINDEX_TIMEOUT", "30"))


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache: dict[str, dict]) -> None:
    temporary = CACHE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(CACHE_PATH)


def _headers() -> dict[str, str]:
    return {"api_key": PAGEINDEX_API_KEY}


def _request(method: str, path: str, **kwargs) -> dict:
    response = requests.request(
        method,
        f"{PAGEINDEX_BASE_URL}{path}",
        headers=_headers(),
        timeout=(10, PAGEINDEX_TIMEOUT),
        **kwargs,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("PageIndex trả về dữ liệu không hợp lệ")
    return payload


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được cấu hình")

    cache = _load_cache()
    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        if path.name in cache and cache[path.name].get("doc_id"):
            continue
        with path.open("rb") as file_handle:
            payload = _request(
                "POST",
                "/doc/",
                files={"file": (path.name, file_handle, "application/pdf")},
                data={"if_retrieval": "true"},
            )
        doc_id = payload.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"PageIndex không trả doc_id cho {path.name}")
        cache[path.name] = {
            "doc_id": doc_id,
            "source": path.name,
            "title": path.stem.replace("-", " "),
            "doc_type": "legal",
            "url": LEGAL_DOCUMENTS.get(path.name),
        }
        _save_cache(cache)
        print(f"Uploaded: {path.name} -> {doc_id}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip() or not PAGEINDEX_API_KEY:
        return []
    cache = _load_cache()
    candidates: list[dict] = []

    for entry in cache.values():
        doc_id = entry.get("doc_id")
        if not doc_id:
            continue
        tree = _request("GET", f"/doc/{doc_id}/?type=tree&summary=false")
        if not tree.get("retrieval_ready", False):
            continue
        submitted = _request(
            "POST",
            "/retrieval/",
            json={"doc_id": doc_id, "query": query, "thinking": False},
        )
        retrieval_id = submitted.get("retrieval_id")
        if not retrieval_id:
            continue

        deadline = time.monotonic() + PAGEINDEX_TIMEOUT
        response: dict = {}
        while time.monotonic() < deadline:
            response = _request("GET", f"/retrieval/{retrieval_id}/")
            status = response.get("status")
            if status == "completed":
                break
            if status in {"failed", "error"}:
                response = {}
                break
            time.sleep(0.5)
        else:
            continue

        for node in response.get("retrieved_nodes", []):
            node_id = str(node.get("node_id") or len(candidates))
            title = str(node.get("title") or entry.get("title") or entry["source"])
            for content_index, relevant in enumerate(node.get("relevant_contents", [])):
                content = str(relevant.get("relevant_content") or "").strip()
                if not content:
                    continue
                rank = len(candidates) + 1
                candidates.append(
                    {
                        "id": f"pageindex:{doc_id}:{node_id}:{content_index}",
                        "content": content,
                        "score": 1.0 / rank,
                        "metadata": {
                            "source": entry["source"],
                            "title": title,
                            "doc_type": entry.get("doc_type", "legal"),
                            "url": entry.get("url"),
                            "chunk_index": rank - 1,
                        },
                        "retrieval_method": "pageindex",
                    }
                )

    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    upload_documents()
