"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.

API của SDK pageindex==0.2.8 (đọc từ pageindex/client.py):
    submit_document(file_path)   -> {"doc_id": ...}      (chỉ nhận PDF)
    submit_query(doc_id, query)  -> {"retrieval_id": ...}
    get_retrieval(retrieval_id)  -> {"status": ..., "retrieved_nodes": [...]}
SDK không đặt timeout cho request, nên pageindex_search tự giới hạn thời gian.

Luôn upload nội dung đã chuẩn hoá (Markdown -> PDF) thay vì PDF gốc: PDF gốc là
bản scan toàn văn, còn standardized có thể là bản khác, nên fallback phải tìm
trên cùng corpus với ChromaDB.
"""

import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import STANDARDIZED_DIR, parse_markdown


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
ROOT_DIR = Path(__file__).parent.parent
DOC_ID_CACHE = ROOT_DIR / "data" / "pageindex_docs.json"

# Tổng thời gian tối đa cho một lần fallback (giây).
PAGEINDEX_TIMEOUT = float(os.getenv("PAGEINDEX_TIMEOUT") or 20)
POLL_INTERVAL = 1.0

# fpdf2 cần font Unicode để hiển thị tiếng Việt khi đổi Markdown sang PDF.
FONT_CANDIDATES = [
    os.getenv("PAGEINDEX_FONT", ""),
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not set")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def load_doc_ids() -> dict:
    """Đọc mapping source -> {doc_id, metadata} đã upload."""
    if not DOC_ID_CACHE.exists():
        return {}
    return json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))


def _save_doc_ids(doc_ids: dict) -> None:
    DOC_ID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DOC_ID_CACHE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _markdown_to_pdf(text: str, output_path: Path) -> None:
    from fpdf import FPDF

    font_path = next((path for path in FONT_CANDIDATES if path and Path(path).exists()), None)
    if font_path is None:
        raise RuntimeError("No Unicode TTF font found; set PAGEINDEX_FONT in .env")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("body", fname=font_path)
    pdf.set_font("body", size=11)
    # Chèn khoảng trắng vào token quá dài (URL) để multi_cell xuống dòng được.
    pdf.multi_cell(0, 6, re.sub(r"(\S{60})", r"\1 ", text))
    pdf.output(str(output_path))


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    client = _client()
    doc_ids = load_doc_ids()

    with tempfile.TemporaryDirectory() as temp_dir:
        for md_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
            key = md_path.relative_to(STANDARDIZED_DIR).as_posix()
            if key in doc_ids:
                continue
            metadata, content = parse_markdown(md_path)
            if not content:
                continue
            pdf_path = Path(temp_dir) / f"{md_path.stem}.pdf"
            _markdown_to_pdf(f"{metadata['title']}\n\n{content}", pdf_path)

            response = client.submit_document(str(pdf_path))
            doc_ids[key] = {"doc_id": response["doc_id"], "metadata": metadata}
            # Lưu sau mỗi file để lỗi giữa chừng không làm mất các upload trước.
            _save_doc_ids(doc_ids)
            print(f"Uploaded {key} -> {response['doc_id']}")


def _node_text(node: dict) -> str:
    contents = node.get("relevant_contents") or []
    texts = [
        item.get("relevant_content", "") if isinstance(item, dict) else str(item)
        for item in contents
    ]
    text = "\n".join(part for part in texts if part.strip())
    return text or node.get("text") or node.get("content") or ""


def _query_document(client, doc_id: str, query: str, deadline: float) -> list[dict]:
    retrieval_id = client.submit_query(doc_id, query)["retrieval_id"]
    while time.monotonic() < deadline:
        response = client.get_retrieval(retrieval_id)
        status = response.get("status")
        if status == "completed":
            return response.get("retrieved_nodes") or []
        if status == "failed":
            return []
        time.sleep(POLL_INTERVAL)
    return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    doc_ids = load_doc_ids()
    if top_k <= 0 or not query.strip() or not doc_ids:
        return []

    client = _client()
    deadline = time.monotonic() + PAGEINDEX_TIMEOUT
    executor = ThreadPoolExecutor(max_workers=min(8, len(doc_ids)))
    futures = {
        executor.submit(_query_document, client, entry["doc_id"], query, deadline): key
        for key, entry in doc_ids.items()
    }
    done, _ = wait(futures, timeout=PAGEINDEX_TIMEOUT + 1)
    # Không chờ request còn treo; thread sẽ tự kết thúc sau.
    executor.shutdown(wait=False, cancel_futures=True)

    results = []
    for future in done:
        if future.exception() is not None:
            continue
        key = futures[future]
        metadata = doc_ids[key]["metadata"]
        for rank, node in enumerate(future.result(), 1):
            content = _node_text(node).strip()
            if not content:
                continue
            node_id = node.get("node_id") or str(rank)
            results.append({
                "id": f"pageindex::{key}::{node_id}",
                "content": content,
                # API không trả score; gán score giảm dần theo rank trong tài liệu.
                "score": 1.0 / rank,
                "metadata": {**metadata, "chunk_index": rank - 1},
                "retrieval_method": "pageindex",
            })

    unique = {item["id"]: item for item in results}
    return sorted(unique.values(), key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    upload_documents()
