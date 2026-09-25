"""
Task 4 — Chunking, embedding và indexing.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# [BÁO CÁO]: Giải thích quyết định kỹ thuật về chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

load_dotenv()

# [BÁO CÁO]: Giữ BAAI/bge-m3 làm baseline (chạy trên CPU)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
EMBEDDING_LOCAL_FILES_ONLY = os.getenv(
    "EMBEDDING_LOCAL_FILES_ONLY", "false"
).lower() in {"1", "true", "yes"}

# [BÁO CÁO]: Khai báo mô hình thử nghiệm (A/B testing). 
# [TÙY CHỈNH]: Đổi "minhdh" thành tên bạn hoặc tên mô hình bạn thực sự test
ALTERNATIVE_EMBEDDING_MODELS = {
    "minhdh/vietnamese-embedding": 768, 
}

_model_slug = EMBEDDING_MODEL.lower().replace("/", "_").replace("-", "_")
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION", f"rag_documents_{_model_slug}_{EMBEDDING_DIM}"
)

# ... (Giữ nguyên các hàm embed_texts, _embedding_model, get_collection, load_documents, _parse_front_matter, chunk_documents, embed_chunks, index_to_vectorstore như cũ) ...

def run_pipeline() -> None:
    """
    [BÁO CÁO]: Re-index được chuyển sang incremental.
    Đồng bộ index, chỉ embed chunk mới hoặc có nội dung thay đổi để tiết kiệm thời gian index.
    """
    documents = load_documents()
    chunks = chunk_documents(documents)
    collection = get_collection()
    current = collection.get(include=["documents"])
    existing = dict(zip(current.get("ids", []), current.get("documents", [])))
    active_ids = {chunk["id"] for chunk in chunks}
    
    # Chỉ lấy ra những chunk có sự thay đổi
    changed = [
        chunk for chunk in chunks if existing.get(chunk["id"]) != chunk["content"]
    ]
    stale_ids = sorted(set(existing) - active_ids)

    if changed:
        index_to_vectorstore(embed_chunks(changed))
    if stale_ids:
        collection.delete(ids=stale_ids)
        
    print(
        f"Indexed {len(chunks)} chunks "
        f"({len(changed)} updated, {len(stale_ids)} removed)"
    )

if __name__ == "__main__":
    run_pipeline()