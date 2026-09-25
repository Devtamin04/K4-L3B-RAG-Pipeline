"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
EMBEDDING_LOCAL_FILES_ONLY = os.getenv(
    "EMBEDDING_LOCAL_FILES_ONLY", "false"
).lower() in {"1", "true", "yes"}

# Phương án A/B bổ sung; không dùng làm model mặc định.
ALTERNATIVE_EMBEDDING_MODELS = {
    "dangvantuan/vietnamese-embedding": 768,
}

_model_slug = EMBEDDING_MODEL.lower().replace("/", "_").replace("-", "_")
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION", f"rag_documents_{_model_slug}_{EMBEDDING_DIM}"
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text locally on CPU with the configured SentenceTransformer."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings")

    vectors = _embedding_model().encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > EMBEDDING_BATCH_SIZE,
    )
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
            f"got {vectors.shape}"
        )
    return vectors.tolist()


@lru_cache(maxsize=1)
def _embedding_model():
    """Load the large model once; lazy loading keeps unit tests lightweight."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
        local_files_only=EMBEDDING_LOCAL_FILES_ONLY,
    )


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        metadata, content = _parse_front_matter(raw)
        if not content.strip():
            continue
        relative = path.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append(
            {
                "id": relative,
                "content": content.strip(),
                "metadata": {
                    "source": str(metadata.get("source") or path.name),
                    "title": str(metadata.get("title") or path.stem),
                    "doc_type": str(
                        metadata.get("doc_type")
                        or ("legal" if "legal" in path.parts else "news")
                    ),
                    "url": metadata.get("url") or None,
                },
            }
        )
    return documents


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Parse the small JSON-valued front matter emitted by Task 3."""
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---\n", 4)
    if end < 0:
        return {}, raw
    metadata: dict = {}
    for line in raw[4:end].splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.strip()
        try:
            metadata[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            metadata[key.strip()] = value.strip('"')
    return metadata, raw[end + 5 :]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    return [
        {**chunk, "embedding": vector}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    metadatas = [
        {key: ("" if value is None else value) for key, value in chunk["metadata"].items()}
        for chunk in chunks
    ]
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=metadatas,
    )


def run_pipeline() -> None:
    """Đồng bộ index, chỉ embed chunk mới hoặc có nội dung thay đổi."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    collection = get_collection()
    current = collection.get(include=["documents"])
    existing = dict(zip(current.get("ids", []), current.get("documents", [])))
    active_ids = {chunk["id"] for chunk in chunks}
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
