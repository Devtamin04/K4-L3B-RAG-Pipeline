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
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or "sentence_transformers"
DEFAULT_EMBEDDING_MODELS = {
    "sentence_transformers": "BAAI/bge-m3",
    "openai": "text-embedding-3-small",
    "gemini": "text-embedding-004",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS.get(
    EMBEDDING_PROVIDER, "BAAI/bge-m3"
)
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 64

COLLECTION_NAME = "rag_documents"

# Task 3 ghi metadata dạng front matter, mỗi giá trị được json.dumps:
# ---\ntitle: "..."\nsource: "..."\nurl: "..." | null\ndoc_type: "..."\n---
FRONT_MATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


@lru_cache(maxsize=1)
def _sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_PROVIDER == "sentence_transformers":
        vectors = _sentence_transformer().encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [list(item.values) for item in response.embeddings]

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start:start + EMBEDDING_BATCH_SIZE]))
    return vectors


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _parse_front_matter_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip().strip("\"'")


def parse_markdown(path: Path) -> tuple[dict, str]:
    """Tách front matter của Task 3 thành metadata và trả về phần nội dung."""
    text = path.read_text(encoding="utf-8")
    fields = {}
    match = FRONT_MATTER_PATTERN.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = _parse_front_matter_value(value.strip())
        text = text[match.end():]

    relative = path.relative_to(STANDARDIZED_DIR)
    default_doc_type = "legal" if relative.parts[0] == "legal" else "news"
    metadata = {
        "source": str(fields.get("source") or path.name),
        "title": str(fields.get("title") or path.stem),
        "doc_type": str(fields.get("doc_type") or default_doc_type),
        "url": str(fields["url"]) if fields.get("url") else None,
    }
    return metadata, text.strip()


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        metadata, content = parse_markdown(path)
        if not content:
            continue
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": metadata,
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Không cắt tại ". " đứng sau chữ số để giữ nguyên "1. ", "Điều 11. ";
        # keep_separator="end" giữ dấu chấm ở cuối câu trước.
        separators=[r"\n\n", r"\n", r"(?<!\d)\. ", r" ", r""],
        is_separator_regex=True,
        keep_separator="end",
    )
    chunks = []
    for document in documents:
        texts = [text.strip() for text in splitter.split_text(document["content"])]
        for index, text in enumerate(text for text in texts if text):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB và xoá chunks cũ không còn trong corpus."""
    collection = get_collection()
    current_ids = {chunk["id"] for chunk in chunks}
    stale_ids = [item_id for item_id in collection.get(include=[])["ids"]
                 if item_id not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)
    if not chunks:
        return

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        # Chroma không nhận metadata None; Task 5/6 đổi "" về lại None.
        metadatas=[
            {**chunk["metadata"], "url": chunk["metadata"]["url"] or ""}
            for chunk in chunks
        ],
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
