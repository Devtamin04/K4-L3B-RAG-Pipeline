"""
Chỉ dùng cho demo Streamlit: chạy thử thêm dangvantuan/vietnamese-embedding.

Pipeline lab trong src/ vẫn giữ một embedding model (EMBEDDING_MODEL). Module
này tạm thay embed_texts()/get_collection() của Task 4/5 trong phạm vi
embedding_scope(), nên Task 9/10 chạy nguyên vẹn với model thay thế.

Build index cho model thay thế:
    python -m demo_embeddings
"""

import threading
from contextlib import contextmanager
from functools import lru_cache

from src import task4_chunking_indexing as indexing
from src import task5_semantic_search as semantic


DEFAULT_MODEL = indexing.EMBEDDING_MODEL
ALTERNATIVE_MODEL = "dangvantuan/vietnamese-embedding"
ALTERNATIVE_DIM = indexing.ALTERNATIVE_EMBEDDING_MODELS[ALTERNATIVE_MODEL]
ALTERNATIVE_COLLECTION = (
    "rag_documents_"
    + ALTERNATIVE_MODEL.lower().replace("/", "_").replace("-", "_")
    + f"_{ALTERNATIVE_DIM}"
)

# Swapping module globals is process-wide, so every request (either model)
# runs one at a time under this lock.
_scope_lock = threading.Lock()


@lru_cache(maxsize=1)
def _alternative_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        ALTERNATIVE_MODEL,
        device=indexing.EMBEDDING_DEVICE,
        local_files_only=indexing.EMBEDDING_LOCAL_FILES_ONLY,
    )
    # The model config declares max_seq_length=512 but PhoBERT only has 258
    # positions (RoBERTa reserves 2), so long chunks would crash.
    positions = model[0].auto_model.config.max_position_embeddings
    model.max_seq_length = min(model.max_seq_length, positions - 2)
    return model


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings")
    from pyvi import ViTokenizer

    # PhoBERT-based model expects word-segmented input (see model card).
    vectors = _alternative_model().encode(
        [ViTokenizer.tokenize(text) for text in texts],
        batch_size=indexing.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > indexing.EMBEDDING_BATCH_SIZE,
    )
    if vectors.ndim != 2 or vectors.shape[1] != ALTERNATIVE_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {ALTERNATIVE_DIM}, "
            f"got {vectors.shape}"
        )
    return vectors.tolist()


def _get_collection():
    import chromadb

    indexing.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(indexing.CHROMA_DIR))
    return client.get_or_create_collection(
        name=ALTERNATIVE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


@contextmanager
def embedding_scope(model_name: str):
    """Chạy Task 4-10 với model đã chọn trong phạm vi khối with."""
    with _scope_lock:
        if model_name == DEFAULT_MODEL:
            yield
            return
        if model_name != ALTERNATIVE_MODEL:
            raise ValueError(f"Unknown embedding model: {model_name}")
        originals = [
            (module, module.embed_texts, module.get_collection)
            for module in (indexing, semantic)
        ]
        for module, _, _ in originals:
            module.embed_texts = _embed_texts
            module.get_collection = _get_collection
        try:
            yield
        finally:
            for module, embed_texts, get_collection in originals:
                module.embed_texts = embed_texts
                module.get_collection = get_collection


def indexed_chunk_count(model_name: str) -> int:
    with embedding_scope(model_name):
        return indexing.get_collection().count()


if __name__ == "__main__":
    with embedding_scope(ALTERNATIVE_MODEL):
        indexing.run_pipeline()
