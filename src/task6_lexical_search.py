"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []
_BM25_INDEX = None
_BM25_SIGNATURE: tuple | None = None


def _tokenize(text: str) -> list[str]:
    """Vietnamese word segmentation with a safe whitespace fallback."""
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text.lower()).split()
    except Exception:
        return text.lower().split()


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(item["content"]) for item in corpus])


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import numpy as np

    global CORPUS, _BM25_INDEX, _BM25_SIGNATURE
    if top_k <= 0 or not query.strip():
        return []
    if not CORPUS:
        from .task4_chunking_indexing import chunk_documents, load_documents

        CORPUS = chunk_documents(load_documents())
    if not CORPUS:
        return []

    signature = tuple(
        (item["id"], len(item["content"]), hash(item["content"])) for item in CORPUS
    )
    if _BM25_INDEX is None or signature != _BM25_SIGNATURE:
        _BM25_INDEX = build_bm25_index(CORPUS)
        _BM25_SIGNATURE = signature
    scores = _BM25_INDEX.get_scores(_tokenize(query))
    # Stable sort makes ties deterministic and keeps corpus order.
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {
            "id": CORPUS[int(index)]["id"],
            "content": CORPUS[int(index)]["content"],
            "score": float(scores[int(index)]),
            "metadata": CORPUS[int(index)]["metadata"],
            "retrieval_method": "bm25",
        }
        for index in indices
    ]


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
