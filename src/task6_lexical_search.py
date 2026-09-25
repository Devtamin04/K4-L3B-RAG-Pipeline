"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import re

from .task4_chunking_indexing import get_collection


# Để trống thì lexical_search tự nạp chunks từ ChromaDB (cùng corpus với Task 5).
CORPUS: list[dict] = []

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

_index_cache: dict = {"corpus": None, "bm25": None}


def tokenize(text: str) -> list[str]:
    """Tách token theo âm tiết; tiếng Việt viết cách nhau bằng khoảng trắng."""
    return TOKEN_PATTERN.findall(text.lower())


def load_corpus_from_vectorstore() -> list[dict]:
    """Đọc toàn bộ chunks đã index để BM25 và dense search dùng chung corpus."""
    response = get_collection().get(include=["documents", "metadatas"])
    return [
        {
            "id": item_id,
            "content": content,
            "metadata": {**metadata, "url": metadata.get("url") or None},
        }
        for item_id, content, metadata in zip(
            response["ids"], response["documents"], response["metadatas"]
        )
    ]


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    # BM25Plus thay vì BM25Okapi: IDF của Okapi bằng 0 khi một từ xuất hiện
    # trong đúng một nửa corpus, làm mất kết quả khớp trên corpus nhỏ.
    from rank_bm25 import BM25Plus

    return BM25Plus([tokenize(item["content"]) for item in corpus])


def _get_bm25(corpus: list[dict]):
    if _index_cache["corpus"] is not corpus:
        _index_cache["corpus"] = corpus
        _index_cache["bm25"] = build_bm25_index(corpus)
    return _index_cache["bm25"]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global CORPUS
    query_tokens = tokenize(query)
    if top_k <= 0 or not query_tokens:
        return []
    if not CORPUS:
        CORPUS = load_corpus_from_vectorstore()
    if not CORPUS:
        return []

    scores = _get_bm25(CORPUS).get_scores(query_tokens)
    query_terms = set(query_tokens)
    results = []
    seen = set()
    for index in sorted(range(len(CORPUS)), key=lambda i: scores[i], reverse=True):
        item = CORPUS[index]
        # BM25Plus cộng delta cho mọi chunk; chỉ giữ chunk có ít nhất một từ khớp.
        if item["id"] in seen or not query_terms & set(tokenize(item["content"])):
            continue
        seen.add(item["id"])
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
