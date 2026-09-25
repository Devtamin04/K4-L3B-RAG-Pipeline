"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        seen_in_list = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            # Một list chỉ được góp điểm một lần cho mỗi ID.
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            items.setdefault(item_id, item)

    # sorted() ổn định: khi hoà điểm, giữ thứ tự xuất hiện đầu tiên.
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [
        {
            **items[item_id],
            "score": scores[item_id],
            "retrieval_method": "hybrid",
        }
        for item_id in ranked_ids[:max(top_k, 0)]
    ]


if __name__ == "__main__":
    dense = [
        {"id": "a", "content": "A", "score": 0.9, "metadata": {}, "retrieval_method": "dense"},
        {"id": "b", "content": "B", "score": 0.8, "metadata": {}, "retrieval_method": "dense"},
    ]
    bm25 = [
        {"id": "b", "content": "B", "score": 7.0, "metadata": {}, "retrieval_method": "bm25"},
        {"id": "c", "content": "C", "score": 5.0, "metadata": {}, "retrieval_method": "bm25"},
    ]
    for result in rerank_rrf([dense, bm25], top_k=3):
        print(result["id"], round(result["score"], 5))
