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
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    counter = 0
    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items.setdefault(item_id, item)
            if item_id not in first_seen:
                first_seen[item_id] = counter
                counter += 1
    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], first_seen[item_id]))
    return [
        {
            **items[item_id],
            "score": scores[item_id],
            "retrieval_method": "hybrid",
        }
        for item_id in ranked_ids[:top_k]
    ]


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
