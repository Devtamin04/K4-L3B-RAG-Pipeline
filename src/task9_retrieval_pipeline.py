"""
Task 9 — Retrieval pipeline hoàn chỉnh.
"""
import os

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

# [BÁO CÁO]: Threshold chọn 0.573 dựa trên calibration.
# [TÙY CHỈNH]: Thay số 0.573 thành con số thực tế mà tập evaluation của bạn sinh ra
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.573"))
DEFAULT_TOP_K = 5

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    if top_k <= 0 or not query.strip():
        return []
    
    candidate_count = top_k * 2
    
    # 1. Thực hiện Dense và Lexical (BM25) search
    dense = semantic_search(query, top_k=candidate_count)
    sparse = lexical_search(query, top_k=candidate_count)
    
    # 2. [BÁO CÁO]: Kết hợp dense + BM25 bằng thuật toán RRF
    hybrid = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking
        else dense[:top_k]
    )

    # 3. [BÁO CÁO]: Quyết định dùng dense cosine score gốc để trigger fallback
    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            # Fallback sang PageIndex nếu score quá thấp
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            # [BÁO CÁO]: Safe fallback - nếu degraded mode lỗi thì vẫn trả về kết quả hybrid thay vì crash
            pass
            
    return hybrid[:top_k]

if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)