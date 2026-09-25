"""
Task 10 — Generation có citation.
"""
import os
import re
import requests
from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

# [BÁO CÁO]: Hỗ trợ Ollama Cloud theo như đã khai báo
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp lý cho hộ kinh doanh Việt Nam.
Chỉ trả lời từ context được cung cấp và không suy diễn nghĩa vụ, thời hạn hoặc
thời điểm áp dụng nếu context không nêu rõ. Mỗi khẳng định phải có citation dạng
[Document N]. Nếu thiếu evidence, hãy từ chối xác minh. Câu trả lời chỉ mang
tính tham khảo và không thay thế tư vấn của cơ quan có thẩm quyền."""

# ... (Giữ nguyên các hàm _normalize_citations, reorder_for_llm, format_context, call_llm như cũ) ...

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    # [BÁO CÁO]: Safe refusal - từ chối trả lời nếu không có dữ liệu
    refusal = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    
    if top_k <= 0 or not query.strip():
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}
    
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}
        
    if not chunks:
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}

    # [BÁO CÁO]: Fix lỗi "citation sai thứ tự sau reorder nên gắn nhãn trước khi reorder"
    # Gắn nhãn `_citation_index` tĩnh dựa trên thứ tự ban đầu của kết quả search
    labeled_chunks = [
        {
            **chunk,
            "metadata": {**chunk["metadata"], "_citation_index": index},
        }
        for index, chunk in enumerate(chunks, 1)
    ]
    
    # Lúc này dù reorder (đảo lộn vị trí để tránh lost-in-the-middle), các chunk vẫn giữ đúng số citation ban đầu
    context = format_context(reorder_for_llm(labeled_chunks))
    
    user_message = (
        f"Context:\n{context}\n\nCâu hỏi: {query}\n\n"
        "Trích dẫn theo dạng [Document N]."
    )
    
    try:
        answer = _normalize_citations(
            call_llm(SYSTEM_PROMPT, user_message), len(chunks)
        )
    except Exception:
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}
        
    if not answer:
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}
        
    lowered_answer = answer.lower()
    
    # [BÁO CÁO]: Kiểm tra fallback từ khóa từ chối của LLM
    if any(
        marker in lowered_answer
        for marker in (
            "không thể xác minh",
            "không thể trả lời",
            "không có đủ thông tin",
            "không có thông tin",
            "không có thông tin trong",
            "không tìm thấy trong",
        )
    ):
        return {"answer": answer, "sources": [], "retrieval_source": "none"}
        
    method = chunks[0]["retrieval_method"]
    retrieval_source = "pageindex" if method == "pageindex" else "hybrid"
    
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }

if __name__ == "__main__":
    print(generate_with_citation("test query"))