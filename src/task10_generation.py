"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
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

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp lý cho hộ kinh doanh Việt Nam.
Chỉ trả lời từ context được cung cấp và không suy diễn nghĩa vụ, thời hạn hoặc
thời điểm áp dụng nếu context không nêu rõ. Mỗi khẳng định phải có citation dạng
[Document N]. Nếu thiếu evidence, hãy từ chối xác minh. Câu trả lời chỉ mang
tính tham khảo và không thay thế tư vấn của cơ quan có thẩm quyền."""


def _normalize_citations(answer: str, source_count: int) -> str:
    """Normalize common bracket variants and drop impossible citation labels."""
    answer = re.sub(
        r"【\s*Document\s+(\d+)\s*】",
        lambda match: f"[Document {match.group(1)}]",
        answer,
        flags=re.IGNORECASE,
    )

    def keep_valid(match: re.Match) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= source_count else ""

    return re.sub(
        r"\[Document\s+(\d+)\]", keep_valid, answer, flags=re.IGNORECASE
    ).strip()


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        url = metadata.get("url") or "N/A"
        citation_index = metadata.get("_citation_index", index)
        parts.append(
            f"[Document {citation_index} | ID: {chunk['id']} | Title: {metadata['title']} | "
            f"Source: {metadata['source']} | URL: {url}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL chưa được cấu hình")

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY chưa được cấu hình")
        response = OpenAI().responses.create(
            model=LLM_MODEL,
            instructions=system_prompt,
            input=user_message,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.output_text.strip()

    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY chưa được cấu hình")
        response = genai.Client(api_key=os.environ["GEMINI_API_KEY"]).models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return (response.text or "").strip()

    if LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY chưa được cấu hình")
        response = Anthropic().messages.create(
            model=LLM_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1200,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

    if LLM_PROVIDER == "ollama":
        api_key = os.getenv("OLLAMA_API_KEY", "")
        chat_url = os.getenv("OLLAMA_CHAT_URL", "https://ollama.com/api/chat")
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY chưa được cấu hình")
        response = requests.post(
            chat_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": TEMPERATURE, "top_p": TOP_P},
            },
            timeout=(10, 180),
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama không trả về nội dung hợp lệ")
        return content.strip()

    raise ValueError(f"LLM_PROVIDER không được hỗ trợ: {LLM_PROVIDER}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    refusal = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    if top_k <= 0 or not query.strip():
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}
    if not chunks:
        return {"answer": refusal, "sources": [], "retrieval_source": "none"}

    # Keep citation numbers tied to the original, score-sorted sources even
    # though context blocks are reordered to reduce lost-in-the-middle.
    labeled_chunks = [
        {
            **chunk,
            "metadata": {**chunk["metadata"], "_citation_index": index},
        }
        for index, chunk in enumerate(chunks, 1)
    ]
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
