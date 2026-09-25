"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.

Citation: mỗi source có field "citation" = vị trí (bắt đầu từ 1) trong danh sách
sources. Context gắn nhãn [n] theo field này nên [n] trong answer luôn map về
sources[n - 1], dù thứ tự trong context đã bị reorder.
"""

import logging
import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

logger = logging.getLogger(__name__)

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 1024
LLM_TIMEOUT = 60

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5",
}
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_LLM_MODELS.get(LLM_PROVIDER, "")

REFUSAL_MESSAGE = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = f"""Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh.

Quy tắc:
- Mỗi đoạn context có nhãn dạng [n]. Sau mỗi khẳng định, ghi citation bằng đúng
  nhãn đó, ví dụ: "Học phí được đóng theo học kỳ [2]." Có thể ghi nhiều nhãn [1][3].
- Không dùng kiến thức ngoài context, không tự tạo nhãn không có trong context.
- Nếu context không chứa thông tin để trả lời, chỉ trả lời đúng câu:
  "{REFUSAL_MESSAGE}"
- Trả lời bằng ngôn ngữ của câu hỏi, ngắn gọn và đúng trọng tâm."""

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    # Input sort theo score giảm dần: hạng 1, 3, 5... ở đầu; hạng 2, 4... đảo
    # ngược ở cuối, nên chunk kém nhất nằm giữa context.
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for position, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        label = chunk.get("citation", position)
        header = f"[{label}] Title: {metadata['title']} | Source: {metadata['source']}"
        if metadata.get("url"):
            header += f" | URL: {metadata['url']}"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=LLM_TIMEOUT)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content or ""

    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        return response.text or ""

    if LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=LLM_TIMEOUT
        )
        # Model Claude mới không cho truyền đồng thời temperature và top_p.
        response = client.messages.create(
            model=LLM_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def remove_invalid_citations(answer: str, source_count: int) -> str:
    """Xoá citation [n] không map được về sources."""
    return CITATION_PATTERN.sub(
        lambda match: match.group(0) if 1 <= int(match.group(1)) <= source_count else "",
        answer,
    )


def _refusal() -> dict:
    return {"answer": REFUSAL_MESSAGE, "sources": [], "retrieval_source": "none"}


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    if not query.strip():
        return _refusal()

    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception as error:
        logger.warning("Retrieval failed: %s", error)
        return _refusal()
    if not chunks:
        return _refusal()

    sources = [{**chunk, "citation": index} for index, chunk in enumerate(chunks, 1)]
    context = format_context(reorder_for_llm(sources))
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message).strip()
    except Exception as error:
        logger.warning("LLM provider %s failed: %s", LLM_PROVIDER, error)
        return _refusal()

    answer = remove_invalid_citations(answer, len(sources)).strip()
    # Chỉ coi là từ chối khi toàn bộ câu trả lời là câu refusal; câu trả lời
    # một phần có citation vẫn được giữ.
    if not answer or answer.strip('"“” ') == REFUSAL_MESSAGE:
        return _refusal()

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": (
            "pageindex" if sources[0]["retrieval_method"] == "pageindex" else "hybrid"
        ),
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
