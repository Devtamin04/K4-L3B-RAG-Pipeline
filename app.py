import re

import streamlit as st
from dotenv import load_dotenv

from src.task9_retrieval_pipeline import SCORE_THRESHOLD
from src.task10_generation import (
    LLM_MODEL,
    LLM_PROVIDER,
    REFUSAL_MESSAGE,
    generate_with_citation,
)


load_dotenv()

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# Mỗi phương pháp có thang score riêng, cần ghi rõ khi hiển thị.
SCORE_LABELS = {
    "dense": "Cosine",
    "bm25": "BM25",
    "hybrid": "RRF",
    "pageindex": "PageIndex rank",
}
RETRIEVAL_SOURCE_LABELS = {
    "hybrid": "Hybrid (Dense + BM25 + RRF)",
    "pageindex": "PageIndex fallback",
    "none": "Không có nguồn phù hợp",
}

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def cited_numbers(answer: str) -> set[int]:
    return {int(number) for number in CITATION_PATTERN.findall(answer)}


def render_answer(answer: str) -> None:
    # In đậm [n] để dễ đối chiếu với danh sách nguồn. Chèn khoảng trắng phía
    # trước để hai citation liền nhau ([1][2]) không tạo ra "****" làm hỏng markdown.
    highlighted = CITATION_PATTERN.sub(r" **[\1]**", answer)
    st.markdown(re.sub(r" +(?= \*\*\[\d+\]\*\*)", "", highlighted))


def render_sources(message: dict) -> None:
    sources = message.get("sources") or []
    retrieval_source = message.get("retrieval_source", "none")
    st.caption(f"Retrieval: {RETRIEVAL_SOURCE_LABELS.get(retrieval_source, retrieval_source)}")
    if not sources:
        return

    cited = cited_numbers(message["content"])
    with st.expander(f"Nguồn tham khảo ({len(sources)})", expanded=bool(cited)):
        for position, source in enumerate(sources, 1):
            number = source.get("citation", position)
            metadata = source["metadata"]
            method = source["retrieval_method"]
            score_label = SCORE_LABELS.get(method, "Score")
            marker = "✅ được trích dẫn" if number in cited else "không được trích dẫn"

            title = metadata["title"]
            if metadata.get("url"):
                title = f"[{title}]({metadata['url']})"
            st.markdown(f"**[{number}] {title}** · {marker}")
            st.caption(
                f"{metadata['source']} · {metadata['doc_type']} · "
                f"chunk {metadata.get('chunk_index', '-')} · "
                f"{method} · {score_label}: {source['score']:.4f}"
            )
            st.text(source["content"])
            if position < len(sources):
                st.divider()


with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Thay mô tả theo đề tài của nhóm")
    top_k = st.slider("Số chunks", 3, 10, 5)
    st.divider()
    st.caption(f"LLM: {LLM_PROVIDER} · {LLM_MODEL or 'chưa cấu hình'}")
    st.caption(f"Ngưỡng fallback (cosine): {SCORE_THRESHOLD}")
    if st.button("Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("RAG Chatbot")
st.caption(
    "Thay tiêu đề và hướng dẫn sử dụng. Câu trả lời chỉ dựa trên tài liệu đã "
    "thu thập; mỗi ý có citation [n] trỏ tới nguồn bên dưới."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message["content"])
            render_sources(message)
        else:
            st.markdown(message["content"])

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            try:
                result = generate_with_citation(query, top_k=top_k)
            except Exception as error:
                # Lưới an toàn cuối cùng: UI không được crash vì pipeline.
                st.error(f"Pipeline gặp lỗi: {error}")
                result = {"answer": REFUSAL_MESSAGE, "sources": [], "retrieval_source": "none"}

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
        render_answer(assistant_message["content"])
        render_sources(assistant_message)

    st.session_state.messages.append(assistant_message)
