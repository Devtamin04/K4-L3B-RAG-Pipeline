import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="Trợ lý hộ kinh doanh",
    page_icon="📋",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Trợ lý hộ kinh doanh")
    st.caption("Tra cứu đăng ký, thuế và hóa đơn điện tử từ nguồn chính thức")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("Trợ lý pháp lý cho hộ kinh doanh")
st.caption("Câu trả lời chỉ mang tính tham khảo và luôn kèm nguồn văn bản.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Nguồn tham khảo"):
                for source in message["sources"]:
                    metadata = source["metadata"]
                    st.markdown(
                        f"- **{metadata['title']}** — `{source['score']:.3f}` "
                        f"({source['retrieval_method']})  \n"
                        f"  {metadata.get('url') or metadata['source']}"
                    )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu nguồn..."):
            result = generate_with_citation(query, top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        st.markdown(answer)

        if sources:
            with st.expander("Nguồn tham khảo"):
                for source in sources:
                    metadata = source["metadata"]
                    st.markdown(
                        f"- **{metadata['title']}** — `{source['score']:.3f}` "
                        f"({source['retrieval_method']})  \n"
                        f"  {metadata.get('url') or metadata['source']}"
                    )

    st.session_state.messages.extend(
        [
            {"role": "assistant", "content": answer, "sources": sources},
        ]
    )
