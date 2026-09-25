import html
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from demo_embeddings import (
    ALTERNATIVE_MODEL,
    DEFAULT_MODEL,
    embedding_scope,
    indexed_chunk_count,
)
from src.task1_collect_legal_docs import LEGAL_DOCUMENTS
from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import SCORE_THRESHOLD
from src.task10_generation import generate_with_citation


load_dotenv()

ROOT = Path(__file__).parent
EVALUATION_DIR = ROOT / "group_project" / "evaluation"
STYLE_PATH = ROOT / "demo_style.css"

EMBEDDING_TABS = {
    "BGE-M3": DEFAULT_MODEL,
    "Vietnamese Embedding": ALTERNATIVE_MODEL,
}
RRF_K = 60
OUT_OF_DOMAIN_QUESTION = "Thời tiết Hà Nội ngày mai thế nào?"
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
EMBLEM_SVG = (
    '<svg viewBox="0 0 100 100" class="emblem">'
    '<circle cx="50" cy="50" r="47" fill="#c00000" stroke="#f59e0b" stroke-width="2.5"/>'
    '<circle cx="50" cy="50" r="42" fill="none" stroke="#fef08a" stroke-width="1" '
    'stroke-dasharray="2,2" opacity="0.7"/>'
    '<circle cx="50" cy="50" r="37" fill="none" stroke="#fef08a" stroke-width="0.8" opacity="0.6"/>'
    '<path d="M 20 62 C 14 42, 25 22, 50 15 C 75 22, 86 42, 80 62" fill="none" '
    'stroke="#facc15" stroke-width="3" stroke-linecap="round"/>'
    '<path d="M 23 60 C 18 44, 28 26, 50 20 C 72 26, 82 44, 77 60" fill="none" '
    'stroke="#fef08a" stroke-width="1.5" stroke-linecap="round"/>'
    '<circle cx="50" cy="72" r="13" fill="#ca8a04" stroke="#fef08a" stroke-width="1.5"/>'
    '<circle cx="50" cy="72" r="7" fill="#c00000"/>'
    '<path d="M 26 78 Q 50 85 74 78 L 70 86 Q 50 90 30 86 Z" fill="#b91c1c" '
    'stroke="#fef08a" stroke-width="1"/>'
    '<polygon points="50,26 54.5,39 68,39 57,47.5 61,60.5 50,52 39,60.5 43,47.5 32,39 45.5,39" '
    'fill="#facc15" stroke="#ca8a04" stroke-width="0.5"/>'
    "</svg>"
)


st.set_page_config(
    page_title="Trợ lý Pháp lý Hộ Kinh Doanh",
    page_icon="⚖️",
    layout="wide",
)
st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = {model: [] for model in EMBEDDING_TABS.values()}
    st.session_state.pending_query = None


# ---------------------------------------------------------------- data


@st.cache_data
def load_golden_questions() -> list[dict]:
    path = EVALUATION_DIR / "golden_dataset.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@st.cache_data
def load_ab_summary() -> dict | None:
    path = EVALUATION_DIR / "evaluation_results.json"
    if not path.exists():
        return None
    cases = json.loads(path.read_text(encoding="utf-8")).values()
    summary = {}
    for config in ("dense", "hybrid"):
        runs = [case[config] for case in cases if not case[config].get("error")]
        metrics = {
            name: sum(run["metrics"][name] for run in runs) / len(runs)
            for name in runs[0]["metrics"]
        }
        metrics["average"] = sum(metrics.values()) / len(metrics)
        latency = sum(run["retrieval_latency_seconds"] for run in runs) / len(runs)
        summary[config] = {"metrics": metrics, "latency": latency, "cases": len(runs)}
    return summary


def document_label(metadata: dict) -> str:
    """'141-2026-ND-CP-....pdf' -> 'Nghị định 141/2026/NĐ-CP'."""
    source = str(metadata.get("source") or "")
    if match := re.match(r"(\d+)-(\d{4})-ND-CP", source):
        return f"Nghị định {match[1]}/{match[2]}/NĐ-CP"
    if match := re.match(r"(\d+)-(\d{4})-VBHN-TT-BTC", source):
        return f"VBHN {match[1]}/{match[2]}/VBHN-BTC"
    return str(metadata.get("title") or source)


# ---------------------------------------------------------------- pipeline


def answer_query(query: str, model_name: str, top_k: int) -> dict:
    """Chạy pipeline lab với model đã chọn, kèm dữ liệu truy vết cho UI."""
    started = time.perf_counter()
    with embedding_scope(model_name):
        dense = semantic_search(query, top_k=1)
        result = generate_with_citation(query, top_k=top_k)
    return {
        **result,
        "trace": {
            "model": model_name,
            "best_dense_score": dense[0]["score"] if dense else 0.0,
            "elapsed": time.perf_counter() - started,
            "top_k": top_k,
        },
    }


# ---------------------------------------------------------------- render helpers


def render_header(model_name: str, top_k: int) -> None:
    now = datetime.now()
    today = f"{WEEKDAYS[now.weekday()]}, {now:%d/%m/%Y}"
    llm = html.escape(os.getenv("LLM_MODEL", "LLM"))
    st.markdown(
        '<header class="gov-header">'
        f'<div class="gov-brand">{EMBLEM_SVG}<div>'
        '<div class="gov-kicker"><span>CỔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ</span>'
        '<span class="tag-red">VĂN BẢN HIỆN HÀNH 2026</span></div>'
        '<h1 class="gov-title">XÂY DỰNG CHÍNH SÁCH, PHÁP LUẬT</h1>'
        '<p class="gov-sub">Trợ lý AI tra cứu pháp lý &amp; chính sách thuế dành cho '
        "Hộ kinh doanh Việt Nam</p></div></div>"
        '<div class="gov-meta">'
        f'<div class="gov-date"><span class="dot-live"></span>{today}'
        f'<span class="sep">|</span><span class="mono">LLM: {llm}</span></div>'
        '<div class="gov-badges">'
        f'<span class="badge">Model: <strong class="red">{html.escape(model_name)} (CPU)</strong></span>'
        f'<span class="badge">RRF: <strong class="green">k={RRF_K} (Top {top_k})</strong></span>'
        f'<span class="badge amber">Ngưỡng: <strong>{SCORE_THRESHOLD:.3f}</strong></span>'
        "</div></div></header>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> int:
    golden = load_golden_questions()
    with st.sidebar:
        st.markdown(
            '<div class="card-drum">'
            '<svg class="drum-watermark" viewBox="0 0 100 100">'
            '<circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" stroke-width="3"/>'
            '<circle cx="50" cy="50" r="35" fill="none" stroke="currentColor" stroke-width="2"/>'
            '<polygon points="50,15 58,40 85,50 58,60 50,85 42,60 15,50 42,40" fill="currentColor"/>'
            "</svg>"
            '<h3 class="side-title red">🏛️ Dữ liệu văn bản chuẩn hóa</h3>'
            "<p>Văn bản pháp luật từ datafiles.chinhphu.vn và bài hướng dẫn từ "
            "xaydungchinhsach.chinhphu.vn, chuyển về Markdown để truy hồi.</p></div>",
            unsafe_allow_html=True,
        )

        with st.container(key="params"):
            st.markdown(
                '<div class="side-row"><span>⚙️ THAM SỐ TRUY HỒI</span>'
                '<span class="red mono">Hybrid RRF</span></div>',
                unsafe_allow_html=True,
            )
            top_k = st.slider("Số chunks trích xuất (Top-K)", 3, 10, 5)
            st.markdown(
                '<div class="kv-box">'
                "<div><span>Phương thức:</span><strong>Dense + BM25 (RRF)</strong></div>"
                "<div><span>Fallback PageIndex:</span>"
                f'<strong class="green">Dense &lt; {SCORE_THRESHOLD:.3f}</strong></div>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="side-row section"><span>🎯 CÂU HỎI ĐIỂN HÌNH (GOLDEN)</span>'
            f'<span class="tag-red mono">{len(golden)} câu</span></div>',
            unsafe_allow_html=True,
        )
        for index, case in enumerate(golden[:3], 1):
            preset_button(f"{index}. {case['question']}", case["question"], f"preset_{index}", case["expected_answer"])
        preset_button(
            f"🛡️ Out-of-domain: {OUT_OF_DOMAIN_QUESTION}",
            OUT_OF_DOMAIN_QUESTION,
            "preset_ood",
            "Out-of-domain: kiểm thử cơ chế Safe Refusal",
        )
        if len(golden) > 3:
            with st.expander(f"Xem thêm {len(golden) - 3} câu"):
                for index, case in enumerate(golden[3:], 4):
                    preset_button(f"{index}. {case['question']}", case["question"], f"preset_{index}", case["expected_answer"])

        items = "".join(
            f'<li><a href="{html.escape(url)}" target="_blank">📜 '
            f"{html.escape(document_label({'source': filename}))}</a>"
            '<span class="muted">PDF</span></li>'
            for filename, url in LEGAL_DOCUMENTS.items()
        )
        news_count = len(list((ROOT / "data" / "standardized" / "news").glob("*.md")))
        st.markdown(
            '<div class="corpus"><h3 class="side-title">📂 DANH MỤC VĂN BẢN CƠ SỞ</h3>'
            f'<ul>{items}<li><span>📰 Bài hướng dẫn chinhphu.vn</span>'
            f'<span class="muted">{news_count} bài</span></li></ul></div>',
            unsafe_allow_html=True,
        )
    return top_k


def preset_button(label: str, question: str, key: str, hint: str) -> None:
    def ask() -> None:
        st.session_state.pending_query = question

    st.button(label, key=key, help=hint, on_click=ask, width="stretch")


def render_welcome(model_name: str) -> None:
    indexed = indexed_chunk_count(model_name)
    st.markdown(
        '<div class="welcome"><div class="welcome-icon">⚖️</div><div>'
        '<div class="welcome-title"><h2>Trợ lý Pháp lý Hộ Kinh Doanh Việt Nam</h2>'
        '<span class="tag-red">CÓ TRÍCH DẪN</span></div>'
        "<p>Hệ thống hỗ trợ tra cứu các quy định về đăng ký kinh doanh, nghĩa vụ thuế và "
        "hóa đơn điện tử năm 2026. Câu trả lời được dẫn xuất từ văn bản trong corpus và "
        'đánh dấu thẻ trích dẫn <span class="cite-static">[Document N]</span>.</p>'
        '<div class="chips">'
        '<span class="chip"><span class="red">●</span> Click thẻ <strong>[Document N]</strong> '
        "để làm nổi bật nguồn</span>"
        f'<span class="chip"><span class="green">●</span> Embedding <code>{html.escape(model_name)}</code>'
        f" — {indexed} chunks đã index</span>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )
    if not indexed:
        command = (
            "python -m src.task4_chunking_indexing"
            if model_name == DEFAULT_MODEL
            else "python -m demo_embeddings"
        )
        st.warning(f"Chưa có index cho model này. Chạy: `{command}`")


def render_user(text: str) -> None:
    st.markdown(
        f'<div class="row-user"><div class="bubble-user">{html.escape(text)}</div>'
        '<div class="avatar avatar-user">Bạn</div></div>',
        unsafe_allow_html=True,
    )


def render_trace(result: dict) -> None:
    trace = result["trace"]
    score = trace["best_dense_score"]
    in_domain = score >= SCORE_THRESHOLD
    source = result["retrieval_source"]
    verdict = (
        f"Cosine Score: {score:.3f} ≥ {SCORE_THRESHOLD:.3f} (In-Domain)"
        if in_domain
        else f"Cosine Score: {score:.3f} &lt; {SCORE_THRESHOLD:.3f} (Fallback)"
    )
    if source == "pageindex":
        step3 = ("3. PageIndex Fallback", "Dense dưới ngưỡng", "Tìm theo cây mục lục văn bản")
    else:
        step3 = (f"3. RRF Fusion", f"Reciprocal Rank k={RRF_K}", f"Top {trace['top_k']} chunks sau hợp nhất")
    st.markdown(
        '<details class="trace" open><summary>'
        '<span><span class="red strong">⚡ TRUY VẾT PIPELINE:</span> '
        f"Hybrid Retrieval (Dense + BM25 → RRF k={RRF_K})</span>"
        f'<span class="score-pill {"ok" if in_domain else "warn"}">{verdict}</span>'
        '</summary><div class="trace-grid">'
        '<div><span class="red strong">1. Dense Search</span>'
        f'<span class="muted">Embedding: {html.escape(trace["model"])}</span>'
        f"<span>Top-1 cosine: {score:.3f}</span></div>"
        '<div><span class="amber strong">2. Lexical Search</span>'
        '<span class="muted">Tokenizer: PyVi Tiếng Việt</span>'
        f"<span>Top {trace['top_k'] * 2} BM25 Okapi matches</span></div>"
        f'<div><span class="green strong">{step3[0]}</span>'
        f'<span class="muted">{step3[1]}</span><span class="green strong">{step3[2]}</span></div>'
        '</div><div class="trace-foot">'
        "<span>Reordering Context: đưa chunks trọng tâm về đầu và cuối để tránh Lost-in-the-middle</span>"
        f'<span class="strong">Tổng thời gian: {trace["elapsed"]:.2f}s</span>'
        "</div></details>",
        unsafe_allow_html=True,
    )


def render_answer(result: dict, message_id: str) -> None:
    sources = result["sources"]
    if not sources:
        render_refusal(result)
        return

    def cite(match: re.Match) -> str:
        number = int(match[1])
        return (
            f'<a class="cite" href="#src-{message_id}-{number}" target="_self">'
            f"📄 [Document {number}]</a>"
        )

    body = re.sub(r"\[Document\s+(\d+)\]", cite, result["answer"].replace("<", "&lt;"))
    avatar, content = st.columns([1, 24], gap="small")
    avatar.markdown('<div class="avatar avatar-ai">AI</div>', unsafe_allow_html=True)
    with content:
        with st.container(key=f"answer_{message_id}"):
            st.markdown(
                '<div class="answer-head"><span>🏛️ KẾT QUẢ TRA CỨU PHÁP LUẬT</span>'
                '<span class="muted mono">Trích xuất tự động</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(body, unsafe_allow_html=True)
            st.markdown(
                '<div class="note"><span>⚠️ Lưu ý: Thông tin tra cứu mang tính chất tham khảo, '
                "không thay thế văn bản hướng dẫn nghiệp vụ của cơ quan thuế quản lý trực tiếp."
                '</span><span class="strong">Nguồn: Cổng TTĐT Chính phủ</span></div>',
                unsafe_allow_html=True,
            )
        render_sources(sources, message_id)


def render_sources(sources: list[dict], message_id: str) -> None:
    cards = []
    for number, source in enumerate(sources, 1):
        metadata = source["metadata"]
        url = metadata.get("url") or metadata.get("source") or ""
        method = source["retrieval_method"]
        score_name = {"hybrid": "RRF", "dense": "Cosine"}.get(method, method.title())
        snippet = " ".join(source["content"].split())
        cards.append(
            f'<div class="src-card" id="src-{message_id}-{number}">'
            f'<div class="src-top"><span class="cite-static">[Document {number}]</span>'
            f'<span class="score-pill ok small">{score_name}: {source["score"]:.3f}</span></div>'
            f"<h4>{html.escape(document_label(metadata))}</h4>"
            f'<p>"{html.escape(snippet[:220])}{"…" if len(snippet) > 220 else ""}"</p>'
            f'<a class="src-link" href="{html.escape(url)}" target="_blank">🔗 {html.escape(url)}</a>'
            "</div>"
        )
    st.markdown(
        '<div class="src-head"><span><span class="red">📑</span> CĂN CỨ VĂN BẢN ĐÃ DÙNG '
        "(Click thẻ để làm nổi bật căn cứ):</span>"
        f'<span class="muted">{len(sources)} tài liệu liên quan</span></div>'
        f'<div class="src-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_refusal(result: dict) -> None:
    trace = result["trace"]
    score = trace["best_dense_score"]
    if score < SCORE_THRESHOLD:
        title = "⚠️ NGOÀI PHẠM VI TÀI LIỆU (SAFE REFUSAL)"
        pill = f"Cosine Score: {score:.3f} &lt; Ngưỡng {SCORE_THRESHOLD:.3f}"
        tip = ""
    else:
        # Retrieval found related documents, but the LLM judged the top-k
        # chunks insufficient and declined (Task 10 then drops the sources).
        title = "⚠️ KHÔNG ĐỦ CĂN CỨ TRONG NGUỒN TRUY HỒI"
        pill = f"Cosine Score: {score:.3f} ≥ Ngưỡng {SCORE_THRESHOLD:.3f}"
        tip = (
            '<p class="refusal-tip">💡 Đã tìm thấy tài liệu liên quan nhưng '
            f"{trace['top_k']} chunks lấy về chưa chứa câu trả lời. Thử tăng "
            "<strong>Top-K</strong> ở thanh bên hoặc hỏi cụ thể hơn "
            "(nêu số nghị định, mẫu biểu, mốc thời gian).</p>"
        )
    st.markdown(
        '<div class="row-ai"><div class="avatar avatar-shield">🛡️</div>'
        f'<div class="refusal"><div class="refusal-head"><span>{title}</span>'
        f'<span class="score-pill warn">{pill}</span></div>'
        f"<p>{html.escape(result['answer'])}</p>{tip}</div></div>",
        unsafe_allow_html=True,
    )


@st.dialog("📊 Báo cáo Đánh giá A/B (Ragas Framework)", width="large")
def show_ab_report() -> None:
    summary = load_ab_summary()
    if not summary:
        st.info("Chưa có evaluation_results.json. Chạy group_project/evaluation/run_evaluation.py.")
        return
    dense, hybrid = summary["dense"], summary["hybrid"]
    labels = {
        "faithfulness": "Faithfulness (Độ trung thực)",
        "answer_relevance": "Answer Relevance (Độ phù hợp đáp án)",
        "context_recall": "Context Recall (Độ bao phủ ngữ cảnh)",
        "context_precision": "Context Precision (Độ chính xác ngữ cảnh)",
        "average": "Điểm trung bình (Average Score)",
    }
    rows = []
    for name, label in labels.items():
        a, b = dense["metrics"][name], hybrid["metrics"][name]
        tone = "green" if b >= a else "red"
        rows.append(
            f'<tr class="{"avg" if name == "average" else ""}"><td class="sans">{label}</td>'
            f"<td>{a:.3f}</td><td class='{tone} strong'>{b:.3f}</td>"
            f"<td class='right {tone} strong'>{b - a:+.3f}</td></tr>"
        )
    st.markdown(
        f'<p class="muted">So sánh Dense-only và Hybrid RRF trên {hybrid["cases"]} golden cases '
        "(embedding BAAI/bge-m3).</p>"
        '<table class="ab-table"><thead><tr><th>Tiêu chí đo lường (Metric)</th>'
        "<th>Config A (Dense)</th><th>Config B (Hybrid RRF)</th>"
        '<th class="right">Chênh lệch (Delta)</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '<div class="note block"><p><strong>Độ trễ truy hồi trung bình:</strong> '
        f'Dense-only: {dense["latency"]:.3f}s | Hybrid RRF: {hybrid["latency"]:.3f}s.</p>'
        "<p>Chi tiết từng câu và phân tích lỗi: group_project/evaluation/RESULT.md.</p></div>",
        unsafe_allow_html=True,
    )


def reset_chat() -> None:
    st.session_state.messages = {model: [] for model in EMBEDDING_TABS.values()}
    st.session_state.pending_query = None


# ---------------------------------------------------------------- page

top_k = render_sidebar()
active_label = st.session_state.get("embedding_tab", next(iter(EMBEDDING_TABS)))
model_name = EMBEDDING_TABS[active_label]
render_header(model_name, top_k)

with st.container(key="navbar", horizontal=True, vertical_alignment="center"):
    st.tabs(list(EMBEDDING_TABS), key="embedding_tab", on_change="rerun")
    if st.button("📊 Xem A/B Benchmark", key="ab_button"):
        show_ab_report()
    st.button("Làm mới", key="reset_button", on_click=reset_chat)

query = st.chat_input(
    "Nhập câu hỏi về chính sách thuế, đăng ký kinh doanh, hóa đơn năm 2026..."
)
query = query or st.session_state.pending_query
st.session_state.pending_query = None

with st.container(key="chat"):
    render_welcome(model_name)
    messages = st.session_state.messages[model_name]
    for index, message in enumerate(messages):
        render_user(message["query"])
        render_trace(message["result"])
        render_answer(message["result"], f"{index}")

    if query:
        render_user(query)
        with st.spinner("Đang tra cứu nguồn..."):
            try:
                result = answer_query(query, model_name, top_k)
            except Exception as error:
                st.error(f"Không thể tạo câu trả lời: {error}")
                result = None
        if result:
            render_trace(result)
            render_answer(result, f"{len(messages)}")
            messages.append({"query": query, "result": result})

    st.markdown(
        '<div class="footer-hint"><span>💡 Nhấn vào câu hỏi bên trái hoặc nhập thử: '
        "<em>\"Thời tiết hôm nay\"</em> để xem tính năng Safe Refusal.</span>"
        '<span class="mono">RAG Pipeline • Lab K4</span></div>',
        unsafe_allow_html=True,
    )
