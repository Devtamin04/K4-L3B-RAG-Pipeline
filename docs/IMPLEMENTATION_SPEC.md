# Implementation spec — RAG pháp lý cho hộ kinh doanh (CPU)

## 1. Mục tiêu và phạm vi

Hoàn thiện chatbot RAG tiếng Việt về đăng ký, thuế và hóa đơn điện tử dành cho
hộ kinh doanh theo đúng
`docs/MODULE_CONTRACTS.md` và rubric của lab:

- Thu thập tối thiểu 3 tài liệu chính sách/quy định và 5 bài viết công khai.
- Chuẩn hóa dữ liệu về Markdown, chia chunk và lập chỉ mục ChromaDB.
- Truy hồi dense + BM25, hợp nhất đúng một lần bằng RRF.
- Dùng PageIndex làm fallback tùy chọn khi cosine score gốc thấp.
- Sinh câu trả lời có citation, safe refusal và giao diện Streamlit.
- Xây dựng tối thiểu 15 golden cases, đo 4 metric và so sánh A/B.

Chủ đề đã chốt là **Trợ lý pháp lý cho hộ kinh doanh Việt Nam**. Phạm vi gồm:
đăng ký/thay đổi/tạm ngừng hộ kinh doanh, chính sách và thủ tục thuế năm 2026,
hóa đơn điện tử. Không mở rộng sang tư vấn tranh chấp, lao động hoặc giấy phép
chuyên ngành. Contract giữa các module không thay đổi.

Corpus phải ghi rõ số hiệu, ngày ban hành, ngày hiệu lực và quan hệ sửa đổi. Khi
hai nguồn mâu thuẫn, ưu tiên văn bản có hiệu lực áp dụng mới hơn và phải nêu rõ
căn cứ. Đặc biệt, `141/2026/NĐ-CP` sửa `68/2026/NĐ-CP`, nâng ngưỡng liên quan
từ 500 triệu lên 1 tỷ đồng/năm và áp dụng từ 01/01/2026.

## 2. Ràng buộc kỹ thuật

### 2.1 Runtime

- Python 3.12 (repo hỗ trợ `>=3.10,<3.14`).
- Mọi package Python được cài trong `.venv`; không cài global.
- Không có GPU: mọi embedding chạy với `device="cpu"`, không dùng CUDA,
  `device_map="auto"` hay mixed precision.
- Trọng số model không nằm trong `.venv`. Đặt `HF_HOME=.cache/huggingface` để
  cache model trong workspace; `.cache/` đã được gitignore.
- API key chỉ đặt trong `.env`, không commit.

### 2.2 Embedding baseline và phương án bổ sung

- Model mặc định theo skeleton: `BAAI/bge-m3`.
- Backend: `sentence-transformers`.
- Dimension mặc định: 1024.
- Truyền nguyên văn tiếng Việt vào tokenizer của BGE-M3; không áp dụng PyVi cho
  embedding baseline.
- Encode với `normalize_embeddings=True`, `convert_to_numpy=True`,
  `show_progress_bar=False`, batch mặc định 8 và cuối cùng đổi sang `list`.
- Khởi tạo model theo lazy singleton để Streamlit không load lại ở mỗi query.
- Sau khi tải model lần đầu, có thể đặt `EMBEDDING_LOCAL_FILES_ONLY=true` để
  app không kiểm tra Hugging Face Hub ở mỗi lần khởi động.
- ChromaDB dùng cosine distance. Task 4 và Task 5 bắt buộc gọi chung
  `embed_texts()`; không tạo embedding function thứ hai.

Pseudo-code chuẩn:

```python
model = SentenceTransformer(MODEL_NAME, device="cpu")
prepared = [text.strip() for text in texts]
vectors = model.encode(
    prepared,
    batch_size=8,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)
return vectors.astype("float32").tolist()
```

Phải xử lý `texts == []` mà không load model và kiểm tra mỗi vector baseline có
đúng 1024 phần tử. Chunk được giới hạn theo ký tự ở bản đầu, nhưng cần kiểm tra
thêm số token để tránh âm thầm truncate nội dung dài.

`dangvantuan/vietnamese-embedding` được giữ như một cấu hình **thử nghiệm thêm,
chưa dùng mặc định**. Model này trả vector 768 chiều và model card gợi ý tiền xử
lý bằng `pyvi.ViTokenizer.tokenize`. Khi thử model này phải tạo collection Chroma
riêng hoặc rebuild toàn bộ index; không được trộn vector 768 và 1024 chiều trong
cùng collection. Nếu làm A/B embedding, document và query của từng nhánh phải
dùng cùng model và cùng preprocessing.

## 3. Thiết kế pipeline

```text
PDF/DOCX + URL
      -> landing/{legal,news}
      -> standardized Markdown
      -> stable chunks
      -> Vietnamese embeddings (CPU) -> Chroma cosine -> dense results
      -> same chunks -> BM25 --------------------------> sparse results
      -> RRF (một lần) -> threshold bằng dense cosine -> PageIndex fallback
      -> reorder + context labels -> LLM -> answer + citations
      -> Streamlit + evaluation report
```

### Task 1 — Legal documents

- Chọn ít nhất 3 URL chính thức, khai báo nguồn tập trung và tải bằng
  `requests` với timeout, user-agent và `raise_for_status()`.
- Tên file ổn định, không dấu; ghi file theo kiểu idempotent.
- Kiểm tra content type/kích thước để không lưu nhầm HTML thành PDF.

**Done:** `data/landing/legal/` có ít nhất 3 PDF/DOCX hợp lệ, mỗi file lớn hơn
1 KB và có nguồn kiểm chứng được.

Corpus pháp lý đã chọn gồm 5 văn bản chính thức:

1. `168/2025/NĐ-CP` — đăng ký doanh nghiệp và hộ kinh doanh.
2. `68/2026/NĐ-CP` — chính sách, quản lý thuế cho hộ/cá nhân kinh doanh.
3. `141/2026/NĐ-CP` — sửa đổi Nghị định 68, phải được ưu tiên khi xung đột.
4. `254/2026/NĐ-CP` — hóa đơn điện tử và chứng từ điện tử.
5. `24/2026/VBHN-TT-BTC` — văn bản hợp nhất thủ tục quản lý thuế.

Danh mục URL và metadata chi tiết nằm tại `docs/DATA_SOURCES.md`.

### Task 2 — News/articles

- Khai báo tối thiểu 5 URL công khai cùng chủ đề.
- Crawl tuần tự hoặc giới hạn concurrency để ổn định; lỗi một URL không dừng cả
  batch.
- Mỗi JSON chứa đủ `url`, `title`, `date_crawled`, `content_markdown`.

**Done:** có ít nhất 5 JSON không rỗng và chạy lại ghi đúng file tương ứng.

### Task 3 — Standardization

- Dùng MarkItDown cho PDF/DOCX; JSON bài viết được đưa về Markdown.
- Thêm front matter tối thiểu: title, source/url, doc type, crawl/publish date.
- Loại output rỗng, chuẩn hóa whitespace nhưng giữ heading/list/table có ích.

**Done:** `standardized/legal` có >=3 file và `standardized/news` có >=5 file;
mỗi file tối thiểu 200 ký tự.

### Task 4 — Chunk, embed, index

- `load_documents()` tạo ID từ relative path, giữ `source`, `title`, `doc_type`,
  `url`.
- Recursive chunking với baseline 500 ký tự, overlap 50; ưu tiên biên heading,
  đoạn, câu, whitespace.
- Chunk ID ổn định theo `{document_id}::chunk-{index}`; không mutate input.
- Embed theo batch bằng model baseline ở mục 2.2.
- Chroma persistent collection dùng cosine và `upsert` để re-index không trùng.
- Metadata gửi vào Chroma phải tránh giá trị `None` nếu version Chroma không hỗ
  trợ; khi đọc ra phải phục hồi đúng contract (`url: None` nếu thiếu).

**Done:** contract tests pass, dimension baseline là 1024, số record không tăng khi chạy
index hai lần trên cùng corpus.

### Task 5 — Dense retrieval

- Validate query không rỗng và `top_k > 0`.
- Embed query qua đúng `embed_texts()` của Task 4.
- Query Chroma, đổi cosine distance thành similarity `1 - distance`.
- Loại ID trùng, sort score giảm dần, trả tối đa `top_k`, method `dense`.

### Task 6 — BM25

- Dùng đúng tập chunks của Task 4, không duy trì corpus thủ công khác biệt.
- Tokenize tiếng Việt nhất quán (PyVi + lowercase) cho corpus và query.
- Có thể cache BM25 index theo phiên chạy sau khi corpus đã ổn định.
- Trả tối đa `top_k`, kể cả score bằng 0 để contract luôn có thứ hạng xác định;
  sort giảm dần và đặt method `bm25`.

### Task 7 — RRF

- Tính `sum(1 / (k + rank))`, rank bắt đầu từ 1, mặc định `k=60`.
- Deduplicate theo stable chunk ID; giữ content/metadata và đặt method `hybrid`.
- RRF chỉ chạy một lần. RRF score không được dùng cho fallback threshold.

### Task 8/9 — Fallback và orchestration

- Chạy dense và BM25 với candidate pool `top_k * 2`, sau đó fuse.
- So sánh threshold với **best dense cosine score gốc**.
- Nếu thấp hơn threshold, thử PageIndex khi có key/config; timeout hoặc provider
  lỗi thì trả hybrid result thay vì crash.
- Nếu không cấu hình PageIndex, coi đây là optional degraded mode và tiếp tục
  hybrid/safe refusal.
- Threshold đã được hiệu chỉnh trên 15 query in-domain và 5 query out-of-domain;
  kết quả hiện tại là `0.573` (balanced accuracy `1.0` trên tập calibration).

### Task 10 — Generation và citation

- Reorder không mutate input, không mất/đổi ID.
- Mỗi context block có số thứ tự, title, source và chunk ID.
- Prompt yêu cầu chỉ dùng context và citation dạng `[Document N]`.
- Kiểm tra citation tham chiếu đúng `sources`; không có evidence hoặc provider
  lỗi thì trả safe refusal với `retrieval_source="none"`.
- Chỉ bật một LLM provider cần dùng; các provider khác là optional.

### Streamlit

- Gọi `generate_with_citation(query, top_k)` và lưu cả answer + sources trong
  session state.
- Hiển thị title, URL/source, retrieval method và score cho từng citation.
- Bắt lỗi ở biên UI và hiển thị thông báo hữu ích, không làm app crash.

### Evaluation

- Tạo >=15 câu hỏi grounded từ corpus, gồm factual, multi-chunk, lexical-heavy
  và vài câu out-of-domain.
- Cùng golden dataset, generator, evaluator, prompt và `top_k` cho hai cấu hình:
  A dense-only; B dense + BM25 + RRF.
- Báo cáo faithfulness, answer relevance, context recall, context precision,
  latency và ít nhất 3 failure cases.

## 4. Thư viện

### Bắt buộc cho baseline

| Thư viện | Vai trò |
| --- | --- |
| `sentence-transformers` | Load và encode model embedding tiếng Việt |
| `pyvi` | Tokenize BM25 và hỗ trợ model tiếng Việt thử nghiệm |
| `torch` (dependency gián tiếp) | Inference tensor trên CPU |
| `chromadb` | Vector store persistent, cosine search |
| `langchain-text-splitters` | Recursive chunking |
| `rank-bm25` | Lexical retrieval |
| `numpy`, `scikit-learn` | Xử lý vector và phép đo/evaluation phụ trợ |
| `markitdown[pdf]` | Chuyển PDF/DOCX sang Markdown |
| `requests` | Tải tài liệu nguồn |
| `python-dotenv` | Đọc cấu hình `.env` |
| `streamlit` | Chat UI |
| `pytest` | Contract và acceptance tests |

### Theo chức năng, chỉ cần khi sử dụng

| Thư viện | Khi nào cần |
| --- | --- |
| `crawl4ai` + Playwright Chromium | Crawl các trang bài viết động |
| `openai` / `google-genai` / `anthropic` / `requests` | Provider generation được chọn, gồm Ollama Cloud API |
| `pageindex` | Vectorless fallback có API key |
| `ragas`, `datasets` | Chạy 4 metric và quản lý evaluation set |
| `fpdf2` | Chuyển Markdown sang PDF tạm cho PageIndex nếu cần |
| `langchain-community`, `langchain-openai` | Adapter/evaluation nếu implementation dùng LangChain |

`pyproject.toml` là dependency manifest duy nhất. Không tạo một
`requirements.txt` lệch phiên bản; khi cần reproducibility đầy đủ, sinh lock
file sau khi baseline đã chạy ổn.

## 5. Thiết lập môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Cài wheel CPU trước để resolver không kéo CUDA runtime từ PyPI.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m playwright install chromium  # chỉ cần cho Crawl4AI
cp .env.example .env
```

Smoke test model trên CPU (lần đầu cần tải trọng số):

```bash
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('BAAI/bge-m3', device='cpu'); print(m.get_sentence_embedding_dimension(), m.device)"
```

Kỳ vọng: dimension `1024`, device `cpu`.

## 6. Thứ tự triển khai và cổng kiểm thử

1. Setup `.venv`, `.env`; chạy `pytest tests/test_contracts.py -q` để ghi nhận
   baseline TODO.
2. Chốt chủ đề/nguồn; hoàn thiện Task 1–3 và acceptance tests về corpus.
3. Hoàn thiện Task 4–7; chạy contract tests sau mỗi task.
4. Hoàn thiện Task 9 trước; Task 8 có thể ở degraded mode nếu chưa có API key.
5. Hoàn thiện Task 10 và UI, test một query đúng domain và một query ngoài domain.
6. Tạo golden dataset, chạy A/B, điền `group_project/evaluation/RESULT.md`.
7. Chạy toàn bộ `pytest -q`, rà secret/cache, rồi demo.

## 7. Definition of Done

- Tất cả test contract và acceptance pass.
- App chạy end-to-end trên CPU từ một checkout sạch theo README.
- Dense index và query cùng model, cùng preprocessing, đúng dimension baseline
  1024.
- Re-index idempotent; search result đúng schema, thứ tự và giới hạn.
- Hybrid dùng RRF đúng một lần; fallback dùng dense cosine gốc.
- Answer có citation map được về sources hoặc safe refusal.
- Golden dataset >=15 case và báo cáo A/B không còn `TODO`.
- Repo không chứa `.env`, API key, model weights, Chroma DB hoặc cache runtime.

## 8. Rủi ro cần kiểm chứng sớm

- BGE-M3 tương đối lớn khi chạy CPU; lần đầu tải/load và thời gian index có thể
  chậm. Đo latency, RAM và cân nhắc batch 4 nếu máy thiếu bộ nhớ.
- Chỉ dùng PyVi trong nhánh `dangvantuan/vietnamese-embedding` hoặc BM25. Nếu thử
  model bổ sung, document và query tuyệt đối phải dùng cùng preprocessing.
- `CHUNK_SIZE=500` là ký tự trong splitter, không phải token; kiểm tra truncation
  ở model với các chunk dài.
- Chroma metadata và PageIndex SDK có thể khác theo version; test bằng response
  thực, không suy đoán field.
- Evaluation bằng LLM phát sinh API cost; cố định sample/config và cache kết quả
  để phép so sánh A/B công bằng.
