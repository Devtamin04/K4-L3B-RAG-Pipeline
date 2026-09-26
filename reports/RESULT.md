# RAG evaluation results

## Thông tin nhóm

| Field | Value |
| --- | --- |
| Nhóm | K4-L3B |
| Repository/branch | `K4-L3B-RAG-Pipeline` / `main` |
| Bài toán | Trợ lý tra cứu pháp lý, thuế và hóa đơn điện tử cho hộ kinh doanh Việt Nam |
| Corpus | 5 văn bản pháp luật (PDF + HTML toàn văn) và 5 bài hướng dẫn từ chinhphu.vn |

## Thành viên và phân công

Mỗi thành viên tự điền dòng của mình và chỉ kê khai phần việc đối chiếu được bằng file, commit, test hoặc kết quả evaluation. Thêm hoặc bớt dòng theo số thành viên thực tế.

| # | Họ và tên | Mã học viên | Module phụ trách | Individual report |
| --: | --- | --- | --- | --- |
| 1 | Nguyễn Quang Tuấn | 2A202602470 | Data pipeline, Retrieval, Orchestration, Generation/UI, Evaluation | `reports/2A202602470-nqtuan.md` |
| 2 | Đỗ Trương Thành An | 2A202602889 | Data pipeline, Retrieval (BM25Plus & Chunking), Orchestration (Fallback), Generation/UI (Reordering) | `reports/2A202602889-thanhan.md` |
| 3 | Lê Minh Hiếu | 2A202602828 | Retrieval, Orchestration | `reports/2A202602828-hieulm.md` |
| 4 |  |  |  |  |

### Ownership theo module

| Module | Phụ trách chính | Thành viên hỗ trợ | Bằng chứng | Trạng thái |
| --- | --- | --- | --- | --- |
| Data pipeline (Task 1–3) | Nguyễn Quang Tuấn | Đỗ Trương Thành An | `src/task1_*` đến `src/task3_*`, `data/` | Done |
| Retrieval (Task 4–7) | Nguyễn Quang Tuấn | Lê Minh Hiếu | `src/task4_*` đến `src/task7_*` | Done |
| Orchestration (Task 8–9) | Nguyễn Quang Tuấn | Lê Minh Hiếu | `src/task8_*`, `src/task9_*` | Done |
| Generation/UI (Task 10) | Nguyễn Quang Tuấn | Đỗ Trương Thành An | `src/task10_generation.py`, `app.py` | Done |
| Evaluation | Nguyễn Quang Tuấn | Đỗ Trương Thành An | `group_project/evaluation/` | Done |
|  |  |  |  |  |

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-23 |
| Framework and version | Ragas 0.4.3 |
| Evaluator model | `gpt-oss:120b` qua Ollama Cloud |
| Generator model | `gpt-oss:120b` qua Ollama Cloud |
| Embedding model | `BAAI/bge-m3`, CPU, 1024 chiều |
| Corpus version | Corpus hộ kinh doanh chuẩn hóa 2026-09-23 |
| Golden dataset size | 15 |
| `top_k` | 5 |
| Fallback threshold calibration | `0.573`; balanced accuracy `1.000` |

## Configurations

- **Config A — dense-only:** Chroma cosine, lấy trực tiếp top 5.
- **Config B — hybrid + RRF:** dense và BM25 lấy 10 ứng viên mỗi nhánh, RRF `k=60`, trả top 5.
- Hai cấu hình dùng cùng corpus, golden dataset, generator, evaluator, prompt và `top_k`.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.674 | 0.680 | +0.006 |
| Answer relevance | 0.718 | 0.755 | +0.037 |
| Context recall | 0.800 | 0.867 | +0.067 |
| Context precision | 0.840 | 0.868 | +0.028 |
| **Average** | 0.758 | 0.792 | +0.035 |

## A/B comparison

- Cấu hình có điểm trung bình cao hơn: **hybrid**.
- Retrieval latency trung bình: dense `0.796s`, hybrid `2.005s`.
- Hai nhánh generation được chạy đồng thời trong lượt đo hiện tại, vì vậy latency API riêng lẻ chỉ mang tính quan sát và không dùng để kết luận A/B.
- Hybrid thêm BM25 và RRF tại máy, nhưng vẫn chỉ phát sinh một API generation cho mỗi câu như dense.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Hộ tính thuế theo phương pháp thu nhập tính thuế nhân thuế suất phải lập bảng kê tài sản nào khi chuyển đổi? | dense | 0.000 | 0.000 | 0.000 | 0.250 | retrieval/data | Top-k không chứa đủ bằng chứng cần thiết để trả lời |
| 2 | Có thể nộp hồ sơ khai thuế hộ kinh doanh bằng phương thức điện tử ở đâu? | dense | 0.167 | 0.000 | 0.000 | 0.500 | retrieval/data | Top-k không chứa đủ bằng chứng cần thiết để trả lời |
| 3 | Hộ tính thuế theo phương pháp thu nhập tính thuế nhân thuế suất phải lập bảng kê tài sản nào khi chuyển đổi? | hybrid | 0.000 | 0.000 | 0.000 | 0.887 | retrieval/data | Top-k không chứa đủ bằng chứng cần thiết để trả lời |

## Recommendations

| Priority | Action | Evidence | Expected impact | How to verify |
| --: | --- | --- | --- | --- |
| 1 | Dùng threshold `0.573` | Calibration đạt balanced accuracy `1.000` | Fallback đúng hơn | Chạy lại tập in/out-domain |
| 2 | Deduplicate nguồn cùng nội dung và tăng đa dạng top-k | Các lỗi precision thường do nhiều chunk gần nhau | Tăng context precision | So sánh precision trước/sau |
| 3 | Thêm metadata ngày hiệu lực và ưu tiên văn bản sửa đổi | NĐ 141 sửa ngưỡng của NĐ 68 | Tránh trả quy định cũ | Test riêng câu hỏi ngưỡng thuế |

## Artifacts

- `evaluation_results.json`: câu trả lời, context, nguồn, latency và điểm từng case.
- `threshold_calibration.json`: score của câu in-domain/out-of-domain và threshold được chọn.
- `golden_dataset.json`: 15 câu hỏi/đáp án/context chuẩn.

## Bonus experiments

| Hạng mục | Trạng thái | Bằng chứng | Ghi chú |
| --- | --- | --- | --- |
| UI citation/source highlighting | Chạy được trong demo | `app.py`, `demo_style.css` | Click `[Document N]` để làm nổi bật card nguồn tương ứng |
| Embedding thay thế `dangvantuan/vietnamese-embedding` | Có trong demo, chưa đánh giá Ragas | `demo_embeddings.py`, tab "Vietnamese Embedding" | Collection riêng 768 chiều; threshold `0.573` chưa calibrate lại cho model này |
|  |  |  |  |

Pipeline lab trong `src/` vẫn chỉ dùng `BAAI/bge-m3`; model thay thế không được trộn vào collection BGE-M3.

## Kiểm thử

- `pytest -q`: 21 passed (chạy lại ngày 2026-09-25).

## Ghi chú bổ sung của thành viên

Dành cho các thành viên khác bổ sung phân tích lỗi, thí nghiệm hoặc đề xuất thuộc phần mình phụ trách.

### Thành viên 2 — Đỗ Trương Thành An (2A202602889)

- Nội dung:
  - **Data Pipeline (Task 1–3):** Bổ sung hỗ trợ đa định dạng đầu vào (PDF/DOCX/HTML) vào Task 3; xây dựng cơ chế kiểm tra và thông báo theo từng file trong `convert_legal_docs` và `convert_news_articles`, bổ sung cờ `--force` để hỗ trợ re-standardize dữ liệu sạch khi có văn bản mới.
  - **Cải tiến Lexical Search (Task 6):** Phân tích lỗi triệt tiêu điểm do IDF âm/bằng 0 trên corpus câu ngắn của `BM25Okapi`; thay thế bằng `BM25Plus` với hằng số $\delta=1$ giúp mọi từ khóa khớp đều đạt điểm dương và tuân thủ contract sorting.
  - **Orchestration & Fallback (Task 8–9):** Hiệu chỉnh cơ chế fallback PageIndex kích hoạt dựa trên Cosine Similarity gốc thay vì điểm RRF để phản ánh chính xác ngữ nghĩa out-of-domain; bọc try-except graceful fallback tránh crash khi thiếu key dịch vụ ngoài (degraded mode).
  - **Chống Lost-in-the-middle (Task 10):** Triển khai thuật toán `reorder_for_llm` luân phiên đưa các chunk điểm cao về đầu và cuối context trước khi gửi đến LLM, bảo toàn citations map chính xác với sources.
- Bằng chứng:
  - Báo cáo cá nhân: `reports/2A202602889-thanhan.md`
  - Mã nguồn: `src/task3_convert_markdown.py`, `src/task6_lexical_search.py`, `src/task8_pageindex_vectorless.py`, `src/task9_retrieval_pipeline.py`, `src/task10_generation.py`
  - Kết quả kiểm thử: `pytest -q` pass 21/21 tests; kết quả đánh giá A/B cho thấy nhánh Hybrid đạt điểm trung bình 0.792 (vượt trội hơn Dense-only 0.758; context recall đạt 0.867 so với 0.800 của dense).

### Thành viên 3 — Lê Minh Hiếu (2A202602828)

- Nội dung:
  - **Retrieval (Task 4–7):** Hỗ trợ triển khai và rà soát chunking recursive (`CHUNK_SIZE=500`, overlap 50), index Chroma, dense search, BM25 và RRF `k=60`; đối chiếu output với contract `SearchResult` (sort score giảm dần, giữ `id`/`metadata`).
  - **Orchestration (Task 8–9):** Hỗ trợ luồng `retrieve()`: mỗi nhánh lấy 10 ứng viên trước khi RRF trả top 5; xét fallback bằng cosine dense gốc (threshold `0.573`) thay vì RRF score; PageIndex lỗi/thiếu key thì trả hybrid thay vì crash.
- Bằng chứng:
  - Báo cáo cá nhân: `reports/2A202602828-hieulm.md`
  - Mã nguồn: `src/task4_chunking_indexing.py` đến `src/task7_reranking.py`, `src/task8_pageindex_vectorless.py`, `src/task9_retrieval_pipeline.py`
  - Kết quả: hybrid đạt context recall 0.867 so với 0.800 của dense; điểm trung bình 0.792 so với 0.758.

### Thành viên 4 —

- Nội dung:
- Bằng chứng:
