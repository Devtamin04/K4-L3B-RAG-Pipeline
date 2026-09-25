# RAG evaluation results

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

`dangvantuan/vietnamese-embedding` được giữ làm phương án A/B bổ sung nhưng chưa dùng làm baseline hay trộn vào collection BGE-M3.
