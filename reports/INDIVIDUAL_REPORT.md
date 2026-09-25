# Individual contribution report

Mỗi thành viên copy template này thành:

```text
reports/<student-id>-<short-name>.md
```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---

## Thông tin

- Họ và tên: Nguyễn Quang Tuấn
- Mã học viên: 2A202602470
- Nhóm: K4-L3B
- Repository/branch: `K4-L3B-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Data pipeline | Thu thập 5 văn bản, 5 page; fallback HTML cho PDF scan; chuẩn hóa Markdown | `src/task1_*` đến `src/task3_*`, `data/` | Done |
| Retrieval | Chunk ổn định, BGE-M3 CPU, Chroma, dense, BM25 và RRF | `src/task4_*` đến `src/task7_*` | Done |
| Orchestration | Dense threshold, PageIndex degraded mode, safe fallback | `src/task8_*`, `src/task9_*` | Done |
| Generation/UI | Ollama Cloud, citation mapping, safe refusal, Streamlit sources | `src/task10_generation.py`, `app.py` | Done |
| Evaluation | 15 golden cases, dense-vs-hybrid, 4 Ragas metrics và calibration | `group_project/evaluation/` | Done |


## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Giữ `BAAI/bge-m3` làm baseline và chỉ khai báo `dangvantuan/vietnamese-embedding` như model thử nghiệm.
   **Lý do/evidence:** Skeleton gợi ý BGE-M3; index thực tế có 1.091 vector 1024 chiều và chạy hoàn toàn trên CPU.
   **Trade-off:** Model lớn, lần index đầu mất khoảng 10 phút; đổi lại chất lượng tiếng Việt và đa ngôn ngữ tốt hơn.

2. **Quyết định:** Kết hợp dense + BM25 bằng RRF và dùng dense cosine gốc để quyết định fallback.
   **Lý do/evidence:** Hybrid đạt trung bình 0.792, cao hơn dense 0.758 trên 15 golden cases.
   **Trade-off:** Retrieval tăng từ khoảng 0.796 giây lên 2.005 giây mỗi query.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest -q`; 15 câu trong `golden_dataset.json`; 5 câu out-of-domain để calibration.
- Kết quả trước/sau nếu có: 20/20 test pass; hybrid +0.035 điểm trung bình so với dense; threshold chọn 0.573.
- Lỗi đã phát hiện và cách xử lý: PDF scan không có text layer nên bổ sung HTML chính thức; citation sai thứ tự sau reorder nên gắn nhãn trước khi reorder; re-index được chuyển sang incremental.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: PageIndex cloud chưa được gọi thật vì không có `PAGEINDEX_API_KEY`; pipeline hiện kiểm thử degraded mode.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: tăng diversity của top-k và ghép chunk lân cận cho các câu hỏi có bằng chứng nằm sát biên chunk.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-23
- Tên thành viên: Nguyễn Quang Tuấn
