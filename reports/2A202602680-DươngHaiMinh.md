# Individual contribution report

Mỗi thành viên copy template này thành:

```text
reports/<student-id>-<short-name>.md
```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---

## Thông tin

- Họ và tên: Dương Hải Minh
- Mã học viên: 2A202602680
- Nhóm: K4-L3B
- Repository/branch: `K4-L3B-RAG-Pipeline` / `minhdh`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Evaluation | 15 golden cases, dense-vs-hybrid, 4 Ragas metrics và calibration | group_project/evaluation/ | Done |

Chỉ kê khai công việc có thể đối chiếu bằng file, commit, pull request, test hoặc kết quả evaluation.

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Dung hợp kết quả tìm kiếm (Dense + BM25) bằng thuật toán RRF, nhưng vẫn dùng điểm cosine nguyên bản của Dense để xét điều kiện fallback[cite: 2].
   - **Lý do/Evidence:** Khi thử nghiệm trên tập 15 golden cases, chiến lược hybrid mang lại điểm trung bình 0.792, cải thiện rõ rệt so với mức 0.758 nếu chỉ dùng dense search thuần túy[cite: 2].
   - **Trade-off:** Đánh đổi lớn nhất là độ trễ, thời gian truy xuất (retrieval) bị đẩy từ mức 0.796 giây lên khoảng 2.005 giây cho mỗi truy vấn[cite: 2].

## Kiểm thử và kết quả

- **Kịch bản test:** Tôi đã chạy kiểm thử toàn bộ hệ thống bằng lệnh `pytest -q`, đồng thời dùng 15 câu hỏi trong `golden_dataset.json` cùng 5 câu ngoài miền (out-of-domain) để dò ngưỡng (calibration)[cite: 2].
- **Kết quả:** Hệ thống vượt qua đủ 20/20 test cases. Việc dùng hybrid giúp điểm trung bình tăng thêm 0.035 điểm; và ngưỡng fallback tối ưu được chốt ở mức 0.573[cite: 2].
- **Fix bugs:** 
  - *Lỗi dữ liệu:* Phát hiện các file PDF scan bị thiếu text layer, tôi đã chuyển sang parse văn bản từ nguồn HTML chính thức[cite: 2].
  - *Lỗi hiển thị:* Trích dẫn (citation) bị lệch thứ tự sau khi hàm reorder chạy. Tôi đã fix bằng cách đánh nhãn tĩnh (labeling) trước khi sắp xếp lại[cite: 2]. Cải tiến thêm việc re-index sang dạng incremental (chỉ cập nhật phần mới)[cite: 2].

## Điều còn hạn chế

- **Hạn chế trong công việc của tôi:** Pipeline hiện tại chưa gọi được API đám mây thực tế của PageIndex do thiếu `PAGEINDEX_API_KEY`, nên chỉ đang chạy giả lập ở trạng thái suy giảm (degraded mode)[cite: 2].
- **Định hướng cải tiến:** Nếu có thêm thời gian dự án, tôi sẽ cải thiện độ đa dạng (diversity) của top-k kết quả trả về, đồng thời bổ sung logic nối các chunk lân cận nhau để cung cấp ngữ cảnh đầy đủ hơn cho các câu hỏi có đáp án nằm vắt ngang giữa hai chunk[cite: 2].

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Dương Hải Minh
