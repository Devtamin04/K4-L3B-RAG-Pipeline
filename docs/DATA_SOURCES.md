# Data sources — Hộ kinh doanh Việt Nam

Ngày chốt corpus: **2026-09-23**.

## Phạm vi

Chatbot hỗ trợ tra cứu thông tin công khai về:

- Thành lập, thay đổi, tạm ngừng và chấm dứt hộ kinh doanh.
- Chính sách, hồ sơ và thủ tục thuế áp dụng trong năm 2026.
- Hóa đơn điện tử và hóa đơn điện tử khởi tạo từ máy tính tiền.

Chatbot không thay thế tư vấn của luật sư/cơ quan thuế và phải từ chối kết luận
nếu corpus không đủ căn cứ hoặc câu hỏi nằm ngoài phạm vi.

## Tài liệu pháp lý tải về

| File | Văn bản | Ban hành/hiệu lực | Vai trò |
| --- | --- | --- | --- |
| `168-2025-ND-CP-dang-ky-doanh-nghiep.pdf` | 168/2025/NĐ-CP | 30/06/2025; 01/07/2025 | Đăng ký doanh nghiệp, hộ kinh doanh |
| `68-2026-ND-CP-thue-ho-kinh-doanh.pdf` | 68/2026/NĐ-CP | 05/03/2026; 05/03/2026 | Chính sách và quản lý thuế |
| `141-2026-ND-CP-sua-doi-thue-ho-kinh-doanh.pdf` | 141/2026/NĐ-CP | 29/04/2026; áp dụng 01/01/2026 | Sửa NĐ 68, nâng ngưỡng lên 1 tỷ đồng |
| `254-2026-ND-CP-hoa-don-dien-tu.pdf` | 254/2026/NĐ-CP | 30/06/2026; 01/07/2026 | Hóa đơn/chứng từ điện tử |
| `24-2026-VBHN-TT-BTC-thu-tuc-quan-ly-thue.pdf` | 24/2026/VBHN-TT-BTC | 05/08/2026 | Hợp nhất thủ tục quản lý thuế |

URL trực tiếp được khai báo trong `src/task1_collect_legal_docs.py`. Trang tra cứu
chính thức:

- https://vanban.chinhphu.vn/?classid=1&docid=214334&pageid=27160
- https://vanban.chinhphu.vn/?classid=0&docid=217111&pageid=27160
- https://vanban.chinhphu.vn/?classid=1&docid=217960&pageid=27160
- https://vanban.chinhphu.vn/?docid=218689&pageid=27160&typegroupid=4
- https://vanban.chinhphu.vn/?docid=219107&pageid=27160

Các PDF ký số là bản scan nên pipeline đồng thời lưu bản toàn văn HTML từ chuyên
trang Xây dựng Chính sách của Cổng TTĐT Chính phủ. Task 3 ưu tiên text trong PDF;
nếu không có text layer thì dùng HTML chính thức làm fallback, còn metadata và
citation vẫn trỏ về PDF/văn bản gốc.

## Bài viết/page để crawl

| Nội dung | Cơ quan nguồn | Ghi chú |
| --- | --- | --- |
| Thành lập và hoạt động của hộ kinh doanh | Cổng TTĐT Chính phủ | Tổng hợp thủ tục hành chính theo NĐ 168/2025 |
| Chuyển đổi hộ kinh doanh sang doanh nghiệp | Cổng TTĐT Chính phủ | Hồ sơ, trình tự và xử lý đăng ký hộ cũ |
| Thủ tục khai thuế cho hộ/cá nhân kinh doanh | Cổng TTĐT Chính phủ | Có biểu mẫu và trình tự |
| Hóa đơn điện tử với doanh thu dưới 1 tỷ đồng | Cổng TTĐT Chính phủ | Phân biệt bắt buộc/tự nguyện |
| Hồ sơ, thủ tục quản lý thuế theo VBHN 24/2026 | Cổng TTĐT Chính phủ | Nguồn tổng hợp mới nhất |

URL đầy đủ được khai báo trong `src/task2_crawl_news.py`.

## Quy tắc phiên bản dữ liệu

- Metadata mỗi document/chunk cần thêm `document_number`, `issued_date`,
  `effective_date`, `status`, `amends` khi triển khai schema nội bộ.
- Citation phải hiển thị số hiệu văn bản và ngày hiệu lực, không chỉ tên file.
- `141/2026/NĐ-CP` được áp dụng khi nội dung của `68/2026/NĐ-CP` còn ghi ngưỡng
  500 triệu đồng.
- Không dùng bài dự thảo làm căn cứ trả lời nếu đã có văn bản ban hành.
- Những câu hỏi không nói rõ thời điểm được trả lời theo corpus hiện hành tại
  ngày chốt; câu hỏi về quá khứ phải đối chiếu hiệu lực tại thời điểm được hỏi.

## Golden questions gợi ý

1. Cá nhân có thể đăng ký hộ kinh doanh trực tuyến không?
2. Hồ sơ đăng ký hộ kinh doanh gồm những gì?
3. Có được đăng ký nhiều địa điểm kinh doanh không?
4. Khi tạm ngừng kinh doanh cần thông báo cho cơ quan nào?
5. Từ năm 2026 hộ kinh doanh còn áp dụng thuế khoán không?
6. Ngưỡng doanh thu không phải nộp thuế trong năm 2026 là bao nhiêu?
7. Vì sao Nghị định 68 ghi 500 triệu nhưng quy định hiện hành là 1 tỷ đồng?
8. Hộ có doanh thu trên 1 tỷ đồng dùng loại hóa đơn nào?
9. Hộ dưới 1 tỷ đồng có được tự nguyện dùng hóa đơn điện tử không?
10. Khi doanh thu lũy kế vượt 1 tỷ đồng, thời hạn đăng ký hóa đơn điện tử là bao lâu?
11. Mẫu kê khai dành cho hộ có doanh thu từ 1 tỷ đồng trở xuống là mẫu nào?
12. Hộ kinh doanh có nhiều cửa hàng ghi mã số thuế trên hóa đơn thế nào?
13. Kinh doanh trên nền tảng số kê khai doanh thu ra sao?
14. Văn bản nào quy định thủ tục quản lý thuế hiện hành?
15. Công chức có được đứng tên thành lập hộ kinh doanh không?

Các expected answer chỉ được viết sau khi PDF đã convert và được đối chiếu trực
tiếp với điều/khoản tương ứng.
