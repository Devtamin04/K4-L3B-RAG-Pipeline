"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Nguồn chính thức, hiện hành tại thời điểm chốt corpus (2026-09-23).
# Giữ cả Nghị định 68 và văn bản sửa đổi 141 để đánh giá xử lý phiên bản.
LEGAL_DOCUMENTS = {
    "168-2025-ND-CP-dang-ky-doanh-nghiep.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/7/168nd.signed.pdf"
    ),
    "68-2026-ND-CP-thue-ho-kinh-doanh.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/68-ndcp.signed.pdf"
    ),
    "141-2026-ND-CP-sua-doi-thue-ho-kinh-doanh.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/4/141-ndcp.signed.pdf"
    ),
    "254-2026-ND-CP-hoa-don-dien-tu.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/7/254-ndcp.signed.pdf"
    ),
    "24-2026-VBHN-TT-BTC-thu-tuc-quan-ly-thue.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/8/24-vbhn-btc.signed.pdf"
    ),
}

# Các PDF ký số ở trên là bản scan. Lưu thêm bản toàn văn HTML từ Cổng TTĐT
# Chính phủ để chuẩn hóa text; citation vẫn trỏ về văn bản pháp lý gốc.
LEGAL_TEXT_SOURCES = {
    "168-2025-ND-CP-dang-ky-doanh-nghiep.pdf": (
        "https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-168-2025-nd-cp-ve-dang-ky-doanh-nghiep-119250702175708554.htm"
    ),
    "68-2026-ND-CP-thue-ho-kinh-doanh.pdf": (
        "https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-68-2026-nd-cp-quy-dinh-ve-chinh-sach-thue-quan-ly-thue-voi-ho-kinh-doanh-119260306102906789.htm"
    ),
    "141-2026-ND-CP-sua-doi-thue-ho-kinh-doanh.pdf": (
        "https://xaydungchinhsach.chinhphu.vn/nghi-dinh-so-141-2026-nd-cp-quy-dinh-moi-ve-chinh-sach-thue-doi-voi-ho-kinh-doanh-doanh-nghiep-119260430091642895.htm"
    ),
    "254-2026-ND-CP-hoa-don-dien-tu.pdf": (
        "https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-so-254-2026-nd-cp-ve-hoa-don-dien-tu-chung-tu-dien-tu-119260713164251972.htm"
    ),
    "24-2026-VBHN-TT-BTC-thu-tuc-quan-ly-thue.pdf": (
        "https://xaydungchinhsach.chinhphu.vn/van-ban-hop-nhat-24-2026-vbhn-tt-btc-ve-ho-so-thu-tuc-quan-ly-thue-119260814155824299.htm"
    ),
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    setup_directory()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Lab/1.0)"}

    for filename, url in LEGAL_DOCUMENTS.items():
        output = DATA_DIR / filename
        if output.exists() and output.stat().st_size > 1024:
            print(f"Exists: {output}")
            continue

        temporary = output.with_suffix(output.suffix + ".part")
        try:
            with requests.get(
                url,
                headers=headers,
                timeout=(15, 120),
                stream=True,
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as target:
                    for block in response.iter_content(chunk_size=64 * 1024):
                        if block:
                            target.write(block)

            if temporary.stat().st_size <= 1024:
                raise ValueError("downloaded file is unexpectedly small")
            if output.suffix.lower() == ".pdf":
                with temporary.open("rb") as source:
                    if source.read(5) != b"%PDF-":
                        raise ValueError("response is not a PDF")

            temporary.replace(output)
            print(f"Saved: {output}")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    for filename, url in LEGAL_TEXT_SOURCES.items():
        output = DATA_DIR / f"{Path(filename).stem}.html"
        if output.exists() and output.stat().st_size > 1024:
            print(f"Exists: {output}")
            continue
        response = requests.get(url, headers=headers, timeout=(15, 60))
        response.raise_for_status()
        if len(response.content) <= 1024:
            raise ValueError(f"HTML source is unexpectedly small: {url}")
        output.write_bytes(response.content)
        print(f"Saved: {output}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
