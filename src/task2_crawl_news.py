"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # Nguồn hướng dẫn chính thức; ưu tiên bài đã phản ánh sửa đổi năm 2026.
    "https://xaydungchinhsach.chinhphu.vn/thu-tuc-hanh-chinh-trong-linh-vuc-thanh-lap-va-hoat-dong-cua-ho-kinh-doanh-119250708142514155.htm",
    "https://xaydungchinhsach.chinhphu.vn/huong-dan-chuyen-doi-ho-kinh-doanh-sang-doanh-nghiep-119250818113746859.htm",
    "https://xaydungchinhsach.chinhphu.vn/huong-dan-thu-tuc-khai-thue-doi-voi-ho-kinh-doanh-ca-nhan-kinh-doanh-119260530072620314.htm",
    "https://xaydungchinhsach.chinhphu.vn/huong-dan-su-dung-hoa-don-dien-tu-voi-ho-kinh-doanh-co-doanh-thu-duoi-1-ty-dong-nam-119260727150121544.htm",
    "https://xaydungchinhsach.chinhphu.vn/quy-dinh-ve-ho-so-thu-tuc-quan-ly-thue-voi-ho-kinh-doanh-ca-nhan-kinh-doanh-119260812153118737.htm",
]


async def crawl_article(url: str) -> dict:
    """Tải một trang công khai và giữ phần nội dung bài viết ở dạng Markdown."""

    def fetch() -> dict:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Lab/1.0)"},
            timeout=(15, 60),
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.select(
            "script, style, noscript, nav, footer, header, aside, form, iframe"
        ):
            tag.decompose()

        title_node = soup.find("h1") or soup.find("title")
        title = title_node.get_text(" ", strip=True) if title_node else url
        content_node = (
            soup.select_one('[data-role="content"]')
            or soup.select_one(".article-content, .detail-content, .content-detail")
            or soup.find("article")
            or soup.find("main")
            or soup.body
        )
        if content_node is None:
            raise ValueError("page does not contain readable content")

        for tag in content_node.select(
            ".kbwscwl-relatedbox, [type='RelatedNewsBox'], .box-related, .ads, figure"
        ):
            tag.decompose()
        content = markdownify(str(content_node), heading_style="ATX").strip()
        if len(content) < 200:
            raise ValueError("extracted article is unexpectedly short")

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
        }

    return await asyncio.to_thread(fetch)


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
