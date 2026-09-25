"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path

import json

from bs4 import BeautifulSoup
from markitdown import MarkItDown
from markdownify import markdownify

from .task1_collect_legal_docs import LEGAL_DOCUMENTS


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX sang Markdown và gắn metadata nguồn."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        result = converter.convert(str(path))
        content = result.text_content.strip()
        if not content:
            html_path = path.with_suffix(".html")
            if html_path.exists():
                soup = BeautifulSoup(
                    html_path.read_text(encoding="utf-8", errors="replace"),
                    "html.parser",
                )
                for tag in soup.select(
                    "script, style, noscript, nav, footer, header, aside, form, iframe"
                ):
                    tag.decompose()
                # The Government policy portal wraps the actual body in this
                # node.  Prefer it over <body>, otherwise menus/recommendations
                # become chunks and hurt retrieval quality.
                content_node = (
                    soup.select_one('[data-role="content"]')
                    or soup.select_one(".detail-content")
                    or soup.find("article")
                    or soup.find("main")
                    or soup.body
                )
                if content_node is not None:
                    for tag in content_node.select(
                        ".kbwscwl-relatedbox, [type='RelatedNewsBox'], "
                        ".box-related, .ads, figure"
                    ):
                        tag.decompose()
                    content = markdownify(
                        str(content_node), heading_style="ATX"
                    ).strip()
        if not content:
            raise ValueError(f"empty conversion result: {path}")

        source_url = LEGAL_DOCUMENTS.get(path.name)
        title = path.stem.replace("-", " ")
        header = (
            "---\n"
            f"title: {json.dumps(title, ensure_ascii=False)}\n"
            f"source: {json.dumps(path.name, ensure_ascii=False)}\n"
            f"url: {json.dumps(source_url, ensure_ascii=False)}\n"
            'doc_type: "legal"\n'
            "---\n\n"
        )
        output = output_dir / f"{path.stem}.md"
        output.write_text(header + content + "\n", encoding="utf-8")
        print(f"Saved: {output}")


def convert_news_articles() -> None:
    """Convert JSON bài hướng dẫn sang Markdown có front matter."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        required = ("url", "title", "date_crawled", "content_markdown")
        if any(not str(data.get(key, "")).strip() for key in required):
            raise ValueError(f"missing required article data: {path}")

        header = (
            "---\n"
            f"title: {json.dumps(data['title'], ensure_ascii=False)}\n"
            f"source: {json.dumps(path.name, ensure_ascii=False)}\n"
            f"url: {json.dumps(data['url'], ensure_ascii=False)}\n"
            'doc_type: "news"\n'
            f"date_crawled: {json.dumps(data['date_crawled'])}\n"
            "---\n\n"
        )
        output = output_dir / f"{path.stem}.md"
        output.write_text(
            header + data["content_markdown"].strip() + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {output}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
