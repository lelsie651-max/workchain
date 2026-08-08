from __future__ import annotations

from io import BytesIO

from docx import Document
from pypdf import PdfReader


MAX_EXTRACTED_TEXT_LENGTH = 50_000


def _limit_text(text: str) -> tuple[str, str]:
    if len(text) <= MAX_EXTRACTED_TEXT_LENGTH:
        return text, ""
    return text[:MAX_EXTRACTED_TEXT_LENGTH], f"已截取前 {MAX_EXTRACTED_TEXT_LENGTH} 个字符"


def _extract_pdf_text(payload: bytes) -> tuple[str | None, str]:
    try:
        reader = PdfReader(BytesIO(payload))
        page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        return None, f"这份 PDF 暂时无法读取({type(exc).__name__})"

    text = "\n\n".join(item for item in page_texts if item).strip()
    if not text:
        return None, "这份 PDF 看起来是扫描件,没有可提取的文字"
    return _limit_text(text)


def _extract_docx_text(payload: bytes) -> tuple[str | None, str]:
    try:
        document = Document(BytesIO(payload))
    except Exception as exc:
        return None, f"这份 docx 暂时无法读取({type(exc).__name__})"

    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n".join(parts).strip()
    if not text:
        return None, "这份 docx 里没有可提取的文字"
    return _limit_text(text)


def _extract_txt_text(payload: bytes) -> tuple[str | None, str]:
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None, "这份 txt 暂时无法按常见编码读取"

    stripped = text.strip()
    if not stripped:
        return None, "这份 txt 里没有可提取的文字"
    return _limit_text(stripped)


def extract_text(payload: bytes, media_type: str, filename: str) -> tuple[str | None, str]:
    try:
        lower_name = (filename or "").lower()
        if media_type == "image":
            return None, "图片内容识别暂未开放"
        if lower_name.endswith(".pdf"):
            return _extract_pdf_text(payload)
        if lower_name.endswith(".docx"):
            return _extract_docx_text(payload)
        if lower_name.endswith(".txt"):
            return _extract_txt_text(payload)
        return None, "暂不支持提取这类文件中的文字"
    except Exception as exc:  # pragma: no cover
        return None, f"提取失败({type(exc).__name__})"
