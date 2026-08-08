from __future__ import annotations

import base64
from io import BytesIO

from docx import Document
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from app.extract import extract_text


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
)


def _build_pdf_bytes(text: str | None = None, *, image_only: bool = False) -> bytes:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setFont("STSong-Light", 14)
    if text:
        pdf.drawString(72, 720, text)
    if image_only:
        pdf.drawImage(ImageReader(BytesIO(PNG_BYTES)), 72, 650, width=80, height=80)
    pdf.save()
    return buffer.getvalue()


def _build_docx_bytes(*parts: str) -> bytes:
    document = Document()
    for part in parts:
        document.add_paragraph(part)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_extract_pdf_text_returns_chinese_text():
    text, note = extract_text(_build_pdf_bytes("渠道复盘数据"), "file", "demo.pdf")

    assert text is not None
    assert "渠道复盘数据" in text
    assert note == ""


def test_extract_docx_text_returns_paragraphs():
    text, note = extract_text(_build_docx_bytes("第一段", "第二段"), "file", "demo.docx")

    assert text == "第一段\n第二段"
    assert note == ""


def test_extract_txt_supports_utf8_and_gbk():
    utf8_text, utf8_note = extract_text("渠道复盘数据".encode("utf-8"), "file", "demo.txt")
    gbk_text, gbk_note = extract_text("渠道复盘数据".encode("gbk"), "file", "demo.txt")

    assert utf8_text == "渠道复盘数据"
    assert gbk_text == "渠道复盘数据"
    assert utf8_note == ""
    assert gbk_note == ""


def test_extract_image_only_pdf_returns_scan_note():
    text, note = extract_text(_build_pdf_bytes(image_only=True), "file", "scan.pdf")

    assert text is None
    assert note == "这份 PDF 看起来是扫描件,没有可提取的文字"


def test_extract_broken_pdf_returns_note_without_raising():
    text, note = extract_text(b"%PDF-broken", "file", "broken.pdf")

    assert text is None
    assert "这份 PDF 暂时无法读取" in note


def test_extract_image_without_ocr_config_returns_configured_note(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    text, note = extract_text(PNG_BYTES, "image", "demo.png")

    assert text is None
    assert note == "图片识别未配置(DASHSCOPE_API_KEY 未设置)"
