from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

import httpx
from PIL import Image

from app import ocr


def _build_png_bytes(width: int, height: int, color: tuple[int, int, int] = (32, 96, 160)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_prepare_image_for_ocr_resizes_large_image_to_2000_max_side():
    prepared_bytes, mime_type = ocr._prepare_image_for_ocr(_build_png_bytes(4000, 3000))

    assert mime_type == "image/jpeg"
    with Image.open(BytesIO(prepared_bytes)) as image:
        assert image.size == (2000, 1500)
        assert image.format == "JPEG"


def test_image_to_text_without_key_returns_configured_note(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    text, note = ocr.image_to_text(_build_png_bytes(120, 80), "image/png")

    assert text is None
    assert note == "图片识别未配置"


def test_image_to_text_calls_dashscope_compatible_openai_and_preserves_high_detail(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["kwargs"] = kwargs
            return _fake_response("第一行\n第二行")

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note = ocr.image_to_text(_build_png_bytes(4000, 3000), "image/png")

    assert text == "第一行\n第二行"
    assert note == ""
    assert captured["api_key"] == "dashscope-test-key"
    assert captured["base_url"] == ocr.OCR_BASE_URL
    kwargs = captured["kwargs"]
    assert kwargs["model"] == ocr.OCR_MODEL
    assert kwargs["timeout"] == ocr.OCR_TIMEOUT_SECONDS
    content = kwargs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": ocr.OCR_PROMPT}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["detail"] == "high"
    data_url = content[1]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")

    prepared_bytes = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(BytesIO(prepared_bytes)) as image:
        assert image.size == (2000, 1500)
        assert image.format == "JPEG"


def test_image_to_text_short_result_is_treated_as_no_text(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            return _fake_response("abc")

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note = ocr.image_to_text(_build_png_bytes(120, 80), "image/png")

    assert text is None
    assert note == "这张图里没有识别到文字,原件已完整保存"


def test_image_to_text_catches_timeout_without_raising(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            raise httpx.TimeoutException("boom")

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note = ocr.image_to_text(_build_png_bytes(120, 80), "image/png")

    assert text is None
    assert note == "图片识别暂不可用(TimeoutException)"
