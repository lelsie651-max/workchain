from __future__ import annotations

import base64
import json
from io import BytesIO
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

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


def test_prepare_image_for_ocr_with_metadata_reports_safe_preprocess_fields():
    prepared_bytes, mime_type, metadata = ocr._prepare_image_for_ocr_with_metadata(
        _build_png_bytes(4000, 3000),
        "image/png",
    )

    assert mime_type == "image/jpeg"
    assert metadata == {
        "original_mime": "image/png",
        "original_width": 4000,
        "original_height": 3000,
        "prepared_mime": "image/jpeg",
        "prepared_width": 2000,
        "prepared_height": 1500,
        "resized": True,
        "png_to_jpeg": True,
    }
    with Image.open(BytesIO(prepared_bytes)) as image:
        assert image.size == (2000, 1500)
        assert image.format == "JPEG"


def test_image_to_text_without_key_returns_configured_note(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    text, note = ocr.image_to_text(_build_png_bytes(120, 80), "image/png")

    assert text is None
    assert note == "图片识别未配置(DASHSCOPE_API_KEY 未设置)"


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


def test_image_to_text_success_emits_structured_log_metadata(monkeypatch, capsys):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            return _fake_response("第一行\n第二行")

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note, metadata = ocr.image_to_text_with_metadata(
        _build_png_bytes(4000, 3000),
        "image/png",
        evidence_id="ev_log_success",
    )

    assert text == "第一行\n第二行"
    assert note == ""
    assert metadata["original_width"] == 4000
    stderr = capsys.readouterr().err
    log_lines = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
    assert len(log_lines) == 2
    assert log_lines[0]["status"] == "started"
    assert log_lines[1] == {
        "event": "image_extraction",
        "evidence_id": "ev_log_success",
        "provider": "dashscope",
        "model": ocr.OCR_MODEL,
        "original_mime": "image/png",
        "original_width": 4000,
        "original_height": 3000,
        "prepared_mime": "image/jpeg",
        "prepared_width": 2000,
        "prepared_height": 1500,
        "resized": True,
        "png_to_jpeg": True,
        "status": "succeeded",
        "latency_ms": log_lines[1]["latency_ms"],
        "transcript_chars": 7,
        "warning_types": [],
        "error_type": None,
    }
    assert isinstance(log_lines[1]["latency_ms"], int)
    assert log_lines[1]["latency_ms"] >= 0
    assert "第一行" not in stderr


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
    assert note == "图片识别超时"


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "图片识别鉴权失败(状态码 401)"),
        (403, "图片识别鉴权失败(状态码 403)"),
        (404, "图片识别模型不可用(状态码 404,可能未开通该模型)"),
        (429, "图片识别调用过于频繁"),
    ],
)
def test_image_to_text_maps_status_errors_to_locatable_notes(monkeypatch, status_code, expected):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    class FakeStatusError(Exception):
        def __init__(self, code: int):
            super().__init__(f"HTTP {code}")
            self.status_code = code

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            raise FakeStatusError(status_code)

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note = ocr.image_to_text(_build_png_bytes(120, 80), "image/png")

    assert text is None
    assert note == expected


def test_image_to_text_connection_error_is_locatable_and_does_not_leak_key(monkeypatch):
    fake_key = "dashscope-secret-key"
    monkeypatch.setenv("DASHSCOPE_API_KEY", fake_key)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            raise httpx.ConnectError(f"boom {fake_key}")

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note = ocr.image_to_text(_build_png_bytes(120, 80), "image/png")

    assert text is None
    assert note == "无法连接图片识别服务:ConnectError"
    assert fake_key not in note


def test_image_to_text_failure_log_does_not_leak_api_key(monkeypatch, capsys):
    fake_key = "dashscope-super-secret-key"
    monkeypatch.setenv("DASHSCOPE_API_KEY", fake_key)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            raise httpx.ConnectError(f"boom {fake_key}")

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note, _ = ocr.image_to_text_with_metadata(
        _build_png_bytes(120, 80),
        "image/png",
        evidence_id="ev_log_failure",
    )

    assert text is None
    assert note == "无法连接图片识别服务:ConnectError"
    stderr = capsys.readouterr().err
    assert fake_key not in stderr
    log_lines = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
    assert len(log_lines) == 2
    assert log_lines[1]["status"] == "failed"
    assert log_lines[1]["warning_types"] == ["connection_error"]
    assert log_lines[1]["error_type"] == "ConnectError"


def test_image_to_text_other_api_error_contains_type_and_status(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    class FakeStatusError(Exception):
        def __init__(self):
            super().__init__("upstream exploded")
            self.status_code = 500

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            raise FakeStatusError()

    monkeypatch.setattr("app.ocr.OpenAI", FakeOpenAI)

    text, note = ocr.image_to_text(_build_png_bytes(120, 80), "image/png")

    assert text is None
    assert note == "图片识别失败:FakeStatusError(500)"
