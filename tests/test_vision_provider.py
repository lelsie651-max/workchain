from __future__ import annotations

import json

import httpx

from app import vision_provider


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload=None,
        json_exc: Exception | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._payload = payload
        self._json_exc = json_exc
        self.headers = headers or {}

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


def test_ark_timeout_defaults_and_invalid_values_fallback(monkeypatch):
    monkeypatch.delenv("ARK_TEXT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("ARK_VISION_TIMEOUT_SECONDS", raising=False)

    assert vision_provider.get_ark_text_timeout_seconds() == 20.0
    assert vision_provider.get_ark_vision_timeout_seconds() == 90.0

    monkeypatch.setenv("ARK_TEXT_TIMEOUT_SECONDS", "bad")
    monkeypatch.setenv("ARK_VISION_TIMEOUT_SECONDS", "0")
    assert vision_provider.get_ark_text_timeout_seconds() == 20.0
    assert vision_provider.get_ark_vision_timeout_seconds() == 90.0


def test_ark_timeout_env_overrides(monkeypatch):
    monkeypatch.setenv("ARK_TEXT_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("ARK_VISION_TIMEOUT_SECONDS", "120")

    assert vision_provider.get_ark_text_timeout_seconds() == 12.5
    assert vision_provider.get_ark_vision_timeout_seconds() == 120.0


def test_diagnose_visual_evidence_reports_missing_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "config"
    assert result["status_code"] is None
    assert result["error_code"] == "not_configured"
    assert result["error_type"] == "config"
    assert result["safe_message"] == "ARK_API_KEY 未设置"
    assert result["latency_ms"] == 0
    assert result["timeout_seconds"] == 90.0
    assert result["thinking_mode"] == "disabled"
    assert result["extraction"] is None


def test_diagnose_visual_evidence_reports_401(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=401,
            payload={"error": {"code": "invalid_api_key", "type": "auth_error", "message": "bad key"}},
            headers={"x-request-id": "req-401"},
        ),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "http"
    assert result["status_code"] == 401
    assert result["error_code"] == "invalid_api_key"
    assert result["error_type"] == "auth_error"
    assert result["safe_message"] == "bad key"
    assert result["request_id"] == "req-401"
    assert result["timeout_seconds"] == 90.0
    assert result["thinking_mode"] == "disabled"


def test_diagnose_visual_evidence_reports_403(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=403,
            payload={"error": {"code": "access_denied", "type": "permission_error", "message": "not allowed"}},
        ),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "http"
    assert result["status_code"] == 403
    assert result["error_code"] == "access_denied"
    assert result["error_type"] == "permission_error"
    assert result["safe_message"] == "not allowed"
    assert result["timeout_seconds"] == 90.0
    assert result["thinking_mode"] == "disabled"


def test_diagnose_visual_evidence_reports_400_and_redacts_key_and_data_url(monkeypatch):
    fake_key = "ark-test-key"
    monkeypatch.setenv("ARK_API_KEY", fake_key)

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=400,
            payload={
                "error": {
                    "code": "invalid_parameter",
                    "type": "request_error",
                    "message": f"bad request {fake_key} data:image/png;base64,AAAABBBB",
                }
            },
        ),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "http"
    assert result["status_code"] == 400
    assert result["error_code"] == "invalid_parameter"
    assert result["error_type"] == "request_error"
    assert fake_key not in result["safe_message"]
    assert "base64" not in result["safe_message"]
    assert "[redacted]" in result["safe_message"]
    assert result["timeout_seconds"] == 90.0


def test_diagnose_visual_evidence_reports_timeout(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("boom")),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "http"
    assert result["error_code"] == "timeout"
    assert result["error_type"] == "TimeoutException"
    assert result["safe_message"] == "请求超过当前超时上限 90 秒"
    assert result["timeout_seconds"] == 90.0
    assert result["thinking_mode"] == "disabled"


def test_diagnose_visual_evidence_reports_200_non_json_response(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(status_code=200, json_exc=json.JSONDecodeError("bad", "x", 0)),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "response_json"
    assert result["status_code"] == 200
    assert result["error_code"] == "invalid_http_json"
    assert result["error_type"] == "JSONDecodeError"
    assert result["timeout_seconds"] == 90.0


def test_diagnose_visual_evidence_reports_missing_output_text(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=200,
            payload={"id": "resp_1", "output": [{"content": [{"type": "refusal"}]}]},
        ),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "output_text"
    assert result["error_code"] == "missing_output_text"
    assert result["response_shape"]["top_level_keys"] == ["id", "output"]
    assert result["response_shape"]["output_type"] == "list"
    assert "dict" in result["response_shape"]["output_item_types"]
    assert "refusal" in result["response_shape"]["content_types"]
    assert result["timeout_seconds"] == 90.0


def test_diagnose_visual_evidence_reports_invalid_model_json(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=200,
            payload={"output_text": "not valid json"},
        ),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "model_json"
    assert result["error_code"] == "invalid_model_json"
    assert result["error_type"] == "JSONDecodeError"
    assert "model_json" in result["safe_message"]
    assert result["timeout_seconds"] == 90.0


def test_diagnose_visual_evidence_reports_invalid_contract(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=200,
            payload={"output_text": json.dumps({"transcript": None, "observations": [], "warnings": ["无"]}, ensure_ascii=False)},
        ),
    )

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is False
    assert result["stage"] == "contract"
    assert result["error_code"] == "invalid_contract"
    assert result["error_type"] == "contract_validation_failed"
    assert "contract" in result["safe_message"]
    assert result["timeout_seconds"] == 90.0


def test_diagnose_visual_evidence_successfully_normalizes_result(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    monkeypatch.delenv("ARK_VISION_MODEL", raising=False)
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    captured = {}

    def fake_post(*args, **kwargs):
        captured["url"] = args[0]
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        captured["timeout"] = kwargs["timeout"]
        return FakeResponse(
            status_code=200,
            payload={
                "output_text": json.dumps(
                    {
                        "transcript": "请周五前补齐渠道复盘数据",
                        "observations": [
                            {
                                "kind": "reaction",
                                "content": "有人对该消息显示👍反应",
                                "confidence": 0.74,
                                "actor_name": "不应保留",
                            }
                        ],
                        "warnings": ["画面上半部分存在遮挡"],
                    },
                    ensure_ascii=False,
                )
            },
            headers={"x-request-id": "req-ok"},
        )

    monkeypatch.setattr(vision_provider.httpx, "post", fake_post)

    result = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert result["success"] is True
    assert result["stage"] == "contract"
    assert result["status_code"] == 200
    assert result["request_id"] == "req-ok"
    assert result["timeout_seconds"] == 90.0
    assert result["thinking_mode"] == "disabled"
    assert result["response_shape"]["top_level_keys"] == ["output_text"]
    assert result["extraction"] == {
        "transcript": "请周五前补齐渠道复盘数据",
        "observations": [
            {
                "kind": "reaction",
                "content": "有人对该消息显示👍反应",
                "confidence": 0.74,
            }
        ],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": ["画面上半部分存在遮挡"],
    }
    assert captured["url"] == "https://ark.cn-beijing.volces.com/api/v3/responses"
    assert captured["headers"]["Authorization"] == "Bearer ark-test-key"
    assert captured["timeout"] == 90.0
    assert captured["json"]["model"] == "doubao-seed-2-0-lite-260215"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["input"][0]["role"] == "system"
    assert "reaction 存在" in captured["json"]["input"][0]["content"][0]["text"]
    assert captured["json"]["input"][1]["content"][0]["type"] == "input_text"
    assert captured["json"]["input"][1]["content"][1]["type"] == "input_image"
    assert captured["json"]["input"][1]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_diagnose_text_preflight_uses_text_only_input(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["json"] = kwargs["json"]
        captured["timeout"] = kwargs["timeout"]
        return FakeResponse(status_code=200, payload={"output_text": "pong"})

    monkeypatch.setattr(vision_provider.httpx, "post", fake_post)

    result = vision_provider.diagnose_text_preflight()

    assert result["success"] is True
    assert result["stage"] == "output_text"
    assert result["status_code"] == 200
    assert result["timeout_seconds"] == 20.0
    assert result["thinking_mode"] == "disabled"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "ping",
                }
            ],
        }
    ]
    assert captured["timeout"] == 20.0


def test_text_and_vision_use_different_timeouts(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    captured_timeouts = []

    def fake_post(*args, **kwargs):
        captured_timeouts.append(kwargs["timeout"])
        return FakeResponse(status_code=200, payload={"output_text": json.dumps({"transcript": "文字", "observations": [], "warnings": []}, ensure_ascii=False)})

    monkeypatch.setattr(vision_provider.httpx, "post", fake_post)

    preflight = vision_provider.diagnose_text_preflight()
    vision = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png")

    assert preflight["timeout_seconds"] == 20.0
    assert vision["timeout_seconds"] == 90.0
    assert captured_timeouts == [20.0, 90.0]


def test_extract_visual_evidence_remains_production_compatible(monkeypatch):
    monkeypatch.setattr(
        vision_provider,
        "diagnose_visual_evidence",
        lambda image_bytes, mime_type: {
            "success": True,
            "stage": "contract",
            "status_code": 200,
            "error_code": None,
            "error_type": None,
            "safe_message": None,
            "request_id": None,
            "latency_ms": 1,
            "model": "demo-model",
            "base_url": "https://ark.example.com",
            "response_shape": {"top_level_keys": [], "output_type": None, "output_item_types": [], "content_types": [], "output_text_type": None},
            "extraction": {
                "transcript": "文字",
                "observations": [],
                "provider": "doubao-ark",
                "model": "demo-model",
                "warnings": [],
            },
        },
    )

    success = vision_provider.extract_visual_evidence(b"fake-image", "image/png")

    monkeypatch.setattr(
        vision_provider,
        "diagnose_visual_evidence",
        lambda image_bytes, mime_type: {
            "success": False,
            "stage": "http",
            "status_code": 401,
            "error_code": "unauthorized",
            "error_type": "auth_error",
            "safe_message": "bad key",
            "request_id": None,
            "latency_ms": 1,
            "model": "demo-model",
            "base_url": "https://ark.example.com",
            "response_shape": {"top_level_keys": [], "output_type": None, "output_item_types": [], "content_types": [], "output_text_type": None},
            "extraction": None,
        },
    )

    failure = vision_provider.extract_visual_evidence(b"fake-image", "image/png")

    assert success == {
        "transcript": "文字",
        "observations": [],
        "provider": "doubao-ark",
        "model": "demo-model",
        "warnings": [],
    }
    assert failure is None
