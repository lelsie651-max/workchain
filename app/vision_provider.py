from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any

import httpx

from evidence_core.extraction_contract import build_extraction_result


DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_VISION_MODEL = "doubao-seed-2-0-lite-260215"
DEFAULT_ARK_TEXT_TIMEOUT_SECONDS = 20.0
DEFAULT_ARK_VISION_TIMEOUT_SECONDS = 90.0
ARK_TEXT_TIMEOUT_ENV = "ARK_TEXT_TIMEOUT_SECONDS"
ARK_VISION_TIMEOUT_ENV = "ARK_VISION_TIMEOUT_SECONDS"
ARK_PROVIDER_NAME = "doubao-ark"
ARK_RESPONSE_PATH = "/responses"
ARK_THINKING_MODE = "disabled"

VISION_SYSTEM_PROMPT = """你是 WorkChain 的 Visual Extraction 实验 provider。

你必须只输出一个 JSON 对象,不得输出 markdown 代码块或额外说明。

输出契约:
{
  "transcript": "图中可见文字,没有则为 null",
  "observations": [
    {
      "kind": "可观察 UI/视觉事实类型",
      "content": "只描述画面中直接可见的信息",
      "confidence": 0.0
    }
  ],
  "warnings": ["可选提示"]
}

硬性规则:
1. Observation 只描述画面中直接可观察到的 UI/视觉事实,不得推断心理、意图、hidden state 或不可见状态。
2. 不得根据后续对话或上下文去猜测画面里没有直接显示的信息。
3. 如果能看到 reaction 存在,但反应者身份在画面中不可见,只能记录 reaction 存在或身份未知,不得猜是谁点的。
4. transcript 只记录画面中实际能看到的文字; observations 不要把 transcript 改写成结论。
5. 如果某项不确定,宁可省略或写 warning,不要脑补。
6. 如果画面里直接显示了完整年月日,或完整日期+时间,额外增加一条 observation,其中 kind 必须是 "timestamp",content 必须保留画面中可见的完整日期文本。
7. 如果画面里只有 "19:21" 这类时分,不得补出年月日,也不要伪造 timestamp observation。
8. 不得使用上传时间、保存时间或任何画面外时间去推断聊天日期。
"""

VISION_USER_PROMPT = """请基于图片做提取,返回 transcript + observations + warnings 的 JSON。"""
ARK_TEXT_PREFLIGHT_PROMPT = "ping"


def get_ark_api_key() -> str:
    return os.getenv("ARK_API_KEY", "").strip()


def get_ark_base_url() -> str:
    return os.getenv("ARK_BASE_URL", "").strip() or DEFAULT_ARK_BASE_URL


def get_ark_vision_model() -> str:
    return os.getenv("ARK_VISION_MODEL", "").strip() or DEFAULT_ARK_VISION_MODEL


def _timeout_from_env(env_name: str, default: float) -> float:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def get_ark_text_timeout_seconds() -> float:
    return _timeout_from_env(ARK_TEXT_TIMEOUT_ENV, DEFAULT_ARK_TEXT_TIMEOUT_SECONDS)


def get_ark_vision_timeout_seconds() -> float:
    return _timeout_from_env(ARK_VISION_TIMEOUT_ENV, DEFAULT_ARK_VISION_TIMEOUT_SECONDS)


def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _redact_sensitive_text(value: Any, api_key: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if api_key:
        text = text.replace(api_key, "[redacted]")
    text = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer [redacted]", text, flags=re.IGNORECASE)
    text = re.sub(r"data:[^;]+;base64,[A-Za-z0-9+/=]+", "data:[redacted]", text)
    if len(text) > 300:
        text = f"{text[:300]}..."
    return text


def _coerce_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _empty_response_shape() -> dict[str, Any]:
    return {
        "top_level_keys": [],
        "output_type": None,
        "output_item_types": [],
        "content_types": [],
        "output_text_type": None,
    }


def _build_response_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _empty_response_shape()

    output_item_types: list[str] = []
    content_types: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            output_item_types.append(type(item).__name__)
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    content_types.append(type(content_item).__name__)
                    continue
                content_types.append(_coerce_text(content_item.get("type")) or "dict")

    return {
        "top_level_keys": sorted(str(key) for key in payload.keys()),
        "output_type": type(output).__name__ if output is not None else None,
        "output_item_types": output_item_types,
        "content_types": content_types,
        "output_text_type": type(payload.get("output_text")).__name__ if "output_text" in payload else None,
    }


def _request_id_from_response(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for key in ("x-request-id", "request-id", "x-tt-logid", "x-ark-request-id"):
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _http_error_details(payload: Any, api_key: str) -> tuple[str | None, str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None, None

    error = payload.get("error")
    if isinstance(error, dict):
        code = _coerce_text(error.get("code")) or _coerce_text(payload.get("code"))
        error_type = _coerce_text(error.get("type")) or _coerce_text(payload.get("type"))
        message = (
            _coerce_text(error.get("message"))
            or _coerce_text(error.get("msg"))
            or _coerce_text(payload.get("message"))
            or _coerce_text(payload.get("msg"))
        )
        return code, error_type, _redact_sensitive_text(message, api_key)

    code = _coerce_text(payload.get("code"))
    error_type = _coerce_text(payload.get("type"))
    message = _coerce_text(payload.get("message")) or _coerce_text(payload.get("msg"))
    return code, error_type, _redact_sensitive_text(message, api_key)


def _status_error_code(status_code: int | None) -> str | None:
    if status_code is None:
        return None
    if status_code == 400:
        return "bad_request"
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "server_error"
    return f"http_{status_code}"


def _diagnostic_result(
    *,
    success: bool,
    stage: str,
    latency_ms: int,
    model: str,
    base_url: str,
    timeout_seconds: float,
    status_code: int | None = None,
    error_code: str | None = None,
    error_type: str | None = None,
    safe_message: str | None = None,
    request_id: str | None = None,
    response_shape: dict[str, Any] | None = None,
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "stage": stage,
        "status_code": status_code,
        "error_code": error_code,
        "error_type": error_type,
        "safe_message": safe_message,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "model": model,
        "base_url": base_url,
        "timeout_seconds": timeout_seconds,
        "thinking_mode": ARK_THINKING_MODE,
        "response_shape": response_shape if response_shape is not None else _empty_response_shape(),
        "extraction": extraction,
    }


def _call_ark_responses(
    input_payload: list[dict[str, Any]],
    *,
    expect_contract: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    api_key = get_ark_api_key()
    base_url = get_ark_base_url().rstrip("/")
    model = get_ark_vision_model()
    if not api_key:
        return _diagnostic_result(
            success=False,
            stage="config",
            latency_ms=0,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            error_code="not_configured",
            error_type="config",
            safe_message="ARK_API_KEY 未设置",
        )

    start = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url}{ARK_RESPONSE_PATH}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "thinking": {"type": ARK_THINKING_MODE},
                "input": input_payload,
            },
            timeout=timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _diagnostic_result(
            success=False,
            stage="http",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            error_code="timeout",
            error_type=type(exc).__name__,
            safe_message=f"请求超过当前超时上限 {timeout_seconds:g} 秒",
        )
    except httpx.RequestError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _diagnostic_result(
            success=False,
            stage="http",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            error_code="network_error",
            error_type=type(exc).__name__,
            safe_message=f"无法连接 Ark /responses: {type(exc).__name__}",
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _diagnostic_result(
            success=False,
            stage="http",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            error_code="request_error",
            error_type=type(exc).__name__,
            safe_message=_redact_sensitive_text(str(exc), api_key) or type(exc).__name__,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    status_code = getattr(response, "status_code", None)
    request_id = _request_id_from_response(response)

    if status_code != 200:
        payload = None
        try:
            payload = response.json()
        except Exception:
            payload = None
        parsed_error_code, parsed_error_type, safe_message = _http_error_details(payload, api_key)
        return _diagnostic_result(
            success=False,
            stage="http",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            status_code=status_code,
            error_code=parsed_error_code or _status_error_code(status_code),
            error_type=parsed_error_type,
            safe_message=safe_message or f"Ark /responses 返回 HTTP {status_code}",
            request_id=request_id,
            response_shape=_build_response_shape(payload),
        )

    try:
        payload = response.json()
    except Exception as exc:
        return _diagnostic_result(
            success=False,
            stage="response_json",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            status_code=status_code,
            error_code="invalid_http_json",
            error_type=type(exc).__name__,
            safe_message="Ark 返回了 200,但响应体不是合法 JSON",
            request_id=request_id,
        )

    response_shape = _build_response_shape(payload)
    output_text = _extract_text_from_output(payload)
    if output_text is None:
        return _diagnostic_result(
            success=False,
            stage="output_text",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            status_code=status_code,
            error_code="missing_output_text",
            error_type="missing_output_text",
            safe_message="Ark 已正常返回,但响应中找不到可解析的 output text",
            request_id=request_id,
            response_shape=response_shape,
        )

    if not expect_contract:
        return _diagnostic_result(
            success=True,
            stage="output_text",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            status_code=status_code,
            request_id=request_id,
            response_shape=response_shape,
        )

    parsed = _parse_json_text(output_text)
    if parsed is None:
        return _diagnostic_result(
            success=False,
            stage="model_json",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            status_code=status_code,
            error_code="invalid_model_json",
            error_type="JSONDecodeError",
            safe_message="Ark 已正常返回,但 WorkChain 在 model_json 阶段无法解析结果",
            request_id=request_id,
            response_shape=response_shape,
        )

    extraction = _normalize_visual_result(parsed)
    if extraction is None or (extraction["transcript"] is None and not extraction["observations"]):
        return _diagnostic_result(
            success=False,
            stage="contract",
            latency_ms=latency_ms,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            status_code=status_code,
            error_code="invalid_contract",
            error_type="contract_validation_failed",
            safe_message="Ark 已正常返回,但 WorkChain 在 contract 阶段无法解析结果",
            request_id=request_id,
            response_shape=response_shape,
        )

    return _diagnostic_result(
        success=True,
        stage="contract",
        latency_ms=latency_ms,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        status_code=status_code,
        request_id=request_id,
        response_shape=response_shape,
        extraction=extraction,
    )


def _extract_text_from_output(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return None

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content_items = item.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                parts.append(content["text"])
                continue
            nested_text = content.get("text")
            if isinstance(nested_text, dict) and isinstance(nested_text.get("value"), str):
                parts.append(nested_text["value"])
    text = "\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip()).strip()
    return text or None


def _parse_json_text(raw_text: str | None) -> Any:
    if not isinstance(raw_text, str):
        return None
    stripped = raw_text.strip()
    candidates = [stripped]
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(stripped[start : end + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_visual_result(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return build_extraction_result(
        transcript=payload.get("transcript"),
        observations=payload.get("observations"),
        provider=ARK_PROVIDER_NAME,
        model=get_ark_vision_model(),
        warnings=payload.get("warnings"),
    )


def diagnose_text_preflight() -> dict[str, Any]:
    return _call_ark_responses(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": ARK_TEXT_PREFLIGHT_PROMPT,
                    }
                ],
            }
        ],
        expect_contract=False,
        timeout_seconds=get_ark_text_timeout_seconds(),
    )


def diagnose_visual_evidence(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    return _call_ark_responses(
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": VISION_SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": VISION_USER_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": _build_data_url(image_bytes, mime_type),
                    },
                ],
            },
        ],
        expect_contract=True,
        timeout_seconds=get_ark_vision_timeout_seconds(),
    )


def extract_visual_evidence(image_bytes: bytes, mime_type: str) -> dict[str, Any] | None:
    diagnostic = diagnose_visual_evidence(image_bytes, mime_type)
    if not diagnostic["success"]:
        return None
    return diagnostic["extraction"]
