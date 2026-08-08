from __future__ import annotations

import base64
import json
import os
from typing import Any

import httpx

from evidence_core.extraction_contract import build_extraction_result


DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_VISION_MODEL = "doubao-seed-2-0-lite-260215"
ARK_TIMEOUT_SECONDS = 30.0
ARK_PROVIDER_NAME = "doubao-ark"
ARK_RESPONSE_PATH = "/responses"

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
"""

VISION_USER_PROMPT = """请基于图片做提取,返回 transcript + observations + warnings 的 JSON。"""


def get_ark_api_key() -> str:
    return os.getenv("ARK_API_KEY", "").strip()


def get_ark_base_url() -> str:
    return os.getenv("ARK_BASE_URL", "").strip() or DEFAULT_ARK_BASE_URL


def get_ark_vision_model() -> str:
    return os.getenv("ARK_VISION_MODEL", "").strip() or DEFAULT_ARK_VISION_MODEL


def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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


def extract_visual_evidence(image_bytes: bytes, mime_type: str) -> dict[str, Any] | None:
    api_key = get_ark_api_key()
    if not api_key:
        return None

    try:
        response = httpx.post(
            f"{get_ark_base_url().rstrip('/')}{ARK_RESPONSE_PATH}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": get_ark_vision_model(),
                "input": [
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
            },
            timeout=ARK_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return None

        payload = response.json()
        parsed = _parse_json_text(_extract_text_from_output(payload))
        return _normalize_visual_result(parsed)
    except Exception:
        return None
