from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

from app.evidence_extractor import ARK_VISION_EXTRACTION_PROVIDER, extract_image_evidence
from app.vision_provider import get_ark_api_key


def _guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate visual extraction locally without touching DB.")
    parser.add_argument("image_path", help="Local image path")
    parser.add_argument(
        "--provider",
        default=ARK_VISION_EXTRACTION_PROVIDER,
        help="Extraction provider to use (default: ark_vision)",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists() or not image_path.is_file():
        print(f"图片不存在: {image_path}", file=sys.stderr)
        return 1

    if args.provider == ARK_VISION_EXTRACTION_PROVIDER and not get_ark_api_key():
        print(
            "ARK_API_KEY 未设置，无法运行实验视觉 provider。生产默认 OCR 行为不受影响。",
            file=sys.stderr,
        )
        return 1

    result = extract_image_evidence(
        image_path.read_bytes(),
        _guess_mime_type(image_path),
        provider=args.provider,
    )
    if result is None:
        print("视觉提取失败：provider 未配置或返回了无效响应。", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
