from __future__ import annotations

import base64
import json
from typing import Any, Mapping


class ToolImage(dict[str, Any]):
    """Binary image content kept out of text transcripts until API serialization."""

    def __init__(self, mime_type: str, data: bytes, detail: str = "auto"):
        super().__init__(type="image", mime_type=mime_type, bytes=len(data))
        self.mime_type = mime_type
        self.data = data
        self.detail = detail

    @property
    def data_uri(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"

    def metadata(self) -> dict[str, Any]:
        return dict(self)


WIRE_IMAGE_MARKER = "_harnesseval_image"


def wire_tool_result(value: Any) -> Any:
    """Encode images for the product HTTP hop while preserving metadata-only JSON."""
    if isinstance(value, ToolImage):
        return {
            **value.metadata(),
            WIRE_IMAGE_MARKER: {
                "data": base64.b64encode(value.data).decode("ascii"),
                "mime_type": value.mime_type,
                "detail": value.detail,
            },
        }
    if isinstance(value, Mapping):
        return {str(key): wire_tool_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [wire_tool_result(item) for item in value]
    if isinstance(value, tuple):
        return [wire_tool_result(item) for item in value]
    return value


def json_safe(value: Any) -> Any:
    """Make tool values safe for JSON traces/prompts without embedding image bytes."""
    if isinstance(value, ToolImage):
        return value.metadata()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def tool_result_content(value: Any, *, prefix: str = "Observation: ") -> str | list[dict[str, Any]]:
    """Render a tool result as text plus native image content when present."""
    images: list[ToolImage] = []

    def collect(item: Any) -> None:
        if isinstance(item, ToolImage):
            images.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    text = prefix + json.dumps(json_safe(value), ensure_ascii=False)
    if not images:
        return text
    return [
        {"type": "text", "text": text},
        *({"type": "image", "image": image} for image in images),
    ]
