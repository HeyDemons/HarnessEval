from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _manifest() -> dict[str, Any]:
    value = json.loads(Path(_required("HARNESSEVAL_TOOL_MANIFEST")).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tools"), list):
        raise RuntimeError("HarnessEval manifest tools must be an array")
    return value


def _execute(endpoint: str, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    request = Request(
        f"{endpoint.rstrip('/')}/execute",
        data=json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=None) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload, response.status >= 400 or payload.get("ok") is False
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"HarnessEval tool bridge HTTP {exc.code}: {detail}"}, True
    except (URLError, OSError) as exc:
        return {"ok": False, "error": f"HarnessEval tool bridge transport failed: {exc}"}, True


def _tool_schema(entry: dict[str, Any]) -> dict[str, Any]:
    read_only = bool(entry.get("read_only"))
    return {
        "name": str(entry["name"]),
        "description": str(entry.get("description") or f"HarnessEval tool {entry['name']}"),
        "inputSchema": entry.get("parameters") or {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": not read_only,
            "idempotentHint": read_only,
            "openWorldHint": str(entry["name"]) == "web_search",
        },
    }


def _result(request: dict[str, Any], manifest: dict[str, Any], endpoint: str) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        requested = (request.get("params") or {}).get("protocolVersion")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": requested or "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "harnesseval", "version": "1"},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": [_tool_schema(entry) for entry in manifest["tools"]]},
        }
    if method == "tools/call":
        params = request.get("params") or {}
        name = str(params.get("name") or "")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        known = {str(entry.get("name")) for entry in manifest["tools"]}
        if name not in known:
            payload, is_error = {"ok": False, "error": f"Unknown HarnessEval tool: {name}"}, True
        else:
            payload, is_error = _execute(endpoint, name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
                ],
                "isError": is_error,
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    manifest = _manifest()
    endpoint = _required("HARNESSEVAL_TOOL_ENDPOINT")
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = _result(request, manifest, endpoint)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
            }
        if response is not None:
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
