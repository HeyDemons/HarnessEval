from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _response(payload: dict) -> dict:
    messages = payload.get("messages") or []
    system = "\n".join(
        str(item.get("content") or "")
        for item in messages
        if item.get("role") == "system"
    )
    contents = "\n".join(str(item.get("content") or "") for item in messages)
    if "<scenario>" in system:
        if "created successfully" in contents.lower():
            message = {"role": "assistant", "content": "###STOP###"}
        elif any(item.get("role") == "tool" for item in messages):
            message = {
                "role": "assistant",
                "content": "Please create a task called Important Meeting for user_1.",
            }
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "user-check-1",
                        "type": "function",
                        "function": {"name": "check_notifications", "arguments": "{}"},
                    }
                ],
            }
    elif "Observation:" in contents:
        message = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "tool": "send_message_to_user",
                    "arguments": {"content": "The task was created successfully."},
                }
            ),
        }
    else:
        message = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "tool": "create_task",
                    "arguments": {"user_id": "user_1", "title": "Important Meeting"},
                }
            ),
        }
    return {
        "id": "mock-completion",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        body = json.dumps(_response(payload)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock tau OpenAI server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
