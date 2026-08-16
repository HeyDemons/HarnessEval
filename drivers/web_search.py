#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from ddgs import DDGS


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured DDGS search for benchmark agents.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--max-results", type=int, default=10)
    args = parser.parse_args()
    if args.query:
        query = args.query
        max_results = args.max_results
    else:
        payload = json.load(sys.stdin)
        query = str(payload.get("query", "")).strip()
        max_results = int(payload.get("max_results", args.max_results))
    if not query:
        raise SystemExit("query is required")
    results = list(DDGS().text(query, max_results=max_results))
    print(json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
