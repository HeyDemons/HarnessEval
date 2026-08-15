#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ddgs import DDGS


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured DDGS search for benchmark agents.")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=10)
    args = parser.parse_args()
    results = list(DDGS().text(args.query, max_results=args.max_results))
    print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
