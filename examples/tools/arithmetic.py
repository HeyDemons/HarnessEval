#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> None:
    operation = sys.argv[1]
    arguments = json.load(sys.stdin)
    if operation == "lookup":
        values = {"alpha": 6, "beta": 7}
        key = str(arguments.get("key", ""))
        if key not in values:
            raise SystemExit(f"unknown key: {key}")
        result = {"key": key, "value": values[key]}
    elif operation == "multiply":
        result = {"product": arguments["a"] * arguments["b"]}
    else:
        raise SystemExit(f"unknown operation: {operation}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
