from __future__ import annotations

import json
from pathlib import Path

from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker


def main() -> None:
    root = Path("/opt/gorilla/berkeley-function-call-leaderboard")
    data_files = sorted((root / "bfcl_eval" / "data").rglob("*.json"))
    checker = agentic_checker("The final answer is Paris, France.", ["Paris"])
    categories = TEST_COLLECTION_MAPPING["all_scoring"]
    complete = checker["valid"] and bool(data_files) and bool(categories)
    payload = {
        "source_root": str(root),
        "data_json_files": len(data_files),
        "official_scoring_categories": categories,
        "agentic_oracle": checker,
        "excluded_profiles": [
            "AST and executable scorer runtime dependencies",
            "vector-memory (sentence-transformers/torch)",
        ],
        "scores": {"official_core_scorer_integrity": float(complete)},
        "oracle_smoke": True,
    }
    output = Path("/job/payload.json")
    pending = output.with_name(".payload.json.tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(output)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
