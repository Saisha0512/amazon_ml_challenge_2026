"""Validate generated outputs against all required test TSV records."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.config import FINAL_OUTPUT_DIR, TEST_FILES
from business_entity_resolution.submission import validate_submission


def main():
    result = validate_submission(
        FINAL_OUTPUT_DIR / "matching_results.tsv",
        FINAL_OUTPUT_DIR / "candidate_pairs.tsv",
        TEST_FILES["source1"], TEST_FILES["source2"], TEST_FILES["source3"],
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
