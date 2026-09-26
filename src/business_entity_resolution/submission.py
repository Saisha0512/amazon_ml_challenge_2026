"""Submission writing helpers and an independent stdlib format validator."""

from __future__ import annotations

import csv
from pathlib import Path
from itertools import zip_longest


def validate_submission(matching_path: str | Path, candidates_path: str | Path,
                        test_source1: str | Path, test_source2: str | Path,
                        test_source3: str | Path) -> dict:
    """Validate headers, full S1 coverage, references, duplicates, and subset rule."""
    for path in (matching_path, candidates_path, test_source1, test_source2, test_source3):
        if not Path(path).is_file():
            return {"passed": False, "test_source1_entities": 0, "matching_rows": 0,
                    "candidate_rows": 0, "errors": [f"Missing required file: {path}"], "error_count": 1}
    # A set is more memory efficient than retaining every output row and allows each
    # candidate ID to be checked against the supplied test targets.
    target_ids = set()
    for path in (test_source2, test_source3):
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            target_ids.update(row["entity_id"].strip() for row in reader)
    errors = []
    counts = {"test_source1_entities": 0, "matching_rows": 0, "candidate_rows": 0}
    with (Path(test_source1).open("r", encoding="utf-8-sig", newline="") as s1_stream,
          Path(matching_path).open("r", encoding="utf-8-sig", newline="") as match_stream,
          Path(candidates_path).open("r", encoding="utf-8-sig", newline="") as candidate_stream):
        s1_reader = csv.DictReader(s1_stream, delimiter="\t")
        match_reader = csv.DictReader(match_stream, delimiter="\t")
        candidate_reader = csv.DictReader(candidate_stream, delimiter="\t")
        if match_reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"matching: unexpected header {match_reader.fieldnames}")
        if candidate_reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"candidate: unexpected header {candidate_reader.fieldnames}")
        sentinel = object()
        for line, (s1_row, match_row, candidate_row) in enumerate(zip_longest(
                s1_reader, match_reader, candidate_reader, fillvalue=sentinel), start=2):
            if s1_row is sentinel:
                errors.append("Output has more rows than test Source 1")
                break
            if match_row is sentinel or candidate_row is sentinel:
                errors.append("Output has fewer rows than test Source 1")
                break
            expected_s1 = s1_row["entity_id"].strip()
            match_s1 = match_row.get("source1_entity_id", "").strip()
            candidate_s1 = candidate_row.get("source1_entity_id", "").strip()
            if match_s1 != expected_s1 or candidate_s1 != expected_s1:
                errors.append(f"line {line}: output Source 1 IDs do not match test input order")
            matched = [x.strip() for x in match_row.get("matched_entity_ids", "").split(",") if x.strip()]
            candidates = [x.strip() for x in candidate_row.get("candidate_entity_ids", "").split(",") if x.strip()]
            if len(matched) != len(set(matched)):
                errors.append(f"matching:{line}: duplicate matched IDs")
            if len(candidates) != len(set(candidates)):
                errors.append(f"candidate:{line}: duplicate candidate IDs")
            candidate_set = set(candidates)
            for entity_id in candidates:
                if not entity_id.startswith(("S2-", "S3-")) or entity_id not in target_ids:
                    errors.append(f"candidate:{line}: invalid or unknown candidate ID {entity_id}")
                    if len(errors) >= 100: break
            if not set(matched).issubset(candidate_set):
                errors.append(f"matching:{line}: prediction is not a subset of candidates")
            for entity_id in matched:
                if not entity_id.startswith(("S2-", "S3-")) or entity_id not in target_ids:
                    errors.append(f"matching:{line}: invalid or unknown match ID {entity_id}")
                    if len(errors) >= 100: break
            counts["test_source1_entities"] += 1
            counts["matching_rows"] += 1
            counts["candidate_rows"] += 1
            if len(errors) >= 100:
                break
    return {"passed": not errors, **counts, "errors": errors[:100], "error_count": len(errors)}
