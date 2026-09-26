"""TSV loading, streaming validation, and entity-level split helpers."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from . import config


def read_tsv(path: str | Path):
    """Load a challenge TSV as a pandas DataFrame, always using tab separation."""
    import pandas as pd

    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def iter_tsv(path: str | Path):
    """Yield dictionaries from a TSV without materializing the full file."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty TSV: {path}")
        yield from reader


def validate_source_file(path: str | Path, expected_source: str) -> dict:
    """Stream a source TSV and report schema, missingness, IDs, and basic EDA."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = reader.fieldnames or []
        missing_columns = [c for c in config.SOURCE_COLUMNS if c not in columns]
        if missing_columns:
            raise ValueError(f"{path}: missing required columns {missing_columns}; found {columns}")
        row_count = 0
        seen_ids: set[str] = set()
        duplicate_ids = 0
        id_prefix_errors = 0
        missing = Counter()
        countries = Counter()
        name_counts = Counter()
        name_sample_size = 50_000
        name_length_total = 0
        address_length_total = 0
        for row in reader:
            row_count += 1
            entity_id = row["entity_id"].strip()
            if entity_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(entity_id)
            if not entity_id.startswith(expected_source + "-"):
                id_prefix_errors += 1
            for field in config.SOURCE_COLUMNS:
                if not row[field].strip():
                    missing[field] += 1
            country = row["country"].strip()
            countries[country or "<missing>"] += 1
            name = " ".join(row["business_name"].casefold().split())
            if row_count <= name_sample_size and name:
                name_counts[name] += 1
            name_length_total += len(row["business_name"])
            address_length_total += len(row["business_address"])
    dup_name_groups = sum(count > 1 for count in name_counts.values())
    dup_name_rows = sum(count for count in name_counts.values() if count > 1)
    return {
        "path": str(path), "rows": row_count, "columns": columns,
        "duplicate_entity_ids": duplicate_ids, "id_prefix_errors": id_prefix_errors,
        "missing": dict(missing), "countries": dict(countries),
        "sample_size_for_name_duplicates": min(row_count, name_sample_size),
        "sample_duplicate_name_groups_casefold": dup_name_groups,
        "sample_rows_in_duplicate_name_groups_casefold": dup_name_rows,
        "mean_name_chars": round(name_length_total / row_count, 2) if row_count else 0,
        "mean_address_chars": round(address_length_total / row_count, 2) if row_count else 0,
    }


def validate_ground_truth(path: str | Path) -> dict:
    """Stream labels and summarize zero/one/many match counts and label integrity."""
    path = Path(path)
    distribution = Counter()
    seen_s1: set[str] = set()
    duplicate_s1 = 0
    duplicate_match_rows = 0
    row_count = 0
    malformed_matches = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = reader.fieldnames or []
        missing = [c for c in config.GROUND_TRUTH_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}; found {columns}")
        for row in reader:
            row_count += 1
            s1_id = row["source1_entity_id"].strip()
            if s1_id in seen_s1:
                duplicate_s1 += 1
            seen_s1.add(s1_id)
            values = [value.strip() for value in row["matched_entity_ids"].split(",") if value.strip()]
            if len(values) != len(set(values)):
                duplicate_match_rows += 1
            distribution[min(len(values), 2)] += 1
            if any(not (value.startswith("S2-") or value.startswith("S3-")) for value in values):
                malformed_matches += 1
    return {
        "path": str(path), "rows": row_count, "columns": columns,
        "duplicate_source1_ids": duplicate_s1,
        "rows_with_duplicate_matched_ids": duplicate_match_rows,
        "match_count_distribution": {
            "zero": distribution[0], "one": distribution[1], "many_2_plus": distribution[2]
        },
        "rows_with_invalid_match_prefix": malformed_matches,
    }


def make_source1_split(source1_path: str | Path, output_path: str | Path,
                       validation_fraction: float = config.VALIDATION_FRACTION,
                       seed: int = config.RANDOM_SEED) -> dict:
    """Create a deterministic Source-1 entity split and persist it as TSV."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    entity_ids = [row["entity_id"].strip() for row in iter_tsv(source1_path)]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("Source 1 contains duplicate IDs; cannot create an unambiguous split")
    random.Random(seed).shuffle(entity_ids)
    n_validation = round(len(entity_ids) * validation_fraction)
    validation_ids = set(entity_ids[:n_validation])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "split"])
        writer.writerows((entity_id, "validation" if entity_id in validation_ids else "train")
                         for entity_id in entity_ids)
    return {"train_entities": len(entity_ids) - n_validation,
            "validation_entities": n_validation, "total_entities": len(entity_ids),
            "seed": seed, "validation_fraction": validation_fraction,
            "split_file": str(output_path)}


def run_phase1_audit() -> dict:
    """Validate every supplied TSV, check cross-file labels, and write split."""
    source_reports = {}
    train_id_sets: dict[str, set[str]] = {}
    for split, files in (("train", config.TRAIN_FILES), ("test", config.TEST_FILES)):
        for source in ("source1", "source2", "source3"):
            report = validate_source_file(files[source], f"S{source[-1]}")
            source_reports[f"{split}_{source}"] = report
            if split == "train":
                train_id_sets[source] = {r["entity_id"].strip() for r in iter_tsv(files[source])}
    ground_truth = validate_ground_truth(config.TRAIN_FILES["ground_truth"])
    train_s1_ids = train_id_sets["source1"]
    train_s2_ids = train_id_sets["source2"]
    train_s3_ids = train_id_sets["source3"]
    label_s1_ids: set[str] = set()
    invalid_s1_refs = invalid_match_refs = 0
    for row in iter_tsv(config.TRAIN_FILES["ground_truth"]):
        s1_id = row["source1_entity_id"].strip()
        label_s1_ids.add(s1_id)
        invalid_s1_refs += s1_id not in train_s1_ids
        for matched_id in (item.strip() for item in row["matched_entity_ids"].split(",") if item.strip()):
            invalid_match_refs += matched_id not in (train_s2_ids if matched_id.startswith("S2-") else train_s3_ids)
    if label_s1_ids != train_s1_ids:
        raise ValueError(f"Ground truth S1 coverage differs: missing={len(train_s1_ids-label_s1_ids)}, extra={len(label_s1_ids-train_s1_ids)}")
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    split = make_source1_split(config.TRAIN_FILES["source1"], config.VALIDATION_DIR / "source1_split.tsv")
    report = {"sources": source_reports, "ground_truth": ground_truth,
              "label_reference_errors": {"unknown_source1": invalid_s1_refs, "unknown_match_ids": invalid_match_refs},
              "validation_split": split}
    report_path = config.VALIDATION_DIR / "phase1_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["audit_report_file"] = str(report_path)
    return report
