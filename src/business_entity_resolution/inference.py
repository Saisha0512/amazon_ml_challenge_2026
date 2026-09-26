"""Candidate feature scoring for validation and final inference."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from . import config
from .blocking import generate_candidates
from .data_loader import iter_tsv
from .features import FEATURE_NAMES, pair_features
from .normalization import normalize_address, normalize_country, normalize_name


def reservoir_sample(values, size: int, rng: random.Random) -> list[str]:
    reservoir = []
    for idx, value in enumerate(values):
        if idx < size:
            reservoir.append(value)
        else:
            j = rng.randrange(idx + 1)
            if j < size:
                reservoir[j] = value
    return reservoir


def sampled_split_ids(source1_path: str | Path, split_path: str | Path,
                      train_n: int, validation_n: int, seed: int = 42) -> tuple[set[str], set[str]]:
    assignments = {r["source1_entity_id"]: r["split"] for r in iter_tsv(split_path)}
    rng = random.Random(seed)
    train_ids = reservoir_sample((s1 for s1, sp in assignments.items() if sp == "train"), train_n, rng)
    valid_ids = reservoir_sample((s1 for s1, sp in assignments.items() if sp == "validation"), validation_n, rng)
    # Sets make filtering and label lookup efficient; stable row order follows input TSV.
    return set(train_ids), set(valid_ids)


def read_source_rows(path: str | Path, wanted_ids: set[str] | None = None):
    for row in iter_tsv(path):
        if wanted_ids is None or row["entity_id"].strip() in wanted_ids:
            yield row


def read_truth(path: str | Path, wanted_ids: set[str] | None = None) -> dict[str, set[str]]:
    truth = {}
    for row in iter_tsv(path):
        s1 = row["source1_entity_id"].strip()
        if wanted_ids is None or s1 in wanted_ids:
            truth[s1] = {mid.strip() for mid in row["matched_entity_ids"].split(",") if mid.strip()}
    return truth


def candidate_feature_matrix(connection, source1_rows):
    """Create one bounded matrix for a sequence of source1 rows."""
    feature_rows, groups = [], []
    candidate_sets = {}
    for source1 in source1_rows:
        source1 = dict(source1)
        source1["_name_norm"] = normalize_name(source1["business_name"])
        source1["_address_norm"] = normalize_address(source1["business_address"])
        source1["_country_norm"] = normalize_country(source1["country"])
        s1_id = source1["entity_id"].strip()
        candidates = generate_candidates(connection, source1["business_name"],
                                         source1["business_address"], source1["country"])
        candidate_sets[s1_id] = {c["entity_id"] for c in candidates}
        start = len(feature_rows)
        feature_rows.extend([pair_features(source1, candidate) for candidate in candidates])
        groups.append((s1_id, candidates, start, len(feature_rows)))
    matrix = np.asarray([[features[name] for name in FEATURE_NAMES] for features in feature_rows], dtype=np.float32)
    if not feature_rows:
        matrix = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return matrix, groups, candidate_sets


def training_arrays(connection, source1_path, ground_truth_path, selected_ids: set[str],
                    chunk_entities: int = 500):
    feature_chunks, label_chunks = [], []
    truth = read_truth(ground_truth_path, selected_ids)
    batch = []
    n_candidates = 0
    processed = 0
    for row in read_source_rows(source1_path, selected_ids):
        batch.append(row)
        if len(batch) >= chunk_entities:
            matrix, groups, csets = candidate_feature_matrix(connection, batch)
            labels = []
            for s1, candidates, start, end in groups:
                labels.extend(int(c["entity_id"] in truth.get(s1, set())) for c in candidates)
            feature_chunks.append(matrix)
            label_chunks.append(np.asarray(labels, dtype=np.int8))
            n_candidates += len(labels)
            processed += len(batch)
            if processed % 2_000 == 0:
                print(f"Training pairs: processed {processed:,}/{len(selected_ids):,} Source 1 entities", flush=True)
            batch.clear()
    if batch:
        matrix, groups, csets = candidate_feature_matrix(connection, batch)
        labels = []
        for s1, candidates, start, end in groups:
            labels.extend(int(c["entity_id"] in truth.get(s1, set())) for c in candidates)
        feature_chunks.append(matrix)
        label_chunks.append(np.asarray(labels, dtype=np.int8))
        n_candidates += len(labels)
        processed += len(batch)
    features = np.concatenate(feature_chunks) if feature_chunks else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.concatenate(label_chunks) if label_chunks else np.empty(0, dtype=np.int8)
    return features, labels, n_candidates


def score_rows(connection, source1_path, model, threshold: float,
               matching_path: str | Path, candidates_path: str | Path,
               chunk_entities: int = 500):
    """Stream every Source 1 record and write both required TSVs."""
    matching_path, candidates_path = Path(matching_path), Path(candidates_path)
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    total_entities = total_pairs = total_matches = no_candidate_entities = 0
    with matching_path.open("w", encoding="utf-8", newline="") as mf, candidates_path.open("w", encoding="utf-8", newline="") as cf:
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        batch = []
        def flush_batch(rows):
            nonlocal total_entities, total_pairs, total_matches, no_candidate_entities
            matrix, groups, _ = candidate_feature_matrix(connection, rows)
            scores = model.predict_proba(matrix)[:, 1] if len(matrix) else np.empty(0)
            for s1_id, candidates, start, end in groups:
                total_entities += 1
                total_pairs += len(candidates)
                if not candidates:
                    no_candidate_entities += 1
                    mw.writerow([s1_id, ""])
                    cw.writerow([s1_id, ""])
                    if total_entities % 2_000 == 0:
                        print(f"Final inference: processed {total_entities:,} Source 1 entities; {total_pairs:,} candidates", flush=True)
                    continue
                matched = [candidate["entity_id"] for candidate, score in zip(candidates, scores[start:end]) if score >= threshold]
                cw.writerow([s1_id, ",".join(c["entity_id"] for c in candidates)])
                mw.writerow([s1_id, ",".join(matched)])
                total_matches += len(matched)
                if total_entities % 2_000 == 0:
                    print(f"Final inference: processed {total_entities:,} Source 1 entities; {total_pairs:,} candidates", flush=True)
        for source1 in read_source_rows(source1_path):
            batch.append(source1)
            if len(batch) >= chunk_entities:
                flush_batch(batch)
                batch.clear()
        if batch:
            flush_batch(batch)
    return {"source1_entities": total_entities, "candidate_pairs": total_pairs,
            "predicted_matches": total_matches, "entities_without_candidates": no_candidate_entities,
            "matching_path": str(matching_path), "candidate_path": str(candidates_path)}
