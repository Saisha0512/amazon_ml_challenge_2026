"""Train, tune, and generate final Amazon ML Challenge submission files."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from pathlib import Path

import numpy as np

from . import config
from .blocking import build_index, connect_index
from .data_loader import make_source1_split
from .evaluation import candidate_recall, threshold_sweep
from .features import FEATURE_NAMES
from .inference import (candidate_feature_matrix, read_source_rows, read_truth,
                        sampled_split_ids, score_rows, training_arrays)
from .model import fit_logistic, load_model, save_model
from .submission import validate_submission


def _validation_sample(connection, ids: set[str], source1_path: Path,
                       ground_truth_path: Path, model):
    truth = read_truth(ground_truth_path, ids)
    pairs = []
    candidate_sets = {}
    candidate_count = 0
    batch = []
    processed = 0
    for row in read_source_rows(source1_path, ids):
        batch.append(row)
        if len(batch) >= 250:
            matrix, groups, csets = candidate_feature_matrix(connection, batch)
            score_array = model.predict_proba(matrix)[:, 1] if len(matrix) else np.empty(0)
            for s1, candidates, start, end in groups:
                candidate_sets[s1] = csets[s1]
                candidate_count += len(candidates)
                pairs.extend((s1, candidate["entity_id"], float(score))
                             for candidate, score in zip(candidates, score_array[start:end]))
            batch.clear()
            processed += 250
            if processed % 5_000 == 0:
                print(f"Validation pairs: processed {processed:,}/{len(ids):,} Source 1 entities", flush=True)
    if batch:
        matrix, groups, csets = candidate_feature_matrix(connection, batch)
        score_array = model.predict_proba(matrix)[:, 1] if len(matrix) else np.empty(0)
        for s1, candidates, start, end in groups:
            candidate_sets[s1] = csets[s1]
            candidate_count += len(candidates)
            pairs.extend((s1, candidate["entity_id"], float(score))
                         for candidate, score in zip(candidates, score_array[start:end]))
        processed += len(batch)
        print(f"Validation pairs: processed {processed:,}/{len(ids):,} Source 1 entities", flush=True)
    return truth, pairs, candidate_sets, candidate_count


def _write_experiment(report: dict) -> None:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / "phase2_4_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (config.REPORT_DIR / "experiments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["threshold", "macro_f0_5", "candidate_recall",
                                                     "candidate_pairs", "validation_entities", "train_entities"])
        writer.writeheader()
        for row in report["threshold_sweep"]["grid"]:
            writer.writerow({"threshold": row["threshold"], "macro_f0_5": row["macro_f0_5"],
                             "candidate_recall": report["candidate_recall"],
                             "candidate_pairs": report["validation_candidate_pairs"],
                             "validation_entities": report["validation_entities"],
                             "train_entities": report["train_entities"]})


def train_and_tune() -> dict:
    started = time.time()
    split_file = config.VALIDATION_DIR / "source1_split.tsv"
    if not split_file.exists():
        make_source1_split(config.TRAIN_FILES["source1"], split_file)
    train_index = config.BLOCK_INDEX_DIR / "train_candidates.sqlite"
    build_index({"source2": config.TRAIN_FILES["source2"], "source3": config.TRAIN_FILES["source3"]}, train_index)
    con = connect_index(train_index)
    try:
        train_ids, validation_ids = sampled_split_ids(
            config.TRAIN_FILES["source1"], split_file,
            config.TRAIN_ENTITY_SAMPLE, config.VALIDATION_ENTITY_SAMPLE, config.RANDOM_SEED)
        train_x, train_y, train_candidate_count = training_arrays(
            con, config.TRAIN_FILES["source1"], config.TRAIN_FILES["ground_truth"], train_ids)
        train_positive_count = int(train_y.sum())
        train_positive_fraction = float(train_y.mean()) if len(train_y) else 0.0
        model = fit_logistic(train_x, train_y, config.RANDOM_SEED)
        del train_x, train_y
        gc.collect()
        truth, score_rows, candidate_sets, validation_candidate_count = _validation_sample(
            con, validation_ids, config.TRAIN_FILES["source1"], config.TRAIN_FILES["ground_truth"], model)
        sweep = threshold_sweep(score_rows, truth, config.PREDICTION_THRESHOLD_GRID)
        total_possible = len(validation_ids) * (5_034_616 + 5_285_603)
        report = {
            "seed": config.RANDOM_SEED,
            "train_entities": len(train_ids), "validation_entities": len(validation_ids),
            "train_candidate_pairs": train_candidate_count,
            "train_positive_candidate_pairs": train_positive_count,
            "train_candidate_recall_not_evaluated": "Training sample recall is not used for tuning.",
            "validation_candidate_pairs": validation_candidate_count,
            "candidate_recall": candidate_recall(truth, candidate_sets),
            "candidate_reduction_ratio": 1 - validation_candidate_count / total_possible,
            "threshold_sweep": sweep,
            "feature_names": FEATURE_NAMES,
            "training_positive_fraction": train_positive_fraction,
            "note": "Validation and training use disjoint Source 1 entities sampled from the fixed 80/20 split.",
            "elapsed_seconds": round(time.time() - started, 1),
        }
        _write_experiment(report)
        save_model(model, config.MODEL_DIR / "entity_matcher.joblib", sweep["best"]["threshold"], report)
        print(json.dumps(report, indent=2))
        return report
    finally:
        con.close()


def final_inference(threshold: float | None = None) -> dict:
    artifact = load_model(config.MODEL_DIR / "entity_matcher.joblib")
    if threshold is None:
        threshold = artifact["threshold"]
    test_index = config.BLOCK_INDEX_DIR / "test_candidates.sqlite"
    build_index({"source2": config.TEST_FILES["source2"], "source3": config.TEST_FILES["source3"]}, test_index)
    con = connect_index(test_index)
    try:
        result = {}
        result.update(score_rows(
            con, config.TEST_FILES["source1"], artifact["model"], threshold,
            config.FINAL_OUTPUT_DIR / "matching_results.tsv",
            config.FINAL_OUTPUT_DIR / "candidate_pairs.tsv", chunk_entities=250))
    finally:
        con.close()
    result["threshold"] = threshold
    validation = validate_submission(
        config.FINAL_OUTPUT_DIR / "matching_results.tsv",
        config.FINAL_OUTPUT_DIR / "candidate_pairs.tsv",
        config.TEST_FILES["source1"], config.TEST_FILES["source2"], config.TEST_FILES["source3"])
    result["submission_validation"] = validation
    if not validation["passed"]:
        raise RuntimeError(f"Submission validation failed: {validation['errors'][:5]}")
    (config.FINAL_OUTPUT_DIR / "inference_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "train", "infer"), default="all")
    args = parser.parse_args()
    if args.phase in ("all", "train"):
        train_and_tune()
    if args.phase in ("all", "infer"):
        final_inference()


if __name__ == "__main__":
    main()
