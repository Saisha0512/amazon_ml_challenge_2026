"""Entity-level macro F0.5 and candidate-recall evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def entity_fbeta(true_ids: set[str], predicted_ids: set[str], beta: float = 0.5) -> float:
    if not true_ids and not predicted_ids:
        return 1.0
    if not true_ids or not predicted_ids:
        return 0.0
    tp = len(true_ids & predicted_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom else 0.0


def macro_fbeta(truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], beta: float = 0.5) -> float:
    if not truth:
        return 0.0
    return sum(entity_fbeta(true_ids, predictions.get(s1, set()), beta) for s1, true_ids in truth.items()) / len(truth)


def candidate_recall(truth: Mapping[str, set[str]], candidates: Mapping[str, set[str]]) -> float:
    positives = sum(len(ids) for ids in truth.values())
    recovered = sum(len(ids & candidates.get(s1, set())) for s1, ids in truth.items())
    return recovered / positives if positives else 0.0


def threshold_sweep(rows: Sequence[tuple[str, str, float]], truth: Mapping[str, set[str]],
                    thresholds: Sequence[float]) -> dict:
    """Tune a single score threshold against entity macro F0.5."""
    best = None
    all_s1 = set(truth)
    scores = []
    for threshold in thresholds:
        predictions = {s1: set() for s1 in all_s1}
        for s1, candidate_id, score in rows:
            if score >= threshold:
                predictions.setdefault(s1, set()).add(candidate_id)
        f05 = macro_fbeta(truth, predictions, beta=0.5)
        n_pred = sum(map(len, predictions.values()))
        record = {"threshold": float(threshold), "macro_f0_5": f05, "predicted_links": n_pred}
        scores.append(record)
        if best is None or (record["macro_f0_5"], -record["threshold"]) > (best["macro_f0_5"], -best["threshold"]):
            best = record
    return {"best": best, "grid": scores}
