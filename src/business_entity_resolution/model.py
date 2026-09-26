"""Supervised pair classifier and persisted model metadata."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES


def fit_logistic(features: np.ndarray, labels: np.ndarray, seed: int = 42):
    if len(features) == 0 or len(set(labels.tolist())) < 2:
        raise ValueError("Training candidates must contain positive and negative labels")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=300, random_state=seed,
                           solver="lbfgs", C=1.0),
    )
    classifier.fit(features, labels)
    return classifier


def save_model(model, path: str | Path, threshold: float, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                 "threshold": float(threshold), "metadata": metadata or {}}, path, compress=3)


def load_model(path: str | Path) -> dict:
    artifact = joblib.load(path)
    if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Saved model feature schema does not match current code")
    return artifact
