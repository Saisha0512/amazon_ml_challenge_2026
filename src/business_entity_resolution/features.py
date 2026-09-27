"""Pairwise string and country comparison features."""

from __future__ import annotations

import re
from functools import lru_cache
from rapidfuzz import fuzz

from .normalization import normalize_address, normalize_country, normalize_name, text_tokens

_NUMBERS = re.compile(r"\d+")


FEATURE_NAMES = (
    "name_exact", "name_compact_exact", "name_ratio", "name_token_sort",
    "name_token_set", "name_wratio", "name_jaccard", "name_shared_tokens",
    "name_token_containment", "name_length_ratio", "address_exact",
    "address_ratio", "address_token_sort", "address_token_set",
    "address_jaccard", "address_shared_tokens", "address_numeric_overlap",
    "country_exact", "source2_candidate", "source3_candidate",
    "block_exact_name", "block_compact_name", "block_exact_address",
    "block_fts_name", "block_fts_address",
)


@lru_cache(maxsize=131_072)
def _token_set(value: str) -> frozenset[str]:
    return frozenset(text_tokens(value))


@lru_cache(maxsize=131_072)
def _number_set(value: str) -> frozenset[str]:
    return frozenset(_NUMBERS.findall(value))


def pair_feature_vector(source1: dict, candidate: dict) -> tuple[float, ...]:
    """Return the model features in fixed order without per-pair dict allocation."""
    n1 = source1.get("_name_norm") or normalize_name(source1.get("business_name"))
    n2 = candidate.get("name_norm") or normalize_name(candidate.get("business_name"))
    a1 = source1.get("_address_norm") or normalize_address(source1.get("business_address"))
    a2 = candidate.get("address_norm") or normalize_address(candidate.get("business_address"))
    c1 = source1.get("_country_norm") or normalize_country(source1.get("country"))
    c2 = candidate.get("country") or normalize_country(candidate.get("country"))
    t1, t2 = _token_set(n1), _token_set(n2)
    at1, at2 = _token_set(a1), _token_set(a2)
    nums1, nums2 = _number_set(a1), _number_set(a2)
    shared_name = t1 & t2
    shared_addr = at1 & at2
    union_name = t1 | t2
    union_addr = at1 | at2
    rules = candidate.get("rules", set())
    return (
        float(bool(n1) and n1 == n2),
        float(bool(n1) and n1.replace(" ", "") == n2.replace(" ", "")),
        fuzz.ratio(n1, n2) / 100 if n1 and n2 else 0.0,
        fuzz.token_sort_ratio(n1, n2) / 100 if n1 and n2 else 0.0,
        fuzz.token_set_ratio(n1, n2) / 100 if n1 and n2 else 0.0,
        fuzz.WRatio(n1, n2) / 100 if n1 and n2 else 0.0,
        len(shared_name) / len(union_name) if union_name else 0.0,
        float(len(shared_name)),
        len(shared_name) / max(1, min(len(t1), len(t2))),
        min(len(n1), len(n2)) / max(1, len(n1), len(n2)),
        float(bool(a1) and a1 == a2),
        fuzz.ratio(a1, a2) / 100 if a1 and a2 else 0.0,
        fuzz.token_sort_ratio(a1, a2) / 100 if a1 and a2 else 0.0,
        fuzz.token_set_ratio(a1, a2) / 100 if a1 and a2 else 0.0,
        len(shared_addr) / len(union_addr) if union_addr else 0.0,
        float(len(shared_addr)),
        len(nums1 & nums2) / max(1, len(nums1 | nums2)),
        float(bool(c1) and c1 == c2),
        float(candidate.get("source") == "S2"),
        float(candidate.get("source") == "S3"),
        float("name_exact" in rules),
        float("name_compact" in rules),
        float("address_exact" in rules),
        float("fts_name" in rules),
        float("fts_address" in rules),
    )


def pair_features(source1: dict, candidate: dict) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, pair_feature_vector(source1, candidate)))
