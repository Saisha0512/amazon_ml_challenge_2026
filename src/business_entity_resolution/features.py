"""Pairwise string and country comparison features."""

from __future__ import annotations

import re
from rapidfuzz import fuzz

from .normalization import normalize_address, normalize_country, normalize_name, text_tokens

_NUMBERS = re.compile(r"\d+")


def _jaccard(left: str, right: str) -> float:
    a, b = set(text_tokens(left)), set(text_tokens(right))
    return len(a & b) / len(a | b) if a or b else 0.0


def pair_features(source1: dict, candidate: dict) -> dict[str, float]:
    n1 = source1.get("_name_norm") or normalize_name(source1.get("business_name"))
    n2 = candidate.get("name_norm") or normalize_name(candidate.get("business_name"))
    a1 = source1.get("_address_norm") or normalize_address(source1.get("business_address"))
    a2 = candidate.get("address_norm") or normalize_address(candidate.get("business_address"))
    c1 = source1.get("_country_norm") or normalize_country(source1.get("country"))
    c2 = candidate.get("country") or normalize_country(candidate.get("country"))
    t1, t2 = set(text_tokens(n1)), set(text_tokens(n2))
    at1, at2 = set(text_tokens(a1)), set(text_tokens(a2))
    nums1, nums2 = set(_NUMBERS.findall(a1)), set(_NUMBERS.findall(a2))
    shared_name = t1 & t2
    shared_addr = at1 & at2
    rules = candidate.get("rules", set())
    return {
        "name_exact": float(bool(n1) and n1 == n2),
        "name_compact_exact": float(bool(n1) and n1.replace(" ", "") == n2.replace(" ", "")),
        "name_ratio": fuzz.ratio(n1, n2) / 100 if n1 and n2 else 0.0,
        "name_token_sort": fuzz.token_sort_ratio(n1, n2) / 100 if n1 and n2 else 0.0,
        "name_token_set": fuzz.token_set_ratio(n1, n2) / 100 if n1 and n2 else 0.0,
        "name_wratio": fuzz.WRatio(n1, n2) / 100 if n1 and n2 else 0.0,
        "name_jaccard": _jaccard(n1, n2),
        "name_shared_tokens": float(len(shared_name)),
        "name_token_containment": len(shared_name) / max(1, min(len(t1), len(t2))),
        "name_length_ratio": min(len(n1), len(n2)) / max(1, len(n1), len(n2)),
        "address_exact": float(bool(a1) and a1 == a2),
        "address_ratio": fuzz.ratio(a1, a2) / 100 if a1 and a2 else 0.0,
        "address_token_sort": fuzz.token_sort_ratio(a1, a2) / 100 if a1 and a2 else 0.0,
        "address_token_set": fuzz.token_set_ratio(a1, a2) / 100 if a1 and a2 else 0.0,
        "address_jaccard": _jaccard(a1, a2),
        "address_shared_tokens": float(len(shared_addr)),
        "address_numeric_overlap": len(nums1 & nums2) / max(1, len(nums1 | nums2)),
        "country_exact": float(bool(c1) and c1 == c2),
        "source2_candidate": float(candidate.get("source") == "S2"),
        "source3_candidate": float(candidate.get("source") == "S3"),
        "block_exact_name": float("name_exact" in rules),
        "block_compact_name": float("name_compact" in rules),
        "block_exact_address": float("address_exact" in rules),
        "block_fts_name": float("fts_name" in rules),
        "block_fts_address": float("fts_address" in rules),
    }


FEATURE_NAMES = tuple(pair_features(
    {"business_name": "A", "business_address": "1 Main St", "country": "US"},
    {"name_norm": "a", "address_norm": "1 main st", "country": "us", "source": "S2", "rules": set()},
).keys())
