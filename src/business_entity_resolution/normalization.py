"""Local, deterministic text normalization for business records."""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SPACE = re.compile(r"\s+")
_ADDRESS_REPLACEMENTS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "apartment": "apt",
    "suite": "ste", "floor": "fl", "north": "n", "south": "s",
    "east": "e", "west": "w",
}
_NAME_REPLACEMENTS = {"corporation": "corp", "limited": "ltd", "private": "pvt"}


def _ascii_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def normalize_name(value: str | None) -> str:
    text = _ascii_text(value).replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    tokens = [_NAME_REPLACEMENTS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def compact_name(value: str | None) -> str:
    return normalize_name(value).replace(" ", "")


def normalize_address(value: str | None) -> str:
    text = _ascii_text(value).replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    tokens = [_ADDRESS_REPLACEMENTS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def normalize_country(value: str | None) -> str:
    """Normalize superficial formatting while allowing unseen country labels."""
    return _SPACE.sub(" ", _ascii_text(value).strip())


def text_tokens(value: str | None, min_length: int = 2) -> list[str]:
    return [t for t in (value or "").split() if len(t) >= min_length]


def fts_or_query(field: str, text: str | None) -> str | None:
    """Build a safely quoted FTS5 OR query from normalized tokens."""
    terms = list(dict.fromkeys(text_tokens(text, min_length=2)))
    if not terms:
        return None
    return f'{field} : (' + " OR ".join('"' + term.replace('"', '""') + '"' for term in terms) + ")"
