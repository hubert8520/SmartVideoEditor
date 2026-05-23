"""Polish-friendly text normalization helpers."""

from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    """Remove combining marks while preserving base letters."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparisons and heuristics."""
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9ąćęłńóśźż]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text."""
    return [token for token in normalize_text(text).split() if token]
