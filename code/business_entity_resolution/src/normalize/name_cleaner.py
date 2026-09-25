"""clean_name(): structural cleaning of business_name into schema.NAME_FIELDS.

Scope: structural cleaning only (suffix/prefix/DBA extraction, accent and
duplicate-token noise removal). Spelling/typo correction is a fuzzy-matching
concern for Stage 4 features, not this module.
"""
import re

from src.common.schema import NAME_FIELDS
from src.normalize import text_utils as tu
from src.normalize.legal_suffixes import (
    DBA_MARKER_RE,
    HONORIFIC_PREFIXES,
    SUFFIX_CANONICAL_MAP,
)

_WORD_SPLIT_RE = re.compile(r"([\s\-]+)")  # keeps separators so we can drop them alongside a matched suffix
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _split_dba(text: str) -> tuple[str, str]:
    m = DBA_MARKER_RE.search(text)
    if not m:
        return text, ""
    before, after = m.group("before").strip(" -:"), m.group("after").strip(" -:")
    if not before or not after:
        return text, ""
    return before, after


def _extract_suffixes(text: str) -> tuple[str, tuple[str, ...]]:
    """Remove legal-suffix tokens anywhere in the string; return (remaining, canonical suffixes).

    Splits on hyphens as well as whitespace — real data glues a suffix directly
    onto an adjacent word with a hyphen (e.g. "PLLC-PARTNERS"), and a plain
    ``text.split()`` would treat that as one unmatched token and miss the suffix
    entirely. Dropping the separator alongside a matched suffix (the ``i += 2``
    below) avoids leaving a dangling hyphen/space in the remaining text.
    """
    parts = _WORD_SPLIT_RE.split(text)
    found = []
    kept = []
    i = 0
    while i < len(parts):
        word = parts[i]
        key = word.strip(".,()[]").lower().replace(".", "")
        if word and key in SUFFIX_CANONICAL_MAP:
            found.append(SUFFIX_CANONICAL_MAP[key])
            i += 2  # also drop the following separator
            continue
        kept.append(word)
        i += 1
    remaining = "".join(kept)
    remaining = _MULTI_SPACE_RE.sub(" ", remaining).strip(" -,()[]")
    return remaining, tuple(dict.fromkeys(found))  # dedupe, preserve order


def _strip_honorific(text: str) -> str:
    tokens = text.split()
    if len(tokens) >= 3 and tokens[0].strip(".,").lower() in HONORIFIC_PREFIXES:
        return " ".join(tokens[1:])
    return text


def _build_acronym(text: str) -> str:
    letters = []
    for tok in text.split():
        clean_tok = tok.strip("()[]{}.,")
        if not clean_tok or clean_tok.lower() in tu.NAME_STOPWORDS:
            continue
        letters.append(clean_tok[0].upper())
    return "".join(letters)


def clean_name(raw: str) -> dict:
    raw = raw or ""
    deaccented = tu.strip_accents(raw)
    deaccented = tu.dedupe_consecutive_tokens(deaccented)

    name_a, name_b = _split_dba(deaccented)

    working = _strip_honorific(name_a)
    stripped_suffix, suffixes = _extract_suffixes(working)

    core_name = stripped_suffix if stripped_suffix.strip() else working
    if not core_name.strip():
        core_name = deaccented  # never emit an empty core name

    compact_name = _NON_ALNUM_RE.sub("", core_name.lower())
    skeleton = tu.transliterate_skeleton(core_name)
    acronym = _build_acronym(core_name)

    result = {
        "raw": raw,
        "deaccented": deaccented,
        "core_name": core_name,
        "legal_suffix": suffixes,
        "alt_name": name_b,
        "compact_name": compact_name,
        "skeleton": skeleton,
        "acronym": acronym,
    }
    assert set(result.keys()) == set(NAME_FIELDS)
    return result
