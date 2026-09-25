"""Legal-suffix canonicalization, honorific prefixes, and DBA-marker detection."""
import re

SUFFIX_CANONICAL_MAP = {
    "pvt": "private", "private": "private",
    "ltd": "limited", "limited": "limited",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "llc": "llc", "l.l.c": "llc",
    "llp": "llp",
    "pllc": "pllc",
    "lp": "lp",
    "co": "co", "company": "co",
    # French
    "sarl": "sarl",
    "sas": "sas",
}

HONORIFIC_PREFIXES = frozenset({"mr", "mrs", "smt", "shri", "m/s"})
# Deliberately excludes bare "ms" (no slash): "M/s" (Messrs, the Indian filing
# convention) is unambiguous, but bare "Ms" collides with names that legitimately
# start with the token "MS" (e.g. "MS Consultancy Corp", promoter's initials) —
# see the name_cleaner test that pins this exact case.

DBA_MARKER_RE = re.compile(
    r"(?i)^(?P<before>.*?)\b(?:dba|d/b/a|trading as|née)\b:?\s*(?P<after>.*)$"
)
