"""Low-level, dependency-light text helpers shared by name and address cleaners."""
import re
import unicodedata

try:
    from unidecode import unidecode
except ImportError:
    def unidecode(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(c for c in decomposed if not unicodedata.combining(c))

NAME_STOPWORDS = frozenset({"and", "the", "of", "&"})

_NULL_TOKEN_RE = re.compile(
    r"(?i)(?:^|(?<=[,\s]))(?:<null>|null|n/a|na|none)(?:$|(?=[,\s]))"
)
_MULTI_COMMA_RE = re.compile(r"\s*,\s*,+\s*")
_LEADING_TRAILING_COMMA_RE = re.compile(r"^\s*,\s*|\s*,\s*$")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def strip_accents(text: str) -> str:
    """Remove diacritics from Latin-script characters only.

    NFKD-decompose each character individually and keep the decomposition only
    when its non-combining base is pure ASCII (i.e. it was a Latin letter with an
    accent, like 'á' -> 'a'). Devanagari/Telugu/Bengali/Gujarati/Tamil consonants
    AND their combining vowel signs (matras) are themselves Unicode category Mn
    ("combining") — a naive "strip everything unicodedata.combining() flags" pass
    corrupts Indic text by deleting matras. Checking the decomposed base is ASCII
    before accepting it is what keeps non-Latin scripts untouched.
    """
    result = []
    for ch in text:
        decomposed = unicodedata.normalize("NFKD", ch)
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        if base and all(ord(c) < 128 for c in base):
            result.append(base)
        else:
            result.append(ch)
    return "".join(result)


def transliterate_skeleton(text: str) -> str:
    """Romanize text (unidecode), lowercase, and strip to alphanumeric only.

    A compact matching key, not a display string — used to compare names/words
    across scripts (e.g. Devanagari vs. its English equivalent).
    """
    romanized = unidecode(text).lower()
    return _NON_ALNUM_RE.sub("", romanized)


def normalize_nulls(text: str) -> str:
    """Remove whole-token null-like placeholders ('null', 'N/A', '<NULL>', ...)."""
    cleaned = _NULL_TOKEN_RE.sub("", text)
    cleaned = _MULTI_COMMA_RE.sub(", ", cleaned)
    cleaned = _LEADING_TRAILING_COMMA_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def dedupe_consecutive_tokens(text: str) -> str:
    """Collapse immediately-repeated (case-insensitive) whitespace tokens to one."""
    tokens = text.split()
    out = []
    for tok in tokens:
        if out and out[-1].lower() == tok.lower():
            continue
        out.append(tok)
    return " ".join(out)
