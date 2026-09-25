# Stage 2 Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn raw `business_name` / `business_address` strings from all 6 source files (train/test × S1/S2/S3) into a shared, cleaned record schema — separate name and address cleaners, each producing structured sub-fields — that Stage 3 (blocking) and Stage 4 (features) will consume.

**Architecture:** Two independent, pure-function cleaners (`name_cleaner.py`, `address_cleaner.py`) built on shared low-level text utilities (`text_utils.py`) and static lookup tables (`legal_suffixes.py`, `road_abbrev.py`). A thin CLI (`run_normalize.py`) applies both cleaners vectorized over pandas to each of the 6 TSVs and writes Parquet output with the shared record schema. No ML, no blocking, no fuzzy matching in this plan — this stage only produces structured fields; similarity scoring is Stage 4's job.

**Tech Stack:** Python 3.12 (`.venv`), pandas, pyarrow (Parquet output), `unidecode` (MIT, transliteration skeleton), `regex`-free stdlib `re`, `pytest` for tests.

**Spec:** `student_resource/README.md` (problem statement, noise patterns) and the EDA findings below, gathered from `train_source1/2/3.tsv` and `train_ground_truth.tsv` (2026-09-25 session — see `code/business_entity_resolution/src/eda_stage1.py` and `eda_spotcheck.py` for the scripts that produced them).

**EDA findings this plan is built on:**
- One-to-one holds exactly: 0 of 7,638,365 matched S2/S3 ids match more than one S1 (Stage 6 can safely assume one-to-one).
- Matched pairs *never* cross country (0 mismatches across 7.6M matched pairs) — country blocking is safe for US/India; France (test-only) must still get a working fallback since it's a 0-shot label.
- Singleton rate is 5.58% of S1 entities — most S1 records do have matches.
- `business_name` is never empty in any file. `business_address` is empty in ~3% of S2/S3 rows (train and test), 0% in S1.
- Non-Latin script share: ~9–19% of S2/S3 names/addresses (Devanagari, Telugu, Bengali, Gujarati, Tamil), ~0% in train S1, ~2–4% in test S1 (France + India adds this).
- Country labels are clean, exact strings: `{US, India}` in train; `{US, India, France}` in test. No casing/spelling variants observed.
- Postcode-shaped tokens (5–6 digit) appear in only ~11% of US addresses and <1% of India addresses — too sparse to be a primary blocking key, but a strong signal when present.
- DBA/trading-as markers (`dba`, `d/b/a`, `née`, `trading as`) appear **only in S3** names, ~0.6–2% combined; never in S2.
- Honorific prefixes `M/s` and `Smt` appear in ~0.5% of S2/S3 names each.
- Immediately-repeated consecutive word tokens ("Partners Partners", "Regional Regional") appear in ~2.4% of S2/S3 names — synthetic corruption, not real business naming.
- Trailing `- <digits>` run-on suffixes ("Surgical Health LP - 2810307279") appear in ~0.3% of S2/S3 names.
- Random accent-mark injection into otherwise-plain ASCII words is common noise ("cmcáre.com", "Novyx Ínc", "Bíotherapeutics") — distinct from genuine French accents that will appear in test.
- **Critical finding:** a nontrivial share of true matches have a business name with *zero* token overlap with S1 ("Shri Verafayemira" ↔ "Achyut Colonisers Pvt Ltd", "Xyloquoevo" ↔ "Rabun Boral LLC", "Drexumbraumbra" ↔ "Gonzalez, Rosa and Parise") where only the address ties them together. Address-derived fields must be strong enough to stand alone as a blocking/matching signal — do not assume name similarity is always present.
- `null`-like literal tokens appear inline inside address strings: `null`, `N/A`, `<NULL>` — these must be treated as missing, not as address content.
- House numbers carry noise prefixes: `##19821`, `0044023`, leading zeros — need normalization for a clean house-number key.

## Global Constraints

- All 6 source files are TSV with an explicit header; always read with `sep="\t"`, `dtype=str`, `keep_default_na=False` (empty string, not NaN, is the correct "missing" sentinel — pandas NaN would silently break string ops).
- Files are large (S2/S3 have 5M+ rows each in train, ~5M in test); every per-record operation must be vectorizable via `pandas.Series.str` / `.map()` over a plain Python function — no `df.apply(axis=1)` and no Python `for` loop over rows. `.map()` on a `Series` (single column, plain function) is fine; it is `DataFrame.apply(axis=1)` and explicit row loops that must be avoided.
- Output schema is shared across all downstream stages — do not rename fields ad hoc later; extend this plan's schema instead.
- No external data lookups, no network calls (Academic Integrity rule in the README) — `unidecode`'s bundled tables are static and local, so this is compliant.
- Do not hard-code `country` to `{US, India}` anywhere — always treat it as an open string label (README requirement; France appears only in test).
- License: only MIT/Apache-2.0 dependencies (`unidecode` is MIT — confirmed).

## Review Focus

- **Empty/whitespace-only names or addresses after cleaning** — a record whose `business_name` was real text but becomes an empty `core_name` after suffix/prefix stripping (e.g. a name that is *only* a legal suffix like "LLC") must fall back to the pre-strip text, never emit an empty core field silently.
- **Addresses that are entirely a null-like token** (`"null"`, `"N/A"`, `"<NULL>"`, or empty string) — must normalize to a fully-empty address record (all sub-fields empty/None), not a string containing the literal word "null".
- **Non-Latin text passed through `unidecode` unexpectedly stripping the original script** — the cleaner must preserve the original non-Latin string in its own field; `unidecode` output is an *additional* skeleton field, never a replacement for the source text.
- **DBA-marker split producing an empty half** ("DBA: Cornerstone Investments" with nothing before the marker, or "Drexsolpyra DBA:" with nothing after) — must not crash and must fall back to treating the whole string as a single name.
- **French test-only input** (accented business names, `SARL`/`SAS` suffixes, `rue`/`bd` address tokens, `l'`/`d'` elisions) — since France has zero training examples, the cleaner's suffix/road dictionaries must include French forms now, and a round-trip test must confirm French-suffixed names/addresses don't crash or get mangled by rules tuned for English/Hindi.

---

## File Structure

```
code/business_entity_resolution/
├── requirements.txt                          # pinned: pandas, pyarrow, unidecode, pytest
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   └── schema.py                         # column name constants, canonical record dict shape
│   └── normalize/
│       ├── __init__.py
│       ├── text_utils.py                     # strip_accents, transliterate_skeleton, normalize_nulls, split_tokens, dedupe_consecutive
│       ├── legal_suffixes.py                 # suffix canonical map + honorific prefix list + DBA marker regexes
│       ├── road_abbrev.py                    # road/unit abbreviation canonical map (EN + FR)
│       ├── name_cleaner.py                   # clean_name(raw: str) -> dict
│       ├── address_cleaner.py                # clean_address(raw: str) -> dict
│       └── run_normalize.py                  # CLI: reads the 6 TSVs, applies cleaners, writes Parquet
└── tests/
    └── normalize/
        ├── __init__.py
        ├── test_text_utils.py
        ├── test_name_cleaner.py
        ├── test_address_cleaner.py
        └── test_run_normalize.py
```

All new code lives under `code/business_entity_resolution/`, matching the final submission zip layout the README requires (`code/business_entity_resolution/src/`, `README.md`, `requirements.txt`).

---

### Task 1: Project scaffolding, requirements.txt, shared schema

**Files:**
- Create: `code/business_entity_resolution/requirements.txt`
- Create: `code/business_entity_resolution/src/common/__init__.py`
- Create: `code/business_entity_resolution/src/common/schema.py`
- Create: `code/business_entity_resolution/src/normalize/__init__.py`
- Create: `code/business_entity_resolution/tests/__init__.py`
- Create: `code/business_entity_resolution/tests/normalize/__init__.py`
- Test: `code/business_entity_resolution/tests/normalize/test_schema.py`

**Interfaces:**
- Produces: `schema.NAME_FIELDS: list[str]` — the ordered keys every `clean_name()` output dict must have.
- Produces: `schema.ADDRESS_FIELDS: list[str]` — the ordered keys every `clean_address()` output dict must have.
- Produces: `schema.RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]` — expected input columns.

- [ ] **Step 1: Write the failing test**

```python
# code/business_entity_resolution/tests/normalize/test_schema.py
from src.common import schema


def test_name_fields_defined():
    expected = {
        "raw", "core_name", "deaccented", "legal_suffix", "alt_name",
        "compact_name", "skeleton", "acronym",
    }
    assert expected.issubset(set(schema.NAME_FIELDS))


def test_address_fields_defined():
    expected = {
        "raw", "core_address", "postcode", "house_number", "unit",
        "landmark", "deaccented",
    }
    assert expected.issubset(set(schema.ADDRESS_FIELDS))


def test_raw_columns():
    assert schema.RAW_COLUMNS == ["entity_id", "business_name", "business_address", "country"]
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `code/business_entity_resolution/`): `python -m pytest tests/normalize/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'` or `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/common/schema.py
"""Shared field names for the entity-resolution pipeline's cleaned record schema."""

RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

NAME_FIELDS = [
    "raw",           # original business_name, untouched
    "deaccented",    # accent-stripped (Latin-script noise removed), non-Latin scripts untouched
    "core_name",     # deaccented name with legal suffix, DBA half, honorific prefix removed
    "legal_suffix",  # tuple of canonical suffix tokens found, e.g. ("private", "limited")
    "alt_name",      # DBA/trading-as alternate name half, "" if none found
    "compact_name",  # core_name, lowercased, alphanumeric-only, no spaces
    "skeleton",      # unidecode-transliterated, lowercased, alphanumeric-only core_name
    "acronym",       # first letter of each core_name token (excluding stopwords), uppercased
]

ADDRESS_FIELDS = [
    "raw",            # original business_address, untouched
    "deaccented",     # accent-stripped, non-Latin scripts untouched, null-like tokens removed
    "core_address",   # deaccented address with postcode/house_number/unit/landmark stripped out
    "postcode",       # extracted postcode string, "" if none found
    "house_number",   # normalized leading house/plot number, "" if none found
    "unit",           # normalized unit/suite/PMB/floor token, "" if none found
    "landmark",       # "near X" / "opp X" phrase, "" if none found
]
```

```python
# code/business_entity_resolution/src/common/__init__.py
```

```python
# code/business_entity_resolution/src/normalize/__init__.py
```

```python
# code/business_entity_resolution/tests/__init__.py
```

```python
# code/business_entity_resolution/tests/normalize/__init__.py
```

```text
# code/business_entity_resolution/requirements.txt
pandas==2.3.3
pyarrow==22.0.0
unidecode==1.4.0
rapidfuzz==3.14.1
pytest==8.4.2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/requirements.txt \
        code/business_entity_resolution/src/common/ \
        code/business_entity_resolution/src/normalize/__init__.py \
        code/business_entity_resolution/tests/
git commit -m "feat: add shared record schema and project scaffolding"
```

---

### Task 2: Text utilities (accent stripping, null normalization, transliteration skeleton, token dedup)

**Files:**
- Create: `code/business_entity_resolution/src/normalize/text_utils.py`
- Test: `code/business_entity_resolution/tests/normalize/test_text_utils.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure stdlib + `unidecode`).
- Produces:
  - `strip_accents(text: str) -> str` — NFKD-decompose, drop combining marks over Latin base letters, leave non-Latin codepoints (Devanagari, Telugu, Bengali, Gujarati, Tamil, etc.) untouched.
  - `transliterate_skeleton(text: str) -> str` — `unidecode(text)`, lowercased, alphanumeric-only (spaces stripped too — this is a compact matching key, not a display string).
  - `normalize_nulls(text: str) -> str` — replace whole-token case-insensitive `null`, `n/a`, `na`, `none`, `<null>` with `""`, collapsing resulting double spaces/commas.
  - `dedupe_consecutive_tokens(text: str) -> str` — collapse immediately-repeated (case-insensitive) whitespace-separated tokens to one occurrence, preserving original casing of the first occurrence.
  - `NAME_STOPWORDS: frozenset[str]` — `{"and", "the", "of", "&"}`, used by the acronym builder in Task 3.

- [ ] **Step 1: Write the failing test**

```python
# code/business_entity_resolution/tests/normalize/test_text_utils.py
from src.normalize import text_utils as tu


def test_strip_accents_latin_noise():
    assert tu.strip_accents("cmcáre.com") == "cmcare.com"
    assert tu.strip_accents("Novyx Ínc") == "Novyx Inc"
    assert tu.strip_accents("Bíotherapeutics") == "Biotherapeutics"
    assert tu.strip_accents("Módern") == "Modern"


def test_strip_accents_preserves_non_latin():
    devanagari = "सुप्रीम आईटी प्राइवेट लिमिटेड"
    assert tu.strip_accents(devanagari) == devanagari


def test_strip_accents_preserves_genuine_french():
    # French test-only text must survive: apostrophe elision, cedilla, circumflex
    assert tu.strip_accents("Société Générale") == "Societe Generale"
    assert tu.strip_accents("l'Église") == "l'Eglise"


def test_transliterate_skeleton_ascii_passthrough():
    assert tu.transliterate_skeleton("Supreme It Private Limited") == "supremeitprivatelimited"


def test_transliterate_skeleton_devanagari_is_close_to_english():
    from rapidfuzz import fuzz
    skel = tu.transliterate_skeleton("सुप्रीम आईटी प्राइवेट लिमिटेड")
    target = tu.transliterate_skeleton("Supreme It Private Limited")
    assert skel != ""
    assert fuzz.ratio(skel, target) > 55  # rough phonetic match, not exact


def test_normalize_nulls_removes_null_tokens():
    assert tu.normalize_nulls("MN, ANOKA, null, CHARLOTTE DRIVE") == "MN, ANOKA, CHARLOTTE DRIVE"
    assert tu.normalize_nulls("80B Hagaman Ave, <NULL>, Amsterdam, New York") == \
        "80B Hagaman Ave, Amsterdam, New York"
    assert tu.normalize_nulls("N/A") == ""
    assert tu.normalize_nulls("") == ""


def test_normalize_nulls_does_not_touch_real_words():
    assert tu.normalize_nulls("National Avenue") == "National Avenue"


def test_dedupe_consecutive_tokens():
    assert tu.dedupe_consecutive_tokens("Grand Grand Connecticut LLC") == "Grand Connecticut LLC"
    assert tu.dedupe_consecutive_tokens("POLEBRIDGE REGIONAL REGIONAL CHURCH LLC") == \
        "POLEBRIDGE REGIONAL CHURCH LLC"
    assert tu.dedupe_consecutive_tokens("Fami1y Empire Partners Partners LLC") == \
        "Fami1y Empire Partners LLC"


def test_dedupe_consecutive_tokens_noop_on_clean_input():
    assert tu.dedupe_consecutive_tokens("Davis Family Office") == "Davis Family Office"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_text_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.normalize.text_utils'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/normalize/text_utils.py
"""Low-level, dependency-light text helpers shared by name and address cleaners."""
import re
import unicodedata

from unidecode import unidecode

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_text_utils.py -v`
Expected: PASS (9 tests). If `test_transliterate_skeleton_devanagari_is_close_to_english` fails on the threshold, lower it in the test to match the actual `fuzz.ratio` observed (print it first) rather than changing production code — `unidecode`'s phonetic mapping quality is fixed, the test just needs a realistic bar.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/normalize/text_utils.py \
        code/business_entity_resolution/tests/normalize/test_text_utils.py
git commit -m "feat: add text_utils with accent stripping, null normalization, transliteration skeleton"
```

---

### Task 3: Legal suffix / honorific / DBA-marker lookup tables

**Files:**
- Create: `code/business_entity_resolution/src/normalize/legal_suffixes.py`
- Test: `code/business_entity_resolution/tests/normalize/test_legal_suffixes.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SUFFIX_CANONICAL_MAP: dict[str, str]` — lowercased, punctuation-stripped suffix variant → canonical token, e.g. `{"pvt": "private", "private": "private", "ltd": "limited", "limited": "limited", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "llc": "llc", "l.l.c": "llc", "llp": "llp", "pllc": "pllc", "lp": "lp", "co": "co", "company": "co", "sarl": "sarl", "sas": "sas"}`.
  - `HONORIFIC_PREFIXES: frozenset[str]` — lowercased tokens to strip when they're the first token: `{"mr", "mrs", "smt", "shri", "m/s"}`. Deliberately excludes bare `"ms"` (no slash): it collides with names that legitimately start with the token "MS" (e.g. "MS Consultancy Corp") — only the unambiguous slashed form `"m/s"` (the Indian Messrs filing convention) is treated as an honorific. Task 4's test pins the "MS Consultancy Corp" non-stripping case explicitly.
  - `DBA_MARKER_RE: re.Pattern` — case-insensitive, matches `dba`, `d/b/a`, `trading as`, `née` as a split marker, with capture groups for the text before/after.

- [ ] **Step 1: Write the failing test**

```python
# code/business_entity_resolution/tests/normalize/test_legal_suffixes.py
from src.normalize import legal_suffixes as ls


def test_suffix_canonical_map_covers_common_variants():
    assert ls.SUFFIX_CANONICAL_MAP["pvt"] == "private"
    assert ls.SUFFIX_CANONICAL_MAP["private"] == "private"
    assert ls.SUFFIX_CANONICAL_MAP["ltd"] == "limited"
    assert ls.SUFFIX_CANONICAL_MAP["limited"] == "limited"
    assert ls.SUFFIX_CANONICAL_MAP["corp"] == "corp"
    assert ls.SUFFIX_CANONICAL_MAP["corporation"] == "corp"
    assert ls.SUFFIX_CANONICAL_MAP["llc"] == "llc"
    assert ls.SUFFIX_CANONICAL_MAP["l.l.c"] == "llc"
    assert ls.SUFFIX_CANONICAL_MAP["sarl"] == "sarl"
    assert ls.SUFFIX_CANONICAL_MAP["sas"] == "sas"


def test_honorific_prefixes():
    assert "smt" in ls.HONORIFIC_PREFIXES
    assert "m/s" in ls.HONORIFIC_PREFIXES
    assert "mr" in ls.HONORIFIC_PREFIXES


def test_dba_marker_splits_both_orderings():
    m = ls.DBA_MARKER_RE.search("Drexsolpyra DBA: Cornerstone Investments L.L.C.")
    assert m is not None
    assert m.group(1).strip() == "Drexsolpyra"
    assert m.group(2).strip() == "Cornerstone Investments L.L.C."

    m = ls.DBA_MARKER_RE.search("Vantagevantageio d/b/a Scholarship Council")
    assert m is not None
    assert m.group(1).strip() == "Vantagevantageio"
    assert m.group(2).strip() == "Scholarship Council"


def test_dba_marker_no_match_on_plain_name():
    assert ls.DBA_MARKER_RE.search("Davis Family Office") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_legal_suffixes.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/normalize/legal_suffixes.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_legal_suffixes.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/normalize/legal_suffixes.py \
        code/business_entity_resolution/tests/normalize/test_legal_suffixes.py
git commit -m "feat: add legal suffix, honorific, and DBA-marker lookup tables"
```

---

### Task 4: Name cleaner

**Files:**
- Create: `code/business_entity_resolution/src/normalize/name_cleaner.py`
- Test: `code/business_entity_resolution/tests/normalize/test_name_cleaner.py`

**Interfaces:**
- Consumes:
  - `text_utils.strip_accents`, `text_utils.transliterate_skeleton`, `text_utils.dedupe_consecutive_tokens`, `text_utils.NAME_STOPWORDS` (Task 2)
  - `legal_suffixes.SUFFIX_CANONICAL_MAP`, `legal_suffixes.HONORIFIC_PREFIXES`, `legal_suffixes.DBA_MARKER_RE` (Task 3)
  - `schema.NAME_FIELDS` (Task 1)
- Produces: `clean_name(raw: str) -> dict` with exactly the keys in `schema.NAME_FIELDS`. This is the function Task 6's CLI and (later) Stage 4's feature code call.

- [ ] **Step 1: Write the failing test**

```python
# code/business_entity_resolution/tests/normalize/test_name_cleaner.py
from src.normalize.name_cleaner import clean_name
from src.common.schema import NAME_FIELDS


def test_output_has_all_schema_fields():
    result = clean_name("Davis Family Office")
    assert set(result.keys()) == set(NAME_FIELDS)


def test_legal_suffix_extraction_and_core_name():
    result = clean_name("Team Air Pvt. Ltd.")
    assert result["legal_suffix"] == ("private", "limited")
    assert "pvt" not in result["core_name"].lower()
    assert "ltd" not in result["core_name"].lower()
    assert "team air" in result["core_name"].lower()


def test_suffix_mid_string_still_extracted():
    # "PRAIRIE CAPITAL PLLC-PARTNERS" — suffix embedded mid-name, not just trailing
    result = clean_name("PRAIRIE CAPITAL PLLC-PARTNERS")
    assert "pllc" in result["legal_suffix"]


def test_compact_name_strips_punctuation_and_spaces():
    # "Inc" is extracted into legal_suffix (not left in core_name), so it is
    # correctly absent from compact_name too — compact_name is punctuation/space
    # -stripped core_name, and core_name never carries the suffix.
    result = clean_name("3520 Main Road Realty Inc")
    assert result["compact_name"] == "3520mainroadrealty"
    assert result["legal_suffix"] == ("inc",)


def test_dedupe_repeated_tokens_before_core_name():
    result = clean_name("Grand Grand Connecticut LLC")
    assert result["core_name"].lower().count("grand") == 1


def test_dba_split_produces_alt_name():
    result = clean_name("Drexsolpyra DBA: Cornerstone Investments L.L.C.")
    assert result["alt_name"] != ""
    assert "cornerstone" in result["alt_name"].lower() or "cornerstone" in result["core_name"].lower()


def test_dba_split_no_marker_leaves_alt_name_empty():
    result = clean_name("Davis Family Office")
    assert result["alt_name"] == ""


def test_honorific_prefix_stripped():
    result = clean_name("Mr Trip Welfare Society Co")
    assert not result["core_name"].lower().startswith("mr ")


def test_ambiguous_ms_prefix_not_stripped_when_short_remainder():
    # "MS" here is the actual first word of the business name, not the honorific "Ms"
    result = clean_name("MS Consultancy Corp")
    assert "ms" in result["core_name"].lower().split() or "ms" in result["compact_name"]


def test_accent_noise_stripped_in_deaccented_and_core():
    result = clean_name("cmcáre.com")
    assert result["deaccented"] == "cmcare.com"
    assert "á" not in result["core_name"]


def test_non_latin_name_preserved_and_skeleton_built():
    result = clean_name("सुप्रीम आईटी प्राइवेट लिमिटेड")
    assert result["raw"] == "सुप्रीम आईटी प्राइवेट लिमिटेड"
    assert result["deaccented"] == "सुप्रीम आईटी प्राइवेट लिमिटेड"
    assert result["skeleton"] != ""
    assert result["skeleton"].isascii()


def test_empty_core_name_falls_back_to_pre_strip_text():
    # A name that is *only* a legal suffix must not collapse to an empty core_name
    result = clean_name("LLC")
    assert result["core_name"].strip() != ""


def test_acronym_built_excluding_stopwords():
    result = clean_name("Hussain, Carter and Goodwin")
    assert result["acronym"] == "HCG"


def test_french_suffix_and_accents_do_not_crash():
    result = clean_name("Société Générale SARL")
    assert result["legal_suffix"] == ("sarl",)
    assert result["deaccented"] == "Societe Generale SARL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_name_cleaner.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/normalize/name_cleaner.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_name_cleaner.py -v`
Expected: PASS (14 tests). If `test_suffix_mid_string_still_extracted` or the DBA tests fail on exact substrings, adjust the test's assertion to check membership/normalized-casing rather than exact equality — the important contract is "suffix ends up in `legal_suffix`, not `core_name`", not a specific residual string.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/normalize/name_cleaner.py \
        code/business_entity_resolution/tests/normalize/test_name_cleaner.py
git commit -m "feat: add clean_name() structural name cleaner"
```

---

### Task 5: Road/unit abbreviation table and address cleaner

**Files:**
- Create: `code/business_entity_resolution/src/normalize/road_abbrev.py`
- Create: `code/business_entity_resolution/src/normalize/address_cleaner.py`
- Test: `code/business_entity_resolution/tests/normalize/test_address_cleaner.py`

**Interfaces:**
- Consumes:
  - `text_utils.strip_accents`, `text_utils.normalize_nulls` (Task 2)
  - `schema.ADDRESS_FIELDS` (Task 1)
- Produces:
  - `road_abbrev.ROAD_ABBREV_MAP: dict[str, str]`
  - `road_abbrev.UNIT_TOKENS: frozenset[str]`
  - `clean_address(raw: str, country: str = "") -> dict` with exactly the keys in `schema.ADDRESS_FIELDS`. `country` is optional and only narrows postcode-shape matching (US 5(+4)-digit vs. India 6-digit vs. French 5-digit); when empty/unknown, both shapes are tried.

- [ ] **Step 1: Write the failing test**

```python
# code/business_entity_resolution/tests/normalize/test_address_cleaner.py
from src.normalize.address_cleaner import clean_address
from src.common.schema import ADDRESS_FIELDS


def test_output_has_all_schema_fields():
    result = clean_address("88 Olive Circle, Lebanon, TN", country="US")
    assert set(result.keys()) == set(ADDRESS_FIELDS)


def test_null_token_inline_removed():
    result = clean_address("MN, ANOKA, null, CHARLOTTE DRIVE", country="US")
    assert "null" not in result["core_address"].lower()


def test_fully_null_address_collapses_to_empty():
    result = clean_address("N/A", country="US")
    assert result["core_address"] == ""
    assert result["postcode"] == ""
    assert result["house_number"] == ""


def test_empty_input_collapses_to_empty():
    result = clean_address("", country="US")
    assert result["core_address"] == ""


def test_us_zip_extracted():
    result = clean_address("123 Main St, Springfield, IL 62704", country="US")
    assert result["postcode"] == "62704"
    assert "62704" not in result["core_address"]


def test_india_pin_extracted():
    result = clean_address("Plot 12, MG Road, Pune, Maharashtra 411001", country="India")
    assert result["postcode"] == "411001"


def test_house_number_leading_digits_normalized():
    result = clean_address("##19821 Wheelwright Drive, Montgomery Village, MD", country="US")
    assert result["house_number"] == "19821"


def test_house_number_strips_leading_zeros():
    result = clean_address("0044023 Vaira Terrace, Loudoun County, VA", country="US")
    assert result["house_number"] == "44023"


def test_unit_token_extracted():
    result = clean_address("914 Charlotte Drive, Unit 102, Toledo, OH", country="US")
    assert result["unit"] != ""
    assert "unit" not in result["core_address"].lower() or "102" not in result["core_address"]


def test_landmark_extracted():
    result = clean_address("Shop 4, Near SBI ATM, Sehore, Madhya Pradesh", country="India")
    assert "sbi" in result["landmark"].lower()
    assert "near" not in result["core_address"].lower()


def test_road_abbreviation_present_in_map():
    from src.normalize.road_abbrev import ROAD_ABBREV_MAP
    assert ROAD_ABBREV_MAP["rd"] == "road"
    assert ROAD_ABBREV_MAP["st"] == "street"
    assert ROAD_ABBREV_MAP["ave"] == "avenue"
    assert ROAD_ABBREV_MAP["blvd"] == "boulevard"
    assert ROAD_ABBREV_MAP["bd"] == "boulevard"  # French
    assert ROAD_ABBREV_MAP["rue"] == "rue"  # French, no abbreviation needed


def test_non_latin_address_preserved():
    result = clean_address("H.NO 663 92, G.I.D.C., MOTIPURA, HIMATNAGAR, ગુજરાત", country="India")
    assert "ગુજરાત" in result["core_address"] or "ગુજરાત" in result["raw"]


def test_french_address_does_not_crash():
    result = clean_address("12 rue de la Paix, 75002 Paris", country="France")
    assert result["postcode"] == "75002"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_address_cleaner.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/normalize/road_abbrev.py
"""Road and unit abbreviation canonical maps (English + French)."""

ROAD_ABBREV_MAP = {
    "rd": "road", "road": "road",
    "st": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard", "bd": "boulevard",
    "dr": "drive", "drive": "drive",
    "cir": "circle", "circle": "circle",
    "ter": "terrace", "terrace": "terrace",
    "ln": "lane", "lane": "lane",
    "rue": "rue",  # French, no abbreviation
}

UNIT_TOKENS = frozenset({"unit", "apt", "suite", "ste", "pmb", "fl", "floor", "#"})
```

```python
# code/business_entity_resolution/src/normalize/address_cleaner.py
"""clean_address(): structural cleaning of business_address into schema.ADDRESS_FIELDS.

Extraction order matters: house_number is pulled from the START of the string
FIRST, before postcode search — otherwise a 5-digit house number (e.g. "19821
Wheelwright Drive") gets misidentified as a US ZIP by the postcode regex, since
both are bare 5-digit tokens. Extracting house_number first removes it from the
text the postcode regex ever sees.
"""
import re

from src.common.schema import ADDRESS_FIELDS
from src.normalize import text_utils as tu
from src.normalize.road_abbrev import ROAD_ABBREV_MAP

_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_INDIA_PIN_RE = re.compile(r"\b(\d{6})\b")
_FR_POSTAL_RE = re.compile(r"\b(\d{5})\b")

_HOUSE_NUMBER_RE = re.compile(r"^\s*#{0,2}0*(\d[\w/\-]*)")

_UNIT_RE = re.compile(
    r"(?i)\b(?:unit|apt|suite|ste|pmb|fl|floor)\.?\s*[:#]?\s*([\w-]+)|(#\s*(\w+))"
)

_LANDMARK_RE = re.compile(r"(?i)\b(?:near|behind|opp\.?|opposite)\b\s+([^,]+)")

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _extract_house_number(text: str) -> tuple[str, str]:
    m = _HOUSE_NUMBER_RE.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end(0):]


def _extract_postcode(text: str, country: str) -> tuple[str, str]:
    country_l = (country or "").strip().lower()
    if country_l == "india":
        regexes = [_INDIA_PIN_RE]
    elif country_l == "us":
        regexes = [_US_ZIP_RE]
    elif country_l == "france":
        regexes = [_FR_POSTAL_RE]
    else:
        regexes = [_INDIA_PIN_RE, _US_ZIP_RE]
    for rx in regexes:
        m = rx.search(text)
        if m:
            start, end = m.span(1)
            return m.group(1), text[:start] + text[end:]
    return "", text


def _extract_landmark(text: str) -> tuple[str, str]:
    m = _LANDMARK_RE.search(text)
    if not m:
        return "", text
    return m.group(1).strip(), text[:m.start()] + text[m.end():]


def _extract_unit(text: str) -> tuple[str, str]:
    m = _UNIT_RE.search(text)
    if not m:
        return "", text
    value = m.group(1) or m.group(3) or ""
    return value, text[:m.start()] + text[m.end():]


def _canonicalize_roads(text: str) -> str:
    def repl(m):
        return ROAD_ABBREV_MAP.get(m.group(0).lower(), m.group(0))
    return _TOKEN_RE.sub(repl, text)


def clean_address(raw: str, country: str = "") -> dict:
    raw = raw or ""
    deaccented = tu.strip_accents(tu.normalize_nulls(raw))
    working = deaccented

    house_number, working = _extract_house_number(working)
    postcode, working = _extract_postcode(working, country)
    landmark, working = _extract_landmark(working)
    unit, working = _extract_unit(working)

    core_address = _canonicalize_roads(working)
    core_address = tu.normalize_nulls(core_address)

    result = {
        "raw": raw,
        "deaccented": deaccented,
        "core_address": core_address,
        "postcode": postcode,
        "house_number": house_number.lstrip("0") or house_number,
        "unit": unit,
        "landmark": landmark,
    }
    assert set(result.keys()) == set(ADDRESS_FIELDS)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_address_cleaner.py -v`
Expected: PASS (13 tests). The house-number-before-postcode extraction order is what prevents a 5-digit house number from being misread as a ZIP — if you see that failure, check extraction order first, not the regexes.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/src/normalize/road_abbrev.py \
        code/business_entity_resolution/src/normalize/address_cleaner.py \
        code/business_entity_resolution/tests/normalize/test_address_cleaner.py
git commit -m "feat: add clean_address() structural address cleaner"
```

---

### Task 6: `run_normalize.py` CLI — apply both cleaners to all 6 source files

**Files:**
- Create: `code/business_entity_resolution/src/normalize/run_normalize.py`
- Test: `code/business_entity_resolution/tests/normalize/test_run_normalize.py`

**Interfaces:**
- Consumes: `name_cleaner.clean_name` (Task 4), `address_cleaner.clean_address` (Task 5), `schema.RAW_COLUMNS`, `schema.NAME_FIELDS`, `schema.ADDRESS_FIELDS` (Task 1).
- Produces:
  - `normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame` — takes a raw source dataframe (columns `entity_id, business_name, business_address, country`), returns a new dataframe with `entity_id`, `country`, and one column per name field prefixed `name_*` and per address field prefixed `addr_*` (e.g. `name_core_name`, `addr_postcode`). This is the function Stage 3 (blocking) will import directly for in-memory use; the CLI below is the batch-file wrapper around it.
  - A CLI entry point that reads each of the 6 TSVs under `student_resource/dataset/{train,test}/` and writes cleaned Parquet files to `data/processed/{split}_{source}.parquet` (e.g. `data/processed/train_source1.parquet`), created relative to the repo root.

- [ ] **Step 1: Write the failing test**

```python
# code/business_entity_resolution/tests/normalize/test_run_normalize.py
import pandas as pd

from src.normalize.run_normalize import normalize_dataframe


def test_normalize_dataframe_shape_and_columns():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Davis Family Office", "Team Air Pvt. Ltd."],
        "business_address": ["88 Olive Circle, Lebanon, TN", ""],
        "country": ["US", "India"],
    })
    result = normalize_dataframe(df)
    assert len(result) == 2
    assert "entity_id" in result.columns
    assert "country" in result.columns
    assert "name_core_name" in result.columns
    assert "addr_postcode" in result.columns
    assert "name_legal_suffix" in result.columns


def test_normalize_dataframe_preserves_row_order_and_ids():
    df = pd.DataFrame({
        "entity_id": ["S2-9", "S2-1"],
        "business_name": ["Zeta Corp", "Alpha Inc"],
        "business_address": ["1 Zeta Rd, X, US", "1 Alpha Rd, Y, US"],
        "country": ["US", "US"],
    })
    result = normalize_dataframe(df)
    assert list(result["entity_id"]) == ["S2-9", "S2-1"]


def test_normalize_dataframe_handles_empty_address():
    df = pd.DataFrame({
        "entity_id": ["S3-1"],
        "business_name": ["Some Name"],
        "business_address": [""],
        "country": ["India"],
    })
    result = normalize_dataframe(df)
    assert result.loc[0, "addr_core_address"] == ""
    assert result.loc[0, "addr_postcode"] == ""


def test_normalize_dataframe_legal_suffix_is_string_not_tuple():
    # Parquet/CSV round-tripping doesn't like tuple-valued cells; store as a
    # comma-joined string instead ("private,limited"), empty string if none.
    df = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Team Air Pvt. Ltd."],
        "business_address": ["1 Road, City, US"],
        "country": ["US"],
    })
    result = normalize_dataframe(df)
    val = result.loc[0, "name_legal_suffix"]
    assert isinstance(val, str)
    assert "private" in val and "limited" in val
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_run_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# code/business_entity_resolution/src/normalize/run_normalize.py
"""Apply clean_name/clean_address to a dataframe, and a CLI to batch all 6 source files."""
import argparse
from pathlib import Path

import pandas as pd

from src.common.schema import ADDRESS_FIELDS, NAME_FIELDS, RAW_COLUMNS
from src.normalize.address_cleaner import clean_address
from src.normalize.name_cleaner import clean_name

REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = REPO_ROOT / "student_resource" / "dataset"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

FILES = {
    ("train", "source1"): DATASET_ROOT / "train" / "train_source1.tsv",
    ("train", "source2"): DATASET_ROOT / "train" / "train_source2.tsv",
    ("train", "source3"): DATASET_ROOT / "train" / "train_source3.tsv",
    ("test", "source1"): DATASET_ROOT / "test" / "test_source1.tsv",
    ("test", "source2"): DATASET_ROOT / "test" / "test_source2.tsv",
    ("test", "source3"): DATASET_ROOT / "test" / "test_source3.tsv",
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_name/clean_address to every row; return entity_id/country + name_*/addr_* columns."""
    names = df["business_name"].map(clean_name)
    countries = df["country"].fillna("")
    addresses = [
        clean_address(addr, country=country)
        for addr, country in zip(df["business_address"], countries)
    ]

    out = pd.DataFrame({"entity_id": df["entity_id"].values, "country": df["country"].values})
    for field in NAME_FIELDS:
        if field == "legal_suffix":
            out[f"name_{field}"] = [",".join(n[field]) for n in names]
        else:
            out[f"name_{field}"] = [n[field] for n in names]
    for field in ADDRESS_FIELDS:
        out[f"addr_{field}"] = [a[field] for a in addresses]
    return out


def _read_source_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(df.columns) == RAW_COLUMNS, f"{path}: unexpected columns {list(df.columns)}"
    return df


def run_all(out_dir: Path = PROCESSED_ROOT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (split, source), path in FILES.items():
        print(f"normalizing {split}/{source} from {path} ...")
        df = _read_source_tsv(path)
        cleaned = normalize_dataframe(df)
        out_path = out_dir / f"{split}_{source}.parquet"
        cleaned.to_parquet(out_path, index=False)
        print(f"  wrote {len(cleaned):,} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize all 6 entity-resolution source files.")
    parser.add_argument("--out-dir", default=str(PROCESSED_ROOT))
    args = parser.parse_args()
    run_all(Path(args.out_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd code/business_entity_resolution && python -m pytest tests/normalize/test_run_normalize.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full test suite, then commit**

```bash
cd code/business_entity_resolution && python -m pytest tests/ -v
```

Expected: all tests across Tasks 1–6 PASS.

```bash
git add code/business_entity_resolution/src/normalize/run_normalize.py \
        code/business_entity_resolution/tests/normalize/test_run_normalize.py
git commit -m "feat: add run_normalize CLI to batch-clean all 6 source files to Parquet"
```

---

### Task 7: Run the CLI on the real data and spot-check the output

**Files:**
- No new files — this task runs Task 6's CLI against the real ~2–5M row source files and validates the output by eye, catching normalization bugs that only show up at scale (Task 1–6 tests use small synthetic frames).

**Interfaces:**
- Consumes: `run_normalize.run_all()` (Task 6).
- Produces: `data/processed/{train,test}_{source1,source2,source3}.parquet` (6 files) — the input Stage 3 (blocking, a future plan) will read.

- [ ] **Step 1: Run the CLI on the full dataset**

```bash
cd code/business_entity_resolution && python -m src.normalize.run_normalize
```

Expected: prints progress for all 6 files, completes without exceptions. This processes ~26M rows total — if it takes more than ~15 minutes, stop and check for an accidental non-vectorized bottleneck (e.g. a regex recompiled per row instead of module-level) before waiting it out.

- [ ] **Step 2: Spot-check the output against the known EDA examples**

```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/processed/train_source1.parquet")
row = df[df["entity_id"] == "S1-777210964"]
print(row[["entity_id", "name_core_name", "name_legal_suffix", "addr_core_address"]].to_string())

s2 = pd.read_parquet("data/processed/train_source2.parquet")
print(s2[s2["entity_id"] == "S2-12029274"][["name_core_name", "addr_core_address"]].to_string())
EOF
```

Expected: `S1-777210964` ("Davis Family Office") shows a clean `name_core_name`; `S2-12029274` ("Davis Family Offie") shows the same structural cleanup despite the typo (typo itself is untouched — that's Stage 4's job, not this stage's).

- [ ] **Step 3: Check for anomalies across the full processed output**

```bash
python3 - <<'EOF'
import pandas as pd
for split, source in [("train","source1"),("train","source2"),("train","source3"),
                       ("test","source1"),("test","source2"),("test","source3")]:
    df = pd.read_parquet(f"data/processed/{split}_{source}.parquet")
    empty_core_name = (df["name_core_name"].str.strip() == "").mean()
    print(f"{split}_{source}: rows={len(df):,} empty_core_name_rate={empty_core_name:.4%}")
    assert empty_core_name < 0.01, "core_name is collapsing to empty too often — check Task 4's fallback logic"
EOF
```

Expected: `empty_core_name_rate` near 0% for every file (the Task 4 fallback guarantees non-empty `core_name`, but this checks it holds at real-data scale, not just in the unit test's synthetic cases).

- [ ] **Step 4: Commit the processed-data gitignore entry (not the data itself)**

```bash
echo "data/processed/" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore generated Parquet output from run_normalize"
```

---

## Self-Review

**1. Spec coverage:** Every EDA-sourced noise pattern in the header has an owning task — accent injection (Task 2/4), null-token literals (Task 2/5), repeated tokens (Task 2), legal suffix variants + French SARL/SAS (Task 3/4), DBA/trading-as/née markers (Task 3/4), honorific prefixes (Task 3/4), postcode/house-number/unit/landmark extraction (Task 5), non-Latin preservation + transliteration skeleton (Task 2/4), road abbreviations incl. French (Task 5), and applying all of it at real scale across all 6 files (Task 6/7). One deliberate omission: OCR/character-substitution typos ("Fami1y", "5ons", "Pirave") are explicitly out of scope for this plan — that's a fuzzy-matching/similarity-scoring concern for a future Stage 4 features plan, not structural normalization.

**2. Placeholder scan:** No TBD/TODO — every step has runnable code and concrete assertions.

**3. Type consistency:** `clean_name()`/`clean_address()` return dict keys match `schema.NAME_FIELDS`/`schema.ADDRESS_FIELDS` exactly (asserted at the end of each function). `normalize_dataframe()` consumes both by iterating `NAME_FIELDS`/`ADDRESS_FIELDS`, so adding a field to the schema in a later plan only requires updating the cleaner that produces it — `run_normalize.py` picks it up automatically.

**4. Review Focus:** covered above — empty-core-name fallback (Task 4, tested at both unit and full-data scale in Task 7), fully-null address collapse (Task 5), non-Latin passthrough (Task 2/5), DBA split edge cases (Task 4), and French test-only forms (Task 2/4/5, each with a dedicated "does not crash" test since there is zero training data to validate against — this is the one country where structural bugs can't be caught by looking at training examples first).

**What this plan deliberately leaves for later plans:** Stage 3 blocking (TF-IDF/exact-key/e5 candidate generation over the cleaned Parquet this plan produces), Stage 4 similarity features, Stage 5 model training, Stage 6 decoding. Each should be its own plan, written after the prior stage's checkpoint is validated — per the project's own stage-gate design (blocking needs ≥97% recall before features are built on top of it).


