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
