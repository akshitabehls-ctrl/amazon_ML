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
