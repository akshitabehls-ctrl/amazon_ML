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
