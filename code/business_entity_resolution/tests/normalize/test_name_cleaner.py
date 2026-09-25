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
