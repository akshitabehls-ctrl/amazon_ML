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
