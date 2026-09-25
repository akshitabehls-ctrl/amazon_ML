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
