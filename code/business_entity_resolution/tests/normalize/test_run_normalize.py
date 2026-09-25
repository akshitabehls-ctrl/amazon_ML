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
