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
