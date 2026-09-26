"""Vectorized feature extraction for candidate pairs.

Extracts string similarity, numeric, postal, token overlap, flag, and structural features
for every (s1_id, cand_id) pair. Optimized for vectorized multi-million row processing.
"""
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz


FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_skeleton_ratio",
    "name_deaccented_ratio",
    "name_jaro_winkler",
    "name_ngram_jaccard",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_deaccented_ratio",
    "addr_jaro_winkler",
    "addr_ngram_jaccard",
    "postcode_exact",
    "postcode_ratio",
    "postcode_missing",
    "housenum_exact",
    "housenum_missing",
    "unit_exact",
    "country_match",
    "acronym_match",
    "alt_name_ratio",
    "name_len_ratio",
    "addr_len_ratio",
    "n_methods",
    "cand_source",
]


def _char_ngram_jaccard(s1_arr, s2_arr, n=3):
    scores = []
    for a, b in zip(s1_arr, s2_arr):
        if not a or not b:
            scores.append(0.0)
            continue
        g1 = set(a[i : i + n] for i in range(max(1, len(a) - n + 1)))
        g2 = set(b[i : i + n] for i in range(max(1, len(b) - n + 1)))
        union_len = len(g1 | g2)
        if union_len == 0:
            scores.append(0.0)
        else:
            scores.append(len(g1 & g2) / union_len)
    return scores


def build_feature_table(cands: pd.DataFrame, all_records: pd.DataFrame) -> pd.DataFrame:
    """Compute vectorized feature matrix for candidate pairs."""
    s1_info = all_records.loc[cands["s1_id"].values].reset_index(drop=True)
    cand_info = all_records.loc[cands["cand_id"].values].reset_index(drop=True)

    # Extract strings as lists/arrays for rapid execution
    s1_name_compact = s1_info["name_compact_name"].tolist()
    cand_name_compact = cand_info["name_compact_name"].tolist()

    s1_name_core = s1_info["name_core_name"].tolist()
    cand_name_core = cand_info["name_core_name"].tolist()

    s1_name_skel = s1_info["name_skeleton"].tolist()
    cand_name_skel = cand_info["name_skeleton"].tolist()

    s1_name_deacc = s1_info["name_deaccented"].tolist()
    cand_name_deacc = cand_info["name_deaccented"].tolist()

    s1_addr_core = s1_info["addr_core_address"].tolist()
    cand_addr_core = cand_info["addr_core_address"].tolist()

    s1_addr_deacc = s1_info["addr_deaccented"].tolist()
    cand_addr_deacc = cand_info["addr_deaccented"].tolist()

    # Name rapidfuzz metrics
    name_ratio = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_name_compact, cand_name_compact)]
    name_token_sort_ratio = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_name_core, cand_name_core)]
    name_token_set_ratio = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_name_core, cand_name_core)]
    name_partial_ratio = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_name_compact, cand_name_compact)]
    name_skeleton_ratio = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_name_skel, cand_name_skel)]
    name_deaccented_ratio = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_name_deacc, cand_name_deacc)]
    name_jaro_winkler = [distance.JaroWinkler.similarity(a, b) for a, b in zip(s1_name_compact, cand_name_compact)]
    name_ngram_jaccard = _char_ngram_jaccard(s1_name_compact, cand_name_compact, n=3)

    # Address rapidfuzz metrics
    addr_ratio = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_addr_core, cand_addr_core)]
    addr_token_sort_ratio = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_addr_core, cand_addr_core)]
    addr_token_set_ratio = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_addr_core, cand_addr_core)]
    addr_partial_ratio = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_addr_core, cand_addr_core)]
    addr_deaccented_ratio = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_addr_deacc, cand_addr_deacc)]
    addr_jaro_winkler = [distance.JaroWinkler.similarity(a, b) for a, b in zip(s1_addr_core, cand_addr_core)]
    addr_ngram_jaccard = _char_ngram_jaccard(s1_addr_core, cand_addr_core, n=3)

    # Postal and numeric fields
    s1_pc = s1_info["addr_postcode"].values
    cand_pc = cand_info["addr_postcode"].values
    postcode_exact = ((s1_pc != "") & (s1_pc == cand_pc)).astype(float)
    postcode_missing = ((s1_pc == "") | (cand_pc == "")).astype(float)
    postcode_ratio = [
        fuzz.ratio(a, b) / 100.0 if a and b else 0.0 for a, b in zip(s1_pc, cand_pc)
    ]

    s1_hn = s1_info["addr_house_number"].values
    cand_hn = cand_info["addr_house_number"].values
    housenum_exact = ((s1_hn != "") & (s1_hn == cand_hn)).astype(float)
    housenum_missing = ((s1_hn == "") | (cand_hn == "")).astype(float)

    s1_unit = s1_info["addr_unit"].values
    cand_unit = cand_info["addr_unit"].values
    unit_exact = ((s1_unit != "") & (s1_unit == cand_unit)).astype(float)

    # Country & acronym flags
    s1_country = s1_info["country"].values
    cand_country = cand_info["country"].values
    country_match = (s1_country == cand_country).astype(float)

    s1_acronym = s1_info["name_acronym"].values
    cand_acronym = cand_info["name_acronym"].values
    acronym_match = ((s1_acronym != "") & (s1_acronym == cand_acronym)).astype(float)

    # Alternate name match
    s1_alt = s1_info["name_alt_name"].tolist()
    cand_alt = cand_info["name_alt_name"].tolist()
    alt_name_ratio = [
        max(fuzz.ratio(a, b1) / 100.0, fuzz.ratio(a1, b) / 100.0) if a or a1 else 0.0
        for a, b, a1, b1 in zip(s1_alt, cand_name_core, s1_name_core, cand_alt)
    ]

    # Length ratio features
    s1_nlen = np.array([len(x) for x in s1_name_compact])
    cand_nlen = np.array([len(x) for x in cand_name_compact])
    max_nlen = np.maximum(s1_nlen, cand_nlen)
    min_nlen = np.minimum(s1_nlen, cand_nlen)
    name_len_ratio = np.where(max_nlen > 0, min_nlen / max_nlen, 0.0)

    s1_alen = np.array([len(x) for x in s1_addr_core])
    cand_alen = np.array([len(x) for x in cand_addr_core])
    max_alen = np.maximum(s1_alen, cand_alen)
    min_alen = np.minimum(s1_alen, cand_alen)
    addr_len_ratio = np.where(max_alen > 0, min_alen / max_alen, 0.0)

    # Candidate source (2 for S2, 3 for S3)
    cand_ids = cands["cand_id"].values
    cand_source = np.array([2.0 if str(cid).startswith("S2-") else 3.0 for cid in cand_ids])

    # n_methods feature
    n_methods = cands["n_methods"].values.astype(float) if "n_methods" in cands.columns else np.ones(len(cands))

    out = cands.reset_index(drop=True).copy()
    out["name_ratio"] = name_ratio
    out["name_token_sort_ratio"] = name_token_sort_ratio
    out["name_token_set_ratio"] = name_token_set_ratio
    out["name_partial_ratio"] = name_partial_ratio
    out["name_skeleton_ratio"] = name_skeleton_ratio
    out["name_deaccented_ratio"] = name_deaccented_ratio
    out["name_jaro_winkler"] = name_jaro_winkler
    out["name_ngram_jaccard"] = name_ngram_jaccard

    out["addr_ratio"] = addr_ratio
    out["addr_token_sort_ratio"] = addr_token_sort_ratio
    out["addr_token_set_ratio"] = addr_token_set_ratio
    out["addr_partial_ratio"] = addr_partial_ratio
    out["addr_deaccented_ratio"] = addr_deaccented_ratio
    out["addr_jaro_winkler"] = addr_jaro_winkler
    out["addr_ngram_jaccard"] = addr_ngram_jaccard

    out["postcode_exact"] = postcode_exact
    out["postcode_ratio"] = postcode_ratio
    out["postcode_missing"] = postcode_missing

    out["housenum_exact"] = housenum_exact
    out["housenum_missing"] = housenum_missing
    out["unit_exact"] = unit_exact

    out["country_match"] = country_match
    out["acronym_match"] = acronym_match
    out["alt_name_ratio"] = alt_name_ratio

    out["name_len_ratio"] = name_len_ratio
    out["addr_len_ratio"] = addr_len_ratio

    out["n_methods"] = n_methods
    out["cand_source"] = cand_source

    out["s1_country"] = s1_country

    return out
