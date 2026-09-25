"""Stage 3 blocking: exact-key candidate generation, unioned across several keys.

Iteration-1 scope: exact-match blocking keys (compact name, transliteration
skeleton, name prefix, acronym, postcode+house_number, house_number+first
address word) PLUS a set of character-3-gram MinHash signature keys on the
name and address. Pure exact-key blocking alone measured 68% pair recall
against ground truth (see block.py git history) — well short of the 97%
target — because this dataset's typos are scattered through the string, not
confined to a prefix, so exact/prefix keys miss them. MinHash signatures
(the minimum hash of a record's n-gram set, for several independent hash
seeds) survive scattered typos as long as at least one n-gram is untouched,
and — like the exact keys — are implemented as a plain pandas merge on
(country, signature), so they scale the same way without needing TF-IDF +
nearest-neighbor search infrastructure.

Candidates from S2 and S3 are generated and capped independently (top-K each),
so one source can't crowd out the other, per the project plan.
"""
import zlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

MAX_GROUP_SIZE = 100  # drop overly generic keys (e.g. very common short names) before merging
MAX_CANDIDATES_PER_SOURCE = 10  # cap per (s1, source) after union, per plan's top-K-per-source rule
# Lowered from 200/50 after the first full run produced a 127M-row candidate file that
# OOM'd this 16GB machine downstream (feature scoring, recap). 10/source x 2 sources x
# ~2.2M S1 rows bounds the working set to ~tens of millions instead of ~127M.

_MINHASH_SEEDS = ["mh0", "mh1", "mh2", "mh3"]
_NGRAM_N = 3


def _prefix4(s: pd.Series) -> pd.Series:
    return s.str[:4]


def _first_addr_word(s: pd.Series) -> pd.Series:
    return s.str.split(",").str[0].str.split().str[0].fillna("").str.lower()


def _ngrams(s: str, n: int = _NGRAM_N) -> list[str]:
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _minhash_sigs(s: str) -> tuple:
    """Minimum CRC32 hash of the string's n-gram set, once per seed.

    Deterministic across process runs (unlike builtin hash(), which is
    randomized per-process for strings) — required since blocking train and
    test run as separate invocations and must produce comparable signatures.
    Returns -1 for every seed when the string has no n-grams (empty/blank).
    """
    grams = _ngrams(s)
    if not grams:
        return tuple(-1 for _ in _MINHASH_SEEDS)
    return tuple(min(zlib.crc32((seed + g).encode()) for g in grams) for seed in _MINHASH_SEEDS)


def _add_minhash_columns(df: pd.DataFrame, source_col: str, prefix: str) -> None:
    sigs = df[source_col].map(_minhash_sigs)
    for i, seed in enumerate(_MINHASH_SEEDS):
        df[f"{prefix}_{seed}"] = sigs.map(lambda t, i=i: t[i])


def _drop_oversized_groups(df: pd.DataFrame, key_cols: list[str], max_size: int) -> pd.DataFrame:
    """Drop rows whose (key_cols) group is bigger than max_size, on both sides before merging.

    A key like an empty string or a very generic short name would otherwise fan out
    into a huge cross-join; this keeps merges bounded without a global cap.
    """
    sizes = df.groupby(key_cols)[key_cols[0]].transform("size")
    return df[sizes <= max_size]


def _block_one_key(s1: pd.DataFrame, other: pd.DataFrame, key_cols: list[str], method: str) -> pd.DataFrame:
    s1_f = s1[(s1[key_cols[-1]] != "") if len(key_cols) == 1 else (s1[key_cols[1]] != "")]
    other_f = other[(other[key_cols[-1]] != "") if len(key_cols) == 1 else (other[key_cols[1]] != "")]
    s1_f = _drop_oversized_groups(s1_f, key_cols, MAX_GROUP_SIZE)
    other_f = _drop_oversized_groups(other_f, key_cols, MAX_GROUP_SIZE)
    merged = s1_f[["entity_id"] + key_cols].merge(
        other_f[["entity_id"] + key_cols], on=key_cols, suffixes=("_s1", "_cand")
    )
    merged["method"] = method
    return merged[["entity_id_s1", "entity_id_cand", "method"]].rename(
        columns={"entity_id_s1": "s1_id", "entity_id_cand": "cand_id"}
    )


def build_candidates_one_source(s1: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Union several exact-key and MinHash-signature blocking strategies between S1 and one other source."""
    s1 = s1.copy()
    other = other.copy()
    for df in (s1, other):
        df["_prefix4"] = _prefix4(df["name_compact_name"])
        df["_addr_word"] = _first_addr_word(df["addr_core_address"])
        _add_minhash_columns(df, "name_compact_name", "_name_mh")
        _add_minhash_columns(df, "addr_core_address", "_addr_mh")

    parts = [
        _block_one_key(s1, other, ["country", "name_compact_name"], "compact_name"),
        _block_one_key(s1, other, ["country", "name_skeleton"], "skeleton"),
        _block_one_key(s1, other, ["country", "_prefix4"], "name_prefix4"),
        _block_one_key(s1, other, ["country", "name_acronym"], "acronym"),
        _block_one_key(s1, other, ["country", "addr_postcode", "addr_house_number"], "postcode_housenum"),
        _block_one_key(s1, other, ["country", "addr_house_number", "_addr_word"], "housenum_addrword"),
    ]
    for seed in _MINHASH_SEEDS:
        parts.append(_block_one_key(s1, other, ["country", f"_name_mh_{seed}"], f"name_minhash_{seed}"))
        parts.append(_block_one_key(s1, other, ["country", f"_addr_mh_{seed}"], f"addr_minhash_{seed}"))
    all_cands = pd.concat(parts, ignore_index=True)

    # Each _block_one_key call contributes at most one row per (s1_id, cand_id) pair
    # (merge is 1:1 on unique entity_ids), so a pair appears once per method that found
    # it. A plain groupby().size() therefore gives an exact n_methods count in one
    # vectorized (C-level) pass — no Python-level per-group string join/set needed,
    # which is what made the first version of this function too slow at 2M+ S1 rows.
    counts = all_cands.groupby(["s1_id", "cand_id"]).size().rename("n_methods").reset_index()

    # Cap per s1 at MAX_CANDIDATES_PER_SOURCE, keeping candidates found by more methods first.
    counts = counts.sort_values(["s1_id", "n_methods"], ascending=[True, False])
    result = counts.groupby("s1_id").head(MAX_CANDIDATES_PER_SOURCE).reset_index(drop=True)
    return result


def build_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """Build the unioned S2+S3 candidate set for every S1 record."""
    from_s2 = build_candidates_one_source(s1, s2)
    from_s3 = build_candidates_one_source(s1, s3)
    return pd.concat([from_s2, from_s3], ignore_index=True)


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source1.parquet")
    s2 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source2.parquet")
    s3 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source3.parquet")
    return s1, s2, s3


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    s1, s2, s3 = load_split(split)
    print(f"{split}: s1={len(s1):,} s2={len(s2):,} s3={len(s3):,}")
    cands = build_candidates(s1, s2, s3)
    print(f"candidates: {len(cands):,} rows, {cands['s1_id'].nunique():,} distinct S1 ids covered")
    out_path = PROCESSED_ROOT / f"{split}_candidates.parquet"
    cands.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")
