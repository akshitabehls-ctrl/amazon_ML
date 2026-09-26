"""Stage 3 blocking: exact-key + MinHash + TF-IDF candidate generation, unioned across strategies.

Candidate generation strategies:
1. Exact-match keys (compact_name, skeleton, prefix4, acronym, first_name_word, postcode+housenum, housenum+addrword, postcode).
2. Character 3-gram MinHash signatures on name and address.
3. TF-IDF character n-gram cosine similarity top-K search, partitioned by country.

Candidates from S2 and S3 are unioned and capped independently (top-K each per S1 entity).
"""
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.path_utils import PROCESSED_ROOT

MAX_GROUP_SIZE = 150  # drop overly generic keys before merging
MAX_CANDIDATES_PER_SOURCE = 20  # top-K per (s1, source) after union

_MINHASH_SEEDS = ["mh0", "mh1", "mh2", "mh3"]
_NGRAM_N = 3


def _prefix4(s: pd.Series) -> pd.Series:
    return s.str[:4]


def _first_word(s: pd.Series) -> pd.Series:
    return s.str.split().str[0].fillna("").str.lower()


def _first_addr_word(s: pd.Series) -> pd.Series:
    return s.str.split(",").str[0].str.split().str[0].fillna("").str.lower()


def _ngrams(s: str, n: int = _NGRAM_N) -> list[str]:
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _minhash_sigs(s: str) -> tuple:
    grams = _ngrams(s)
    if not grams:
        return tuple(-1 for _ in _MINHASH_SEEDS)
    return tuple(min(zlib.crc32((seed + g).encode()) for g in grams) for seed in _MINHASH_SEEDS)


def _add_minhash_columns(df: pd.DataFrame, source_col: str, prefix: str) -> None:
    sigs = df[source_col].map(_minhash_sigs)
    for i, seed in enumerate(_MINHASH_SEEDS):
        df[f"{prefix}_{seed}"] = sigs.map(lambda t, i=i: t[i])


def _drop_oversized_groups(df: pd.DataFrame, key_cols: list[str], max_size: int) -> pd.DataFrame:
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


def _block_tfidf_country(s1: pd.DataFrame, other: pd.DataFrame, top_k: int = 15, min_sim: float = 0.18) -> pd.DataFrame:
    """Vectorized TF-IDF character n-gram cosine top-K candidate generation partitioned by country."""
    results = []
    for country in s1["country"].unique():
        s1_c = s1[s1["country"] == country]
        other_c = other[other["country"] == country]
        if s1_c.empty or other_c.empty:
            continue

        s1_text = (s1_c["name_core_name"].fillna("") + " " + s1_c["addr_core_address"].fillna("")).values
        other_text = (other_c["name_core_name"].fillna("") + " " + other_c["addr_core_address"].fillna("")).values

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=40000)
        all_text = list(s1_text) + list(other_text)
        vec.fit(all_text)

        s1_mat = vec.transform(s1_text)
        other_mat = vec.transform(other_text)

        # Batch matrix multiplication to prevent OOM
        batch_size = 5000
        s1_ids = s1_c["entity_id"].values
        other_ids = other_c["entity_id"].values

        for i in range(0, s1_mat.shape[0], batch_size):
            s1_batch = s1_mat[i : i + batch_size]
            sim_mat = s1_batch.dot(other_mat.T)  # sparse dot product

            for row_idx in range(sim_mat.shape[0]):
                row = sim_mat.getrow(row_idx)
                if row.nnz == 0:
                    continue
                data = row.data
                indices = row.indices
                if len(data) > top_k:
                    top_indices = np.argpartition(data, -top_k)[-top_k:]
                    data = data[top_indices]
                    indices = indices[top_indices]
                mask = data >= min_sim
                indices = indices[mask]

                cur_s1 = s1_ids[i + row_idx]
                for cand_idx in indices:
                    results.append({"s1_id": cur_s1, "cand_id": other_ids[cand_idx], "method": "tfidf_ngram"})

    if not results:
        return pd.DataFrame(columns=["s1_id", "cand_id", "method"])
    return pd.DataFrame(results)


def build_candidates_one_source(s1: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Union exact-key, MinHash, and TF-IDF blocking strategies between S1 and one other source."""
    s1 = s1.copy()
    other = other.copy()
    for df in (s1, other):
        df["_prefix4"] = _prefix4(df["name_compact_name"])
        df["_first_name_word"] = _first_word(df["name_core_name"])
        df["_first_addr_word"] = _first_addr_word(df["addr_core_address"])
        _add_minhash_columns(df, "name_compact_name", "_name_mh")
        _add_minhash_columns(df, "addr_core_address", "_addr_mh")

    parts = [
        _block_one_key(s1, other, ["country", "name_compact_name"], "compact_name"),
        _block_one_key(s1, other, ["country", "name_skeleton"], "skeleton"),
        _block_one_key(s1, other, ["country", "_prefix4"], "name_prefix4"),
        _block_one_key(s1, other, ["country", "_first_name_word"], "first_name_word"),
        _block_one_key(s1, other, ["country", "name_acronym"], "acronym"),
        _block_one_key(s1, other, ["country", "addr_postcode", "addr_house_number"], "postcode_housenum"),
        _block_one_key(s1, other, ["country", "addr_house_number", "_first_addr_word"], "housenum_addrword"),
        _block_one_key(s1, other, ["country", "addr_postcode"], "postcode_exact"),
    ]
    for seed in _MINHASH_SEEDS:
        parts.append(_block_one_key(s1, other, ["country", f"_name_mh_{seed}"], f"name_minhash_{seed}"))
        parts.append(_block_one_key(s1, other, ["country", f"_addr_mh_{seed}"], f"addr_minhash_{seed}"))

    # Add TF-IDF n-gram cosine candidates
    parts.append(_block_tfidf_country(s1, other, top_k=15, min_sim=0.18))

    all_cands = pd.concat(parts, ignore_index=True)

    # Count how many distinct methods surfaced each candidate pair
    counts = all_cands.groupby(["s1_id", "cand_id"]).size().rename("n_methods").reset_index()

    # Cap per s1 at MAX_CANDIDATES_PER_SOURCE, keeping candidates found by more methods first
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
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    s1, s2, s3 = load_split(split)
    print(f"Blocking {split}: s1={len(s1):,} s2={len(s2):,} s3={len(s3):,}")
    cands = build_candidates(s1, s2, s3)
    print(f"Candidates: {len(cands):,} rows, {cands['s1_id'].nunique():,} distinct S1 ids covered")
    out_path = PROCESSED_ROOT / f"{split}_candidates.parquet"
    cands.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
