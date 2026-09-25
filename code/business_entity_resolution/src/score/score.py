"""Iteration-1 scoring and decoding: a few fast similarity features combined
into one weighted score, with a threshold picked to maximize macro F_0.5 on
a held-out slice of train.

Deliberately rule-based rather than an ML-trained two-stage LightGBM model
(the master plan's eventual design) — iteration 1's priority is a complete,
validated, end-to-end submission fast; the full feature/model design is an
iteration-2 follow-up once this baseline is working and scored.
"""
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

WEIGHTS = {"name": 0.45, "addr": 0.35, "postcode": 0.10, "housenum": 0.10}


def load_normalized(split: str):
    s1 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source1.parquet")
    s2 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source2.parquet")
    s3 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source3.parquet")
    all_records = pd.concat([s1, s2, s3], ignore_index=True).set_index("entity_id")
    return s1, all_records


def add_features(cands: pd.DataFrame, all_records: pd.DataFrame) -> pd.DataFrame:
    """Join each candidate pair to its two records and compute the score."""
    s1_info = all_records.loc[cands["s1_id"].values].reset_index(drop=True)
    cand_info = all_records.loc[cands["cand_id"].values].reset_index(drop=True)

    name_ratio = [
        fuzz.ratio(a, b) for a, b in zip(s1_info["name_compact_name"], cand_info["name_compact_name"])
    ]
    addr_ratio = [
        fuzz.ratio(a, b) for a, b in zip(s1_info["addr_core_address"], cand_info["addr_core_address"])
    ]
    postcode_match = (
        (s1_info["addr_postcode"].values != "") & (s1_info["addr_postcode"].values == cand_info["addr_postcode"].values)
    ).astype(int)
    housenum_match = (
        (s1_info["addr_house_number"].values != "")
        & (s1_info["addr_house_number"].values == cand_info["addr_house_number"].values)
    ).astype(int)

    out = cands.reset_index(drop=True).copy()
    out["name_ratio"] = name_ratio
    out["addr_ratio"] = addr_ratio
    out["postcode_match"] = postcode_match
    out["housenum_match"] = housenum_match
    out["score"] = (
        WEIGHTS["name"] * out["name_ratio"] / 100
        + WEIGHTS["addr"] * out["addr_ratio"] / 100
        + WEIGHTS["postcode"] * out["postcode_match"]
        + WEIGHTS["housenum"] * out["housenum_match"]
    )
    return out


def decode(scored: pd.DataFrame, threshold: float) -> dict[str, set[str]]:
    """Predicted matches per s1_id: every candidate at/above threshold, deduped."""
    kept = scored[scored["score"] >= threshold]
    return kept.groupby("s1_id")["cand_id"].apply(set).to_dict()


def macro_f05(predictions: dict[str, set[str]], truth: dict[str, set[str]], all_s1_ids) -> float:
    """Macro-average F_0.5 across every S1 id (0 matches predicted+true -> score 1.0)."""
    scores = []
    for s1 in all_s1_ids:
        pred = predictions.get(s1, set())
        true = truth.get(s1, set())
        if not pred and not true:
            scores.append(1.0)
            continue
        if not pred or not true:
            scores.append(0.0)
            continue
        tp = len(pred & true)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / len(pred)
        recall = tp / len(true)
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        scores.append(f05)
    return sum(scores) / len(scores)


def load_truth(path: Path) -> dict[str, set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s1: set(ids.split(",")) - {""}
        for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }
