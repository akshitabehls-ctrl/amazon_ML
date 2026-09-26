"""Tune LightGBM matching model and per-country decoding thresholds on train split.

- Splits S1 entities 80/20 deterministically using MD5 entity hashing.
- Generates binary match labels for 80% train candidates.
- Trains LightGBM classifier with early stopping on 20% validation candidates.
- Sweeps per-country threshold grids to optimize exact macro F_0.5 score on held-out 20% slice.
- Saves model to model.pkl and threshold map to chosen_thresholds.json.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.path_utils import PROCESSED_ROOT, get_dataset_root
from src.score.score import (
    add_features,
    decode_conflict_aware,
    load_normalized,
    load_truth,
    macro_f05,
    train_lgb_model,
)
from src.score.features import FEATURE_COLUMNS


def _is_val(s1_id: str) -> bool:
    return int(hashlib.md5(s1_id.encode()).hexdigest(), 16) % 5 == 0  # 20% validation split


def main():
    print("Loading candidate pairs and normalized entity records...", flush=True)
    cands_path = PROCESSED_ROOT / "train_candidates.parquet"
    if not cands_path.exists():
        print(f"Candidates file {cands_path} does not exist. Please run block.py first.", flush=True)
        sys.exit(1)

    cands = pd.read_parquet(cands_path)
    s1, all_records = load_normalized("train")
    truth_path = get_dataset_root() / "train" / "train_ground_truth.tsv"
    truth = load_truth(truth_path)

    print(f"Total candidates: {len(cands):,} rows across {s1['entity_id'].nunique():,} S1 entities.", flush=True)

    # Split S1 entities into 80% train / 20% validation
    s1["is_val"] = s1["entity_id"].map(_is_val)
    val_ids = set(s1.loc[s1["is_val"], "entity_id"])
    train_ids = set(s1.loc[~s1["is_val"], "entity_id"])

    print(f"Entity split: {len(train_ids):,} train S1 entities, {len(val_ids):,} validation S1 entities.", flush=True)

    train_cands = cands[cands["s1_id"].isin(train_ids)].reset_index(drop=True)
    val_cands = cands[cands["s1_id"].isin(val_ids)].reset_index(drop=True)

    # Create binary target is_match for training set candidates
    print("Generating match labels for training set candidates...", flush=True)
    is_match_list = []
    for s1_id, cand_id in zip(train_cands["s1_id"], train_cands["cand_id"]):
        true_set = truth.get(s1_id, set())
        is_match_list.append(1 if cand_id in true_set else 0)
    train_cands["is_match"] = is_match_list

    val_is_match_list = []
    for s1_id, cand_id in zip(val_cands["s1_id"], val_cands["cand_id"]):
        true_set = truth.get(s1_id, set())
        val_is_match_list.append(1 if cand_id in true_set else 0)
    val_cands["is_match"] = val_is_match_list

    print(f"Train matches: {train_cands['is_match'].sum():,} / {len(train_cands):,} ({train_cands['is_match'].mean():.2%})")
    print(f"Val matches  : {val_cands['is_match'].sum():,} / {len(val_cands):,} ({val_cands['is_match'].mean():.2%})")

    # Compute features for train and val splits
    print("Computing feature tables for train candidates...", flush=True)
    train_scored = add_features(train_cands, all_records)

    print("Computing feature tables for val candidates...", flush=True)
    val_scored = add_features(val_cands, all_records)

    # Train LightGBM model
    model = train_lgb_model(train_scored, val_scored, FEATURE_COLUMNS)

    # Predict scores for val candidates
    print("Predicting match probability scores for validation slice...", flush=True)
    val_scored["score"] = model.predict_proba(val_scored[FEATURE_COLUMNS])[:, 1]

    # Model save
    model_path = PROCESSED_ROOT / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained LightGBM model to {model_path}", flush=True)

    # Sweep threshold optimization per country
    val_scored["s1_country"] = val_scored["s1_country"].fillna("US")
    countries = list(val_scored["s1_country"].unique())
    print(f"\nSweeping thresholds for countries: {countries} ...", flush=True)

    best_thresholds = {}
    best_overall_f05 = -1.0

    # Grid search thresholds per country
    t_grid = [round(x * 0.05, 2) for x in range(5, 19)]  # 0.25 to 0.90

    # First optimize a single global threshold
    best_global_t = 0.50
    best_global_f05 = -1.0
    for t in t_grid:
        preds = decode_conflict_aware(val_scored, country_thresholds={}, default_threshold=t, min_confidence=t)
        f05 = macro_f05(preds, truth, val_ids)
        print(f"  Global threshold t={t:.2f} -> macro F_0.5 = {f05:.5f}")
        if f05 > best_global_f05:
            best_global_f05 = f05
            best_global_t = t

    print(f"\nBest Global Threshold: t={best_global_t:.2f} with Macro F_0.5 = {best_global_f05:.5f}")

    # Now fine-tune per country threshold
    for c in countries:
        best_c_t = best_global_t
        best_c_f05 = best_global_f05
        for t in t_grid:
            temp_thresh = {c: t}
            preds = decode_conflict_aware(val_scored, country_thresholds=temp_thresh, default_threshold=best_global_t, min_confidence=0.30)
            f05 = macro_f05(preds, truth, val_ids)
            if f05 > best_c_f05:
                best_c_f05 = f05
                best_c_t = t
        best_thresholds[c] = best_c_t
        print(f"Country {c} best threshold: t={best_c_t:.2f}")

    best_thresholds["default"] = best_global_t

    # Final evaluation with conflict-aware decoding & per-country thresholds
    final_preds = decode_conflict_aware(val_scored, country_thresholds=best_thresholds, default_threshold=best_global_t, min_confidence=0.30)
    final_f05 = macro_f05(final_preds, truth, val_ids)

    print("\n" + "=" * 60)
    print(f"FINAL TUNED VALIDATION MACRO F_0.5: {final_f05:.5f}")
    print(f"Chosen thresholds: {best_thresholds}")
    print("=" * 60 + "\n")

    thresh_file = PROCESSED_ROOT / "chosen_thresholds.json"
    with open(thresh_file, "w") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"Wrote thresholds to {thresh_file}")

    (PROCESSED_ROOT / "chosen_threshold.txt").write_text(str(best_global_t))


if __name__ == "__main__":
    main()
