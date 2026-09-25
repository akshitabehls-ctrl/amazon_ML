"""Tune the iteration-1 score threshold on train, using the real macro F_0.5 metric.

Splits S1 entities 80/20 (deterministic hash split, not random, so reruns are
stable), fits nothing on the 80% (the score is a fixed rule, not trained),
and sweeps thresholds on the 20% held-out slice to report an honest local
score — never tune on the same rows you report on.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.score.score import add_features, decode, load_normalized, load_truth, macro_f05

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"


def _is_val(s1_id: str) -> bool:
    return int(hashlib.md5(s1_id.encode()).hexdigest(), 16) % 5 == 0  # ~20%


def main():
    print("loading candidates + normalized records...", flush=True)
    cands = pd.read_parquet(PROCESSED_ROOT / "train_candidates.parquet")
    s1, all_records = load_normalized("train")
    print(f"candidates: {len(cands):,} rows, s1: {len(s1):,} rows", flush=True)

    truth = load_truth(REPO_ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv")

    val_mask = s1["entity_id"].map(_is_val)
    val_ids = set(s1.loc[val_mask, "entity_id"])
    print(f"validation slice: {len(val_ids):,} S1 entities", flush=True)

    val_cands = cands[cands["s1_id"].isin(val_ids)]
    print(f"validation candidates: {len(val_cands):,} rows, computing features...", flush=True)
    scored = add_features(val_cands, all_records)
    print("features done, sweeping thresholds...", flush=True)

    best_t, best_f05 = None, -1
    for t in [round(x * 0.02, 2) for x in range(10, 48)]:  # 0.20 .. 0.94
        preds = decode(scored, t)
        f05 = macro_f05(preds, truth, val_ids)
        print(f"  threshold={t:.2f} macro_F0.5={f05:.4f}", flush=True)
        if f05 > best_f05:
            best_t, best_f05 = t, f05

    print(f"\nBEST threshold={best_t:.2f} macro_F0.5={best_f05:.4f} on {len(val_ids):,}-entity validation slice", flush=True)
    (PROCESSED_ROOT / "chosen_threshold.txt").write_text(str(best_t))
    print(f"wrote {PROCESSED_ROOT / 'chosen_threshold.txt'}", flush=True)


if __name__ == "__main__":
    main()
