"""ML-based scoring, feature computation, LightGBM model training, and conflict-aware 1-to-1 decoding.
"""
import hashlib
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.path_utils import PROCESSED_ROOT, get_dataset_root
from src.score.features import FEATURE_COLUMNS, build_feature_table


def load_normalized(split: str):
    s1 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source1.parquet")
    s2 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source2.parquet")
    s3 = pd.read_parquet(PROCESSED_ROOT / f"{split}_source3.parquet")
    all_records = pd.concat([s1, s2, s3], ignore_index=True).set_index("entity_id")
    return s1, all_records


def load_truth(path: Path) -> dict[str, set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s1: set(ids.split(",")) - {""}
        for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }


def add_features(cands: pd.DataFrame, all_records: pd.DataFrame) -> pd.DataFrame:
    """Build feature table for candidate pairs."""
    return build_feature_table(cands, all_records)


def train_lgb_model(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str] = None):
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    X_train = train_df[feature_cols]
    y_train = train_df["is_match"].values

    X_val = val_df[feature_cols]
    y_val = val_df["is_match"].values

    print(f"Training LightGBM on {len(X_train):,} train pairs, early stopping on {len(X_val):,} val pairs ...")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    print("LightGBM feature importances:")
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in importances[:15]:
        print(f"  {feat:25s}: {imp}")

    return model


def decode_conflict_aware(
    scored_df: pd.DataFrame,
    country_thresholds: dict[str, float] = None,
    default_threshold: float = 0.50,
    min_confidence: float = 0.35,
) -> dict[str, set[str]]:
    """Conflict-aware decoding respecting the 1-to-1 matching constraint on S2/S3 candidates.

    - Filters by per-country threshold and min_confidence.
    - Greedily assigns each S2/S3 candidate to the highest-scoring S1 entity.
    """
    if country_thresholds is None:
        country_thresholds = {}

    if "score" not in scored_df.columns:
        raise ValueError("scored_df must contain 'score' column")

    # Determine dynamic threshold per row based on country
    countries = scored_df.get("s1_country", pd.Series([""] * len(scored_df))).values
    thresholds = np.array([
        max(country_thresholds.get(c, default_threshold), min_confidence) for c in countries
    ])

    scores = scored_df["score"].values
    valid_mask = scores >= thresholds

    valid_df = scored_df[valid_mask].copy()
    if valid_df.empty:
        return {}

    # Sort descending by score for greedy 1-to-1 assignment
    valid_df = valid_df.sort_values("score", ascending=False)

    claimed_cands = set()
    matches = {}

    for row in valid_df.itertuples():
        s1_id = row.s1_id
        cand_id = row.cand_id

        # 1-to-1 constraint enforcement on S2/S3 side
        if cand_id in claimed_cands:
            continue

        claimed_cands.add(cand_id)
        if s1_id not in matches:
            matches[s1_id] = set()
        matches[s1_id].add(cand_id)

    return matches


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
