"""Generate submission files for test set using tuned LightGBM model and conflict-aware decoding.

Produces:
- output/matching_results.tsv
- output/candidate_pairs.tsv
"""
import json
import pickle
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.path_utils import OUTPUT_ROOT, PROCESSED_ROOT
from src.score.features import FEATURE_COLUMNS
from src.score.score import add_features, decode_conflict_aware, load_normalized


def write_id_list_file(path: Path, required_ids: list, id_map: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "source1_entity_id\tmatched_entity_ids\n"
        if "matching" in path.name
        else "source1_entity_id\tcandidate_entity_ids\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for s1 in required_ids:
            ids = sorted(id_map.get(s1, []))
            f.write(f"{s1}\t{','.join(ids)}\n")


def main():
    thresh_file = PROCESSED_ROOT / "chosen_thresholds.json"
    if thresh_file.exists():
        with open(thresh_file) as f:
            thresholds = json.load(f)
    else:
        thresholds = {"default": 0.50}

    default_t = thresholds.get("default", 0.50)
    print(f"Using thresholds: {thresholds}", flush=True)

    model_path = PROCESSED_ROOT / "model.pkl"
    if not model_path.exists():
        print(f"Model file {model_path} not found. Please run tune.py first.", flush=True)
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Loaded LightGBM model.", flush=True)

    print("Loading test candidates and normalized records...", flush=True)
    cands_path = PROCESSED_ROOT / "test_candidates.parquet"
    if not cands_path.exists():
        print(f"Test candidates file {cands_path} not found. Please run block.py test first.", flush=True)
        sys.exit(1)

    cands = pd.read_parquet(cands_path)
    s1, all_records = load_normalized("test")
    required_ids = s1["entity_id"].tolist()
    print(f"Test S1 entities: {len(required_ids):,}, candidates: {len(cands):,} rows", flush=True)

    print("Computing feature tables for test candidates...", flush=True)
    scored = add_features(cands, all_records)

    print("Predicting match probabilities...", flush=True)
    scored["score"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]

    print("Decoding matches with 1-to-1 conflict-aware logic...", flush=True)
    matches = decode_conflict_aware(
        scored, country_thresholds=thresholds, default_threshold=default_t, min_confidence=0.30
    )

    candidate_map = cands.groupby("s1_id")["cand_id"].apply(set).to_dict()

    print("Writing output files...", flush=True)
    write_id_list_file(OUTPUT_ROOT / "matching_results.tsv", required_ids, matches)
    write_id_list_file(OUTPUT_ROOT / "candidate_pairs.tsv", required_ids, candidate_map)

    n_with_match = sum(1 for s1_id in required_ids if s1_id in matches and matches[s1_id])
    print("=" * 60)
    print("TEST PREDICTION GENERATION COMPLETE")
    print(f"Wrote {OUTPUT_ROOT / 'matching_results.tsv'}")
    print(f"Wrote {OUTPUT_ROOT / 'candidate_pairs.tsv'}")
    print(f"Entities with predicted matches: {n_with_match:,}/{len(required_ids):,} ({n_with_match/len(required_ids):.2%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
