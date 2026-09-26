"""Evaluate recall of generated candidate pairs against train_ground_truth.tsv.

Reports overall recall as well as per-country recall breakdown.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.path_utils import PROCESSED_ROOT, get_dataset_root
from src.score.score import load_truth


def evaluate_candidates(candidates_path: Path, truth_path: Path, s1_path: Path):
    print(f"Loading ground truth from {truth_path} ...")
    truth_map = load_truth(truth_path)

    print(f"Loading S1 metadata from {s1_path} ...")
    s1_df = pd.read_parquet(s1_path)
    s1_country_map = dict(zip(s1_df["entity_id"], s1_df["country"]))

    print(f"Loading candidates from {candidates_path} ...")
    cands_df = pd.read_parquet(candidates_path)

    # Convert candidates to set per s1_id
    cand_map = cands_df.groupby("s1_id")["cand_id"].apply(set).to_dict()

    total_pairs = 0
    found_pairs = 0
    total_entities_with_gt = 0
    found_entities_with_gt = 0

    country_stats = {}

    for s1_id, true_set in truth_map.items():
        if not true_set:
            continue
        country = s1_country_map.get(s1_id, "UNKNOWN")
        if country not in country_stats:
            country_stats[country] = {"total_pairs": 0, "found_pairs": 0, "total_s1": 0, "full_match_s1": 0}

        country_stats[country]["total_s1"] += 1
        country_stats[country]["total_pairs"] += len(true_set)
        total_pairs += len(true_set)
        total_entities_with_gt += 1

        cand_set = cand_map.get(s1_id, set())
        matched = true_set & cand_set
        country_stats[country]["found_pairs"] += len(matched)
        found_pairs += len(matched)

        if matched == true_set:
            country_stats[country]["full_match_s1"] += 1
            found_entities_with_gt += 1

    overall_recall = found_pairs / total_pairs if total_pairs > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"BLOCKING RECALL EVALUATION SUMMARY ({candidates_path.name})")
    print("=" * 60)
    print(f"Total True Match Pairs: {total_pairs:,}")
    print(f"Candidates Found Pairs: {found_pairs:,}")
    print(f"Overall Pair Recall   : {overall_recall:.4%} ({found_pairs}/{total_pairs})")
    print(f"S1 Entities with Full True Match Recall: {found_entities_with_gt/total_entities_with_gt:.4%} ({found_entities_with_gt:,}/{total_entities_with_gt:,})")
    print("-" * 60)
    print("Per-Country Pair Recall Breakdown:")
    for country, stats in sorted(country_stats.items()):
        rec = stats["found_pairs"] / stats["total_pairs"] if stats["total_pairs"] > 0 else 0.0
        full_rec = stats["full_match_s1"] / stats["total_s1"] if stats["total_s1"] > 0 else 0.0
        print(f"  {country:10s}: Pair Recall = {rec:.4%} ({stats['found_pairs']:,}/{stats['total_pairs']:,}), Full S1 Recall = {full_rec:.4%} ({stats['full_match_s1']:,}/{stats['total_s1']:,})")
    print("=" * 60 + "\n")
    return overall_recall, country_stats


def main():
    truth_path = get_dataset_root() / "train" / "train_ground_truth.tsv"
    candidates_path = PROCESSED_ROOT / "train_candidates.parquet"
    s1_path = PROCESSED_ROOT / "train_source1.parquet"

    if not candidates_path.exists():
        print(f"Candidate file {candidates_path} not found. Please run block.py first.")
        sys.exit(1)

    evaluate_candidates(candidates_path, truth_path, s1_path)


if __name__ == "__main__":
    main()
