"""Generate the final submission files for the test set: output/matching_results.tsv
and output/candidate_pairs.tsv, using the threshold tuned on train.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.score.score import add_features, decode, load_normalized

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
OUTPUT_ROOT = REPO_ROOT / "output"


def write_id_list_file(path: Path, required_ids, id_map: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n" if "matching" in path.name
                else "source1_entity_id\tcandidate_entity_ids\n")
        for s1 in required_ids:
            ids = sorted(id_map.get(s1, []))
            f.write(f"{s1}\t{','.join(ids)}\n")


def main():
    threshold_path = PROCESSED_ROOT / "chosen_threshold.txt"
    threshold = float(threshold_path.read_text().strip())
    print(f"using tuned threshold={threshold}", flush=True)

    print("loading test candidates + normalized records...", flush=True)
    cands = pd.read_parquet(PROCESSED_ROOT / "test_candidates.parquet")
    s1, all_records = load_normalized("test")
    required_ids = s1["entity_id"].tolist()
    print(f"test S1 entities: {len(required_ids):,}, candidates: {len(cands):,} rows", flush=True)

    print("computing features...", flush=True)
    scored = add_features(cands, all_records)

    print("decoding matches...", flush=True)
    matches = decode(scored, threshold)

    candidate_map = cands.groupby("s1_id")["cand_id"].apply(set).to_dict()

    write_id_list_file(OUTPUT_ROOT / "matching_results.tsv", required_ids, matches)
    write_id_list_file(OUTPUT_ROOT / "candidate_pairs.tsv", required_ids, candidate_map)

    n_with_match = sum(1 for s1 in required_ids if s1 in matches and matches[s1])
    print(f"wrote output/matching_results.tsv and output/candidate_pairs.tsv", flush=True)
    print(f"{n_with_match:,}/{len(required_ids):,} S1 entities got >=1 predicted match "
          f"({n_with_match/len(required_ids):.2%})", flush=True)


if __name__ == "__main__":
    main()
