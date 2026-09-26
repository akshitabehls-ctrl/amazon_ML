"""Eyeball 50 matched pairs and 50 hard non-matches to catalog noise patterns."""
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / "student_resource" / "dataset" / "train"


def load(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


def main():
    random.seed(42)
    s1 = load(TRAIN / "train_source1.tsv").set_index("entity_id")
    s2 = load(TRAIN / "train_source2.tsv").set_index("entity_id")
    s3 = load(TRAIN / "train_source3.tsv").set_index("entity_id")
    gt = load(TRAIN / "train_ground_truth.tsv")

    def rec(eid):
        if eid.startswith("S1-"):
            row = s1.loc[eid]
        elif eid.startswith("S2-"):
            row = s2.loc[eid]
        else:
            row = s3.loc[eid]
        return row["business_name"], row["business_address"], row["country"]

    gt["match_list"] = gt["matched_entity_ids"].fillna("").apply(lambda x: [i for i in x.split(",") if i])
    non_empty = gt[gt["match_list"].apply(len) > 0]
    sample = non_empty.sample(50, random_state=42)

    print("=" * 100)
    print("50 RANDOM MATCHED PAIRS (S1 vs each match)")
    print("=" * 100)
    for _, row in sample.iterrows():
        s1_name, s1_addr, s1_c = rec(row["source1_entity_id"])
        print(f"\n[{row['source1_entity_id']}] ({s1_c})")
        print(f"  S1: {s1_name!r} | {s1_addr!r}")
        for mid in row["match_list"]:
            m_name, m_addr, m_c = rec(mid)
            print(f"  {mid}: {m_name!r} | {m_addr!r}")

    # Hard non-matches: same country, share a token with S1 name/address, but NOT a true match.
    print("\n\n" + "=" * 100)
    print("50 HARD NON-MATCH CANDIDATES (share first name token + country, not a GT match)")
    print("=" * 100)
    s1_sample = s1.sample(400, random_state=7)
    gt_map = gt.set_index("source1_entity_id")["match_list"].to_dict()
    found = 0
    s2_by_token = {}
    for eid, row in s2.sample(min(len(s2), 300000), random_state=1).iterrows():
        tok = row["business_name"].split()[0].lower() if row["business_name"].split() else ""
        s2_by_token.setdefault((tok, row["country"]), []).append(eid)

    for eid, row in s1_sample.iterrows():
        if found >= 50:
            break
        toks = row["business_name"].split()
        if not toks:
            continue
        key = (toks[0].lower(), row["country"])
        cands = s2_by_token.get(key, [])
        true_matches = set(gt_map.get(eid, []))
        for cid in cands:
            if cid not in true_matches:
                m_name, m_addr, m_c = rec(cid)
                print(f"\nS1 [{eid}] ({row['country']}): {row['business_name']!r} | {row['business_address']!r}")
                print(f"  NON-MATCH {cid}: {m_name!r} | {m_addr!r}")
                found += 1
                break

    print(f"\nTotal hard non-match examples found: {found}")


if __name__ == "__main__":
    main()
