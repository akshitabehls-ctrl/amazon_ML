"""Stage 1 data checks: answer the design questions before building anything.

Run: source .venv/bin/activate && python code/business_entity_resolution/src/eda_stage1.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "student_resource" / "dataset" / "train"
TEST = ROOT / "student_resource" / "dataset" / "test"

pd.set_option("display.width", 140)


def load(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], **kw)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main():
    section("LOADING")
    s1 = load(TRAIN / "train_source1.tsv")
    s2 = load(TRAIN / "train_source2.tsv")
    s3 = load(TRAIN / "train_source3.tsv")
    gt = load(TRAIN / "train_ground_truth.tsv")
    print(f"train S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} GT={len(gt):,}")

    test_s1 = load(TEST / "test_source1.tsv")
    test_s2 = load(TEST / "test_source2.tsv")
    test_s3 = load(TEST / "test_source3.tsv")
    print(f"test  S1={len(test_s1):,} S2={len(test_s2):,} S3={len(test_s3):,}")

    # sanity: entity_id uniqueness
    for name, df in [("s1", s1), ("s2", s2), ("s3", s3), ("test_s1", test_s1),
                      ("test_s2", test_s2), ("test_s3", test_s3)]:
        dupes = df["entity_id"].duplicated().sum()
        print(f"  {name}: duplicate entity_id rows = {dupes}")

    # ---------------------------------------------------------------
    section("Q1: singleton share (S1 with no match)")
    gt["match_list"] = gt["matched_entity_ids"].fillna("").apply(
        lambda x: [i for i in x.split(",") if i]
    )
    gt["n_matches"] = gt["match_list"].apply(len)

    # every S1 in gt? gt has one row per S1 (per problem statement)
    print(f"GT rows: {len(gt):,}, unique source1_entity_id: {gt['source1_entity_id'].nunique():,}")
    missing_s1_in_gt = set(s1["entity_id"]) - set(gt["source1_entity_id"])
    extra_s1_in_gt = set(gt["source1_entity_id"]) - set(s1["entity_id"])
    print(f"S1 ids missing from GT: {len(missing_s1_in_gt)}, GT ids not in S1: {len(extra_s1_in_gt)}")

    singleton_share = (gt["n_matches"] == 0).mean()
    print(f"Singleton share (0 matches): {singleton_share:.4%}")
    print("Distribution of n_matches per S1:")
    print(gt["n_matches"].value_counts().sort_index().head(20))

    # how many matches come from S2 vs S3
    n_s2 = gt["match_list"].apply(lambda ids: sum(1 for i in ids if i.startswith("S2-")))
    n_s3 = gt["match_list"].apply(lambda ids: sum(1 for i in ids if i.startswith("S3-")))
    print(f"Total match links: {gt['n_matches'].sum():,} (S2 side: {n_s2.sum():,}, S3 side: {n_s3.sum():,})")
    print(f"S1 with >=1 S2 match: {(n_s2 > 0).mean():.4%}, S1 with >=1 S3 match: {(n_s3 > 0).mean():.4%}")

    # ---------------------------------------------------------------
    section("Q2: does any S2/S3 record match more than one S1? (one-to-one check)")
    all_matched = gt.explode("match_list")
    all_matched = all_matched[all_matched["match_list"].notna() & (all_matched["match_list"] != "")]
    dup_targets = all_matched["match_list"].value_counts()
    dup_targets = dup_targets[dup_targets > 1]
    print(f"S2/S3 ids that match >1 S1 entity: {len(dup_targets):,} "
          f"out of {all_matched['match_list'].nunique():,} distinct matched ids "
          f"({len(dup_targets) / max(all_matched['match_list'].nunique(),1):.4%})")
    if len(dup_targets):
        print("Examples:")
        print(dup_targets.head(10))

    # ---------------------------------------------------------------
    section("Q3: do matches always share the same country label?")
    s1_country = s1.set_index("entity_id")["country"]
    s2_country = s2.set_index("entity_id")["country"]
    s3_country = s3.set_index("entity_id")["country"]

    def lookup_country(eid):
        if eid.startswith("S2-"):
            return s2_country.get(eid)
        if eid.startswith("S3-"):
            return s3_country.get(eid)
        return None

    pairs = all_matched.copy()
    pairs["s1_country"] = pairs["source1_entity_id"].map(s1_country)
    pairs["match_country"] = pairs["match_list"].map(lookup_country)
    mismatch = pairs[pairs["s1_country"] != pairs["match_country"]]
    print(f"Matched pairs: {len(pairs):,}, country mismatches: {len(mismatch):,} "
          f"({len(mismatch) / max(len(pairs),1):.4%})")
    if len(mismatch):
        print(mismatch[["source1_entity_id", "match_list", "s1_country", "match_country"]].head(10))

    print("\nCountry label distinct values per source (train):")
    for name, df in [("s1", s1), ("s2", s2), ("s3", s3)]:
        print(f"  {name}: {sorted(df['country'].dropna().unique())}")

    # ---------------------------------------------------------------
    section("Q4: empty / missing fields, non-Latin script share")
    non_latin_re = re.compile(r"[^\x00-\x7F]")

    def field_stats(df, name):
        name_empty = (df["business_name"].isna() | (df["business_name"].str.strip() == "")).mean()
        addr_empty = (df["business_address"].isna() | (df["business_address"].str.strip() == "")).mean()
        name_nonlatin = df["business_name"].fillna("").apply(lambda x: bool(non_latin_re.search(x))).mean()
        addr_nonlatin = df["business_address"].fillna("").apply(lambda x: bool(non_latin_re.search(x))).mean()
        print(f"  {name}: name_empty={name_empty:.3%} addr_empty={addr_empty:.3%} "
              f"name_nonlatin={name_nonlatin:.3%} addr_nonlatin={addr_nonlatin:.3%}")

    for name, df in [("train s1", s1), ("train s2", s2), ("train s3", s3),
                      ("test s1", test_s1), ("test s2", test_s2), ("test s3", test_s3)]:
        field_stats(df, name)

    # ---------------------------------------------------------------
    section("Q5: country labels in test vs train, spelling consistency")
    for name, df in [("test s1", test_s1), ("test s2", test_s2), ("test s3", test_s3)]:
        vc = df["country"].value_counts()
        print(f"  {name} country counts:\n{vc}\n")

    # ---------------------------------------------------------------
    section("Row length / name & address length distribution (train s1)")
    print("business_name length chars:")
    print(s1["business_name"].fillna("").str.len().describe())
    print("business_address length chars:")
    print(s1["business_address"].fillna("").str.len().describe())

    section("DONE")


if __name__ == "__main__":
    sys.exit(main())
