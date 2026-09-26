# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution problem.

## Pipeline Architecture

The solution consists of five modular, vectorized stages:

1. **Stage 1 & 2 — Data Ingestion & Normalization (`src/normalize/`)**:
   - Cleans business names (stripping legal suffixes, noise words, honorifics, accent removal, transliteration, acronym generation).
   - Cleans business addresses (core address extraction, postcode/house_number/unit parsing).
   - Exports parquet formatted datasets for fast downstream vectorized I/O.

2. **Stage 3 — High-Recall Blocking / Candidate Generation (`src/blocking/`)**:
   - Multi-signal candidate union:
     - Exact match keys: compact name, skeleton, name prefix, acronym, postcode + house number, house number + first address word, first name word, postcode.
     - MinHash signatures (4 seeds each for name and address n-grams).
     - TF-IDF character (3-5)-gram cosine similarity top-K search, partitioned by country.
   - Evaluated via `src/blocking/eval_blocking.py` to ensure recall target (>97-98%).

3. **Stage 4 — Vectorized Feature Engineering & LightGBM Classification (`src/score/`)**:
   - 28 high-signal features computed without row-wise Python loops:
     - RapidFuzz string metrics (`ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `jaro_winkler`) across name and address representations.
     - Character 3-gram Jaccard similarities.
     - Postal code, house number, unit exact-match and edit distance flags.
     - Country match, acronym match, alternate name match, length ratio features.
     - Candidate source and `n_methods` blocking signal.
   - Binary LightGBM classifier (`LGBMClassifier`) trained on 80% deterministic hash split, early stopped on 20% validation split.

4. **Stage 5 — 1-to-1 Conflict-Aware Decoding & Threshold Tuning (`src/score/`)**:
   - Ground truth constraint: Each S2/S3 entity matches at most ONE S1 entity.
   - Greedy conflict-aware decoding assigns candidates to highest-scoring S1 entity, dropping lower-scoring conflicting claims.
   - Per-country decoding thresholds swept on held-out validation slice to maximize Macro $F_{0.5}$.

5. **Inference & Format Validation (`src/score/predict.py`)**:
   - Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
   - Validated against official rule-checker (`student_resource/utils/validate_submission.py`).

## Quick Start / Reproduce End-to-End

To run the entire pipeline end-to-end:

```bash
python src/run_pipeline.py
```

Or run individual stages:

```bash
# 1. Normalize datasets
python src/normalize/run_normalize.py

# 2. Generate candidate blocking sets
python src/blocking/block.py train
python src/blocking/block.py test

# 3. Evaluate blocking recall
python src/blocking/eval_blocking.py

# 4. Train LightGBM model and tune per-country thresholds
python src/score/tune.py

# 5. Generate final submission test predictions
python src/score/predict.py

# 6. Validate output submission format
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
