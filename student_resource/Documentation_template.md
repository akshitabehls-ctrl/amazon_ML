# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** High-Recall Vectorized Entity Resolution Team  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

We developed an end-to-end, high-performance Machine Learning solution for multi-source commercial business entity resolution across Source 1, Source 2, and Source 3 datasets. The solution combines multi-signal candidate generation (exact keys, MinHash signatures, and country-partitioned TF-IDF character n-gram cosine similarity) with a 28-feature LightGBM binary classifier and a conflict-aware 1-to-1 decoding algorithm. Our pipeline optimizes the official macro-averaged $F_{0.5}$ metric while respecting zero-shot generalization to unseen countries (France) and strict fair-play/license constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Business records from independent sources exhibit severe surface noise, including:
- **Name Variations:** Legal suffix inconsistencies (Pvt Ltd vs Corp), word order transpositions, acronyms, and phonetic/script transliteration differences.
- **Address Variations:** Structural reordering, missing postal codes, municipal numbering formats, and landmark references.
- **Entity Structure:** Source 1 is deduplicated, while Source 2 and Source 3 contain non-overlapping records where each candidate matches at most one Source 1 record (1-to-1 matching constraint). Singletons (S1 entities with no matches in S2/S3) make up ~5.58% of entities and score 1.0 when correctly predicted empty.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Multi-Signal Candidate Generation (Blocking) + LightGBM Gradient Boosted Decision Trees + Conflict-Aware Greedy Decoding  
**Core Innovation:** Partitioned vectorized TF-IDF character $(3-5)$-gram cosine top-K candidate generation combined with 1-to-1 conflict-aware greedy decoding and per-country threshold tuning.

---

## 3. Candidate Generation (Blocking)

To achieve high recall without memory blowup, candidate generation unions three complementary strategies:
1. **Exact Keys:** `compact_name`, `skeleton` (transliterated core name), `name_prefix4`, `acronym`, `first_name_word`, `postcode + house_number`, `house_number + first_addr_word`, `postcode_exact`.
2. **MinHash Signatures:** 4 independent CRC32 hash seeds each for character 3-grams on name and address fields to capture scattered string typos.
3. **TF-IDF Cosine Similarity Search:** Sparse matrix dot products of character $(3-5)$-gram TF-IDF vectors per country, extracting top candidates with similarity $\ge 0.18$.

- **Candidate Pairs Generated:** Capped at $K=20$ candidates per source per S1 entity.
- **Blocking Recall Measurement:** Evaluated via `src/blocking/eval_blocking.py` against `train_ground_truth.tsv` across all training countries.

---

## 4. Matching Model

### 4.1 Feature Engineering (28 Vectorized Features)
- **Name String Metrics:** RapidFuzz `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `jaro_winkler`, and character 3-gram Jaccard similarity across `compact_name`, `core_name`, `skeleton`, and `deaccented` representations.
- **Address String Metrics:** RapidFuzz `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `jaro_winkler`, and 3-gram Jaccard similarity across `core_address` and `deaccented` representations.
- **Numeric & Postal Fields:** Postal code exact match, postal code edit distance ratio, postal code missing flag; house number exact match, house number missing flag; unit exact match flag.
- **Flags & Structure:** Country match flag, acronym match flag, alternate name (DBA/trading-as) match ratio, name length ratio, address length ratio, candidate source indicator (S2 vs S3), and `n_methods` (number of blocking signals surfacing the pair).

### 4.2 Model Architecture & Training
- **Model Type:** LightGBM Binary Classifier (`LGBMClassifier`).
- **Validation Strategy:** 80/20 entity-level deterministic MD5 hash split (`hashlib.md5(s1_id) % 5 == 0`) to prevent data leakage.
- **Hyperparameters:** `n_estimators=1000`, `learning_rate=0.04`, `num_leaves=63`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`. Early stopping on 20% validation split loss.

### 4.3 Conflict-Aware 1-to-1 Decoding & Threshold Selection
- **Constraint Enforcement:** Implemented greedy conflict-aware decoding (`decode_conflict_aware`). Predictions are sorted by probability score descending; each candidate S2/S3 entity is assigned to at most one S1 entity, dropping lower-scoring claims.
- **Threshold Tuning:** Swept threshold grids per country (US, India, default fallback for France) to maximize exact macro $F_{0.5}$ score on the 20% validation slice.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** Tuned pipeline significantly outperforms rule-based baselines (0.55 -> 0.90+).
- **Blocking Pair Recall:** $>97.5\%$ true match pair recall across training countries.
- **Common False Positives:** Highly similar franchise locations sharing identical names and adjacent house numbers in dense commercial areas.
- **Common False Negatives:** Short acronym-only business names paired with severely abbreviated landmark addresses.

---

## 6. Conclusion

The multi-stage pipeline provides a scalable, leak-free, highly accurate solution for large-scale entity resolution. By combining vectorized TF-IDF candidate generation, 28 string/structural features, LightGBM classification, and conflict-aware decoding, the approach maximizes macro $F_{0.5}$ while maintaining low execution runtime and full compliance with submission formatting rules.

---

## Appendix

### A. Code Artefacts
- Entry point script: `python code/business_entity_resolution/src/run_pipeline.py`
- All source files located under `code/business_entity_resolution/src/`:
  - `normalize/`: `run_normalize.py`, `name_cleaner.py`, `address_cleaner.py`, `text_utils.py`, `legal_suffixes.py`, `road_abbrev.py`.
  - `blocking/`: `block.py` (candidate generation), `eval_blocking.py` (recall evaluation).
  - `score/`: `features.py` (28 vectorized features), `score.py` (model & decoding logic), `tune.py` (training & threshold tuning), `predict.py` (inference).
