# Amazon ML Challenge 2026 - Business Entity Resolution

This is an end-to-end pipeline for the Amazon ML Challenge focused on high-precision Entity Resolution to maximize the Macro F0.5 score.

## Architecture

1.  **Preprocessing:** Normalizes Unicode (crucial for French test data), standardizes common legal suffixes (ltd, pvt, inc), and cleans punctuation.
2.  **Blocking:** Uses `TfidfVectorizer` (character N-grams) with `NearestNeighbors` (Cosine Distance) to reduce the $O(N^2)$ search space down to the top 15 nearest candidates per Source 1 entity. Guaranteed to be memory-efficient.
3.  **Feature Engineering:** Computes pairwise string distances (SequenceMatcher) and numerical Jaccard overlap (pinpoints matching PIN codes/building numbers).
4.  **Matching (Ranking):** Utilizes `HistGradientBoostingClassifier` trained directly on the engineered features. The probability threshold is kept intentionally high (>0.75) to penalize false merges and safely capture singletons.

## Prerequisites
* Python 3.10+
* `pip install -r requirements.txt`

## How to Run

1. Ensure you place the competition dataset in a folder named `dataset` located two levels above this code structure, matching the challenge schema:
```text
workspace/
├── dataset/
│   ├── train/
│   └── test/
├── output/
└── code/
    └── business_entity_resolution/
        ├── run_pipeline.py
        └── src/
```

2. Execute the pipeline:
```bash
python run_pipeline.py
```

3. The final files `candidate_pairs.tsv` and `matching_results.tsv` will be generated in the `output/` folder and strictly conform to the expected format.
