# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains a local, reproducible pipeline for matching each Source 1 business record to zero, one, or many records in Sources 2 and 3. It uses only the supplied challenge data. The candidate stage combines normalized exact/compact name matches, country-scoped name prefixes, exact normalized addresses, and SQLite FTS5 name/address retrieval. A pairwise Logistic Regression model scores candidates; its threshold is selected on a held-out Source 1 validation sample using macro F0.5, including singleton entities.

## Data

Keep the supplied files in `data/train/` and `data/test/` with their original TSV names. The `data/` directory is ignored by Git. All readers use tab separators. Country values are treated as open-set labels, including France in test.

## Setup and run

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python utils/run_pipeline.py
```

The complete run builds a disk-backed SQLite index for the train target records, samples disjoint Source 1 entities from the fixed seed-42 80/20 split, fits a Logistic Regression pair classifier, sweeps thresholds against validation macro F0.5, saves the model, indexes test targets, and scores every test Source 1 entity. Candidate generation limits broad blocks to keep the large run tractable. See `reports/phase2_4_validation.json` for measured candidate recall, candidate volume, and the threshold sweep.

To run a stage separately:

```powershell
python utils/run_pipeline.py --phase train
python utils/run_pipeline.py --phase infer
```

Inference requires a model produced by the training stage. The full data audit and split generator are in `notebooks/01_data_exploration.ipynb` and `src/business_entity_resolution/data_loader.py`.

## Outputs

The final submission files are written to `outputs/final/`:

- `matching_results.tsv`: one row per test Source 1 ID; matched IDs may be empty or contain multiple S2/S3 IDs.
- `candidate_pairs.tsv`: the exact candidate IDs scored by the model. Every predicted match is checked to be in this set.
- `inference_summary.json`: row counts, threshold, and validation status.

Training writes `models/entity_matcher.joblib`, validation metrics to `reports/phase2_4_validation.json`, and a threshold-by-threshold CSV to `reports/experiments.csv`. Large local indexes and generated TSVs are ignored by Git and can be rebuilt. Run `python utils/validate_submission.py` after inference for an independent format and referential-integrity check.

## Phases and constraints

The implementation covers data validation/EDA, normalization and blocking, a classical pair model, entity-level macro F0.5 threshold selection, test inference, and output validation. Training and validation samples are controlled by constants in `config.py` so computation can be scaled. No external business lookup, geocoding, or data augmentation is used.
