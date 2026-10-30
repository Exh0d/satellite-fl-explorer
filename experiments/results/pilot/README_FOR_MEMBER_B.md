# Member B: Using the measured FedAvg pilot results

## Source of truth

- `manifest.json` records the actual selected run configurations and split fingerprints.
- `index.csv` indexes the 21 selected JSON runs and includes summary metadata.
- Each study folder has `round_curves.csv`, `summary.csv`, comparison plots, and per-run JSON files.
- `README_HANDOFF.md` documents the run-level schema and metric caveats.

Use `manifest.json` and each run's JSON `config` for actual settings. This document is a navigation guide, not an alternative source of experiment settings.

## The five studies included in this pilot

1. **Data distribution:** IID plus Dirichlet α = 0.1, 0.2, 0.6, and 1.0 (4 clients; 3,000 training images).
2. **Number of satellites:** 2, 4, 8, and 16 clients (3,000 training images).
3. **Local epochs:** 1, 2, and 5 epochs (4 clients; 3,000 training images).
4. **Client participation:** 12.5%, 25%, 50%, 75%, and 100% (8 clients; 5,000 training images).
5. **Training dataset size:** 1,000, 2,000, 4,000, and 6,000 training images (4 clients).

All cases ran for 50 communication rounds. Round 0 is the untrained initial global model; rounds 1–50 follow aggregation rounds.

## Recommended website visualizations

- Compare selected cases as separate lines for `validation_accuracy` and `validation_loss` versus `round` in each study's `round_curves.csv`.
- Inspect client allocation using `classes`, `client_class_counts`, and `client_sample_counts` from a run JSON.
- Inspect partial participation using `round_metrics[*].participating_clients`; client IDs are zero-based.
- Inspect local computation using `client_round_metrics[*].clients[*].local_optimizer_steps` and `train_seconds`.
- Show the configuration from each selected run alongside curves to make clear which variables changed and which were held fixed.

## Important caveats

- These are measured offline results from a local sequential FedAvg simulation on EuroSAT RGB data. They are not measurements from physical satellites, orbital propagation, radio links, or onboard satellite hardware.
- Round curves are **validation accuracy** and **validation cross-entropy loss**. Accuracy is in the range 0–1; format as a percentage for display.
- `final_test_metrics` is null in this pilot. The held-out test split is reserved but no final test evaluation was run. Never label pilot validation curves as test accuracy.
- Model-transfer figures are estimates of uncompressed model payload transfer and are not measured communication time or actual radio bandwidth.
- Use measured results to describe observed behavior. Do not assume the most heterogeneous or most resource-constrained setting must always perform worse.
