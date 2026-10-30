# Measured Satellite FL experiments — Member B handoff

Stage: **pilot**. All included experiments have **50 FedAvg communication rounds**.

Read `index.csv` for the selected cases. Use each study's `round_curves.csv`
for the website learning curves (x = `round`, y = `validation_accuracy` or
`validation_loss`). Accuracy is a fraction from 0 to 1; multiply by 100
for percentages. Round 0 is the untrained global model.

Each individual JSON contains `config`, `client_class_counts`,
`client_sample_counts`, `round_metrics`, and `client_round_metrics`.
`round_metrics[*].participating_clients` uses zero-based satellite IDs.

The results are **real measured runs of a local sequential FL simulation**;
they are not observations of satellite network latency or actual satellites.

For pilot results, round curves are on a separate **validation** split, and
`final_test_metrics` is null. Final-stage results have the same validation
curves plus one held-out test evaluation at the end. Do not label the
validation curves as test accuracy.

`manifest.json` records the *actual* parameter values and split hashes for
each selected case; it takes precedence over default examples in any older guide.
This ZIP includes only the cases listed in the CURRENT `run_studies.py`, not
previous experimental variants left in the source results directory.
