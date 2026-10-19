# Satellite FL Explorer (Streamlit prototype)

A first, runnable front-end + lightweight FedAvg demonstration for the team's satellite federated-learning educational website.

## What works in this prototype

- Streamlit navigation: Overview, Learn, Explore, Results, Research.
- Adjustable client count, IID/non-IID partitioning, Dirichlet concentration `alpha`, local epochs, rounds, and random seed.
- Preview charts showing the class distribution assigned to each simulated satellite.
- An actual lightweight FedAvg loop using a NumPy softmax classifier and synthetic 10-class feature data.
- Global accuracy/loss curves, estimated uncompressed model-upload bytes, local training-work counters, and recent-run comparison.
- Local FAQ-style concept helper and literature starting links.
- Optional JSON integration seam at `experiments/results/latest_results.json`.

## Important dataset caveat

The current runnable demo uses **synthetic land-cover-like feature vectors**, not actual satellite images and not EuroSAT. It is deliberately small so the interface can be developed and tested before the team integrates Member A's EuroSAT/CNN backend. Do not describe its metrics as EuroSAT results.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

The app should open in a browser. In Explore, select a partition and click **Run FedAvg simulation**. Then inspect Results.

## Team integration points

### 1. Replace the prototype experiment

Member A can replace `backend/fl_simulator.py` with the actual EuroSAT/CNN backend, or add a new implementation while preserving the return structure below.

The app currently imports:

```python
from backend.fl_simulator import CLASS_NAMES, make_preview_partition, run_demo_experiment
```

### 2. Expected result structure

The current interface expects `run_demo_experiment(...)` to return a dictionary shaped like:

```json
{
  "config": {
    "num_clients": 4,
    "partition": "Non-IID",
    "alpha": 0.3,
    "rounds": 10,
    "local_epochs": 1,
    "seed": 42
  },
  "class_names": ["Forest", "Residential"],
  "client_class_counts": [
    {"Forest": 70, "Residential": 20},
    {"Forest": 15, "Residential": 75}
  ],
  "metrics": {
    "global_accuracy": [0.1, 0.3, 0.5],
    "global_loss": [2.3, 1.8, 1.2],
    "client_accuracy": [[0.2, 0.3], [0.3, 0.4]]
  },
  "communication": {
    "parameters": 1000,
    "upload_bytes_per_round": 16000,
    "estimated_upload_bytes_total": 160000,
    "local_example_epochs": 4000
  },
  "runtime_seconds": 12.4,
  "status": "complete",
  "result_type": "experiment"
}
```

This is a schema example, not measured results. Accuracy/loss arrays should include the initial global model at index 0, followed by one value per round. `client_class_counts` should be a list of dictionaries whose keys are the actual class names.

The UI is deliberately tolerant of missing optional communication metrics, but please retain `config`, `metrics.global_accuracy`, `metrics.global_loss`, and `client_class_counts`.

### 3. Load precomputed results

If an experiment is too slow to run live, write a result JSON to:

```text
experiments/results/latest_results.json
```

The Results page has an adapter for this file. For the Explore page's Run button, the current local prototype is used until the team decides how to connect the production backend.

## Suggested Member B tasks

1. Run this prototype and check the page navigation.
2. Replace labels, colors and layout to match the agreed mockups.
3. Keep the JSON contract stable with Member A.
4. Add plotting/interpretation improvements.
5. Test on the same computer and Python environment planned for the demo.

## Current limitations / next steps

- Synthetic data, not imagery.
- Linear softmax classifier, not the final CNN.
- Simulated client partitions, not real satellite links or orbital visibility.
- FedAvg only; FedProx/FedAsync and realistic contact scheduling remain extensions.
- The concept helper is a local FAQ, not a generative-AI chatbot.
- Update-volume counters are simplified estimates and exclude protocol overhead, downlink traffic, retransmission, and actual radio/link budgets.
