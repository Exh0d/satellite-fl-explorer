# Satellite FL Explorer

A Streamlit educational dashboard for federated learning in satellite networks.

## Main pages

- **Overview**: navigation cards, conceptual satellite dashboard, and local Ollama tutor.
- **Learn**: guided explanations of FedAvg, IID/non-IID data, and satellite constraints.
- **Explore & Results**: measured pilot studies and a separate lightweight interactive prototype.
- **Knowledge Check**: scored multiple-choice questions with feedback.
- **Research**: literature directions and advanced satellite-FL concepts.

## Measured pilot studies

The repository includes Member A's offline FedAvg results under `experiments/results/pilot/`:

1. Data distribution: IID and Dirichlet alpha values 0.1, 0.2, 0.6, 1.0.
2. Number of clients: 2, 4, 8, 16.
3. Local epochs: 1, 2, 5.
4. Client participation: 12.5%, 25%, 50%, 75%, 100%.
5. Training images: 1,000, 2,000, 4,000, 6,000.

`index.csv` indexes the runs; each study folder contains `round_curves.csv`, per-run JSON files, summaries, and comparison plots. `manifest.json` records the configurations and split fingerprints. The Streamlit app loads the saved curves and run details; it does not retrain these measured experiments.

**Metric caveat:** the pilot round curves are validation accuracy and validation cross-entropy loss. `final_test_metrics` is null; the reserved test split was not evaluated in these pilot runs. Do not label validation curves as test accuracy. These are offline sequential simulations, not measurements from real satellites, orbital passes, or radio links.

## Interactive prototype

The separate Interactive prototype mode runs a lightweight FedAvg example on synthetic features. It is deliberately distinguished from the measured EuroSAT CNN experiments.

## Run locally

Python 3.10+ recommended.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Or activate the environment and run `streamlit run app.py`.

## Optional local AI tutor

The app can use a local Ollama model through the configured Ollama endpoint. For local use, install Ollama and pull the model configured near the top of `app.py`. Do not commit `.streamlit/secrets.toml` or any API credentials. The Streamlit-hosted app cannot reach Ollama on a developer's personal `localhost`; a hosted model endpoint is required for public deployment.

## Repository layout

- `app.py`: main Streamlit application and page routing.
- `backend/`: lightweight interactive prototype and measured-results data access helpers.
- `interface/`: Learn and measured-study visualization components.
- `content/`: quiz questions and explanations.
- `experiments/results/pilot/`: immutable offline measured experiment package.

## Reproducibility

The saved runs contain configuration metadata, random seed, validation/test split fingerprints, per-round validation metrics, client-level local training metrics, and actual participant IDs. The charts should be interpreted together with those recorded configurations; no general claim that one parameter value is always better is implied.
