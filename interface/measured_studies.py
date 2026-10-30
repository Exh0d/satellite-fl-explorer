"""Browser-side presentation for Member A's offline measured FedAvg studies.

The pilot JSON/CSV files are precomputed experimental results. This module only
loads and visualizes them; it never retrains a model or invents missing curves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = PROJECT_ROOT / "experiments" / "results" / "pilot"

STUDIES: dict[str, dict[str, str]] = {
    "alpha": {
        "label": "Data distribution (IID vs. non-IID)",
        "folder": "01_data_distribution",
        "parameter": "Dirichlet concentration α",
        "description": (
            "The concentration parameter α controls how class examples are distributed across clients. "
            "Smaller α values generally create more uneven local class mixtures; IID is the balanced "
            "reference condition. The observed accuracy and loss changes below come from the saved runs, "
            "not from an assumed trend."
        ),
        "fixed": "4 satellites · 3,000 training images · 1 local epoch · 100% participation · 50 rounds",
    },
    "clients": {
        "label": "Number of satellites",
        "folder": "02_number_of_satellites",
        "parameter": "Number of clients K",
        "description": (
            "The same 3,000 training images are divided among different numbers of clients. "
            "As K increases, each satellite receives fewer examples on average. Compare both the "
            "global validation curves and the actual sample allocation per satellite."
        ),
        "fixed": "3,000 training images · α = 2.0 · 1 local epoch · 100% participation · 50 rounds",
    },
    "epochs": {
        "label": "Local epochs",
        "folder": "03_local_epochs",
        "parameter": "Local epochs E",
        "description": (
            "Local epochs are the number of passes over a client's local dataset before its model update "
            "is aggregated. More local epochs increase local computation per communication round. "
            "Inspect the measured curves alongside optimizer steps and training time; more local work "
            "does not guarantee a better outcome in every setting."
        ),
        "fixed": "4 satellites · 3,000 training images · α = 2.0 · 100% participation · 50 rounds",
    },
    "participation": {
        "label": "Client participation",
        "folder": "04_client_participation",
        "parameter": "Participation rate C",
        "description": (
            "The configured participation rate determines what fraction of the 8 clients are selected "
            "for each communication round. The selected clients are randomly resampled each round. "
            "The heatmap shows the actual saved participation schedule."
        ),
        "fixed": "8 satellites · 5,000 training images · α = 2.0 · 1 local epoch · 50 rounds",
    },
    "samples": {
        "label": "Training dataset size",
        "folder": "05_training_data_size",
        "parameter": "Number of unique training images",
        "description": (
            "This study changes the number of training images while retaining the same validation set. "
            "More training images also create more local mini-batches and optimizer work per round. "
            "Compare the learning curves with the client image allocation and measured runtime."
        ),
        "fixed": "4 satellites · α = 2.0 · 1 local epoch · 100% participation · 50 rounds",
    },
}


@st.cache_data(show_spinner=False)
def load_pilot_manifest() -> list[dict[str, Any]]:
    path = PILOT_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Measured-results manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list):
        raise ValueError("manifest.json must contain a list of experiment records")
    return manifest


@st.cache_data(show_spinner=False)
def load_pilot_index() -> pd.DataFrame:
    path = PILOT_DIR / "index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Measured-results index not found: {path}")
    frame = pd.read_csv(path)
    required = {"study", "case", "json_path", "final_validation_accuracy", "final_validation_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"index.csv is missing required columns: {sorted(missing)}")

    # Ensure the CSV index and manifest refer to the same measured cases.
    manifest = load_pilot_manifest()
    indexed = {(str(row.study), str(row.json_path)) for row in frame.itertuples()}
    declared = {(str(row.get("study", "")), str(row.get("json_path", ""))) for row in manifest}
    if indexed != declared:
        raise ValueError("index.csv and manifest.json do not reference the same experiment runs")
    return frame


@st.cache_data(show_spinner=False)
def load_study_curves(study_key: str) -> pd.DataFrame:
    if study_key not in STUDIES:
        raise ValueError(f"Unknown study: {study_key}")
    path = PILOT_DIR / STUDIES[study_key]["folder"] / "round_curves.csv"
    if not path.exists():
        raise FileNotFoundError(f"Round curves not found: {path}")
    frame = pd.read_csv(path)
    required = {"case", "round", "validation_accuracy", "validation_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    frame["validation_accuracy"] = pd.to_numeric(frame["validation_accuracy"], errors="coerce")
    frame["validation_loss"] = pd.to_numeric(frame["validation_loss"], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def load_run_json(json_path: str) -> dict[str, Any]:
    # index.json_path is relative to pilot/. Do not allow absolute or parent paths.
    candidate = (PILOT_DIR / json_path).resolve()
    root = PILOT_DIR.resolve()
    if root not in candidate.parents:
        raise ValueError("Invalid result path in index.csv")
    if not candidate.exists():
        raise FileNotFoundError(f"Measured run JSON not found: {json_path}")
    with candidate.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("result_source") != "measured":
        raise ValueError(f"Run is not tagged as a measured result: {json_path}")
    if result.get("round_metric_split") != "validation":
        raise ValueError(f"Expected validation curves in run: {json_path}")
    return result


def _format_case_order(study_key: str, cases: list[str]) -> list[str]:
    """Preserve the case order from index.csv (which reflects the experiment design)."""
    return cases


def _make_curve_figure(frame: pd.DataFrame, metric: str, selected_round: int) -> go.Figure:
    is_accuracy = metric == "validation_accuracy"
    title = "Global validation accuracy" if is_accuracy else "Global cross-entropy validation loss"
    fig = px.line(
        frame,
        x="round",
        y=metric,
        color="case",
        markers=False,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data={"round": True, metric: ":.4f", "case": True},
    )
    fig.add_vline(x=selected_round, line_dash="dot", line_color="#546e7a", opacity=0.8)
    fig.update_xaxes(title="Communication round", dtick=5, range=[0, 50], showgrid=False)
    if is_accuracy:
        fig.update_yaxes(title="Validation accuracy", tickformat=".0%", rangemode="tozero", range=[0, 1.02])
    else:
        fig.update_yaxes(title="Cross-entropy loss", rangemode="tozero")
    fig.update_layout(
        height=350,
        margin=dict(l=12, r=12, t=58, b=12),
        legend_title_text="Configuration",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def _client_distribution_figure(run: dict[str, Any]) -> go.Figure | None:
    classes = run.get("classes", [])
    counts = run.get("client_class_counts", [])
    if not classes or not counts:
        return None
    rows: list[dict[str, Any]] = []
    for client_id, client_counts in enumerate(counts):
        if isinstance(client_counts, dict):
            pairs = client_counts.items()
        else:
            pairs = zip(classes, client_counts)
        for class_name, value in pairs:
            rows.append({"Satellite": f"Sat {client_id + 1}", "Land-cover class": str(class_name), "Images": int(value)})
    if not rows:
        return None
    fig = px.bar(
        pd.DataFrame(rows), x="Satellite", y="Images", color="Land-cover class",
        barmode="stack", title="Training images by satellite and class",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(height=355, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="Class")
    return fig


def _client_size_figure(run: dict[str, Any]) -> go.Figure | None:
    counts = run.get("client_sample_counts", [])
    if not counts:
        return None
    df = pd.DataFrame({"Satellite": [f"Sat {i + 1}" for i in range(len(counts))], "Training images": counts})
    fig = px.bar(df, x="Satellite", y="Training images", text="Training images", title="Images assigned to each satellite")
    fig.update_traces(textposition="outside", cliponaxis=False, marker_color="#3786df")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
    return fig


def _participation_heatmap(run: dict[str, Any]) -> go.Figure | None:
    config = run.get("config", {})
    n_clients = int(config.get("num_clients", 0))
    round_metrics = run.get("round_metrics", [])
    if n_clients <= 0 or not round_metrics:
        return None
    rounds = [int(row.get("round", 0)) for row in round_metrics if int(row.get("round", 0)) > 0]
    if not rounds:
        return None
    max_round = max(rounds)
    matrix = np.zeros((n_clients, max_round), dtype=int)
    for row in round_metrics:
        round_no = int(row.get("round", 0))
        if round_no <= 0 or round_no > max_round:
            continue
        for client_id in row.get("participating_clients", []):
            client_id = int(client_id)
            if 0 <= client_id < n_clients:
                matrix[client_id, round_no - 1] = 1
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(1, max_round + 1)),
        y=[f"Satellite {i + 1}" for i in range(n_clients)],
        zmin=0, zmax=1,
        colorscale=[[0, "#e8eef5"], [1, "#2478e5"]],
        colorbar=dict(title="Participated", tickvals=[0, 1], ticktext=["No", "Yes"]),
        hovertemplate="%{y}<br>Round %{x}<br>Participated: %{z}<extra></extra>",
    ))
    fig.update_layout(title="Actual client participation by round", height=max(300, n_clients * 28),
                      margin=dict(l=10, r=10, t=55, b=10), xaxis_title="Communication round", yaxis_title=None)
    return fig


def _local_compute_figure(selected_rows: pd.DataFrame) -> go.Figure | None:
    records: list[dict[str, Any]] = []
    for _, row in selected_rows.iterrows():
        try:
            run = load_run_json(str(row["json_path"]))
        except Exception:
            continue
        for round_record in run.get("client_round_metrics", []):
            round_no = int(round_record.get("round", 0))
            clients = round_record.get("clients", [])
            records.append({
                "case": str(row["case"]),
                "round": round_no,
                "Local optimizer steps": sum(int(c.get("local_optimizer_steps", 0)) for c in clients),
                "Client training seconds (summed)": sum(float(c.get("train_seconds", 0.0)) for c in clients),
            })
    if not records:
        return None
    frame = pd.DataFrame(records)
    fig = px.line(frame, x="round", y="Local optimizer steps", color="case", title="Local optimizer steps per communication round",
                  markers=False, color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=55, b=10), xaxis_title="Communication round")
    return fig


def _fixed_setting_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Show values common to the selected study, not all raw manifest columns."""
    if rows.empty:
        return pd.DataFrame()
    first = rows.iloc[0]
    labels = [
        ("Dataset", "dataset"),
        ("Satellites", "num_clients"),
        ("Training images", "train_images"),
        ("Dirichlet α", "alpha"),
        ("Local epochs", "local_epochs"),
        ("Participation", "participation_rate"),
        ("Rounds", "rounds"),
        ("Validation images", "validation_images"),
    ]
    records = []
    for label, field in labels:
        if field not in rows.columns:
            continue
        vals = rows[field].dropna().unique().tolist()
        if len(vals) == 1:
            value = vals[0]
            if field == "participation_rate":
                value = f"{float(value):.0%}"
            elif field == "alpha":
                value = "IID reference" if pd.isna(value) else f"{float(value):g}"
            elif field in ("train_images", "validation_images", "num_clients", "rounds", "local_epochs"):
                value = f"{int(value):,}"
            records.append({"Held constant across this study": label, "Value": value})
    return pd.DataFrame(records)


def _make_tutor_result(row: pd.Series, run: dict[str, Any], selected_round: int) -> dict[str, Any]:
    rounds = run.get("round_metrics", [])
    selected_metrics = next((item for item in rounds if int(item.get("round", -1)) == int(selected_round)), None)
    return {
        "result_source": "measured",
        "config": {**run.get("config", {}), "study": str(row.get("study", "")), "case": str(row.get("case", "")),
                   "selected_round": int(selected_round)},
        "selected_round_metrics": selected_metrics,
        "metrics": {
            "global_accuracy": [float(item.get("accuracy", 0.0)) for item in rounds],
            "global_loss": [float(item.get("loss", 0.0)) for item in rounds],
        },
        "participation": {
            "clients_per_round": [item.get("participating_clients", []) for item in rounds],
        },
        "client_sizes": run.get("client_sample_counts", []),
        "class_names": run.get("classes", []),
        "client_class_counts": run.get("client_class_counts", []),
        "runtime_seconds": float(run.get("total_elapsed_seconds", 0.0)),
        "round_metric_split": run.get("round_metric_split", "validation"),
        "final_test_metrics": run.get("final_test_metrics"),
    }


def load_default_measured_result() -> dict[str, Any] | None:
    """Return a deterministic measured run for overview/tutor context before selection."""
    try:
        index = load_pilot_index()
        rows = index[index["study"] == "alpha"]
        if rows.empty:
            return None
        row = rows[rows["case"].astype(str) == "IID"]
        if row.empty:
            row = rows.iloc[[0]]
        selected = row.iloc[0]
        run = load_run_json(str(selected["json_path"]))
        rounds = run.get("round_metrics", [])
        last_round = max((int(item.get("round", 0)) for item in rounds), default=0)
        return _make_tutor_result(selected, run, last_round)
    except Exception:
        return None


def _satellite_map_figure(
    run: dict[str, Any], selected_round: int, participants: list[int] | None = None
) -> go.Figure:
    """Map the selected measured run's clients and round participation.

    Geographic positions and coverage ellipses are schematic only. The client
    count and participating-client status come from the selected saved run.
    """
    config = run.get("config", {})
    n_clients = int(config.get("num_clients", len(run.get("client_sample_counts", [])) or 1))
    n_clients = max(1, min(n_clients, 32))
    active = {int(x) for x in (participants or [])}

    # Fixed illustrative positions, extended deterministically for larger K.
    base_points = [
        (58, -150), (34, -112), (12, -78), (-20, -62), (52, -35),
        (24, 8), (-18, 33), (42, 72), (9, 105), (-27, 134),
        (52, 151), (67, 45), (-43, 172), (2, 158), (31, 28), (-5, -12),
    ]
    points = list(base_points[:n_clients])
    # The pilot only uses up to 16 clients; this deterministic extension keeps
    # the function safe if future experiments add more clients.
    if n_clients > len(points):
        for i in range(len(points), n_clients):
            lat = -62 + ((i * 29) % 124)
            lon = -170 + ((i * 47) % 340)
            points.append((lat, lon))

    palette = [
        "#3977e8", "#43b978", "#f0a21a", "#df6558", "#8866c7", "#18a6b3",
        "#c35b98", "#6f879c", "#8b9b32", "#d17d36", "#387f8e", "#ba6e9f",
        "#4c8b50", "#9262b5", "#b67c2d", "#3f83bb",
    ]
    ground_lat, ground_lon = 39.7, -104.9  # illustrative reference ground station
    fig = go.Figure()

    # Soft coverage ellipses are visual guides, not calculated RF footprints.
    angles = np.linspace(0, 2 * np.pi, 64)
    for i, (lat, lon) in enumerate(points):
        radius_lat = 7.5
        radius_lon = radius_lat / max(0.45, np.cos(np.radians(lat)))
        ring_lat = np.clip(lat + radius_lat * np.sin(angles), -88, 88)
        ring_lon = ((lon + radius_lon * np.cos(angles) + 180) % 360) - 180
        fig.add_trace(go.Scattergeo(
            lat=list(ring_lat) + [float(ring_lat[0])],
            lon=list(ring_lon) + [float(ring_lon[0])],
            mode="lines", fill="toself",
            fillcolor="rgba(" + ",".join(str(int(palette[i % len(palette)][j:j+2], 16)) for j in (1, 3, 5)) + ",0.10)",
            line=dict(color=palette[i % len(palette)], width=0.8),
            hoverinfo="skip", showlegend=False,
        ))

    # Links are highlighted only for clients recorded in this round's run log.
    link_lat, link_lon = [], []
    for idx in sorted(active):
        if 0 <= idx < n_clients:
            lat, lon = points[idx]
            link_lat.extend([ground_lat, lat, None])
            link_lon.extend([ground_lon, lon, None])
    if link_lat:
        fig.add_trace(go.Scattergeo(
            lat=link_lat, lon=link_lon, mode="lines", name="Participated this round",
            line=dict(color="#38a89b", width=1.8), opacity=0.85,
            hoverinfo="skip",
        ))

    # All clients remain visible; active clients use their color, others are grey.
    for i, (lat, lon) in enumerate(points):
        participating = i in active
        status = "Participated in selected round" if participating else "Did not participate in selected round"
        marker_color = palette[i % len(palette)] if participating else "#a9b5c2"
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers+text", text=[f"S{i + 1}"],
            textposition="top center", name=f"Satellite {i + 1}",
            hovertemplate=(f"Satellite {i + 1}<br>{status}<extra></extra>"),
            marker=dict(size=11, color=marker_color, line=dict(width=1.5, color="white")),
            showlegend=False,
        ))

    fig.add_trace(go.Scattergeo(
        lat=[ground_lat], lon=[ground_lon], mode="markers+text", text=["Ground station"],
        textposition="bottom center", name="Ground station",
        hovertemplate="Illustrative ground station<extra></extra>",
        marker=dict(size=14, symbol="diamond", color="#3659b5", line=dict(width=1.5, color="white")),
    ))
    fig.update_geos(
        scope="world", projection_type="natural earth",
        showland=True, landcolor="#e3ebdf",
        showocean=True, oceancolor="#eaf3fc",
        showcoastlines=True, coastlinecolor="#9db2c5",
        showcountries=True, countrycolor="#c4d0d8",
        showlakes=True, lakecolor="#eaf3fc",
        bgcolor="#f7fbff",
    )
    fig.update_layout(
        title=f"{n_clients} simulated satellite clients · round {selected_round}",
        height=355, margin=dict(l=4, r=4, t=45, b=4),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="left", x=0),
    )
    return fig


def _selected_round_record(run: dict[str, Any], selected_round: int) -> dict[str, Any] | None:
    return next(
        (item for item in run.get("round_metrics", [])
         if int(item.get("round", -1)) == int(selected_round)),
        None,
    )


@st.cache_data(show_spinner=False)
def load_core_configuration_catalog() -> pd.DataFrame:
    """Return unique measured runs for the K/E/data-distribution control grid.

    Only the controlled baseline family is included: EuroSAT RGB, 3,000 train
    images, 500 validation images, 50 rounds, full participation, seed 42.
    Other studies (participation and training-set size) intentionally remain
    separate because their fixed settings differ.
    """
    manifest = load_pilot_manifest()
    index = load_pilot_index()
    index_by_path = {str(row["json_path"]): row for _, row in index.iterrows()}
    records: list[dict[str, Any]] = []
    for item in manifest:
        cfg = item.get("config", {})
        if cfg.get("dataset") != "EuroSAT RGB":
            continue
        if int(cfg.get("train_limit", -1)) != 3000:
            continue
        if int(cfg.get("validation_limit", -1)) != 500:
            continue
        if int(cfg.get("rounds", -1)) != 50 or int(cfg.get("seed", -1)) != 42:
            continue
        if not np.isclose(float(cfg.get("participation_rate", 0.0)), 1.0):
            continue
        if int(cfg.get("num_clients", 0)) not in (2, 4, 8, 16):
            continue
        if int(cfg.get("local_epochs", 0)) not in (1, 2, 5):
            continue
        partition = str(cfg.get("partition", "")).lower()
        if partition not in ("iid", "dirichlet"):
            continue
        alpha = cfg.get("alpha")
        if partition == "iid":
            alpha = None
        else:
            if alpha is None:
                continue
            alpha = float(alpha)
        path = str(item["json_path"])
        index_row = index_by_path.get(path)
        if index_row is None:
            continue
        distribution_label = "IID" if partition == "iid" else f"Dirichlet α={alpha:g}"
        combo_label = (
            f"{int(cfg['num_clients'])} satellites · {int(cfg['local_epochs'])} epoch(s) · {distribution_label}"
        )
        record = index_row.to_dict()
        record.update({
            "num_clients": int(cfg["num_clients"]),
            "local_epochs": int(cfg["local_epochs"]),
            "partition": partition,
            "alpha": alpha,
            "train_images": int(cfg["train_limit"]),
            "validation_images": int(cfg["validation_limit"]),
            "participation_rate": float(cfg["participation_rate"]),
            "rounds": int(cfg["rounds"]),
            "seed": int(cfg["seed"]),
            "combo_label": combo_label,
            "combo_key": f"K{int(cfg['num_clients'])}_E{int(cfg['local_epochs'])}_{partition}_{'iid' if alpha is None else f'{alpha:g}'}",
        })
        records.append(record)

    if not records:
        return pd.DataFrame()

    # The K=4, E=1, alpha=2.0 baseline appears in both the client and epoch
    # studies. Keep one canonical row so the combination is not duplicated.
    priority = {"clients": 0, "epochs": 1, "alpha": 2}
    records.sort(key=lambda r: (priority.get(str(r.get("study")), 9), str(r["json_path"])))
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(str(record["combo_key"]), record)
    frame = pd.DataFrame(list(unique.values()))
    return frame.sort_values(["num_clients", "local_epochs", "partition", "alpha"], na_position="first").reset_index(drop=True)


def _build_curves_from_runs(rows: pd.DataFrame) -> pd.DataFrame:
    """Build chart rows directly from measured JSON run histories."""
    output: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        run = load_run_json(str(row["json_path"]))
        label = str(row["combo_label"])
        for metric in run.get("round_metrics", []):
            output.append({
                "case": label,
                "round": int(metric.get("round", 0)),
                "validation_accuracy": float(metric.get("accuracy", np.nan)),
                "validation_loss": float(metric.get("loss", np.nan)),
            })
    return pd.DataFrame(output)


def _format_distribution(partition: str, alpha: Any) -> str:
    return "IID" if partition == "iid" else f"Dirichlet non-IID (α = {float(alpha):g})"


def render_measured_studies() -> dict[str, Any] | None:
    """Render a combined K / local-epochs / distribution selector.

    The selected triplet is matched to an exact saved offline experiment.
    Missing combinations are reported as unavailable; no interpolation,
    synthetic curve, or new training run is substituted.
    """
    st.markdown(
        "<div class='hero'><div class='eyebrow'>MODULE 02 · EXPERIMENT EXPLORER</div>"
        "<h1>🛰️ Explore & Results: Satellite Federated Learning</h1>"
        "<p>Choose the number of satellites, local epochs, and data distribution together. "
        "The dashboard displays an exact saved EuroSAT/FedAvg run when one exists.</p></div>",
        unsafe_allow_html=True,
    )

    try:
        catalog = load_core_configuration_catalog()
    except Exception as exc:
        st.error(f"Measured experiment data could not be loaded: {type(exc).__name__}: {exc}")
        return None
    if catalog.empty:
        st.error("No measured baseline configurations were found in experiments/results/pilot/.")
        return None

    control_col, clients_col, map_col = st.columns([0.92, 1.18, 1.72], gap="small")
    with control_col:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>⚙️ Simulation Controls</div>", unsafe_allow_html=True)
            selected_k = st.selectbox(
                "Number of satellites (K)", [2, 4, 8, 16], index=1, key="combo_num_satellites"
            )
            selected_epochs = st.selectbox(
                "Local epochs (E)", [1, 2, 5], index=0, key="combo_local_epochs"
            )
            distribution_mode = st.selectbox(
                "Data distribution", ["IID", "Non-IID (Dirichlet)"], index=1,
                key="combo_distribution_mode",
            )
            if distribution_mode == "IID":
                selected_partition = "iid"
                selected_alpha = None
                st.caption("IID uses a broadly balanced class distribution; no Dirichlet α applies.")
            else:
                selected_partition = "dirichlet"
                selected_alpha = st.selectbox(
                    "Dirichlet concentration (α)", [0.1, 0.2, 0.6, 1.0, 2.0], index=1,
                    format_func=lambda x: f"{x:g}", key="combo_alpha",
                    help="Smaller α generally creates more uneven class mixtures across clients.",
                )

            st.markdown("**Other settings held fixed**")
            st.caption("3,000 training images · 500 validation images · 100% client participation · 50 rounds · seed 42")
            st.caption(f"{len(catalog)} of 72 possible K / epoch / distribution combinations currently have saved results.")
            st.caption("The controls select measured runs. They do not retrain the CNN.")
            with st.expander("Show currently available measured combinations"):
                availability_view = catalog[["combo_label", "study", "json_path"]].rename(columns={
                    "combo_label": "Measured combination", "study": "Original parameter study", "json_path": "Result file"
                })
                st.dataframe(availability_view, use_container_width=True, hide_index=True)

    # Match all three user-selected parameters against an exact recorded run.
    matches = catalog[
        (catalog["num_clients"].astype(int) == int(selected_k))
        & (catalog["local_epochs"].astype(int) == int(selected_epochs))
        & (catalog["partition"].astype(str) == selected_partition)
    ].copy()
    if selected_partition == "iid":
        matches = matches[matches["alpha"].isna()]
    else:
        matches = matches[
            matches["alpha"].notna()
            & np.isclose(pd.to_numeric(matches["alpha"], errors="coerce"), float(selected_alpha))
        ]

    focus_run: dict[str, Any] | None = None
    focus_row: pd.Series | None = None
    selected_round = 50
    participants: list[int] = []
    cfg: dict[str, Any] = {
        "num_clients": int(selected_k),
        "local_epochs": int(selected_epochs),
        "partition": selected_partition,
        "alpha": selected_alpha,
        "train_limit": 3000,
        "validation_limit": 500,
        "participation_rate": 1.0,
        "rounds": 50,
        "seed": 42,
    }

    if not matches.empty:
        # Catalog is deduplicated; this should resolve to exactly one run.
        focus_row = matches.iloc[0]
        try:
            focus_run = load_run_json(str(focus_row["json_path"]))
            cfg = focus_run.get("config", cfg)
        except Exception as exc:
            st.error(f"Could not load the matching measured run: {type(exc).__name__}: {exc}")
            focus_run = None

    with control_col:
        if focus_run is not None:
            st.success("Exact measured configuration available")
            st.caption(str(focus_row["combo_label"]))
            max_round = max(
                (int(item.get("round", 0)) for item in focus_run.get("round_metrics", [])),
                default=50,
            )
            selected_round = st.slider(
                "Communication round", min_value=0, max_value=max_round,
                value=min(50, max_round), step=1, key="combo_selected_round",
                help="The selected round updates the map participation and the round-level metrics.",
            )
            round_record = _selected_round_record(focus_run, selected_round)
            participants = (round_record or {}).get("participating_clients", [])
        else:
            st.warning("No measured data for this combination yet.")
            st.caption("Member A can add this exact K / epoch / distribution combination in a future experiment batch.")
            st.caption("No curve, client distribution, or participation schedule is estimated or substituted.")
            st.slider(
                "Communication round (no saved run)", min_value=0, max_value=50,
                value=50, step=1, disabled=True, key="combo_selected_round_missing",
            )

    # Optional comparisons: only actual runs from the same 3,000-image baseline catalog.
    comparison_rows = pd.DataFrame()
    if focus_run is not None and focus_row is not None:
        other_catalog = catalog[catalog["combo_key"] != str(focus_row["combo_key"])].copy()
        with st.expander("Compare with other measured configurations", expanded=False):
            comparison_labels = st.multiselect(
                "Additional saved configurations",
                options=other_catalog["combo_label"].tolist(),
                default=[], key="combo_comparison_labels",
                help="Only measured configurations in the supplied results package are available.",
            )
        comparison_rows = other_catalog[other_catalog["combo_label"].isin(comparison_labels)].copy()
        selected_rows = pd.concat([matches, comparison_rows], ignore_index=True).drop_duplicates("combo_key")
    else:
        selected_rows = pd.DataFrame()

    with clients_col:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>🛰️ Satellite Clients · Local Data Distribution</div>", unsafe_allow_html=True)
            if focus_run is not None:
                st.caption("Actual class counts from the selected measured run.")
                dist_fig = _client_distribution_figure(focus_run)
                if dist_fig:
                    dist_fig.update_layout(height=355, margin=dict(l=4, r=4, t=35, b=4), legend_title_text="Class")
                    st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False}, key="combo_measured_distribution")
                else:
                    st.info("This measured run does not include per-client class counts.")
                client_sizes = focus_run.get("client_sample_counts", [])
                if client_sizes:
                    total = sum(int(x) for x in client_sizes)
                    st.caption(" · ".join(f"Sat {i+1}: {int(n):,}" for i, n in enumerate(client_sizes)))
                    st.caption(f"{total:,} training images distributed across {len(client_sizes)} satellites.")
            else:
                st.info("No measured client distribution is available for the selected combination.")
                st.caption("The distribution chart will populate when Member A supplies this run's client-level class counts.")

    with map_col:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>🌍 Satellite Ground Tracks & Contact Regions</div>", unsafe_allow_html=True)
            # Always render the requested number of clients. If no measured run
            # exists, render neutral markers without claiming participation data.
            map_run = focus_run or {"config": {"num_clients": int(selected_k)}}
            map_fig = _satellite_map_figure(map_run, selected_round, participants)
            st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False}, key="combo_measured_map")
            if focus_run is not None:
                active_n = len(participants)
                st.caption(
                    f"Round {selected_round}: {active_n}/{int(cfg.get('num_clients', selected_k))} clients participated. "
                    "Client status comes from the selected saved run; positions and coverage areas are illustrative, not real orbital propagation."
                )
            else:
                st.caption(
                    f"Showing {int(selected_k)} requested satellite markers only. No participation data exist for this combination yet. "
                    "Positions and coverage areas are illustrative."
                )

    st.markdown("### Measured training performance")
    if focus_run is None or focus_row is None:
        st.warning(
            f"No saved experiment matches K={selected_k}, E={selected_epochs}, "
            f"{_format_distribution(selected_partition, selected_alpha)}. "
            "Change one of the controls to a measured configuration, or wait for the missing combination to be added."
        )
        available_for_k_e = catalog[
            (catalog["num_clients"].astype(int) == int(selected_k))
            & (catalog["local_epochs"].astype(int) == int(selected_epochs))
        ]
        if not available_for_k_e.empty:
            st.markdown("**Measured distributions currently available for this satellite/epoch setting:**")
            st.write(" · ".join(
                _format_distribution(str(row["partition"]), row["alpha"])
                for _, row in available_for_k_e.iterrows()
            ))
        else:
            st.caption("There are currently no measured runs for this exact satellite-count and local-epoch pair.")
        st.session_state["active_measured_result"] = {
            "result_source": "unavailable",
            "config": cfg,
            "metrics": {},
            "availability_note": "No exact measured configuration exists in the current pilot package.",
        }
        return st.session_state["active_measured_result"]

    try:
        selected_curves = _build_curves_from_runs(selected_rows)
    except Exception as exc:
        st.error(f"Could not build measured comparison curves: {type(exc).__name__}: {exc}")
        return None
    if selected_curves.empty:
        st.warning("The selected measured JSON files contain no round histories.")
        return None

    st.caption(
        f"Selected: K={selected_k} satellites · E={selected_epochs} local epoch(s) · "
        f"{_format_distribution(selected_partition, selected_alpha)}. "
        "Additional comparison curves, if selected, are other exact saved runs."
    )
    acc_col, loss_col = st.columns(2, gap="medium")
    with acc_col:
        st.plotly_chart(
            _make_curve_figure(selected_curves, "validation_accuracy", selected_round),
            use_container_width=True, config={"displayModeBar": False}, key="combo_measured_accuracy",
        )
    with loss_col:
        st.plotly_chart(
            _make_curve_figure(selected_curves, "validation_loss", selected_round),
            use_container_width=True, config={"displayModeBar": False}, key="combo_measured_loss",
        )

    # Summaries are taken from the saved JSON histories and the index metadata.
    summary_records: list[dict[str, Any]] = []
    round_records: list[dict[str, Any]] = []
    for _, row in selected_rows.iterrows():
        run = load_run_json(str(row["json_path"]))
        round_item = _selected_round_record(run, selected_round)
        history = run.get("round_metrics", [])
        last = history[-1] if history else {}
        summary_records.append({
            "Configuration": row["combo_label"],
            "Final validation accuracy": float(last.get("accuracy", np.nan)),
            "Final validation loss": float(last.get("loss", np.nan)),
            "Recorded runtime (s)": float(run.get("total_elapsed_seconds", np.nan)),
        })
        if round_item:
            round_records.append({
                "Configuration": row["combo_label"],
                "Validation accuracy": float(round_item.get("accuracy", np.nan)),
                "Validation loss": float(round_item.get("loss", np.nan)),
                "Participating clients": len(round_item.get("participating_clients", [])),
                "Elapsed seconds": float(round_item.get("elapsed_seconds", np.nan)),
            })
    st.markdown("#### Final-round comparison")
    st.dataframe(pd.DataFrame(summary_records).style.format({
        "Final validation accuracy": "{:.1%}",
        "Final validation loss": "{:.4f}",
        "Recorded runtime (s)": "{:.1f} s",
    }), use_container_width=True, hide_index=True)

    st.markdown(f"#### Selected-round comparison · round {selected_round}")
    if round_records:
        st.dataframe(pd.DataFrame(round_records).style.format({
            "Validation accuracy": "{:.1%}",
            "Validation loss": "{:.4f}",
            "Elapsed seconds": "{:.1f} s",
        }), use_container_width=True, hide_index=True)

    with st.expander("Inspect the full 50-round client participation schedule"):
        heatmap = _participation_heatmap(focus_run)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False}, key="combo_participation_heatmap")
        else:
            st.info("No per-round participation schedule is included in this run.")
    with st.expander("Inspect local computation per round"):
        compute_fig = _local_compute_figure(selected_rows)
        if compute_fig:
            st.plotly_chart(compute_fig, use_container_width=True, config={"displayModeBar": False}, key="combo_local_compute")
        else:
            st.info("No client-level optimizer-step records are included in these runs.")

    with st.expander("Run configuration and provenance"):
        st.json(cfg)
        st.write({
            "result_source": focus_run.get("result_source"),
            "round_metric_split": focus_run.get("round_metric_split"),
            "split_fingerprints": focus_run.get("split_fingerprints"),
            "model_parameters": focus_run.get("model_parameters"),
            "model_bytes": focus_run.get("model_bytes"),
            "total_client_updates": focus_run.get("total_client_updates"),
            "final_test_metrics": focus_run.get("final_test_metrics"),
            "notes": focus_run.get("notes", []),
        })

    st.info(
        "All curves and client-level data are loaded from Member A's saved offline EuroSAT/FedAvg runs. "
        "The three controls select an exact recorded combination. If a combination is missing, the app reports no data "
        "instead of estimating or interpolating a curve. The map's satellite positions and coverage regions are illustrative. "
        "Curves show validation accuracy and validation cross-entropy loss; the pilot did not run final held-out test evaluation."
    )
    return _make_tutor_result(focus_row, focus_run, selected_round)
