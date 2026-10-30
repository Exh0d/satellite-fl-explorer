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
        "round_metric_split": run.get("round_metric_split", "validation"),
        "final_test_metrics": run.get("final_test_metrics"),
    }


def render_measured_studies() -> dict[str, Any] | None:
    """Render the saved, measured 50-round parameter studies; return tutor context."""
    st.markdown(
        "<div class='hero'><div class='eyebrow'>MODULE 02 · MEASURED EXPERIMENTS</div>"
        "<h1>Measured Satellite FL Studies</h1>"
        "<p>Compare saved FedAvg experiments. Curves below are loaded from Member A's offline runs; no model is retrained here.</p></div>",
        unsafe_allow_html=True,
    )
    try:
        index = load_pilot_index()
    except Exception as exc:
        st.error(f"Measured experiment data could not be loaded: {type(exc).__name__}: {exc}")
        st.info("Check that experiments/results/pilot/ contains index.csv, manifest.json, and the five study folders.")
        return None

    study_key = st.selectbox(
        "Parameter study",
        options=list(STUDIES),
        format_func=lambda key: STUDIES[key]["label"],
        key="measured_study_key",
    )
    study_meta = STUDIES[study_key]
    study_rows = index[index["study"] == study_key].copy()
    if study_rows.empty:
        st.warning("No measured runs are indexed for this study.")
        return None
    available_cases = _format_case_order(study_key, study_rows["case"].astype(str).tolist())

    st.markdown(f"### {study_meta['parameter']}")
    st.write(study_meta["description"])
    st.caption(f"Experimental controls held fixed: {study_meta['fixed']}")

    selected_cases = st.multiselect(
        "Configurations to compare",
        options=available_cases,
        default=available_cases,
        key=f"measured_cases_{study_key}",
        help="Each selected line is an actual saved experiment. Unselected configurations are hidden, not recomputed.",
    )
    if not selected_cases:
        st.info("Select at least one configuration to display its measured curves.")
        return None

    selected_rows = study_rows[study_rows["case"].astype(str).isin(selected_cases)].copy()
    try:
        curves = load_study_curves(study_key)
        selected_curves = curves[curves["case"].astype(str).isin(selected_cases)].copy()
    except Exception as exc:
        st.error(f"Could not load measured curves: {type(exc).__name__}: {exc}")
        return None
    if selected_curves.empty:
        st.warning("No saved curve rows match the selected configurations. No placeholder curves have been generated.")
        return None

    # A compact end-of-study comparison complements the full round-by-round curves.
    summary_columns = [
        ("case", "Configuration"),
        ("final_validation_accuracy", "Final validation accuracy"),
        ("final_validation_loss", "Final validation loss"),
        ("total_elapsed_seconds", "Recorded run time (s)"),
    ]
    summary_columns = [(source, label) for source, label in summary_columns if source in selected_rows.columns]
    summary = selected_rows[[source for source, _ in summary_columns]].rename(
        columns={source: label for source, label in summary_columns}
    )
    st.markdown("#### Selected run summary · round 50")
    st.dataframe(
        summary.style.format({
            "Final validation accuracy": "{:.1%}",
            "Final validation loss": "{:.4f}",
            "Recorded run time (s)": "{:.1f} s",
        }),
        use_container_width=True,
        hide_index=True,
    )

    max_round = int(selected_curves["round"].max())
    selected_round = st.slider(
        "Inspect communication round",
        min_value=0, max_value=max_round, value=max_round, step=1,
        key=f"measured_round_{study_key}",
        help="Round 0 is the untrained initial global model; round 1 is the first aggregation.",
    )

    acc_col, loss_col = st.columns(2, gap="medium")
    with acc_col:
        st.plotly_chart(_make_curve_figure(selected_curves, "validation_accuracy", selected_round),
                        use_container_width=True, config={"displayModeBar": False}, key=f"measured_acc_{study_key}")
    with loss_col:
        st.plotly_chart(_make_curve_figure(selected_curves, "validation_loss", selected_round),
                        use_container_width=True, config={"displayModeBar": False}, key=f"measured_loss_{study_key}")

    round_rows = selected_curves[selected_curves["round"] == selected_round].copy()
    round_rows = round_rows.rename(columns={
        "case": "Configuration",
        "validation_accuracy": "Validation accuracy",
        "validation_loss": "Validation loss",
        "client_count": "Participating clients",
        "participating_samples": "Participating images",
        "local_optimizer_steps": "Local optimizer steps",
        "elapsed_seconds": "Elapsed seconds",
    })
    display_cols = [c for c in ["Configuration", "Validation accuracy", "Validation loss", "Participating clients",
                                "Participating images", "Local optimizer steps", "Elapsed seconds"] if c in round_rows.columns]
    st.markdown(f"#### Comparison at round {selected_round}")
    st.dataframe(
        round_rows[display_cols].style.format({
            "Validation accuracy": "{:.1%}", "Validation loss": "{:.4f}", "Elapsed seconds": "{:.1f} s",
            "Participating images": "{:,.0f}", "Local optimizer steps": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    case_for_detail = st.selectbox(
        "Inspect one configuration in detail",
        options=selected_cases,
        index=0,
        key=f"measured_focus_{study_key}",
    )
    focus_row = selected_rows[selected_rows["case"].astype(str) == case_for_detail].iloc[0]
    try:
        focus_run = load_run_json(str(focus_row["json_path"]))
    except Exception as exc:
        st.error(f"Could not load detailed run: {type(exc).__name__}: {exc}")
        return None

    rm = next((item for item in focus_run.get("round_metrics", []) if int(item.get("round", -1)) == selected_round), None)
    cfg = focus_run.get("config", {})
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Validation accuracy", f"{float(rm['accuracy']):.1%}" if rm else "—")
    r2.metric("Validation loss", f"{float(rm['loss']):.4f}" if rm else "—")
    r3.metric("Training images", f"{int(focus_run.get('train_size', 0)):,}")
    r4.metric("Recorded run time", f"{float(focus_run.get('total_elapsed_seconds', 0.0)):.1f} s")

    left, right = st.columns([1.15, .85], gap="medium")
    with left:
        if study_key == "alpha":
            fig = _client_distribution_figure(focus_run)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="measured_client_class_distribution")
        elif study_key in ("clients", "samples"):
            fig = _client_size_figure(focus_run)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="measured_client_sizes")
        elif study_key == "participation":
            fig = _participation_heatmap(focus_run)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="measured_participation_heatmap")
        elif study_key == "epochs":
            compute_fig = _local_compute_figure(selected_rows)
            if compute_fig:
                st.plotly_chart(compute_fig, use_container_width=True, config={"displayModeBar": False}, key="measured_local_compute")
        if study_key in ("clients", "samples", "participation"):
            # Distribution composition is useful context for interpreting all three studies.
            dist_fig = _client_distribution_figure(focus_run)
            if dist_fig:
                with st.expander("Inspect land-cover composition by satellite"):
                    st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False}, key="measured_optional_class_mix")

    with right, st.container(border=True):
        st.markdown("<div class='panel-title'>What this run changed</div>", unsafe_allow_html=True)
        st.write(f"**Configuration:** {case_for_detail}")
        st.write(f"**Model:** {cfg.get('dataset', 'EuroSAT RGB')} · {cfg.get('image_size', 32)}×{cfg.get('image_size', 32)} input images")
        st.write(f"**Training:** {int(cfg.get('rounds', 50))} rounds · {int(cfg.get('local_epochs', 1))} local epoch(s) · batch size {int(cfg.get('batch_size', 32))}")
        st.write(f"**Validation split:** {int(focus_run.get('validation_size', 0)):,} images")
        st.write(f"**Held-out test split:** {int(focus_run.get('test_size', 0)):,} images reserved; final test evaluation was not run in this pilot.")
        fixed_df = _fixed_setting_summary(selected_rows)
        if not fixed_df.empty:
            st.markdown("**Settings common to the selected study:**")
            st.dataframe(fixed_df, use_container_width=True, hide_index=True)

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
        "These are measured results from offline, sequential FedAvg simulations on EuroSAT RGB data. "
        "The lines show validation accuracy and validation loss—not held-out test performance. "
        "They are not measurements of real satellite radio links, orbital visibility, or onboard flight hardware."
    )
    return _make_tutor_result(focus_row, focus_run, selected_round)
