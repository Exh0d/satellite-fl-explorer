from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# NEW: optional import so the app still starts if `requests` is missing.
try:
    import requests
except ImportError:
    requests = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.fl_simulator import CLASS_NAMES
from interface.learn import render_learn as render_learn_page, render_knowledge_check_page
from interface.measured_studies import render_measured_studies, load_default_measured_result

st.set_page_config(
    page_title="Satellite FL Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# A cohesive, low-contrast dashboard theme inspired by the project wireframes.
st.markdown(
    """
<style>
:root { --navy: #102b46; --teal: #1f7180; --sky: #eaf4fa; --ink: #183247; }
.block-container { padding-top: .8rem; padding-bottom: 1.25rem; max-width: 1700px; }
.hero {
  padding: .9rem 1.2rem; border-radius: 14px;
  background: linear-gradient(110deg, #edf6ff 0%, #f7fbff 70%, #e7f4f7 100%);
  color: #133c70; margin-bottom: .75rem; border: 1px solid #d9e8f6;
}
.hero h1 { color: #1458b8 !important; margin: .12rem 0 .28rem 0; font-size: 1.85rem; }
.hero p { color: #526b85 !important; margin: .1rem 0; font-size: .96rem; }
.eyebrow { letter-spacing: .10em; font-size: .70rem; font-weight: 800; color: #50739c; }
.soft-card { border: 1px solid #dce7f2; border-radius: 12px; padding: .75rem .85rem; background: #fff; }
.muted { color: #637789; font-size: .84rem; }
.status-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }
.mini-kicker { color:#557292; font-size:.73rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; }
div[data-testid="stMetric"] { border: 1px solid #dce7f2; padding: .55rem .7rem; border-radius: 10px; background: #f8fbff; }
div[data-testid="stSidebar"] { border-right: 1px solid #dce7f2; background: #f8fbff; }
div[data-testid="stRadio"] label { padding-top: .18rem; padding-bottom: .18rem; }
.stButton>button { border-radius: 9px; font-weight: 650; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #dce7f2; border-radius: 12px; }
h1, h2, h3 { letter-spacing: -.02em; }
.panel-title { color: #144f9c; font-weight: 800; font-size: .98rem; margin-bottom: .3rem; }
.client-label { color: #234a70; font-weight: 750; font-size: .88rem; }
.small-note { color: #698096; font-size: .75rem; line-height: 1.35; }
.land-card { border: 1px solid #dce7f2; border-radius: 9px; padding: .35rem; text-align: center; background: #f7fbff; margin-bottom: .35rem; }
.land-emoji { font-size: 1.45rem; line-height: 1.6; }
.nav-rail-brand { text-align:center; font-size:1.35rem; padding:.25rem 0 .75rem 0; }
.nav-rail-caption { text-align:center; color:#7890a8; font-size:.58rem; font-weight:800; letter-spacing:.08em; margin-bottom:.25rem; }
div[class*="st-key-rail_nav_"] button { min-height:46px !important; height:46px !important; padding:0 !important; border-radius:12px !important; }
div[class*="st-key-rail_nav_"] button p { font-size:1.2rem !important; margin:0 !important; }
.nav-card-icon { font-size:1.65rem; line-height:1.2; margin-bottom:.35rem; }
.nav-card-title { color:#144f9c; font-weight:800; font-size:1.05rem; margin:.1rem 0 .25rem 0; }
.nav-card-body { color:#62788e; font-size:.87rem; min-height:3.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# CHANGED: added Knowledge Check page
PAGES = ["Overview", "Learn", "Explore & Results", "Knowledge Check", "Research"]

def _setting(name: str, default: str = "") -> str:
    """Read a setting from environment variables, then Streamlit secrets."""
    value = os.getenv(name, "")
    if not value:
        try:
            value = str(st.secrets.get(name, ""))
        except Exception:
            value = ""  # no secrets file is fine when running locally
    return value.strip() or default


# Local default: Ollama on your own computer.
# Deployed: set OLLAMA_URL, OLLAMA_MODEL, and OLLAMA_API_KEY in Streamlit secrets.
OLLAMA_URL = _setting("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = _setting("OLLAMA_MODEL", "llama3.2")
OLLAMA_API_KEY = _setting("OLLAMA_API_KEY")


def _ollama_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

# NEW: emoji/colour per class; unknown classes get a default card
CLASS_STYLE = {
    "Annual crop": ("🌾", "#f9efd2"), "Forest": ("🌲", "#e5f3df"),
    "Herbaceous veg.": ("🌿", "#e8f5e4"), "Highway": ("🛣️", "#e5e8ef"),
    "Industrial": ("🏭", "#e7e9ec"), "Pasture": ("🐄", "#eef6e0"),
    "Permanent crop": ("🍇", "#f3e6f2"), "Residential": ("🏘️", "#f4e8d6"),
    "River": ("🏞️", "#dceefb"), "Sea / lake": ("🌊", "#d6ebf8"),
}

st.session_state.setdefault("selected_page", "Overview")
st.session_state.setdefault("tutor_messages", [
    {"role": "assistant", "content": (
        "Hi! I'm your Satellite FL tutor. Ask follow-up questions about federated learning, satellite "
        "constraints, or your latest experiment. Answers come from a local Ollama model running on your computer."
    )}
])

def navigate_to(page: str) -> None:
    """Navigate to a page from the icon rail or overview cards."""
    st.session_state["selected_page"] = page


def page_intro(eyebrow: str, title: str, description: str) -> None:
    st.caption(eyebrow.upper())
    st.title(title)
    st.write(description)


def get_current_result() -> dict[str, Any] | None:
    # Prefer the selected measured run for tutor context on other pages.
    measured = st.session_state.get("active_measured_result")
    if measured:
        return measured
    # Show a real measured reference run on Overview even before the learner
    # has opened Explore & Results in this browser session. Do not fall back to
    # stale session-state prototype results from older app versions.
    return load_default_measured_result() or read_preset_results()


# NEW: accept teammates' schema-1.0 JSON as well as this app's format
def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert schema 1.0 fields into the format the charts use."""
    if "class_names" not in result and "classes" in result:
        result["class_names"] = list(result["classes"])
    if "metrics" not in result and "round_metrics" in result:
        rm = sorted(result["round_metrics"], key=lambda r: r["round"])
        result["metrics"] = {
            "global_accuracy": [float(r["accuracy"]) for r in rm],
            "global_loss": [float(r["loss"]) for r in rm],
        }
    counts = result.get("client_class_counts")
    names = result.get("class_names", [])
    if counts and isinstance(counts[0], list):
        result["client_class_counts"] = [dict(zip(names, row)) for row in counts]
    if "runtime_seconds" not in result and "total_elapsed_seconds" in result:
        result["runtime_seconds"] = result["total_elapsed_seconds"]
    return result


def read_preset_results() -> dict[str, Any] | None:
    """Optional integration point for a teammate's result JSON."""
    result_path = ROOT / "experiments" / "results" / "latest_results.json"
    if not result_path.exists():
        return None
    try:
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        if not isinstance(result, dict):
            return None
        result = normalize_result(result)
        return result if "metrics" in result else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def build_distribution_df(counts: list[dict[str, int]]) -> pd.DataFrame:
    records = []
    for i, client in enumerate(counts):
        for class_name, number in client.items():
            records.append({"Satellite": f"Sat {i + 1}", "Class": class_name, "Samples": int(number)})
    return pd.DataFrame(records)


def distribution_figure(df: pd.DataFrame, title: str = "Local training data by satellite") -> go.Figure:
    fig = px.bar(
        df, x="Satellite", y="Samples", color="Class", barmode="stack",
        category_orders={"Class": CLASS_NAMES},
        title=title, color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        legend_title_text="Land-cover class", xaxis_title="Simulated satellite client",
        yaxis_title="Training samples", margin=dict(l=12, r=12, t=55, b=12),
        height=355, legend=dict(orientation="h", yanchor="bottom", y=-.35, xanchor="left", x=0),
    )
    return fig


def map_figure(num_clients: int = 4, participants: list[int] | None = None, seed: int = 42) -> go.Figure:
    """Interactive concept map. Positions are illustrative, not propagated orbits."""
    all_points = [
        (61, -145), (34, -75), (8, -25), (28, 35), (-16, 90), (45, 138), (-34, 151), (69, 65)
    ]
    ground_lat, ground_lon = 39.7, -104.9  # illustrative ground station in central North America
    num_clients = int(np.clip(num_clients, 2, 8))
    points = all_points[:num_clients]
    active = set(participants if participants is not None else list(range(max(1, int(np.ceil(num_clients * .75))))))

    fig = go.Figure()
    # Broad translucent coverage regions are illustrative schematic circles,
    # not true satellite footprints or orbital/visibility calculations.
    client_palette = ["#2582ef", "#39c88b", "#ff9e2c", "#e45f51", "#8a73d6", "#16a6b5", "#d65b9a", "#718096"]
    fill_palette = ["rgba(37,130,239,0.12)", "rgba(57,200,139,0.12)", "rgba(255,158,44,0.12)", "rgba(228,95,81,0.12)",
                    "rgba(138,115,214,0.12)", "rgba(22,166,181,0.12)", "rgba(214,91,154,0.12)", "rgba(113,128,150,0.12)"]
    for i, (lat, lon) in enumerate(points):
        radius_lat = 9.5
        radius_lon = radius_lat / max(0.35, np.cos(np.radians(lat)))
        angles = np.linspace(0, 2 * np.pi, 72)
        ring_lats = np.clip(lat + radius_lat * np.sin(angles), -88, 88)
        ring_lons = ((lon + radius_lon * np.cos(angles) + 180) % 360) - 180
        fig.add_trace(go.Scattergeo(
            lat=list(ring_lats) + [float(ring_lats[0])],
            lon=list(ring_lons) + [float(ring_lons[0])],
            mode="lines", fill="toself", fillcolor=fill_palette[i],
            line=dict(color=client_palette[i], width=1),
            name=f"Coverage region {i + 1}", hoverinfo="skip", showlegend=False,
        ))

    # Draw conceptual ground links for currently available clients.
    link_lats, link_lons = [], []
    for i, (lat, lon) in enumerate(points):
        if i in active:
            link_lats.extend([ground_lat, lat, None])
            link_lons.extend([ground_lon, lon, None])
    if link_lats:
        fig.add_trace(go.Scattergeo(
            lat=link_lats, lon=link_lons, mode="lines", name="Available contact links",
            line=dict(color="#27a99a", width=1.6), opacity=.7, hoverinfo="skip",
        ))

    for i, (lat, lon) in enumerate(points):
        is_online = i in active
        label = "In contact" if is_online else "Unavailable"
        color = client_palette[i] if is_online else "#a7b3bf"
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers+text", text=[f"S{i+1}"],
            textposition="top center", name=f"Satellite {i + 1}",
            hovertext=[f"Satellite {i + 1}<br>Status: {label}<br>Role: simulated FL client"], hoverinfo="text",
            marker=dict(size=12, color=color, line=dict(width=1.5, color="white")),
        ))

    fig.add_trace(go.Scattergeo(
        lat=[ground_lat], lon=[ground_lon], mode="markers+text", text=["Ground station"],
        textposition="bottom center", name="Ground station", hovertext=["Ground station (illustrative)"],
        hoverinfo="text", marker=dict(size=15, symbol="diamond", color="#2867a8", line=dict(width=1.5, color="white")),
    ))
    fig.update_geos(
        projection_type="natural earth", showland=True, landcolor="#e9f0df",
        showocean=True, oceancolor="#cfe8f5", showlakes=True, lakecolor="#cfe8f5",
        showcountries=True, countrycolor="#ffffff", countrywidth=0.8,
        showcoastlines=True, coastlinecolor="#b8cedb", coastlinewidth=0.8,
        lataxis_showgrid=False, lonaxis_showgrid=False,
        bgcolor="#e7f3fb", showframe=False,
    )
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=15, b=5),
        legend=dict(orientation="h", yanchor="bottom", y=-.05, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def metrics_figure(result: dict[str, Any], compact: bool = False) -> None:
    metrics = result.get("metrics", {})
    accuracy, loss = metrics.get("global_accuracy", []), metrics.get("global_loss", [])
    if not accuracy or not loss:
        st.info("The current result does not include accuracy and loss histories.")
        return
    rounds = list(range(len(accuracy)))
    split_label = "validation" if result.get("result_source") == "measured" else "model"
    accuracy_label = f"Global {split_label} accuracy"
    loss_label = f"Global {split_label} loss"
    accuracy_df = pd.DataFrame({"FL round": rounds, accuracy_label: accuracy})
    loss_df = pd.DataFrame({"FL round": list(range(len(loss))), loss_label: loss})
    left, right = st.columns(2)
    with left:
        fig = px.line(accuracy_df, x="FL round", y=accuracy_label, markers=True, title=accuracy_label.title())
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        fig.update_layout(height=270 if compact else 325, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(loss_df, x="FL round", y=loss_label, markers=True, title=loss_label.title())
        fig.update_layout(height=270 if compact else 325, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Round 0 is the untrained baseline evaluation before any federated training. Pilot curves are validation metrics, not held-out test metrics.")


# ---------------------------------------------------------------------------
# AI tutor
# ---------------------------------------------------------------------------

def local_tutor_answer(question: str, result: dict[str, Any] | None, page: str) -> str:
    """Grounded fallback helper that works without an external API or credentials."""
    q = question.lower().strip()
    cfg = result.get("config", {}) if result else {}
    metrics = result.get("metrics", {}) if result else {}
    if any(w in q for w in ["alpha", "α", "dirichlet", "heterogen", "non-iid", "noniid"]):
        raw_alpha = cfg.get("alpha", st.session_state.get("explore_alpha", 0.3))
        alpha_label = "IID reference (no Dirichlet α)" if raw_alpha is None else f"α = {float(raw_alpha):.2f}"
        return (f"The Dirichlet concentration α controls how class samples are split across clients. Smaller α values tend to create more uneven class mixes; larger α values tend to produce more balanced mixes. Your selected run uses {alpha_label}. α changes the data partition, not the model weights directly. The resulting effect on convergence should be measured rather than assumed.")
    if any(w in q for w in ["fedavg", "aggregation", "average", "global model"]):
        if result and result.get("result_source") == "measured":
            return ("FedAvg sends the current global model to selected clients, trains locally, then aggregates client model updates weighted by the clients' local sample counts. The selected curves are from saved offline EuroSAT RGB/CNN pilot runs; the app is displaying those results rather than retraining them.")
        return ("FedAvg sends the current global model to participating clients, lets each train on local data, then averages the updated model parameters weighted by each client's sample count. The Explore & Results page displays saved offline EuroSAT/CNN FedAvg runs; changing a parameter selector chooses among those measured configurations rather than retraining a model.")
    if any(w in q for w in ["accuracy", "loss", "result", "performance", "improve"]):
        if metrics.get("global_accuracy") and metrics.get("global_loss"):
            acc, loss = float(metrics["global_accuracy"][-1]), float(metrics["global_loss"][-1])
            initial = float(metrics["global_accuracy"][0])
            if result and result.get("result_source") == "measured":
                case = cfg.get("case", "selected measured run")
                return (f"For {case}, the saved run finishes at {acc:.1%} validation accuracy (round-0 baseline {initial:.1%}) "
                        f"and validation cross-entropy loss {loss:.4f}. These are measured offline validation results, "
                        "not held-out test metrics; the pilot did not run a final test evaluation.")
            return (f"The current result finishes at {acc:.1%} model accuracy (initial {initial:.1%}) "
                    f"and loss {loss:.3f}. Check the result source and metric split before interpreting it.")
        return "Open Explore & Results and select a parameter study and measured configuration first."
    if any(w in q for w in ["map", "satellite", "contact", "orbit", "connected", "communication"]):
        return ("The map is a conceptual illustration of FL clients and a ground station. When a measured run is selected, client availability is taken from that run's recorded participation in the inspected round. Satellite positions and coverage areas are illustrative markers, not real orbital propagation or visibility calculations.")
    if any(w in q for w in ["stale", "asynchronous", "async"]):
        return "A stale update was computed from an older global-model version than the latest server model. Asynchronous FL can use updates without waiting for every client, but older updates may be less aligned with the current model. This is a research concept in the current website, not yet a full implementation."
    if any(w in q for w in ["split", "sfl", "split learning"]):
        return "Split learning divides a neural network between parties. A satellite may execute early layers and send activations to a ground station, which executes the remaining layers. Split federated learning combines this with multiple clients; it can reduce some onboard computation but introduces activation/gradient communication and intermittent-link challenges. This page illustrates the idea rather than implementing SatSFL."
    if any(w in q for w in ["iid", "independent", "identically"]):
        return "IID means clients' local samples are drawn from broadly similar distributions. Non-IID means the clients' local distributions differ—for example, one satellite may see more forests while another sees more urban areas. Non-IID data can cause local model updates to disagree, but the impact depends on the dataset, model, and training settings."
    if any(w in q for w in ["participation", "dropout", "budget", "resource", "epoch"]):
        rate = float(cfg.get("participation_rate", 1.0))
        count = int(cfg.get("num_clients", 4))
        if result and result.get("result_source") == "measured":
            return (f"The selected measured run has {count} clients and a configured participation rate of {rate:.0%}. "
                    "The saved per-round records identify which client IDs actually participated. Participation was randomly selected in each round; these are offline simulation records, not measured satellite contact windows.")
        return "Open Explore & Results to select a measured client-participation study. Its heatmap shows the clients recorded as participating in each of the 50 offline communication rounds."
    return (f"I can help with FedAvg, IID vs non-IID data, α, training accuracy/loss, satellite contact, participation, stale updates, and split federated learning. You're currently on the {page} page. Try asking, for example, 'What does alpha control?' or 'Explain the latest accuracy.'")


# Tutor instructions sent to the local Ollama model
TUTOR_INSTRUCTIONS = (
    "You are the AI tutor for an educational website about federated learning (FL) and satellite networks. "
    "Have a natural, multi-turn conversation. Explain concepts clearly for university students who know basic "
    "deep learning, use equations or examples when helpful, ask a brief clarifying question when needed, and "
    "connect answers to the current experiment when relevant. Be honest about uncertainty. "
    "Do not invent experiment curves or imply an unmeasured parameter combination was run. For the saved offline runs, call the plotted curves "
    "validation accuracy and validation loss, never test metrics; the pilot has no final held-out test evaluation. "
    "Clearly distinguish implemented features from "
    "research ideas that are only illustrated. Keep most responses under about 150 words unless the user asks "
    "for more depth."
)


def build_tutor_context(result: dict[str, Any] | None, page: str) -> dict[str, Any]:
    cfg = result.get("config", {}) if result else {}
    metrics = result.get("metrics", {}) if result else {}
    accs = metrics.get("global_accuracy", [])
    losses = metrics.get("global_loss", [])
    is_measured = bool(result and result.get("result_source") == "measured")
    return {
        "current_page": page,
        "result_source": result.get("result_source") if result else None,
        "latest_experiment_config": cfg or None,
        "latest_metrics": {
            "initial_global_accuracy": accs[0] if accs else None,
            "final_global_accuracy": accs[-1] if accs else None,
            "final_global_loss": losses[-1] if losses else None,
            "selected_round": cfg.get("selected_round") if is_measured else None,
            "selected_round_metrics": result.get("selected_round_metrics") if is_measured else None,
            "metric_split": result.get("round_metric_split", "validation") if result else None,
        } if result else None,
        "interpretation_limits": (
            "The selected result is an offline measured FedAvg run on EuroSAT RGB data. Its round-by-round metrics "
            "are validation accuracy and validation cross-entropy loss. The pilot reserves a held-out test set but "
            "does not evaluate it (final_test_metrics is null). These experiments do not measure real satellite radio links or orbital motion."
            if is_measured else
            "No measured experiment is currently selected. The map and coverage regions are illustrative, not orbital propagation or real contact calculations."
        ),
    }


def recent_turns(question: str) -> list[dict[str, str]]:
    """Recent turns for follow-ups; the latest user turn is skipped to avoid duplication."""
    turns = []
    for msg in st.session_state.get("tutor_messages", [])[-9:]:
        if msg.get("role") not in ("user", "assistant"):
            continue
        if msg.get("role") == "user" and msg.get("content", "").strip() == question.strip():
            continue
        turns.append({"role": msg["role"], "content": msg["content"]})
    return turns[-8:]


def _context_message(question: str, result: dict[str, Any] | None, page: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"Student question: {question.strip()}\n\nCurrent application context (JSON): "
                   f"{json.dumps(build_tutor_context(result, page), ensure_ascii=False)}",
    }


@st.cache_data(ttl=10, show_spinner=False)
def ollama_available() -> bool:
    """Quick check whether the local Ollama server is reachable (re-checked every 10 s)."""
    if requests is None:
        return False
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", headers=_ollama_headers(), timeout=5).ok
    except Exception:
        return False


def ollama_answer(question: str, result: dict[str, Any] | None, page: str) -> str:
    messages = [{"role": "system", "content": TUTOR_INSTRUCTIONS}]
    messages += recent_turns(question)
    messages.append(_context_message(question, result, page))
    try:
        r = requests.post(
    f"{OLLAMA_URL}/api/chat",
    headers=_ollama_headers(),
    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
          "options": {"num_predict": 600}},
    timeout=180,
)
        if r.status_code != 200:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            return (f"Ollama returned an error: {detail}. If it says the model was not found, "
                    f"run `ollama pull {OLLAMA_MODEL}` or change OLLAMA_MODEL near the top of app.py "
                    "to a name shown by `ollama list`.")
        answer = r.json().get("message", {}).get("content", "").strip()
        return answer or "The local model returned an empty answer. Try rephrasing."
    except requests.Timeout:
        return ("The local model took too long to answer. The first question can be slow while the model "
                "loads into memory; please try again.")
    except Exception as exc:
        return (f"The AI tutor is temporarily unavailable ({type(exc).__name__}). "
        "Please try again in a moment.")


def tutor_mode_label() -> str:
    if ollama_available():
        return f"Ollama ({OLLAMA_MODEL})"
    return "Ollama offline (built-in keyword helper)"


def tutor_answer(question: str, result: dict[str, Any] | None, page: str) -> str:
    """Use the local Ollama model; fall back to the keyword helper only if Ollama is down."""
    if ollama_available():
        return ollama_answer(question, result, page)
    return ("⚠️ Ollama is not running, so this answer comes from the small built-in helper. "
            "Start Ollama with `ollama serve` for full answers.\n\n"
            + local_tutor_answer(question, result, page))


def ask_and_record(question: str, result: dict[str, Any] | None, page: str) -> None:
    question = question.strip()
    st.session_state.tutor_messages.append({"role": "user", "content": question})
    with st.spinner("Tutor is thinking…"):
        answer = tutor_answer(question, result, page)
    st.session_state.tutor_messages.append({"role": "assistant", "content": answer})
    st.session_state.tutor_messages = st.session_state.tutor_messages[-10:]


def tutor_inline(current_page: str) -> None:
    """Visible, collapsible tutor available on Learn, Quiz, and Research pages."""
    page_key = current_page.lower().replace(" ", "_").replace("&", "and")
    with st.expander("🤖 AI Tutor / Helper", expanded=False):
        st.caption(f"Tutor mode: {tutor_mode_label()}. Ask about the current lesson or your latest experiment.")
        for msg in st.session_state.tutor_messages[-4:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        with st.form(f"inline_tutor_form_{page_key}", clear_on_submit=True):
            question = st.text_input(
                "Ask a question", placeholder="e.g. Why does a smaller α create more uneven clients?",
                key=f"inline_tutor_question_{page_key}",
            )
            ask = st.form_submit_button("Ask tutor", use_container_width=True)
        if ask and question.strip():
            ask_and_record(question, get_current_result(), current_page)
            st.rerun()


def tutor_sidebar(current_page: str) -> None:
    result = get_current_result()
    with st.sidebar:
        st.markdown("---")
        with st.expander("🤖 AI Tutor / Helper",
                         expanded=(current_page in ["Learn", "Explore & Results", "Knowledge Check"])):
            st.caption(f"Tutor mode: {tutor_mode_label()}")
            if not ollama_available():
                with st.expander("How to start the Ollama tutor"):
                    st.write("Install Ollama from ollama.com, then run in a terminal:")
                    st.code(f"ollama pull {OLLAMA_MODEL}\nollama serve", language="bash")
            for msg in st.session_state.tutor_messages[-4:]:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Tutor:** {msg['content']}")
            with st.form("tutor_question_form", clear_on_submit=True):
                question = st.text_input("Ask a question", placeholder="e.g. Why does alpha matter?")
                ask = st.form_submit_button("Ask tutor", use_container_width=True)
            if ask and question.strip():
                ask_and_record(question, result, current_page)
                st.rerun()
            quick = st.selectbox("Quick question",
                                 ["Choose a prompt…", "What does alpha control?", "Explain FedAvg",
                                  "Explain the latest accuracy"], key="tutor_quick_question")
            if st.button("Ask selected question", key="tutor_quick_submit",
                         use_container_width=True) and quick != "Choose a prompt…":
                ask_and_record(quick, result, current_page)
                st.rerun()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_hero(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    render_hero(
        "INTERACTIVE DEEP-LEARNING LAB",
        "🛰️ Satellite FL Explorer",
        "Learn, experiment, and investigate the challenges of federated learning across satellite networks.",
    )

    # Four functional cards provide a clear entry point into the main learning areas.
    st.markdown("### Explore the learning lab")
    card_specs = [
        ("📘", "Learn", "Understand FedAvg, IID vs. non-IID data, and satellite constraints."),
        ("🛰️", "Explore & Results", "Configure an experiment and inspect model performance."),
        ("✅", "Knowledge Check", "Test your understanding and get explanations for your answers."),
        ("📚", "Research", "Explore methods and open-source projects from the literature."),
    ]
    card_cols = st.columns(4, gap="small")
    for col, (icon, title, description) in zip(card_cols, card_specs):
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='nav-card-icon'>{icon}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='nav-card-title'>{title}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='nav-card-body'>{description}</div>", unsafe_allow_html=True)
                st.button(
                    f"Open {title} →", key=f"overview_open_{title.lower().replace(' ', '_').replace('&', 'and')}",
                    use_container_width=True, on_click=navigate_to, args=(title,),
                    type="primary" if title == "Explore & Results" else "secondary",
                )

    result = get_current_result()
    cfg = result.get("config", {}) if result else {}
    metrics = result.get("metrics", {}) if result else {}
    clients = int(cfg.get("num_clients", 4))
    last_acc = metrics.get("global_accuracy", [])
    final_acc = float(last_acc[-1]) if last_acc else None
    participants = None
    if result:
        participation = result.get("participation", {}).get("clients_per_round", [])
        if participation:
            participants = participation[-1]
    active_count = len(participants) if participants is not None else max(1, int(np.ceil(clients * .75)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Simulated satellites", clients)
    m2.metric("Clients available", f"{active_count}/{clients}")
    m3.metric("Latest validation accuracy", f"{final_acc:.1%}" if final_acc is not None else "Not available")
    m4.metric("Learning algorithm", "FedAvg")

    map_col, tutor_col = st.columns([1.65, 1])
    with map_col:
        with st.container(border=True):
            st.subheader("Constellation & ground-station links")
            st.plotly_chart(map_figure(clients, participants), use_container_width=True, config={"displayModeBar": False})
            st.caption("Illustrative positions and contact links only—not real orbital propagation. Green = participating/available; grey = unavailable in the displayed round.")
    with tutor_col:
        with st.container(border=True):
            st.subheader("🤖 AI Tutor")
            st.caption(f"Tutor mode: {tutor_mode_label()}. Ask follow-up questions about FL or your latest run.")
            for msg in st.session_state.tutor_messages[-2:]:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.info(msg["content"])
            with st.form("overview_tutor_form", clear_on_submit=True):
                overview_question = st.text_input("Ask the tutor", placeholder="Why does non-IID data matter?")
                overview_ask = st.form_submit_button("Ask question", use_container_width=True)
            if overview_ask and overview_question.strip():
                ask_and_record(overview_question, result, "Overview")
                st.rerun()
            if result and final_acc is not None:
                st.markdown("**Latest run insight**")
                metric_split = "validation" if result.get("result_source") == "measured" else "model"
                st.write(f"Global {metric_split} accuracy changed from {last_acc[0]:.1%} to {last_acc[-1]:.1%} over {cfg.get('rounds', len(last_acc)-1)} rounds.")
            else:
                st.markdown("**Start here**")
                st.write("Choose Learn for the concepts, then open Explore & Results to select and compare the saved EuroSAT/FedAvg experiments. Knowledge Check contains scored questions with feedback.")

    with st.container(border=True):
        st.subheader("Reference model performance")
        if result:
            metrics_figure(result, compact=True)
        else:
            st.write("Measured pilot curves are unavailable. Check that the experiments/results/pilot folder is included in the project.")

    st.caption("Experiment note: the displayed pilot curves are measured offline EuroSAT/CNN validation results. Client map markers are illustrative and do not represent propagated satellite orbits.")

def _render_tutor_compact(result: dict[str, Any] | None, page: str = "Explore & Results") -> None:
    st.markdown("<div class='panel-title'>🤖 AI Tutor Assistant</div>", unsafe_allow_html=True)
    st.caption(f"Tutor mode: {tutor_mode_label()}. Ask about FL or the current run.")
    with st.form("dashboard_tutor_form", clear_on_submit=True):
        question = st.text_input("Ask the tutor", placeholder="Why does non-IID data matter?", label_visibility="collapsed")
        submitted = st.form_submit_button("➤ Ask", use_container_width=True)
    if submitted and question.strip():
        ask_and_record(question, result, page)
    for msg in st.session_state.tutor_messages[-2:]:
        if msg["role"] == "assistant":
            st.info(msg["content"])
        else:
            st.markdown(f"**You:** {msg['content']}")


def render_explore() -> None:
    """One unified experiment workspace backed by saved measured EuroSAT runs."""
    measured_result = render_measured_studies()
    st.session_state["active_measured_result"] = measured_result
    _render_tutor_compact(measured_result, "Explore & Results")


def render_research() -> None:
    page_intro("MODULE 04 · LITERATURE", "Research & design ideas", "Connect the behaviors you can demonstrate to approaches studied in satellite federated-learning literature.")
    items = [
        ("Non-IID local data", "Clients observe different class distributions; local updates can differ.", "FedProx and methods for statistical heterogeneity", "https://github.com/litian96/FedProx"),
        ("Intermittent satellite-ground contact", "A satellite may miss a scheduled update window.", "Visibility-aware participation and asynchronous FL", "https://github.com/AryamanPatronia/satellite-flower"),
        ("Different device resources", "Satellites may have different compute, memory, and link budgets.", "Scheduling, resource allocation and inter-satellite links", "https://github.com/gkim65/federated-satellites"),
        ("Limited onboard training", "Training the full model onboard can be expensive.", "Split learning and SatSFL as advanced case studies", "https://scholar.google.com/scholar?q=SatSFL+satellite+federated+learning"),
    ]
    for problem, why, methods, url in items:
        with st.container(border=True):
            st.markdown(f"### {problem}")
            st.write(why)
            st.markdown(f"**Literature directions:** {methods}")
            st.markdown(f"[Open source / literature starting point]({url})")

    st.subheader("Demonstration status")
    status_df = pd.DataFrame([
        {"Concept": "IID / non-IID data partitioning", "Status": "Measured EuroSAT parameter study available"},
        {"Concept": "FedAvg training loop", "Status": "Measured offline EuroSAT/CNN experiments displayed"},
        {"Concept": "Intermittent participation", "Status": "Measured client participation schedules available"},
        {"Concept": "Orbital visibility", "Status": "Illustrative map only"},
        {"Concept": "FedProx / FedAsync / SatSFL", "Status": "Research direction, not implemented"},
    ])
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    st.caption("Please cite the specific papers used in the final report. Repository links are starting points, not evidence that every feature is implemented in this app.")


def main() -> None:
    pages = [
        ("Overview", "🏠"),
        ("Learn", "📘"),
        ("Explore & Results", "🛰️"),
        ("Knowledge Check", "✅"),
        ("Research", "📚"),
    ]

    # Migrate state from earlier versions of the app.
    legacy_page = st.session_state.get("page_nav", st.session_state.get("selected_page", "Overview"))
    if legacy_page in ("Explore", "Results"):
        legacy_page = "Explore & Results"
    if legacy_page not in [page for page, _ in pages]:
        legacy_page = "Overview"
    st.session_state.setdefault("selected_page", legacy_page)
    if st.session_state.get("selected_page") not in [page for page, _ in pages]:
        st.session_state["selected_page"] = "Overview"

    # Slim in-app icon rail inspired by the team's mockup. Sidebar remains available
    # for supplemental tutor/status content but is collapsed by default.
    nav_col, content_col = st.columns([0.52, 10], gap="small")
    with nav_col:
        with st.container(border=True):
            st.markdown("<div class='nav-rail-brand'>🛰️</div><div class='nav-rail-caption'>NAV</div>", unsafe_allow_html=True)
            for page, icon in pages:
                active = st.session_state.selected_page == page
                st.button(
                    icon,
                    key="rail_nav_" + page.lower().replace(" ", "_").replace("&", "and"),
                    help=page,
                    type="primary" if active else "secondary",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=(page,),
                )
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    selected_page = st.session_state.selected_page

    with content_col:
        if selected_page in ["Learn", "Knowledge Check", "Research"]:
            tutor_inline(selected_page)
        if selected_page == "Overview":
            render_overview()
        elif selected_page == "Learn":
            render_learn_page()
        elif selected_page == "Explore & Results":
            render_explore()
        elif selected_page == "Knowledge Check":
            render_knowledge_check_page()
        elif selected_page == "Research":
            render_research()

        st.divider()
        st.caption("Satellite FL Explorer · Educational toolkit · measured offline EuroSAT/FedAvg studies unless otherwise stated")


if __name__ == "__main__":
    main()