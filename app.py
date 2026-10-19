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

from backend.fl_simulator import CLASS_NAMES, make_preview_partition, run_demo_experiment

st.set_page_config(
    page_title="Satellite FL Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
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
</style>
""",
    unsafe_allow_html=True,
)

# CHANGED: added Knowledge Check page
PAGES = ["Overview", "Learn", "Explore & Results", "Knowledge Check", "Research"]

# NEW: local Ollama tutor settings (override with environment variables if needed)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # must match `ollama list`

# NEW: single source of truth for Explore defaults (used by init and Reset)
EXPLORE_DEFAULTS = {
    "explore_clients": 3,
    "explore_partition": "Non-IID",
    "explore_alpha": 0.3,
    "explore_epochs": 2,
    "explore_rounds": 10,
    "explore_participation": 0.75,
    "explore_seed": 42,
    "explore_hypothesis": "May change convergence/performance",
}

# NEW: emoji/colour per class; unknown classes get a default card
CLASS_STYLE = {
    "Annual crop": ("🌾", "#f9efd2"), "Forest": ("🌲", "#e5f3df"),
    "Herbaceous veg.": ("🌿", "#e8f5e4"), "Highway": ("🛣️", "#e5e8ef"),
    "Industrial": ("🏭", "#e7e9ec"), "Pasture": ("🐄", "#eef6e0"),
    "Permanent crop": ("🍇", "#f3e6f2"), "Residential": ("🏘️", "#f4e8d6"),
    "River": ("🏞️", "#dceefb"), "Sea / lake": ("🌊", "#d6ebf8"),
}

st.session_state.setdefault("last_result", None)
st.session_state.setdefault("run_history", [])
st.session_state.setdefault("selected_page", "Overview")
st.session_state.setdefault("tutor_messages", [
    {"role": "assistant", "content": (
        "Hi! I'm your Satellite FL tutor. Ask follow-up questions about federated learning, satellite "
        "constraints, or your latest experiment. Answers come from a local Ollama model running on your computer."
    )}
])
for _key, _value in EXPLORE_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_preview(num_clients: int, partition: str, alpha: float, seed: int):
    return make_preview_partition(num_clients, partition, alpha, seed)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_experiment(num_clients: int, partition: str, alpha: float, rounds: int,
                      local_epochs: int, seed: int, participation_rate: float):
    return run_demo_experiment(
        num_clients=num_clients, partition=partition, alpha=alpha, rounds=rounds,
        local_epochs=local_epochs, seed=seed, participation_rate=participation_rate,
    )


def page_intro(eyebrow: str, title: str, description: str) -> None:
    st.caption(eyebrow.upper())
    st.title(title)
    st.write(description)


def get_current_result() -> dict[str, Any] | None:
    result = st.session_state.get("last_result")
    if result:
        return result
    return read_preset_results()


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
    accuracy_df = pd.DataFrame({"FL round": rounds, "Global test accuracy": accuracy})
    loss_df = pd.DataFrame({"FL round": list(range(len(loss))), "Global test loss": loss})
    left, right = st.columns(2)
    with left:
        fig = px.line(accuracy_df, x="FL round", y="Global test accuracy", markers=True, title="Global test accuracy")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        fig.update_layout(height=270 if compact else 325, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(loss_df, x="FL round", y="Global test loss", markers=True, title="Global test loss")
        fig.update_layout(height=270 if compact else 325, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Round 0 is the untrained baseline evaluation before any federated training.")


# ---------------------------------------------------------------------------
# AI tutor
# ---------------------------------------------------------------------------

def local_tutor_answer(question: str, result: dict[str, Any] | None, page: str) -> str:
    """Grounded fallback helper that works without an external API or credentials."""
    q = question.lower().strip()
    cfg = result.get("config", {}) if result else {}
    metrics = result.get("metrics", {}) if result else {}
    if any(w in q for w in ["alpha", "α", "dirichlet", "heterogen", "non-iid", "noniid"]):
        alpha = float(cfg.get("alpha", st.session_state.get("explore_alpha", 0.3)))
        return (f"The Dirichlet concentration α controls how class samples are split across clients. Smaller α values tend to create more uneven class mixes; larger values tend to produce more balanced mixes. Your latest run used α = {alpha:.2f}. α changes the data partition, not the model weights directly. The resulting effect on convergence should be measured rather than assumed.")
    if any(w in q for w in ["fedavg", "aggregation", "average", "global model"]):
        return ("FedAvg sends the current global model to participating clients, lets each train on local data, then averages the updated model parameters weighted by each client's sample count. This prototype uses a small softmax classifier; it is not yet a CNN trained on EuroSAT images.")
    if any(w in q for w in ["accuracy", "loss", "result", "performance", "improve"]):
        if metrics.get("global_accuracy") and metrics.get("global_loss"):
            acc, loss = float(metrics["global_accuracy"][-1]), float(metrics["global_loss"][-1])
            initial = float(metrics["global_accuracy"][0])
            return (f"In the latest run, final global test accuracy was {acc:.1%} (initial {initial:.1%}) and final loss was {loss:.3f}. Accuracy measures the fraction of test examples classified correctly; cross-entropy loss also reflects how confident the model is. Interpret these numbers only for this synthetic prototype, not for real satellite imagery.")
        return "Run an experiment from Explore & Results first. The dashboard will then show global test accuracy and loss across rounds. Accuracy is the fraction of held-out examples classified correctly; cross-entropy loss measures prediction quality more continuously."
    if any(w in q for w in ["map", "satellite", "contact", "orbit", "connected", "communication"]):
        return ("The map is a conceptual illustration of simulated FL clients and a ground station. Green satellites are treated as available in the displayed round; grey satellites are unavailable. Positions are illustrative markers, not real orbital propagation or a visibility calculation. The participation setting is what changes availability in the demo.")
    if any(w in q for w in ["stale", "asynchronous", "async"]):
        return "A stale update was computed from an older global-model version than the latest server model. Asynchronous FL can use updates without waiting for every client, but older updates may be less aligned with the current model. This is a research concept in the current website, not yet a full implementation."
    if any(w in q for w in ["split", "sfl", "split learning"]):
        return "Split learning divides a neural network between parties. A satellite may execute early layers and send activations to a ground station, which executes the remaining layers. Split federated learning combines this with multiple clients; it can reduce some onboard computation but introduces activation/gradient communication and intermittent-link challenges. This page illustrates the idea rather than implementing SatSFL."
    if any(w in q for w in ["iid", "independent", "identically"]):
        return "IID means clients' local samples are drawn from broadly similar distributions. Non-IID means the clients' local distributions differ—for example, one satellite may see more forests while another sees more urban areas. Non-IID data can cause local model updates to disagree, but the impact depends on the dataset, model, and training settings."
    if any(w in q for w in ["participation", "dropout", "budget", "resource", "epoch"]):
        rate = float(cfg.get("participation_rate", 1.0))
        count = int(cfg.get("num_clients", 4))
        return f"The current prototype has {count} clients and uses a participation rate of {rate:.0%}; at least one client contributes each round. Reducing participation can reduce simulated upload traffic but also means fewer local datasets contribute to each global update. This is a simplified availability model, not a radio simulation."
    return (f"I can help with FedAvg, IID vs non-IID data, α, training accuracy/loss, satellite contact, participation, stale updates, and split federated learning. You're currently on the {page} page. Try asking, for example, 'What does alpha control?' or 'Explain the latest accuracy.'")


# Tutor instructions sent to the local Ollama model
TUTOR_INSTRUCTIONS = (
    "You are the AI tutor for an educational website about federated learning (FL) and satellite networks. "
    "Have a natural, multi-turn conversation. Explain concepts clearly for university students who know basic "
    "deep learning, use equations or examples when helpful, ask a brief clarifying question when needed, and "
    "connect answers to the current experiment when relevant. Be honest about uncertainty. "
    "Do not claim synthetic-demo metrics are EuroSAT results. Clearly distinguish implemented features from "
    "research ideas that are only illustrated. Keep most responses under about 150 words unless the user asks "
    "for more depth."
)


def build_tutor_context(result: dict[str, Any] | None, page: str) -> dict[str, Any]:
    cfg = result.get("config", {}) if result else {}
    metrics = result.get("metrics", {}) if result else {}
    accs = metrics.get("global_accuracy", [])
    losses = metrics.get("global_loss", [])
    return {
        "current_page": page,
        "result_source": result.get("result_source") if result else None,
        "latest_experiment_config": cfg or None,
        "latest_metrics": {
            "initial_global_accuracy": accs[0] if accs else None,
            "final_global_accuracy": accs[-1] if accs else None,
            "final_global_loss": losses[-1] if losses else None,
        } if result else None,
        "prototype_limitations": (
            "Current training is FedAvg on synthetic feature vectors with a linear softmax classifier, "
            "not EuroSAT image classification or a CNN. The map and coverage regions are illustrative, "
            "not orbital propagation or real contact calculations. Advanced literature methods may be "
            "explained but are not necessarily implemented."
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
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5).ok
    except Exception:
        return False


def ollama_answer(question: str, result: dict[str, Any] | None, page: str) -> str:
    messages = [{"role": "system", "content": TUTOR_INSTRUCTIONS}]
    messages += recent_turns(question)
    messages.append(_context_message(question, result, page))
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
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
        return (f"Couldn't reach the local Ollama tutor ({type(exc).__name__}). Check that Ollama is running "
                "(`ollama serve`).")


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
    render_hero("INTERACTIVE DEEP-LEARNING LAB", "🛰️ Satellite FL Explorer", "Learn, experiment, and investigate the challenges of federated learning across satellite networks.")
    result = get_current_result()
    cfg = result.get("config", {}) if result else {}
    metrics = result.get("metrics", {}) if result else {}
    clients = int(cfg.get("num_clients", st.session_state.get("explore_clients", 4)))
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
    m3.metric("Latest test accuracy", f"{final_acc:.1%}" if final_acc is not None else "Not run yet")
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
            if result:
                st.markdown("**Latest run insight**")
                if final_acc is not None:
                    st.write(f"Global test accuracy changed from {last_acc[0]:.1%} to {last_acc[-1]:.1%} over {cfg.get('rounds', len(last_acc)-1)} rounds.")
            else:
                st.markdown("**Start here**")
                st.write("Open Learn to understand FedAvg, then use Explore & Results to run your first experiment and interpret the outputs.")

    st.subheader("Your learning path")
    cols = st.columns(4)
    cards = [
        ("01", "Learn", "Understand local training, aggregation, and IID vs non-IID."),
        ("02", "Explore & Results", "Predict, run training, and inspect accuracy, loss, and communication."),
        ("03", "Knowledge Check", "Compare your prediction with the measured outcome."),
        ("04", "Research", "Connect observed limitations to satellite-FL literature."),
    ]
    for col, (num, title, description) in zip(cols, cards):
        with col, st.container(border=True):
            st.caption(f"MODULE {num}")
            st.markdown(f"### {title}")
            st.write(description)

    with st.container(border=True):
        st.subheader("Recent model performance")
        if result:
            metrics_figure(result, compact=True)
        else:
            st.write("No experiment yet. Run one from Explore & Results to populate this panel.")
            st.progress(0.0, text="Waiting for first training run")

    st.caption("Prototype limitation: current FedAvg metrics use synthetic feature vectors and a linear classifier, not EuroSAT imagery/CNN. Client map markers are illustrative.")


def render_learn() -> None:
    page_intro("MODULE 01 · CONCEPTS", "Learn", "Build intuition for federated learning before running an experiment.")
    tab_basics, tab_distribution, tab_satellite = st.tabs(["FedAvg basics", "IID vs non-IID", "Why satellites are different"])
    with tab_basics:
        with st.container(border=True):
            st.subheader("One federated-learning round")
            cols = st.columns(4)
            steps = [
                ("01 · Distribute", "The coordinator sends the current global model to available clients."),
                ("02 · Train", "Each client updates a local copy using its own samples."),
                ("03 · Aggregate", "The server averages the participating client parameters."),
                ("04 · Evaluate", "The new global model is assessed on a common held-out set."),
            ]
            for col, (title, detail) in zip(cols, steps):
                with col:
                    st.markdown(f"**{title}**")
                    st.caption(detail)
            st.latex(r"w^{(t+1)} = \sum_{i\in S_t}\frac{|D_i|}{\sum_{j\in S_t}|D_j|}\,w_i^{(t+1)}")
            st.caption("Sₜ is the set of clients participating in round t; each participant is weighted by its local sample count.")
        st.subheader("Check your understanding")
        answer = st.radio("In FedAvg, what is aggregated at the server?", ["Raw images from each satellite", "Updated local model parameters", "The orbit coordinates only"], key="learn_fedavg_quiz")
        if st.button("Check answer", key="check_fedavg"):
            if answer == "Updated local model parameters":
                st.success("Correct. Each client trains locally and sends model parameters or updates for aggregation.")
            else:
                st.warning("Not quite. The key idea is that raw training data stay local; the server aggregates model information.")
    with tab_distribution:
        st.markdown("A satellite constellation may see different places and therefore different label distributions. This is called statistical heterogeneity or non-IID data.")
        alpha_col, client_col = st.columns([1, 1])
        with alpha_col:
            partition = st.radio("Preview partition", ["IID", "Non-IID"], horizontal=True, key="learn_partition")
            alpha = st.slider("Dirichlet concentration α", 0.05, 5.0, 0.3, step=0.05, disabled=(partition == "IID"), key="learn_alpha")
            clients = st.select_slider("Number of clients", [2, 3, 4, 5, 6, 8], value=4, key="learn_clients")
            st.caption("Smaller α usually increases class imbalance. The effect on model accuracy must be measured; it is not guaranteed to be monotonic.")
        with client_col:
            preview = pd.DataFrame(cached_preview(int(clients), partition, float(alpha), 42))
            st.plotly_chart(distribution_figure(preview, "What each client receives"), use_container_width=True)
        st.markdown("**Think before running:** If each satellite has a different class mix, will local updates always point in the same direction? Why or why not?")
    with tab_satellite:
        p1, p2, p3 = st.columns(3)
        with p1, st.container(border=True):
            st.markdown("### 📡 Intermittent contact")
            st.write("A satellite may be out of range of the ground station during some rounds.")
        with p2, st.container(border=True):
            st.markdown("### ⚙️ Limited resources")
            st.write("Onboard energy, memory, compute, and bandwidth can limit local training.")
        with p3, st.container(border=True):
            st.markdown("### 🧭 Different data")
            st.write("Different regions and sensors can produce distribution shift across clients.")
        st.info("These are the target application challenges. The current running simulator implements client participation as a simplified availability setting; orbital geometry and onboard hardware are not physically simulated.")


def reset_explore_settings() -> None:
    """Reset Explore widgets via a Streamlit callback (before widgets rerender)."""
    for key, value in EXPLORE_DEFAULTS.items():
        st.session_state[key] = value


def _client_preview_cards(preview: pd.DataFrame, max_clients: int = 8) -> None:
    """Compact client histograms inspired by the project dashboard sketch."""
    if preview.empty:
        st.info("No client distribution is available yet.")
        return
    clients = list(preview["Satellite"].drop_duplicates())[:max_clients]
    class_colors = px.colors.qualitative.Safe
    for client in clients:
        client_df = preview[preview["Satellite"] == client].copy()
        if client_df.empty:
            continue
        dominant_row = client_df.loc[client_df["Samples"].idxmax()]
        with st.container(border=True):
            info_col, chart_col = st.columns([0.9, 1.45], vertical_alignment="center")
            with info_col:
                st.markdown(f"<div class='client-label'>🛰️ {client}</div>", unsafe_allow_html=True)
                st.caption(f"{dominant_row['Class']} dominant")
                st.caption(f"{int(client_df['Samples'].sum()):,} samples")
            with chart_col:
                fig = px.bar(client_df, x="Class", y="Samples", color="Class", color_discrete_sequence=class_colors)
                fig.update_layout(
                    showlegend=False, height=92, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showticklabels=False, title=None, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, title=None, showgrid=False, zeroline=False),
                    bargap=0.16,
                )
                fig.update_traces(hovertemplate="%{x}<br>Samples: %{y}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"client_bar_{client}_{len(preview)}")


def _dashboard_metric_charts(result: dict[str, Any] | None) -> None:
    """Three concise charts matching the dashboard concept sketch."""
    if not result:
        cols = st.columns(3)
        for col, title, message in [
            (cols[0], "Global training loss", "Run an experiment to view loss by round."),
            (cols[1], "Global model accuracy", "Run an experiment to view held-out accuracy."),
            (cols[2], "Final accuracy comparison", "Run IID and non-IID experiments to compare them."),
        ]:
            with col, st.container(border=True):
                st.markdown(f"<div class='panel-title'>{title}</div>", unsafe_allow_html=True)
                st.caption(message)
        return

    metrics = result.get("metrics", {})
    accuracy = metrics.get("global_accuracy", [])
    loss = metrics.get("global_loss", [])
    if not accuracy or not loss:
        st.info("This result does not include global accuracy/loss histories.")
        return
    rounds = list(range(len(accuracy)))
    col1, col2, col3 = st.columns([1, 1, .78])
    with col1, st.container(border=True):
        st.markdown("<div class='panel-title'>Global test loss</div>", unsafe_allow_html=True)
        loss_df = pd.DataFrame({"Round": list(range(len(loss))), "Loss": loss})
        fig = px.line(loss_df, x="Round", y="Loss", markers=True)
        fig.update_traces(line=dict(color="#f05b55", width=2), marker=dict(size=3))
        fig.update_yaxes(type="log", title=None, showgrid=True, gridcolor="#e4edf5")
        fig.update_xaxes(title=None, showgrid=False)
        fig.update_layout(height=205, margin=dict(l=4, r=4, t=4, b=4), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col2, st.container(border=True):
        st.markdown("<div class='panel-title'>Global model accuracy</div>", unsafe_allow_html=True)
        acc_df = pd.DataFrame({"Round": rounds, "Accuracy": accuracy})
        fig = px.line(acc_df, x="Round", y="Accuracy", markers=True)
        fig.update_traces(line=dict(color="#2478e5", width=2), marker=dict(size=3))
        fig.update_yaxes(title=None, tickformat=".0%", rangemode="tozero", showgrid=True, gridcolor="#e4edf5")
        fig.update_xaxes(title=None, showgrid=False)
        fig.update_layout(height=205, margin=dict(l=4, r=4, t=4, b=4), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col3, st.container(border=True):
        st.markdown("<div class='panel-title'>Final accuracy</div>", unsafe_allow_html=True)
        history = st.session_state.get("run_history", [])
        by_partition: dict[str, dict[str, Any]] = {}
        for run in history:
            rcfg = run.get("config", {})
            rmetrics = run.get("metrics", {})
            partition_name = str(rcfg.get("partition", "Unknown"))
            acc_values = rmetrics.get("global_accuracy", [])
            if acc_values:
                by_partition[partition_name] = {"Experiment": partition_name, "Accuracy": float(acc_values[-1])}
        if len(by_partition) >= 2:
            bar_df = pd.DataFrame(list(by_partition.values()))
            palette = ["#2478e5" if name == "IID" else "#ef5b55" for name in bar_df["Experiment"]]
            fig = px.bar(bar_df, x="Experiment", y="Accuracy", color="Experiment", color_discrete_sequence=palette)
            fig.update_traces(text=[f"{v:.1%}" for v in bar_df["Accuracy"]], textposition="outside", cliponaxis=False)
            caption = "Latest completed run for each partition in this session."
        else:
            cfg = result.get("config", {})
            alpha_value = cfg.get("alpha", 0.3)
            label = (f"{cfg.get('partition', 'Current')} α={alpha_value:.2g}"
                     if isinstance(alpha_value, (float, int)) else str(cfg.get("partition", "Current")))
            bar_df = pd.DataFrame({"Experiment": [label], "Accuracy": [float(accuracy[-1])]})
            fig = px.bar(bar_df, x="Experiment", y="Accuracy", color="Experiment", color_discrete_sequence=["#2478e5"])
            fig.update_traces(text=[f"{float(accuracy[-1]):.1%}"], textposition="outside", cliponaxis=False)
            caption = "Run both IID and non-IID experiments to compare final accuracy here."
        max_acc = float(bar_df["Accuracy"].max())
        fig.update_yaxes(title=None, tickformat=".0%", range=[0, min(1.12, max(.25, max_acc * 1.22))], showgrid=True, gridcolor="#e4edf5")
        fig.update_xaxes(title=None, tickangle=0)
        fig.update_layout(height=205, margin=dict(l=4, r=4, t=4, b=18), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(caption)
    st.caption("Accuracy is shown as a percentage of held-out test samples. Round 0 is the untrained baseline.")


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
    """Dashboard-style exploration page matching the brainstorming mockup."""
    st.markdown(
        """<div class='hero'><div class='eyebrow'>MODULE 02 · INTERACTIVE EXPERIMENT</div>
<h1>🛰️ Explore & Results: Satellite Federated Learning</h1>
<p>Adjust client data distributions and training settings to compare IID and non-IID federated learning.</p></div>""",
        unsafe_allow_html=True,
    )

    # The left navigation remains in Streamlit's sidebar; controls occupy the
    # left panel of the main canvas, as in the team's mockup.
    controls_col, clients_col, map_col = st.columns([0.92, 1.15, 1.55], gap="small")
    with controls_col, st.container(border=True):
        st.markdown("<div class='panel-title'>⚙️ Simulation Controls</div>", unsafe_allow_html=True)
        # CHANGED: no value=/index= here; defaults come from EXPLORE_DEFAULTS in session state
        num_clients = st.select_slider("Number of satellites (K)", options=[2, 3, 4, 5, 6, 8], key="explore_clients")
        partition = st.selectbox("Data distribution", ["Non-IID", "IID"], key="explore_partition")
        alpha = st.slider("Dirichlet alpha (α)", min_value=0.05, max_value=2.0, step=0.05,
                          disabled=(partition == "IID"), help="Smaller alpha usually creates more uneven class mixtures.", key="explore_alpha")
        # CHANGED: max 5 to match the backend's cap
        local_epochs = st.slider("Local epochs (E)", 1, 5, key="explore_epochs",
                                 help="Local training passes per round (capped at 5 to keep runs fast).")
        rounds = st.slider("Communication rounds (T)", 1, 30, key="explore_rounds")
        participation = st.slider("Client contact availability", 0.25, 1.0, step=0.25,
                                  help="Simplified participation rate, not a physical orbit/contact calculation.", key="explore_participation")
        seed = st.number_input("Random seed", min_value=0, max_value=9999, step=1, key="explore_seed")
        hypothesis = st.radio("Prediction: non-IID data will…", ["Probably have no effect", "May change convergence/performance", "Always lower accuracy"], key="explore_hypothesis")
        run_clicked = st.button("▶  Run Simulation", type="primary", use_container_width=True)
        st.button("↺ Reset controls", on_click=reset_explore_settings, use_container_width=True)
        run_error_slot = st.empty()  # NEW: errors appear next to the controls

    preview = pd.DataFrame(cached_preview(int(num_clients), partition, float(alpha), int(seed)))
    with clients_col, st.container(border=True):
        st.markdown("<div class='panel-title'>Satellite Clients · Local Data Distribution</div>", unsafe_allow_html=True)
        st.caption("Live preview of each client's local samples for the current controls (no training).")
        _client_preview_cards(preview, max_clients=int(num_clients))

    result_before_run = st.session_state.get("last_result") or read_preset_results()
    # A fixed, real run of the lightweight synthetic model gives the initial
    # dashboard populated charts; it is clearly labelled as a reference run.
    reference_result = cached_experiment(3, "Non-IID", 0.3, 10, 2, 42, 0.75)
    participants = None
    map_result = result_before_run or reference_result
    if map_result:
        participation_rounds = map_result.get("participation", {}).get("clients_per_round", [])
        participants = participation_rounds[-1] if participation_rounds else None
    with map_col, st.container(border=True):
        st.markdown("<div class='panel-title'>Satellite Ground Tracks & Contact Regions</div>", unsafe_allow_html=True)
        st.plotly_chart(map_figure(int(num_clients), participants, int(seed)), use_container_width=True, config={"displayModeBar": False}, key="explore_satellite_map")
        st.caption("Illustrative positions and contact regions; not real orbital propagation.")

    # A run is performed only when the user presses the button.
    if run_clicked:
        try:
            started = time.perf_counter()
            with st.spinner("Training local clients and aggregating FedAvg updates…"):
                result = cached_experiment(int(num_clients), partition, float(alpha), int(rounds), int(local_epochs), int(seed), float(participation))
            # NEW: honest timing and cache labelling
            st.session_state.last_wall_clock = time.perf_counter() - started
            st.session_state.last_run_was_cached = any(
                r.get("config") == result.get("config") for r in st.session_state.run_history
            )
            st.session_state.last_result = result
            st.session_state.run_history.append(result)
            st.session_state.run_history = st.session_state.run_history[-6:]
            st.session_state.last_hypothesis = hypothesis
            st.session_state.last_hypothesis_correct = hypothesis == "May change convergence/performance"
            st.rerun()
        except Exception as exc:
            run_error_slot.error(f"The simulation failed ({type(exc).__name__}: {exc}). Try different settings.")

    st.markdown("### Training Performance")
    current_result = st.session_state.get("last_result") or read_preset_results()
    shown = current_result or reference_result

    # NEW: illustrative-source banner
    if shown and shown.get("result_source") != "measured":
        st.warning("Illustrative example: these curves come from real FedAvg arithmetic on synthetic feature "
                   "vectors, not measured EuroSAT/CNN training.")

    # NEW: flag results that no longer match the controls
    if current_result:
        last_cfg = current_result.get("config", {})
        form_cfg = {
            "num_clients": int(num_clients), "partition": partition, "alpha": float(alpha),
            "rounds": int(rounds), "local_epochs": int(local_epochs), "seed": int(seed),
            "participation_rate": float(participation),
        }
        if any(last_cfg.get(k) != v for k, v in form_cfg.items()):
            st.info("Controls have changed since this run. The results below belong to the previous configuration "
                    "(see 'Experiment configuration'). Press Run Simulation to generate results for the new settings.")

    if current_result:
        _dashboard_metric_charts(current_result)
    else:
        st.caption("Reference curves below come from a fixed default run of the synthetic prototype; press Run Simulation to replace them with your selected settings.")
        _dashboard_metric_charts(reference_result)

    # Results live on this same page as the controls.
    if current_result:
        metrics = current_result.get("metrics", {})
        cfg = current_result.get("config", {})
        communication = current_result.get("communication", {})
        accuracy_values = metrics.get("global_accuracy", [])
        loss_values = metrics.get("global_loss", [])
        st.markdown("### Results & interpretation")
        r1, r2, r3, r4 = st.columns(4)
        if accuracy_values:
            r1.metric("Final test accuracy", f"{float(accuracy_values[-1]):.1%}",
                      delta=f"{float(accuracy_values[-1]) - float(accuracy_values[0]):+.1%}")
        else:
            r1.metric("Final test accuracy", "—")
        r2.metric("Final test loss", f"{float(loss_values[-1]):.3f}" if loss_values else "—")
        r3.metric("Backend runtime", f"{float(current_result.get('runtime_seconds', 0)):.2f} s")
        r4.metric("Client participation", f"{float(cfg.get('participation_rate', 1.0)):.0%}")
        # NEW: cache + timing notes
        if st.session_state.get("last_run_was_cached"):
            st.caption("ⓘ Same settings as an earlier run: result reused from cache; backend runtime shown is from the original run.")
        if st.session_state.get("last_wall_clock") is not None:
            st.caption(f"Wall-clock time from Run press to result: {st.session_state.last_wall_clock:.2f} s "
                       "(excludes chart rendering).")
        st.caption(f"Run metadata · partition: {cfg.get('partition')} · α: {cfg.get('alpha')} · seed: {cfg.get('seed')} · "
                   f"local epochs: {cfg.get('local_epochs')} · train samples: {sum(current_result.get('client_sizes', [])):,} · "
                   f"source: {current_result.get('result_source', 'unknown')}")

        with st.expander("Communication and compute estimates"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Model parameters", f"{int(communication.get('parameters', 0)):,}")
            c2.metric("Estimated upload", f"{int(communication.get('estimated_upload_bytes_total', 0)):,} bytes")
            c3.metric("Local example-epochs", f"{int(communication.get('local_example_epochs', 0)):,}")
            st.caption("Simplified estimates for uncompressed float32 parameter uploads; they exclude protocol overhead, downlink, retransmission, and real radio constraints.")

        if st.session_state.get("last_hypothesis"):
            st.markdown("**Your prediction:** " + str(st.session_state.last_hypothesis))
            if st.session_state.get("last_hypothesis_correct"):
                st.success("Good hypothesis. Heterogeneity can change convergence and performance; the outcome depends on the data, model, and training settings.")
            else:
                st.info("A safer conclusion is that heterogeneity may change convergence/performance; it does not guarantee lower accuracy in every run.")

        with st.expander("Experiment configuration and caveats"):
            st.json(cfg)
            st.write(current_result.get("note", ""))

        if len(st.session_state.run_history) > 1:
            st.markdown("**Compare recent runs**")
            history_rows = []
            for index, run in enumerate(st.session_state.run_history, start=1):
                run_cfg = run.get("config", {})
                run_metrics = run.get("metrics", {})
                run_acc = run_metrics.get("global_accuracy", [])
                run_loss = run_metrics.get("global_loss", [])
                history_rows.append({
                    "Run": index,
                    "Partition": run_cfg.get("partition", "?"),
                    "α": run_cfg.get("alpha", None),
                    "Clients": run_cfg.get("num_clients", None),
                    "Participation": run_cfg.get("participation_rate", 1.0),
                    "Final accuracy": run_acc[-1] if run_acc else None,
                    "Final loss": run_loss[-1] if run_loss else None,
                })
            history_df = pd.DataFrame(history_rows)
            st.dataframe(history_df.style.format({"α": "{:.2f}", "Participation": "{:.0%}", "Final accuracy": "{:.1%}", "Final loss": "{:.3f}"}), use_container_width=True, hide_index=True)

    class_col, tutor_col = st.columns([1.35, 1], gap="small")
    with class_col, st.container(border=True):
        st.markdown("<div class='panel-title'>Land-Cover Classes</div>", unsafe_allow_html=True)
        st.caption("Class labels used by the current result. These are schematic cards, not real EuroSAT thumbnails.")
        # CHANGED: built from the result's class list, not hardcoded
        class_names = (shown or {}).get("class_names", CLASS_NAMES)
        for row_start in range(0, len(class_names), 5):
            row = class_names[row_start:row_start + 5]
            for col, label in zip(st.columns(5), row):
                emoji, color = CLASS_STYLE.get(label, ("🛰️", "#f7fbff"))
                with col:
                    st.markdown(
                        f"<div class='land-card' style='background:{color}'><div class='land-emoji'>{emoji}</div>"
                        f"<div style='font-size:.70rem;font-weight:700;color:#28445f'>{label}</div></div>",
                        unsafe_allow_html=True,
                    )
    with tutor_col, st.container(border=True):
        _render_tutor_compact(current_result, "Explore & Results")

    st.caption("Prototype note: training currently uses synthetic feature vectors and a linear softmax classifier, not actual EuroSAT images/CNN. Contact areas and satellite positions are illustrative.")


# NEW: Knowledge Check page (quiz content to be supplied by Member C)
def render_knowledge_check() -> None:
    page_intro("MODULE 03 · CHECK", "Knowledge Check",
               "Compare your prediction with what the experiment showed.")
    result = st.session_state.get("last_result")
    with st.container(border=True):
        st.subheader("Your prediction")
        if st.session_state.get("last_hypothesis"):
            st.markdown(f"**Before running you predicted:** non-IID data will… *{st.session_state.last_hypothesis}*")
        else:
            st.caption("Make a prediction on Explore & Results before pressing Run.")
    if not result:
        st.info("Run an experiment in Explore & Results first, then come back here.")
        return
    cfg = result.get("config", {})
    acc = result.get("metrics", {}).get("global_accuracy", [])
    with st.container(border=True):
        st.subheader("Interpret results")
        if result.get("result_source") != "measured":
            st.warning("Illustrative example: synthetic prototype data, not measured EuroSAT training.")
        if acc:
            st.markdown(f"**What happened:** {cfg.get('partition')} data, α = {cfg.get('alpha')}, "
                        f"{cfg.get('num_clients')} satellites, {cfg.get('rounds')} rounds. "
                        f"Global accuracy went from {acc[0]:.1%} (untrained baseline) to {acc[-1]:.1%}.")
        st.caption("Planned: a factual explanation from Member C's explain_result(result) will appear here.")
    with st.container(border=True):
        st.subheader("Quiz")
        st.caption("Planned: Member C will supply questions and feedback here.")


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
        {"Concept": "IID / non-IID data partitioning", "Status": "Implemented in prototype"},
        {"Concept": "FedAvg training loop", "Status": "Implemented on synthetic features"},
        {"Concept": "Intermittent participation", "Status": "Simplified participation model"},
        {"Concept": "Orbital visibility", "Status": "Illustrative map only"},
        {"Concept": "FedProx / FedAsync / SatSFL", "Status": "Research direction, not implemented"},
    ])
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    st.caption("Please cite the specific papers used in the final report. Repository links are starting points, not evidence that every feature is implemented in this app.")


def main() -> None:
    # Migrate old Streamlit session-state values after merging Explore + Results.
    if st.session_state.pop("navigate_to_results", False):
        st.session_state.page_nav = "Explore & Results"
        st.session_state.selected_page = "Explore & Results"
    for key in ("page_nav", "selected_page"):
        if st.session_state.get(key) in ("Explore", "Results"):
            st.session_state[key] = "Explore & Results"
        if key in st.session_state and st.session_state.get(key) not in PAGES:
            st.session_state[key] = "Overview"

    with st.sidebar:
        st.markdown("## 🛰️ Satellite FL")
        st.caption("Interactive learning toolkit")
        st.markdown("### Pages")
        selected_page = st.radio(
            "Navigate", PAGES,
            index=PAGES.index(st.session_state.get("selected_page", "Overview")),
            label_visibility="collapsed", key="page_nav",
        )
        st.session_state.selected_page = selected_page
        # NEW: progress indicator
        page_number = PAGES.index(selected_page) + 1
        st.progress(page_number / len(PAGES), text=f"Module {page_number} of {len(PAGES)}")
        st.divider()
        result = get_current_result()
        if result:
            st.success("Experiment available")
            cfg = result.get("config", {})
            st.caption(f"{cfg.get('num_clients', '?')} clients · {cfg.get('partition', '?')}")
        else:
            st.caption("Run your first experiment from Explore & Results")
        st.caption(f"AI tutor: {tutor_mode_label()}")  # CHANGED
        st.caption("Data: synthetic prototype")

    if selected_page not in ["Overview", "Explore & Results"]:
        tutor_sidebar(selected_page)

    if selected_page == "Overview":
        render_overview()
    elif selected_page == "Learn":
        render_learn()
    elif selected_page == "Explore & Results":
        render_explore()
    elif selected_page == "Knowledge Check":
        render_knowledge_check()
    elif selected_page == "Research":
        render_research()

    st.divider()
    st.caption("Satellite FL Explorer · Educational prototype · synthetic benchmark unless otherwise stated")


if __name__ == "__main__":
    main()