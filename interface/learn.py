"""Learn page for the Satellite FL Explorer (Member B: interface).

Sections: FedAvg basics, IID vs non-IID, Why satellites are different, Knowledge Check.
Quiz wording lives in content/quiz_bank.py so it can be edited without changing this file.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.fl_simulator import CLASS_NAMES, make_preview_partition
from content.quiz_bank import QUIZ_QUESTIONS

LEARN_SECTIONS = ["📘 FedAvg basics", "📊 IID vs non-IID", "🛰️ Why satellites are different"]
SAFE_PREDICTION = "May change convergence/performance"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=24)
def _preview(num_clients: int, partition: str, alpha: float, seed: int):
    return make_preview_partition(num_clients, partition, alpha, seed)


def _distribution_figure(df: pd.DataFrame, title: str):
    fig = px.bar(
        df, x="Satellite", y="Samples", color="Class", barmode="stack",
        category_orders={"Class": CLASS_NAMES}, title=title,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        legend_title_text="Land-cover class", xaxis_title="Simulated satellite client",
        yaxis_title="Training samples", margin=dict(l=12, r=12, t=55, b=12), height=355,
        legend=dict(orientation="h", yanchor="bottom", y=-.35, xanchor="left", x=0),
    )
    return fig


def _reset_question_set(form_key: str, question_ids: list[str]) -> None:
    """Callback: clear stored answers so the questions can be retaken."""
    st.session_state.pop(f"{form_key}_results", None)
    for qid in question_ids:
        st.session_state.pop(f"{form_key}_{qid}", None)


def _render_question_set(questions: list[dict[str, Any]], form_key: str, submit_label: str) -> None:
    """Render a graded multiple-choice form with score, feedback, and retake."""
    questions = [q for q in questions if q.get("answer") in q.get("options", [])]  # skip malformed entries
    if not questions:
        st.info("No questions are available yet.")
        return
    results_key = f"{form_key}_results"

    with st.form(form_key):
        for i, q in enumerate(questions, start=1):
            st.radio(f"**Q{i}. {q['question']}**", q["options"], index=None, key=f"{form_key}_{q['id']}")
        submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True)

    if submitted:
        answers = {q["id"]: st.session_state.get(f"{form_key}_{q['id']}") for q in questions}
        if any(value is None for value in answers.values()):
            st.warning("Please answer every question before submitting.")
            st.session_state.pop(results_key, None)
        else:
            st.session_state[results_key] = answers

    answers = st.session_state.get(results_key)
    if not answers:
        return

    score = sum(answers.get(q["id"]) == q["answer"] for q in questions)
    st.progress(score / len(questions), text=f"Score: {score} / {len(questions)}")
    if score == len(questions):
        st.success("Perfect score! 🎉")
    for i, q in enumerate(questions, start=1):
        chosen = answers.get(q["id"])
        topic = f" *({q['topic']})*" if q.get("topic") else ""
        if chosen == q["answer"]:
            st.success(f"**Q{i} ✓ Correct.**{topic} {q['explanation']}")
        else:
            st.error(f"**Q{i} ✗** You chose \"{chosen}\". **Correct answer:** \"{q['answer']}\".{topic} {q['explanation']}")
    st.button("↺ Retake", key=f"{form_key}_retake", on_click=_reset_question_set,
              args=(form_key, [q["id"] for q in questions]))


def _run_key(result: dict[str, Any]) -> str:
    """Stable ID for a run's config, so a new run gets fresh questions."""
    raw = json.dumps(result.get("config", {}), sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def _run_questions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build questions whose correct answers come from the user's actual run."""
    cfg = result.get("config", {})
    metrics = result.get("metrics", {})
    acc = metrics.get("global_accuracy", [])
    loss = metrics.get("global_loss", [])
    questions: list[dict[str, Any]] = []

    if len(acc) >= 2:
        delta = float(acc[-1]) - float(acc[0])
        options = ["Increased", "Decreased", "Stayed about the same (within 1 percentage point)"]
        answer = options[0] if delta > 0.01 else options[1] if delta < -0.01 else options[2]
        questions.append({
            "id": "run_accuracy", "topic": "Your run", "options": options, "answer": answer,
            "question": "From round 0 to the final round, how did global test accuracy change in your run?",
            "explanation": (f"Round 0 accuracy was {float(acc[0]):.1%} and final accuracy was {float(acc[-1]):.1%} "
                            f"after {len(acc) - 1} rounds. The untrained starting model gives every class equal scores, "
                            "so round 0 sits near chance level (about 10% for 10 classes)."),
        })

    if len(loss) >= 2:
        delta = float(loss[-1]) - float(loss[0])
        options = ["Lower than round 0", "Higher than round 0", "About the same (within 0.01)"]
        answer = options[0] if delta < -0.01 else options[1] if delta > 0.01 else options[2]
        questions.append({
            "id": "run_loss", "topic": "Your run", "options": options, "answer": answer,
            "question": "How did the final global test loss compare with round 0?",
            "explanation": (f"Round 0 loss was {float(loss[0]):.3f} and final loss was {float(loss[-1]):.3f}. "
                            "Lower cross-entropy loss means the model assigns higher probability to the correct classes."),
        })

    k = cfg.get("num_clients")
    per_round = result.get("participation", {}).get("clients_per_round_count")
    if isinstance(k, int) and isinstance(per_round, int) and 1 <= per_round <= k:
        options = [f"{n} of {k}" for n in range(1, k + 1)]
        rate = float(cfg.get("participation_rate", 1.0))
        questions.append({
            "id": "run_participation", "topic": "Your run", "options": options, "answer": options[per_round - 1],
            "question": "In your run, how many satellites contributed to each FedAvg round?",
            "explanation": (f"Contact availability was {rate:.0%} of {k} satellites, which rounds up to {per_round} per round. "
                            "The rest were skipped in each round. That's why some satellites appear grey on the map."),
        })

    counts = result.get("client_class_counts")
    if cfg.get("partition") == "Non-IID" and counts and isinstance(counts[0], dict):
        sat1 = counts[0]
        ranked = sorted(sat1, key=sat1.get, reverse=True)
        if len(ranked) >= 4 and sat1[ranked[0]] > sat1[ranked[1]]:
            top = ranked[0]
            options = sorted([top] + ranked[-3:])
            questions.append({
                "id": "run_dominant", "topic": "Your run", "options": options, "answer": top,
                "question": "Which land-cover class was most common in Sat 1's local data?",
                "explanation": (f"Sat 1 held {sat1[top]} '{top}' samples out of {sum(sat1.values())}. "
                                f"With α = {cfg.get('alpha')}, the Dirichlet split gave this satellite a skewed class mix."),
            })
    return questions


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _render_basics() -> None:
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
    st.info("Ready to test yourself? Open Knowledge Check from the left icon navigation.")
    if st.button("Go to Knowledge Check →", key="learn_go_to_knowledge_check", type="primary"):
        st.session_state["selected_page"] = "Knowledge Check"
        st.rerun()


def _render_distribution() -> None:
    st.markdown("A satellite constellation may see different places and therefore different label distributions. "
                "This is called statistical heterogeneity or non-IID data.")
    alpha_col, client_col = st.columns([1, 1])
    with alpha_col:
        partition = st.radio("Preview partition", ["IID", "Non-IID"], horizontal=True, key="learn_partition")
        alpha = st.slider("Dirichlet concentration α", 0.05, 5.0, 0.3, step=0.05,
                          disabled=(partition == "IID"), key="learn_alpha")
        clients = st.select_slider("Number of clients", [2, 3, 4, 5, 6, 8], value=4, key="learn_clients")
        st.caption("Smaller α usually increases class imbalance. The effect on model accuracy must be measured; "
                   "it is not guaranteed to be monotonic.")
    with client_col:
        preview = pd.DataFrame(_preview(int(clients), partition, float(alpha), 42))
        st.plotly_chart(_distribution_figure(preview, "What each client receives"), use_container_width=True)
    st.markdown("**Think before running:** If each satellite has a different class mix, will local updates always "
                "point in the same direction? Why or why not?")


def _render_satellite() -> None:
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
    st.info("These are the target application challenges. The current simulator models client participation as a "
            "simplified availability setting; orbital geometry and onboard hardware are not physically simulated.")


def _render_knowledge_check() -> None:
    result = st.session_state.get("last_result")

    with st.container(border=True):
        st.subheader("1 · Your prediction")
        hypothesis = st.session_state.get("last_hypothesis")
        if not hypothesis:
            st.caption("On Explore & Results, choose a prediction before pressing Run Simulation. It will appear here.")
        else:
            st.markdown(f"Before running, you predicted that non-IID data will… **{hypothesis.lower()}**")
            if hypothesis == SAFE_PREDICTION:
                st.success("Good hypothesis. Heterogeneity can change convergence and performance, "
                           "but the size and direction of the effect depend on the settings.")
            else:
                st.info("A safer conclusion is that non-IID data *may* change convergence or performance. "
                        "It does not guarantee lower accuracy in every run. Compare an IID and a non-IID run to see.")

    with st.container(border=True):
        st.subheader("2 · Interpret your latest run")
        if not result:
            st.info("Run an experiment in Explore & Results, then come back to answer questions about your own results.")
        else:
            cfg = result.get("config", {})
            if result.get("result_source") != "measured":
                st.warning("Illustrative example: synthetic prototype data, not measured EuroSAT training.")
            st.caption(f"Run settings · {cfg.get('partition')} · α = {cfg.get('alpha')} · "
                       f"{cfg.get('num_clients')} satellites · {cfg.get('rounds')} rounds · "
                       f"{cfg.get('local_epochs')} local epochs · seed {cfg.get('seed')}")
            run_questions = _run_questions(result)
            if run_questions:
                _render_question_set(run_questions, form_key=f"kc_run_{_run_key(result)}",
                                     submit_label="Check my interpretation")
            else:
                st.caption("This result doesn't include enough data to generate questions.")

    with st.container(border=True):
        st.subheader("3 · Concept quiz")
        st.caption(f"{len(QUIZ_QUESTIONS)} questions. Answer them all, then submit to see your score and explanations.")
        _render_question_set(QUIZ_QUESTIONS, form_key="kc_concepts", submit_label="Submit answers")


# ---------------------------------------------------------------------------
# Page entry points (called from app.py)
# ---------------------------------------------------------------------------

def render_learn() -> None:
    """Render the educational concepts page (quiz is a separate main page)."""
    st.caption("MODULE 01 · CONCEPTS")
    st.title("Learn")
    st.write("Build intuition for federated learning, then explore the satellite-specific challenges.")

    # Migrate the old section value from versions where Knowledge Check was nested here.
    if st.session_state.get("learn_section") not in LEARN_SECTIONS:
        st.session_state["learn_section"] = LEARN_SECTIONS[0]
    section = st.radio(
        "Section", LEARN_SECTIONS, horizontal=True, key="learn_section",
        label_visibility="collapsed",
    )
    st.divider()

    if section == LEARN_SECTIONS[0]:
        _render_basics()
    elif section == LEARN_SECTIONS[1]:
        _render_distribution()
    else:
        _render_satellite()


def render_knowledge_check_page() -> None:
    """Standalone Knowledge Check page with a scored concept quiz and run-specific quiz."""
    st.caption("MODULE 04 · PRACTICE")
    st.title("Knowledge Check")
    st.write("Test your understanding of federated learning, then interpret the results from your own experiment.")
    _render_knowledge_check()
