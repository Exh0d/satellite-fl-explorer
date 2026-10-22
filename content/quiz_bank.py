"""Knowledge Check question bank.

Member C: edit wording, options, and explanations here. The app reads
QUIZ_QUESTIONS automatically; no other file needs to change.

Each question is built with _q(id, topic, question, options, correct_index, explanation).
correct_index counts from 0, so 0 = first option, 1 = second option, and so on.
"""
from __future__ import annotations


def _q(qid: str, topic: str, question: str, options: list[str], correct_index: int, explanation: str) -> dict:
    return {
        "id": qid,
        "topic": topic,
        "question": question,
        "options": options,
        "answer": options[correct_index],
        "explanation": explanation,
    }


QUIZ_QUESTIONS = [
    _q(
        "fedavg_payload", "FedAvg basics",
        "In FedAvg, what does each satellite send to the server after local training?",
        ["Its raw satellite images",
         "Its updated model parameters (or the change in its parameters)",
         "Only its test-set accuracy",
         "The labels of its images"],
        1,
        "Raw data stay on the satellite. The server only receives model information and averages it into a new global model.",
    ),
    _q(
        "fedavg_weighting", "FedAvg basics",
        "When FedAvg combines client models, how is each participating client usually weighted?",
        ["Equally, regardless of how much data it has",
         "By how quickly it finished training",
         "By its share of the participating clients' training samples",
         "By its accuracy on the test set"],
        2,
        "In the FedAvg equation, each client is weighted by |D_i| divided by the total samples of all participating clients, "
        "so satellites with more data have more influence.",
    ),
    _q(
        "iid_definition", "IID vs non-IID",
        "Which situation describes an IID data partition?",
        ["Every satellite's class proportions are roughly the same as the full dataset's",
         "One satellite sees mostly forest while another sees mostly sea",
         "Each satellite holds exactly one class",
         "Satellites hold different subsets of the classes"],
        0,
        "IID means each client's data look like a random sample of the whole dataset. The other options all describe non-IID data.",
    ),
    _q(
        "alpha_effect", "IID vs non-IID",
        "In a Dirichlet partition, what does making α smaller usually do?",
        ["Makes every client's data more balanced",
         "Increases the number of satellites",
         "Changes the model architecture",
         "Makes clients' class mixes more uneven"],
        3,
        "Small α tends to concentrate each class on a few clients, producing skewed local datasets. "
        "Large α tends to give mixes closer to IID. α changes the data split, not the model.",
    ),
    _q(
        "client_drift", "IID vs non-IID",
        "Why can non-IID data make FedAvg converge more slowly or less smoothly?",
        ["The server cannot average models of different sizes",
         "Non-IID data is always mislabeled",
         "Local updates can pull the global model in different directions, so their average may suit no client well",
         "Satellites refuse to send updates"],
        2,
        "Each satellite optimizes for its own class mix. When those mixes differ, local updates can disagree "
        "(often called client drift), which can slow or destabilize the averaged model.",
    ),
    _q(
        "noniid_always", "Interpreting results",
        "Does non-IID data always lower final accuracy?",
        ["Yes, in every case",
         "No. It always improves accuracy",
         "No. It can change convergence and performance, but the effect depends on the data, model, and settings, so it should be measured"],
        2,
        "Heterogeneity often hurts, but not in every run. The size of the effect depends on α, rounds, local epochs, "
        "participation, the model, and the dataset. That's why the app measures it instead of assuming it.",
    ),
    _q(
        "contact", "Satellite constraints",
        "In this simulator, what happens to a satellite that is unavailable (out of contact) during a round?",
        ["It skips that round, so its data do not contribute to that global update",
         "Its update is averaged in anyway",
         "Its raw data are sent to the server instead",
         "The whole round is cancelled"],
        0,
        "Only participating satellites are aggregated in each round. The 'Client contact availability' slider controls "
        "how many satellites take part per round.",
    ),
    _q(
        "local_epochs", "Training settings",
        "What is a possible trade-off of increasing local epochs (E)?",
        ["Less computation on each satellite",
         "The server must store all raw images",
         "Aggregation is no longer needed",
         "More progress per communication round, but local models may drift further apart on non-IID data"],
        3,
        "More local epochs mean more onboard computation and can reduce the number of rounds needed. "
        "With non-IID data, though, each local model moves further toward its own data before averaging.",
    ),
    _q(
        "why_satellites", "Satellite constraints",
        "Why is federated learning attractive for satellite constellations?",
        ["Satellites have unlimited bandwidth",
         "Downlinking all raw imagery can be slow or costly; sharing model updates can need less communication and keeps raw data onboard",
         "Ground stations cannot train neural networks",
         "It guarantees higher accuracy than centralized training"],
        1,
        "Short contact windows and limited downlink bandwidth make moving all imagery expensive. "
        "FL does not guarantee better accuracy than centralized training; it is a trade-off.",
    ),
    _q(
        "round_zero", "Reading the charts",
        "In this app's accuracy and loss charts, what does round 0 show?",
        ["The model after one round of training",
         "The best round so far",
         "The untrained starting model, evaluated before any federated training",
         "An average over all rounds"],
        2,
        "Round 0 is a baseline evaluation of the starting model, so you can see how much federated training improved it.",
    ),
]