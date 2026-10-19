"""Lightweight FedAvg demonstration on synthetic land-cover-like features.

This prototype deliberately avoids external image downloads. It validates the
website-to-training workflow while a real EuroSAT/CNN backend is developed.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

CLASS_NAMES = [
    "Annual crop", "Forest", "Herbaceous veg.", "Highway", "Industrial",
    "Pasture", "Permanent crop", "Residential", "River", "Sea / lake",
]

# NEW: Label agreed in the team result schema (schema_version 1.0).
# The FedAvg math is real, but the data are synthetic, so this is not a
# measured EuroSAT result. Change this only after agreeing with the team.
RESULT_SOURCE = "illustrative_only"


def _make_synthetic_dataset(
    seed: int = 42, n_train: int = 1200, n_test: int = 500,
    n_features: int = 16, n_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible, separable feature vectors (not actual satellite imagery)."""
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(0.0, 1.2, size=(n_classes, n_features))
    y_train = np.arange(n_train) % n_classes
    y_test = np.arange(n_test) % n_classes
    rng.shuffle(y_train)
    rng.shuffle(y_test)
    X_train = prototypes[y_train] + rng.normal(0.0, 1.5, size=(n_train, n_features))
    X_test = prototypes[y_test] + rng.normal(0.0, 1.5, size=(n_test, n_features))
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        ((X_train - mean) / std).astype(np.float32), y_train.astype(np.int64),
        ((X_test - mean) / std).astype(np.float32), y_test.astype(np.int64),
    )


def _partition_indices(y: np.ndarray, n_clients: int, partition: str, alpha: float, seed: int) -> list[np.ndarray]:
    """Partition indices IID or by per-class Dirichlet allocation."""
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(y))
    if partition == "IID":
        rng.shuffle(all_indices)
        return [part.astype(np.int64) for part in np.array_split(all_indices, n_clients)]

    clients: list[list[int]] = [[] for _ in range(n_clients)]
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        proportions = rng.dirichlet(np.full(n_clients, max(float(alpha), 0.03)))
        counts = rng.multinomial(len(label_indices), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            clients[client_id].extend(label_indices[start:start + count].tolist())
            start += count

    min_items = max(10, n_clients * 2)
    for client_id in range(n_clients):
        while len(clients[client_id]) < min_items:
            donor = max(range(n_clients), key=lambda k: len(clients[k]))
            if donor == client_id or len(clients[donor]) <= min_items:
                break
            pick = int(rng.integers(0, len(clients[donor])))
            clients[client_id].append(clients[donor].pop(pick))

    out = [np.asarray(sorted(client), dtype=np.int64) for client in clients]
    for part in out:
        rng.shuffle(part)
    return out


def _softmax_logits(X_aug: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = X_aug @ weights
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)


def _metrics(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    X_aug = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
    probs = _softmax_logits(X_aug, weights)
    predicted = probs.argmax(axis=1)
    accuracy = float(np.mean(predicted == y))
    loss = float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)).mean())
    return accuracy, loss


def _local_train(X: np.ndarray, y: np.ndarray, initial_weights: np.ndarray, epochs: int, learning_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = initial_weights.copy()
    X_aug = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
    n_classes = weights.shape[1]
    batch_size = min(32, max(8, len(y)))
    for _ in range(max(1, epochs)):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            xb, yb = X_aug[batch], y[batch]
            probs = _softmax_logits(xb, weights)
            one_hot = np.eye(n_classes, dtype=np.float32)[yb]
            gradient = (xb.T @ (probs - one_hot)) / max(1, len(batch))
            gradient[:-1] += 0.001 * weights[:-1]
            weights -= learning_rate * gradient
    return weights


def run_demo_experiment(
    num_clients: int = 4,
    partition: str = "Non-IID",
    alpha: float = 0.3,
    rounds: int = 10,
    local_epochs: int = 1,
    seed: int = 42,
    participation_rate: float = 1.0,
) -> dict[str, Any]:
    """Run a small FedAvg loop with optional client participation constraints.

    The data and linear softmax model are synthetic demonstration components,
    not EuroSAT image classification. ``participation_rate`` determines the
    fraction of clients available in each round (at least one client).
    """
    start_time = time.perf_counter()
    num_clients = int(np.clip(num_clients, 2, 8))
    rounds = int(np.clip(rounds, 1, 30))
    local_epochs = int(np.clip(local_epochs, 1, 5))
    participation_rate = float(np.clip(participation_rate, 0.25, 1.0))
    partition = "IID" if partition.upper() == "IID" else "Non-IID"

    X_train, y_train, X_test, y_test = _make_synthetic_dataset(seed=seed)
    client_indices = _partition_indices(y_train, num_clients, partition, alpha, seed=seed + 101)
    class_counts, client_sizes = [], []
    for indices in client_indices:
        counts = np.bincount(y_train[indices], minlength=len(CLASS_NAMES))
        class_counts.append({CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))})
        client_sizes.append(int(len(indices)))

    n_features, n_classes = X_train.shape[1], len(CLASS_NAMES)
    global_weights = np.zeros((n_features + 1, n_classes), dtype=np.float32)
    initial_accuracy, initial_loss = _metrics(X_test, y_test, global_weights)
    accuracy_history, loss_history = [initial_accuracy], [initial_loss]
    client_accuracy_history: list[list[float | None]] = [[] for _ in range(num_clients)]
    participation_history: list[list[int]] = []
    client_participation_counts = [0] * num_clients
    local_example_steps = 0
    parameter_count = int(global_weights.size)
    total_upload_bytes = 0
    clients_per_round = max(1, min(num_clients, int(np.ceil(num_clients * participation_rate))))

    # NEW: Per-round records in the team schema format. Round 0 is the
    # untrained baseline evaluation (all-zero weights) before any training.
    round_metrics: list[dict[str, Any]] = [{
        "round": 0, "accuracy": float(initial_accuracy), "loss": float(initial_loss),
        "participating_clients": [], "elapsed_seconds": 0.0,
    }]

    for round_index in range(rounds):
        round_rng = np.random.default_rng(seed + 73_991 + round_index * 997)
        participants = sorted(round_rng.choice(num_clients, size=clients_per_round, replace=False).tolist())
        participation_history.append([int(v) for v in participants])
        local_weights, local_sizes = [], []

        for client_id, indices in enumerate(client_indices):
            if client_id not in participants:
                client_accuracy_history[client_id].append(None)
                continue
            local_X, local_y = X_train[indices], y_train[indices]
            updated = _local_train(
                local_X, local_y, global_weights, epochs=local_epochs,
                learning_rate=0.12, seed=seed + round_index * 1009 + client_id * 31,
            )
            local_weights.append(updated)
            local_sizes.append(len(indices))
            local_example_steps += len(indices) * local_epochs
            client_participation_counts[client_id] += 1
            client_acc, _ = _metrics(X_test, y_test, updated)
            client_accuracy_history[client_id].append(client_acc)

        total = float(sum(local_sizes))
        global_weights = sum(weights * (size / total) for weights, size in zip(local_weights, local_sizes)).astype(np.float32)
        accuracy, loss = _metrics(X_test, y_test, global_weights)
        accuracy_history.append(accuracy)
        loss_history.append(loss)
        # Uncompressed float32 uplink from participating clients; excludes protocol overhead.
        total_upload_bytes += parameter_count * np.dtype(np.float32).itemsize * len(participants)

        # NEW: schema-style round record
        round_metrics.append({
            "round": round_index + 1, "accuracy": float(accuracy), "loss": float(loss),
            "participating_clients": [int(v) for v in participants],
            "elapsed_seconds": float(time.perf_counter() - start_time),
        })

    elapsed = time.perf_counter() - start_time
    return {
        # NEW: team schema fields
        "schema_version": "1.0",
        "run_id": f"synthetic-{partition.lower()}-k{num_clients}-a{float(alpha):g}-seed{int(seed)}",
        "result_source": RESULT_SOURCE,
        "classes": list(CLASS_NAMES),
        "round_metrics": round_metrics,
        "total_elapsed_seconds": float(elapsed),
        "hardware": "local CPU (NumPy)",
        # Existing fields used by the app
        "config": {
            "dataset": "Synthetic land-cover-like features", "num_clients": num_clients,
            "partition": partition, "alpha": float(alpha), "rounds": rounds,
            "local_epochs": local_epochs, "seed": int(seed),
            "participation_rate": participation_rate,
        },
        "class_names": list(CLASS_NAMES),
        "client_class_counts": class_counts,
        "client_sizes": client_sizes,
        "metrics": {
            "global_accuracy": [float(x) for x in accuracy_history],
            "global_loss": [float(x) for x in loss_history],
            "client_accuracy": client_accuracy_history,
        },
        "participation": {
            "clients_per_round": participation_history,
            "participations_per_client": client_participation_counts,
            "clients_per_round_count": clients_per_round,
        },
        "communication": {
            "parameters": parameter_count,
            "upload_bytes_per_round": int(parameter_count * 4 * clients_per_round),
            "estimated_upload_bytes_total": int(total_upload_bytes),
            "local_example_epochs": int(local_example_steps),
        },
        "runtime_seconds": float(elapsed), "status": "complete",
        "result_type": "synthetic_demo",
        "note": (
            "Actual FedAvg parameter aggregation on synthetic feature vectors, not EuroSAT imagery. "
            "Contact availability is represented by simulated client participation; positions on the map are illustrative."
        ),
    }


def make_preview_partition(num_clients: int = 4, partition: str = "Non-IID", alpha: float = 0.3, seed: int = 42) -> list[dict[str, Any]]:
    """Return class counts for a quick client-distribution preview."""
    _, y_train, _, _ = _make_synthetic_dataset(seed=seed)
    indices = _partition_indices(y_train, num_clients, partition, alpha, seed + 101)
    preview = []
    for client_id, client_indices in enumerate(indices):
        counts = np.bincount(y_train[client_indices], minlength=len(CLASS_NAMES))
        for class_index, count in enumerate(counts):
            preview.append({"Satellite": f"Sat {client_id + 1}", "Class": CLASS_NAMES[class_index], "Samples": int(count)})
    return preview