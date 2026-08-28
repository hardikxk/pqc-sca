from __future__ import annotations

import numpy as np


def compute_key_rank(log_probs: np.ndarray, true_label: int) -> int:
    scores = np.asarray(log_probs)
    if scores.ndim != 1:
        raise ValueError("log_probs must be one-dimensional")
    return int(np.sum(scores > scores[int(true_label)]) + 1)


def compute_guessing_entropy(log_probs: np.ndarray, true_labels: np.ndarray) -> float:
    scores = np.asarray(log_probs)
    labels = np.asarray(true_labels)
    if scores.ndim != 2 or labels.shape != (scores.shape[0],) or scores.shape[0] == 0:
        raise ValueError("scores must be (traces, classes) and labels must match traces")
    cumulative = np.zeros(scores.shape[1], dtype=np.float64)
    for row, label in zip(scores, labels):
        cumulative += row
    return float(compute_key_rank(cumulative, int(labels[-1])))


def compute_ge_curve(log_probs: np.ndarray, true_labels: np.ndarray, step: int = 100) -> dict[int, float]:
    scores = np.asarray(log_probs)
    labels = np.asarray(true_labels)
    cumulative = np.zeros(scores.shape[1], dtype=np.float64)
    curve: dict[int, float] = {}
    for index, (row, label) in enumerate(zip(scores, labels), start=1):
        cumulative += row
        if index % step == 0 or index == len(labels):
            curve[index] = float(compute_key_rank(cumulative, int(label)))
    return curve
