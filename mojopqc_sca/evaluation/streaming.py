from __future__ import annotations

import numpy as np


def update_ranks(cumulative_scores: np.ndarray, log_probs: np.ndarray, labels: np.ndarray) -> list[int]:
    """Accumulate attack evidence and return ranks without retaining prior traces."""
    ranks = []
    for scores, label in zip(np.asarray(log_probs), np.asarray(labels)):
        cumulative_scores += scores
        ranks.append(int(np.sum(cumulative_scores > cumulative_scores[int(label)]) + 1))
    return ranks
