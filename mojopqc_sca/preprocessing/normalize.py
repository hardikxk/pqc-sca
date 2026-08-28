from __future__ import annotations
import numpy as np


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    traces = np.asarray(traces, dtype=np.float32)
    mean = traces.mean(axis=1, keepdims=True)
    std = traces.std(axis=1, keepdims=True)
    return (traces - mean) / (std + 1e-8)
