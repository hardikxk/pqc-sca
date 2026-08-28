from __future__ import annotations

import numpy as np

from .filtering import fir_filter
from .normalize import normalize_traces

try:
    from numba import njit
except ImportError:
    njit = None


if njit is not None:
    @njit(cache=True)
    def _normalize_numba(traces):
        output = np.empty_like(traces, dtype=np.float32)
        for row in range(traces.shape[0]):
            mean = np.float32(0.0)
            for column in range(traces.shape[1]): mean += traces[row, column]
            mean /= traces.shape[1]
            variance = np.float32(0.0)
            for column in range(traces.shape[1]):
                delta = traces[row, column] - mean; variance += delta * delta
            std = np.sqrt(variance / traces.shape[1]) + np.float32(1e-8)
            for column in range(traces.shape[1]): output[row, column] = (traces[row, column] - mean) / std
        return output


def preprocess_chunk(traces: np.ndarray, target_len: int = 5000, kernel_size: int = 11) -> np.ndarray:
    """Portable fallback used when the optional Mojo toolchain is unavailable."""
    if njit is None:
        print("Warning: Numba is unavailable; using NumPy/SciPy preprocessing fallback.")
        normalized = normalize_traces(traces)
    else:
        normalized = _normalize_numba(np.asarray(traces, dtype=np.float32))
    filtered = fir_filter(normalized, kernel_size)
    indices = np.linspace(0, filtered.shape[1] - 1, target_len).astype(np.int64)
    return filtered[:, indices]
