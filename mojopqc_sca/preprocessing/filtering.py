from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d


def fir_filter(traces: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return uniform_filter1d(np.asarray(traces, dtype=np.float32), size=kernel_size, axis=1, mode="nearest")
