from __future__ import annotations
import numpy as np


def align_xcorr(traces: np.ndarray, reference: np.ndarray, max_shift: int = 500) -> np.ndarray:
    traces = np.asarray(traces, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if traces.ndim != 2 or reference.ndim != 1 or traces.shape[1] != reference.size:
        raise ValueError("traces must be 2-D and reference must match its sample length")
    ref = reference - reference.mean()
    output = np.empty_like(traces)
    limit = min(max_shift, traces.shape[1] - 1)
    for i, trace in enumerate(traces):
        centered = trace - trace.mean()
        corr = np.correlate(centered, ref, mode="full")
        center = trace.size - 1
        lo, hi = center - limit, center + limit + 1
        shift = int(np.argmax(corr[lo:hi]) - limit)
        output[i] = np.roll(trace, -shift)
    return output
