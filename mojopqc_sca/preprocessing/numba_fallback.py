from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from mojopqc_sca.datasets.hdf5_loader import validate_hdf5_layout
from mojopqc_sca.utils.benchmark import PeakMemoryMonitor
from .alignment import align_xcorr
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
        normalized = normalize_traces(traces)
    else:
        normalized = _normalize_numba(np.asarray(traces, dtype=np.float32))
    filtered = fir_filter(normalized, kernel_size)
    indices = np.linspace(0, filtered.shape[1] - 1, target_len).astype(np.int64)
    return filtered[:, indices]


def preprocess_pipeline_fallback(input_path: str | Path, output_path: str | Path, config: dict) -> dict:
    """Run the bounded-memory fallback with the same stages as the reference path."""
    started = time.perf_counter()
    memory = PeakMemoryMonitor().start()
    validate_hdf5_layout(input_path)
    pcfg = config["preprocessing"]
    chunk_size = int(config["dataset"]["chunk_size"])
    target_length = int(pcfg["target_length"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = None
    try:
        with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
            target.require_group("traces"); target.require_group("labels")
            for split in ("profiling", "attack"):
                source_traces = source[f"traces/{split}"]
                source_labels = source[f"labels/{split}"]
                count = int(source_traces.shape[0])
                output_traces = target.create_dataset(
                    f"traces/{split}", shape=(count, target_length), dtype="f4",
                    chunks=(min(chunk_size, max(1, count)), target_length), compression="lzf",
                )
                target.create_dataset(f"labels/{split}", data=source_labels[:], dtype="u1")
                reference = np.asarray(source_traces[0], dtype=np.float32) if count else np.zeros(source_traces.shape[1], dtype=np.float32)
                for start in range(0, count, chunk_size):
                    stop = min(count, start + chunk_size)
                    batch = np.asarray(source_traces[start:stop], dtype=np.float32)
                    normalized = normalize_traces(batch) if njit is None else _normalize_numba(batch)
                    filtered = fir_filter(normalized, int(pcfg["filter_kernel_size"]))
                    aligned = align_xcorr(filtered, reference, int(pcfg["max_shift"]))
                    indices = np.linspace(0, aligned.shape[1] - 1, target_length).astype(np.int64)
                    output_traces[start:stop] = aligned[:, indices]
            if "metadata/config" in source:
                target.create_dataset("metadata/config", data=source["metadata/config"][()])
        validate_hdf5_layout(output_path)
        result = {
            "execution_time_seconds": time.perf_counter() - started,
            "backend": "numba" if njit is not None else "numpy_scipy",
            "numba_available": njit is not None,
        }
    finally:
        peak_memory_bytes = memory.stop()
    result["peak_memory_bytes"] = peak_memory_bytes
    return result
