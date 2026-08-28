from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import h5py
import numpy as np

from .alignment import align_xcorr
from .filtering import fir_filter
from .normalize import normalize_traces
from mojopqc_sca.utils.benchmark import PeakMemoryMonitor
from mojopqc_sca.datasets.hdf5_loader import validate_hdf5_layout


def downsample_traces(traces: np.ndarray, target_len: int = 5000) -> np.ndarray:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    indices = np.linspace(0, traces.shape[1] - 1, target_len).astype(np.int64)
    return np.asarray(traces)[:, indices]


def preprocess_pipeline(input_path: str | Path, output_path: str | Path, config: dict[str, Any]) -> dict[str, float]:
    started = time.perf_counter()
    memory = PeakMemoryMonitor().start()
    validate_hdf5_layout(input_path)
    pcfg = config["preprocessing"]
    chunk_size = config["dataset"]["chunk_size"]
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
        target.require_group("traces"); target.require_group("labels")
        for split in ("profiling", "attack"):
            source_traces = source[f"traces/{split}"]
            source_labels = source[f"labels/{split}"]
            output_traces = target.create_dataset(f"traces/{split}", shape=(source_traces.shape[0], pcfg["target_length"]), dtype="f4", chunks=(min(chunk_size, source_traces.shape[0]), pcfg["target_length"]), compression="lzf")
            target.create_dataset(f"labels/{split}", data=source_labels[:], dtype="u1")
            reference = np.asarray(source_traces[0], dtype=np.float32)
            for start in range(0, source_traces.shape[0], chunk_size):
                stop = min(source_traces.shape[0], start + chunk_size)
                batch = normalize_traces(source_traces[start:stop])
                batch = fir_filter(batch, pcfg["filter_kernel_size"])
                batch = align_xcorr(batch, reference, pcfg["max_shift"])
                output_traces[start:stop] = downsample_traces(batch, pcfg["target_length"])
        if "metadata/config" in source:
            target.create_dataset("metadata/config", data=source["metadata/config"][()])
    return {"execution_time_seconds": time.perf_counter() - started, "peak_memory_bytes": memory.stop()}
