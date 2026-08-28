from __future__ import annotations

from pathlib import Path
from typing import Iterator
import h5py
import numpy as np


def validate_hdf5_layout(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Validate the public HDF5 contract without reading trace payloads."""
    required = ("traces/profiling", "labels/profiling", "traces/attack", "labels/attack")
    with h5py.File(path, "r") as dataset:
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(f"Missing HDF5 datasets: {missing}")
        result = {}
        for split in ("profiling", "attack"):
            traces = dataset[f"traces/{split}"]; labels = dataset[f"labels/{split}"]
            if traces.ndim != 2 or labels.ndim != 1 or traces.shape[0] != labels.shape[0]:
                raise ValueError(f"Invalid {split} shapes: traces={traces.shape}, labels={labels.shape}")
            if not np.issubdtype(traces.dtype, np.floating) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"Invalid {split} dtypes: traces={traces.dtype}, labels={labels.dtype}")
            result[f"traces/{split}"] = tuple(traces.shape); result[f"labels/{split}"] = tuple(labels.shape)
        if result["traces/profiling"][1] != result["traces/attack"][1]:
            raise ValueError("Profiling and attack traces must have the same sample length")
    return result


def iter_hdf5_split(path: str | Path, split: str = "profiling", chunk_size: int = 256) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield trace/label batches while keeping the HDF5 file open only per iterator."""
    with h5py.File(path, "r") as dataset:
        traces = dataset[f"traces/{split}"]
        labels = dataset[f"labels/{split}"]
        for start in range(0, traces.shape[0], chunk_size):
            stop = min(start + chunk_size, traces.shape[0])
            yield np.asarray(traces[start:stop]), np.asarray(labels[start:stop])
