from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .hdf5_loader import iter_hdf5_split, validate_hdf5_layout


def _metadata_summary(path: str | Path) -> dict[str, Any]:
    """Read optional provenance metadata without loading trace arrays."""
    with h5py.File(path, "r") as dataset:
        if "metadata/config" not in dataset:
            return {"present": False, "json_valid": False}
        value = dataset["metadata/config"][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"present": True, "json_valid": False}
    return {"present": True, "json_valid": isinstance(parsed, dict)}


def _split_summary(path: str | Path, split: str, trace_length: int, chunk_size: int) -> dict[str, Any]:
    trace_count = 0
    trace_min = float("inf")
    trace_max = float("-inf")
    nonfinite_values = 0
    label_values: set[int] = set()
    labels_outside_uint8 = 0

    for traces, labels in iter_hdf5_split(path, split=split, chunk_size=chunk_size):
        trace_count += int(traces.shape[0])
        if traces.size:
            finite = np.isfinite(traces)
            nonfinite_values += int((~finite).sum())
            if finite.any():
                finite_values = traces[finite]
                trace_min = min(trace_min, float(finite_values.min()))
                trace_max = max(trace_max, float(finite_values.max()))
        if labels.size:
            integer_labels = labels.astype(np.int64, copy=False)
            label_values.update(int(value) for value in np.unique(integer_labels))
            labels_outside_uint8 += int(((integer_labels < 0) | (integer_labels > 255)).sum())

    return {
        "trace_count": trace_count,
        "trace_length": trace_length,
        "trace_min": None if trace_min == float("inf") else trace_min,
        "trace_max": None if trace_max == float("-inf") else trace_max,
        "nonfinite_trace_values": nonfinite_values,
        "label_min": min(label_values) if label_values else None,
        "label_max": max(label_values) if label_values else None,
        "unique_labels": sorted(label_values),
        "labels_outside_uint8": labels_outside_uint8,
    }


def summarize_project_hdf5(path: str | Path, chunk_size: int = 256) -> dict[str, Any]:
    """Validate the project HDF5 contract and report streaming quality checks.

    Only one batch per split is held in memory. The report is intentionally
    JSON-serializable so it can be attached to a paper or demo artifact.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    layout = validate_hdf5_layout(path)
    splits = {
        split: _split_summary(path, split, int(layout[f"traces/{split}"][1]), chunk_size)
        for split in ("profiling", "attack")
    }
    metadata = _metadata_summary(path)
    finite = all(item["nonfinite_trace_values"] == 0 for item in splits.values())
    labels_in_range = all(item["labels_outside_uint8"] == 0 for item in splits.values())
    return {
        "path": str(path.resolve()),
        "schema_valid": True,
        "ready_for_pipeline": finite and labels_in_range,
        "chunk_size": chunk_size,
        "layout": {key: list(shape) for key, shape in layout.items()},
        "metadata": metadata,
        "quality_checks": {
            "finite_traces": finite,
            "labels_in_uint8_range": labels_in_range,
        },
        "splits": splits,
    }
