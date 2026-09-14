from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .hdf5_loader import validate_hdf5_layout
from mojopqc_sca.utils.benchmark import PeakMemoryMonitor


def _dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            paths.append(path)
        elif isinstance(value, h5py.Group):
            paths.extend(_dataset_paths(value, path))
    return paths


def inspect_external_hdf5(path: str | Path) -> list[dict[str, Any]]:
    """Return dataset metadata without loading trace payloads."""
    with h5py.File(path, "r") as source:
        result = []
        for dataset_path in _dataset_paths(source):
            dataset = source[dataset_path]
            result.append({"path": dataset_path, "shape": list(dataset.shape), "dtype": str(dataset.dtype), "chunks": list(dataset.chunks) if dataset.chunks else None})
        return result


def _find_first(paths: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(paths)
    return next((candidate for candidate in candidates if candidate in available), None)


def discover_zenodo_layout(path: str | Path) -> dict[str, str]:
    """Discover the common fixed/random trace and input dataset names."""
    paths = [item["path"] for item in inspect_external_hdf5(path)]
    discovered = {
        "profiling_traces": _find_first(paths, ("fixed/traces", "fixed/trace", "profiling/traces")),
        "attack_traces": _find_first(paths, ("random/traces", "random/trace", "attack/traces")),
        "profiling_inputs": _find_first(paths, ("fixed/inputs", "fixed/input", "profiling/inputs")),
        "attack_inputs": _find_first(paths, ("random/inputs", "random/input", "attack/inputs")),
    }
    return {key: value for key, value in discovered.items() if value is not None}


def _labels_from_inputs(inputs: np.ndarray, label_byte: int) -> np.ndarray:
    flattened = np.asarray(inputs).reshape(inputs.shape[0], -1)
    if label_byte < 0 or label_byte >= flattened.shape[1]:
        raise ValueError(f"label_byte={label_byte} is outside input width {flattened.shape[1]}")
    return flattened[:, label_byte].astype(np.uint8, copy=False)


def _copy_split(target: h5py.File, source: h5py.File, split: str, trace_path: str, input_path: str, label_byte: int, chunk_size: int, limit: int | None) -> tuple[int, int]:
    traces = source[trace_path]
    inputs = source[input_path]
    if traces.ndim != 2 or inputs.ndim < 1 or inputs.shape[0] != traces.shape[0]:
        raise ValueError(f"{split}: traces must be 2-D and inputs must have the same first dimension")
    if not np.issubdtype(traces.dtype, np.number):
        raise ValueError(f"{split}: traces must have a numeric dtype, got {traces.dtype}")
    if not np.issubdtype(inputs.dtype, np.integer):
        raise ValueError(f"{split}: inputs must have an integer dtype, got {inputs.dtype}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if label_byte < 0:
        raise ValueError("label_byte must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("split limits must be non-negative")
    input_width = int(np.prod(inputs.shape[1:], dtype=np.int64)) if inputs.ndim > 1 else 1
    if label_byte >= input_width:
        raise ValueError(f"label_byte={label_byte} is outside input width {input_width}")
    count = min(traces.shape[0], limit) if limit is not None else traces.shape[0]
    output = target.create_dataset(f"traces/{split}", shape=(count, traces.shape[1]), dtype="f4", chunks=(min(chunk_size, max(1, count)), min(traces.shape[1], 4096)), compression="lzf")
    labels = target.create_dataset(f"labels/{split}", shape=(count,), dtype="u1", chunks=(min(chunk_size, max(1, count)),))
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        output[start:stop] = np.asarray(traces[start:stop], dtype=np.float32)
        labels[start:stop] = _labels_from_inputs(np.asarray(inputs[start:stop]), label_byte)
    return count, traces.shape[1]


def convert_zenodo_hdf5(input_path: str | Path, output_path: str | Path, label_byte: int, chunk_size: int = 256, profiling_limit: int | None = None, attack_limit: int | None = None, paths: dict[str, str] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert fixed/random trace groups into the project HDF5 contract."""
    started = time.perf_counter()
    memory = PeakMemoryMonitor().start()
    record: dict[str, Any] | None = None
    try:
        discovered = discover_zenodo_layout(input_path)
        selected = {**discovered, **(paths or {})}
        required = ("profiling_traces", "attack_traces", "profiling_inputs", "attack_inputs")
        missing = [key for key in required if key not in selected]
        if missing:
            raise ValueError(f"Could not discover required external datasets: {missing}. Use explicit dataset paths.")
        output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(input_path, "r") as source, h5py.File(output, "w") as target:
            target.require_group("traces"); target.require_group("labels"); target.require_group("metadata")
            profiling_count, trace_length = _copy_split(target, source, "profiling", selected["profiling_traces"], selected["profiling_inputs"], label_byte, chunk_size, profiling_limit)
            attack_count, attack_length = _copy_split(target, source, "attack", selected["attack_traces"], selected["attack_inputs"], label_byte, chunk_size, attack_limit)
            if trace_length != attack_length:
                raise ValueError("Profiling and attack traces have different sample lengths")
            record = {"source_file": str(Path(input_path).resolve()), "source_layout": selected, "label_definition": {"type": "input_byte", "byte_index": label_byte}, "profiling_traces": profiling_count, "attack_traces": attack_count, "trace_length": trace_length, "chunk_size": chunk_size, **(metadata or {})}
            target.create_dataset("metadata/config", data=json.dumps(record))
        validate_hdf5_layout(output)
        return record
    finally:
        peak_memory_bytes = memory.stop()
        if record is not None:
            record.update({"execution_time_seconds": time.perf_counter() - started, "peak_memory_bytes": peak_memory_bytes})
