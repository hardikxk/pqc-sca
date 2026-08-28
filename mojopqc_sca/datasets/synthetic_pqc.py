from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mojopqc_sca.utils.config import validate_config


def hamming_weight(values: np.ndarray) -> np.ndarray:
    return np.unpackbits(values.astype(np.uint8)[:, None], axis=1).sum(axis=1).astype(np.float32)


def generate_split(dataset: h5py.File, name: str, count: int, trace_length: int, classes: int, chunk_size: int, rng: np.random.Generator, noise_std: float, leakage_amplitude: float) -> None:
    traces = dataset.create_dataset(f"traces/{name}", shape=(count, trace_length), dtype="f4", chunks=(min(chunk_size, max(1, count)), min(trace_length, 4096)), compression="lzf")
    labels = dataset.create_dataset(f"labels/{name}", shape=(count,), dtype="u1", chunks=(min(chunk_size, max(1, count)),))
    positions = np.arange(8) * max(1, trace_length // 20)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        batch = stop - start
        batch_labels = rng.integers(0, classes, size=batch, dtype=np.uint8)
        batch_traces = rng.normal(0.0, noise_std, size=(batch, trace_length)).astype(np.float32)
        hw = hamming_weight(batch_labels)
        ntt_start = rng.integers(max(0, trace_length // 5), max(1, trace_length // 3), size=batch)
        jitter = rng.integers(-50, 51, size=batch)
        width = min(200, trace_length)
        for row in range(batch):
            for offset in positions:
                left = int(np.clip(ntt_start[row] + jitter[row] + offset, 0, trace_length - width))
                batch_traces[row, left:left + width] += hw[row] * leakage_amplitude
            batch_traces[row] += rng.normal(0, 0.05)
        traces[start:stop] = batch_traces
        labels[start:stop] = batch_labels


def generate_dataset(output_path: str | Path, config: dict[str, Any]) -> None:
    validate_config(config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dcfg = config["dataset"]
    rng = np.random.default_rng(config.get("seed", 2026))
    with h5py.File(output, "w") as dataset:
        generate_split(dataset, "profiling", dcfg["profiling_traces"], dcfg["trace_length"], dcfg["classes"], dcfg["chunk_size"], rng, dcfg["noise_std"], dcfg["leakage_amplitude"])
        generate_split(dataset, "attack", dcfg["attack_traces"], dcfg["trace_length"], dcfg["classes"], dcfg["chunk_size"], rng, dcfg["noise_std"], dcfg["leakage_amplitude"])
        dataset.create_dataset("metadata/config", data=json.dumps(config))
