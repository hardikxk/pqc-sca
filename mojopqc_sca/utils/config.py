from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"dataset", "preprocessing"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    for key in ("profiling_traces", "attack_traces", "trace_length", "chunk_size", "classes"):
        if int(dataset.get(key, 0)) <= 0:
            raise ValueError(f"dataset.{key} must be positive")
    if int(dataset["classes"]) > 256:
        raise ValueError("The uint8 label format supports at most 256 classes")
    if int(preprocessing.get("target_length", 0)) <= 0 or int(preprocessing["target_length"]) > 5000:
        raise ValueError("preprocessing.target_length must be in [1, 5000]")
    kernel_size = int(preprocessing.get("filter_kernel_size", 0))
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("preprocessing.filter_kernel_size must be a positive odd integer")
    if int(preprocessing.get("max_shift", -1)) < 0:
        raise ValueError("preprocessing.max_shift must be non-negative")


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def ensure_output_dirs(root: str | Path = ".") -> None:
    root = Path(root)
    for relative in ("data/raw", "data/processed", "data/metadata", "results/benchmarks", "results/figures", "results/models", "results/logs", "results/tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
