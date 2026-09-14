from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mojopqc_sca.datasets.validation import summarize_project_hdf5
from mojopqc_sca.utils.config import load_config


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally so large trace files never enter RAM at once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str | Path, project_root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if record["exists"]:
        record["sha256"] = sha256_file(resolved)
        try:
            record["relative_path"] = str(resolved.relative_to(project_root.resolve()))
        except ValueError:
            record["relative_path"] = None
    else:
        record["sha256"] = None
        record["relative_path"] = None
    return record


def _git_record(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=project_root, check=True,
                capture_output=True, text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "revision": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "h5py", "torch", "onnxruntime", "pyyaml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
    except ImportError:
        cuda_available = False
        cuda_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "mojo_available": shutil.which("mojo") is not None,
    }


def build_run_manifest(
    config_path: str | Path,
    dataset_path: str | Path,
    artifact_paths: Iterable[str | Path] = (),
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Create an auditable manifest for one experiment or demonstration run."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path)
    dataset_path = Path(dataset_path)
    config = load_config(config_path)
    dataset_validation = summarize_project_hdf5(dataset_path)
    actual_splits = dataset_validation["splits"]
    consistency_checks = {
        "profiling_count_matches_config": actual_splits["profiling"]["trace_count"] == int(config["dataset"]["profiling_traces"]),
        "attack_count_matches_config": actual_splits["attack"]["trace_count"] == int(config["dataset"]["attack_traces"]),
        "processed_length_matches_target": all(
            item["trace_length"] == int(config["preprocessing"]["target_length"])
            for item in actual_splits.values()
        ),
    }
    artifacts = [_file_record(path, project_root) for path in artifact_paths]
    return {
        "project": "MojoPQC-SCA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_record(project_root),
        "environment": _environment_record(),
        "config": {"file": _file_record(config_path, project_root), "values": config},
        "dataset": {
            "file": _file_record(dataset_path, project_root),
            "validation": dataset_validation,
            "consistency_checks": consistency_checks,
        },
        "artifacts": artifacts,
    }
