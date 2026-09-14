from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mojopqc_sca.utils.benchmark import PeakMemoryMonitor
from mojopqc_sca.utils.reproducibility import build_run_manifest


parser = argparse.ArgumentParser(description="Create a reproducibility manifest for a PQC-SCA run")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--dataset", default="data/processed/python_processed.h5")
parser.add_argument("--output", default="results/benchmarks/run_manifest.json")
parser.add_argument("--artifact", action="append", dest="artifacts", help="Artifact to hash; repeat for multiple files")
args = parser.parse_args()

artifacts = args.artifacts or [
    "results/models/cnn_baseline.pt",
    "results/models/cnn_mps.pt",
    "results/benchmarks/ge_results.json",
    "results/benchmarks/model_comparison.json",
    "results/benchmarks/python_preprocess.json",
    "results/benchmarks/dataset_validation.json",
    "results/benchmarks/local_inference.json",
    "results/models/cnn_mps.onnx",
    "results/models/cnn_mps_int8.onnx",
]
started = time.perf_counter()
memory = PeakMemoryMonitor().start()
try:
    manifest = build_run_manifest(args.config, args.dataset, artifacts)
except (OSError, ValueError, KeyError) as error:
    memory.stop()
    print(f"Manifest generation failed: {error}", file=sys.stderr)
    raise SystemExit(1)
manifest["manifest_generation_seconds"] = time.perf_counter() - started
manifest["peak_memory_bytes"] = memory.stop()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {output}")
print(f"Execution time: {manifest['manifest_generation_seconds']:.3f}s")
print(f"Peak RSS: {manifest['peak_memory_bytes'] / (1024 ** 2):.1f} MiB")
