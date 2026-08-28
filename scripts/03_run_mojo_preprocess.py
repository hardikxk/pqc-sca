from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mojopqc_sca.preprocessing.python_baseline import preprocess_pipeline
from mojopqc_sca.utils.benchmark import save_json
from mojopqc_sca.utils.config import ensure_output_dirs, load_config

parser = argparse.ArgumentParser(description="Run Mojo preprocessing or its portable fallback")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--input", default="data/raw/synthetic_pqc.h5")
parser.add_argument("--output", default="data/processed/mojo_processed.h5")
args = parser.parse_args()
ensure_output_dirs()
config = load_config(args.config)
started = time.perf_counter()
if shutil.which("mojo"):
    print("Mojo detected, but the HDF5 bridge is not enabled yet; using the reference pipeline.")
    backend = "python_reference"
else:
    print("Warning: Mojo is unavailable; using the Python fallback.")
    backend = "python_fallback"
stats = preprocess_pipeline(args.input, args.output, config)
stats.update({"backend": backend, "input": args.input, "output": args.output, "execution_time_seconds": time.perf_counter() - started})
save_json("results/benchmarks/mojo_preprocess.json", stats)
print(f"Processed with {backend} in {stats['execution_time_seconds']:.3f}s")
