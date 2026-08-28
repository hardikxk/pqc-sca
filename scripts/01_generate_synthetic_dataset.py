from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mojopqc_sca.datasets.synthetic_pqc import generate_dataset
from mojopqc_sca.utils.config import ensure_output_dirs, load_config

parser = argparse.ArgumentParser(description="Generate chunked synthetic PQC traces")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--output", default="data/raw/synthetic_pqc.h5")
parser.add_argument("--profiling", type=int)
parser.add_argument("--attack", type=int)
parser.add_argument("--trace-length", type=int)
args = parser.parse_args()
started = time.perf_counter()
config = load_config(args.config)
if args.profiling is not None: config["dataset"]["profiling_traces"] = args.profiling
if args.attack is not None: config["dataset"]["attack_traces"] = args.attack
if args.trace_length is not None: config["dataset"]["trace_length"] = args.trace_length
ensure_output_dirs()
generate_dataset(args.output, config)
import h5py
with h5py.File(args.output, "r") as dataset:
    print(f"Created {args.output}")
    print(f"profiling: {dataset['traces/profiling'].shape}; attack: {dataset['traces/attack'].shape}")
print(f"Execution time: {time.perf_counter() - started:.3f}s")
