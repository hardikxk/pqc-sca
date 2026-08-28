from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mojopqc_sca.preprocessing.python_baseline import preprocess_pipeline
from mojopqc_sca.utils.benchmark import save_json
from mojopqc_sca.utils.config import ensure_output_dirs, load_config

parser = argparse.ArgumentParser(description="Run streaming Python preprocessing baseline")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--input", default="data/raw/synthetic_pqc.h5")
parser.add_argument("--output", default="data/processed/python_processed.h5")
args = parser.parse_args()
ensure_output_dirs()
stats = preprocess_pipeline(args.input, args.output, load_config(args.config))
stats.update({"input": args.input, "output": args.output})
save_json("results/benchmarks/python_preprocess.json", stats)
print(f"Processed {args.input} -> {args.output}")
print(f"Execution time: {stats['execution_time_seconds']:.3f}s")
