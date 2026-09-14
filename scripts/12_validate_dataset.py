from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mojopqc_sca.datasets.validation import summarize_project_hdf5
from mojopqc_sca.utils.benchmark import PeakMemoryMonitor


parser = argparse.ArgumentParser(description="Validate and summarize a project-contract HDF5 dataset")
parser.add_argument("--input", required=True, help="HDF5 file containing traces/{profiling,attack} and labels/{profiling,attack}")
parser.add_argument("--output", help="Optional JSON report path")
parser.add_argument("--chunk-size", type=int, default=256)
parser.add_argument("--strict", action="store_true", help="Return a failure code for non-finite traces or out-of-range labels")
args = parser.parse_args()

started = time.perf_counter()
memory = PeakMemoryMonitor().start()
try:
    report = summarize_project_hdf5(args.input, chunk_size=args.chunk_size)
except (OSError, ValueError, KeyError) as error:
    memory.stop()
    print(f"Dataset validation failed: {error}", file=sys.stderr)
    raise SystemExit(1)
report.update({"execution_time_seconds": time.perf_counter() - started, "peak_memory_bytes": memory.stop()})

if args.output:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"Execution time: {report['execution_time_seconds']:.3f}s")
print(f"Peak RSS: {report['peak_memory_bytes'] / (1024 ** 2):.1f} MiB")
if args.strict and not report["ready_for_pipeline"]:
    raise SystemExit(2)
