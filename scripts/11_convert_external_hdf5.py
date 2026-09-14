from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mojopqc_sca.datasets.external_hdf5 import convert_zenodo_hdf5, inspect_external_hdf5

parser = argparse.ArgumentParser(description="Inspect/convert fixed-random ML-KEM HDF5 datasets")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="data/raw/mlkem_converted.h5")
parser.add_argument("--label-byte", type=int, help="Required for conversion: byte column used as the supervised class label")
parser.add_argument("--chunk-size", type=int, default=256)
parser.add_argument("--profiling-limit", type=int)
parser.add_argument("--attack-limit", type=int)
parser.add_argument("--profiling-traces-path", help="Explicit source dataset path for profiling traces")
parser.add_argument("--attack-traces-path", help="Explicit source dataset path for attack traces")
parser.add_argument("--profiling-inputs-path", help="Explicit source dataset path for profiling inputs")
parser.add_argument("--attack-inputs-path", help="Explicit source dataset path for attack inputs")
parser.add_argument("--inspect", action="store_true", help="Print dataset paths/shapes and stop")
parser.add_argument("--metadata-json", help="Optional JSON object merged into provenance metadata")
args = parser.parse_args()

if args.inspect:
    print(json.dumps(inspect_external_hdf5(args.input), indent=2))
    raise SystemExit(0)
if args.label_byte is None:
    parser.error("--label-byte is required for conversion; inspect the source first and choose a defensible label")
if args.chunk_size <= 0:
    parser.error("--chunk-size must be positive")
metadata = json.loads(args.metadata_json) if args.metadata_json else {}
explicit_paths = {
    key: value for key, value in {
        "profiling_traces": args.profiling_traces_path,
        "attack_traces": args.attack_traces_path,
        "profiling_inputs": args.profiling_inputs_path,
        "attack_inputs": args.attack_inputs_path,
    }.items() if value
}
record = convert_zenodo_hdf5(args.input, args.output, args.label_byte, args.chunk_size, args.profiling_limit, args.attack_limit, paths=explicit_paths, metadata=metadata)
print(f"Converted {args.input} -> {args.output}")
print(json.dumps(record, indent=2))
print(f"Execution time: {record['execution_time_seconds']:.3f}s")
print(f"Peak RSS: {record['peak_memory_bytes'] / (1024 ** 2):.1f} MiB")
