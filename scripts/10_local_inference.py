from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import h5py
import numpy as np
import onnxruntime as ort

from mojopqc_sca.evaluation.streaming import update_ranks
from mojopqc_sca.utils.benchmark import PeakMemoryMonitor

parser = argparse.ArgumentParser(description="Run CPU ONNX inference and streaming GE evaluation")
parser.add_argument("--model", default="results/models/cnn_mps_int8.onnx")
parser.add_argument("--input", default="data/processed/python_processed.h5")
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()
session_options = ort.SessionOptions()
session_options.log_severity_level = 3
session = ort.InferenceSession(args.model, sess_options=session_options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
cumulative = None; ranks = []; trace_count = 0; started = time.perf_counter(); memory = PeakMemoryMonitor().start()
with h5py.File(args.input, "r") as dataset:
    traces = dataset["traces/attack"]; labels = dataset["labels/attack"]
    for start in range(0, traces.shape[0], args.batch_size):
        stop = min(start + args.batch_size, traces.shape[0])
        batch = np.asarray(traces[start:stop], dtype=np.float32)[:, None, :]
        logits = np.asarray(session.run(None, {input_name: batch})[0])
        if cumulative is None: cumulative = np.zeros(logits.shape[1], dtype=np.float64)
        ranks.extend(update_ranks(cumulative, logits, np.asarray(labels[start:stop])))
        trace_count += stop - start
elapsed = time.perf_counter() - started
result = {"model": args.model, "traces": trace_count, "inference_seconds": elapsed, "milliseconds_per_trace": elapsed * 1000 / max(1, trace_count), "peak_memory_bytes": memory.stop(), "ge_all": float(ranks[-1])}
Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
Path("results/benchmarks/local_inference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"Inference time per trace: {result['milliseconds_per_trace']:.4f} ms")
print(f"GE @ all traces: {result['ge_all']:.2f}")
