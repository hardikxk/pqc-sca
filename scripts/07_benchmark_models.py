from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mojopqc_sca.datasets.torch_hdf5 import HDF5TraceDataset
from mojopqc_sca.evaluation.streaming import update_ranks
from mojopqc_sca.models.cnn import LightweightCNN, count_parameters
from mojopqc_sca.models.cnn_mps import CNNWithMPS
from mojopqc_sca.utils.benchmark import PeakMemoryMonitor

parser = argparse.ArgumentParser(description="Create a paper-ready CNN versus CNN+MPS comparison")
parser.add_argument("--input", default="data/processed/python_processed.h5")
parser.add_argument("--cnn-checkpoint", default="results/models/cnn_baseline.pt")
parser.add_argument("--mps-checkpoint", default="results/models/cnn_mps.pt")
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

def load_model(path: str, mps: bool):
    checkpoint = torch.load(path, map_location="cpu")
    kwargs = (checkpoint["input_len"], checkpoint["num_classes"], checkpoint.get("bond_dim") or 8) if mps else (checkpoint["input_len"], checkpoint["num_classes"])
    model = CNNWithMPS(*kwargs) if mps else LightweightCNN(*kwargs)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, checkpoint

def attack_metrics(model):
    dataset = HDF5TraceDataset(args.input, "attack")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cumulative = np.zeros(model.num_classes, dtype=np.float64); ranks = []; started = time.perf_counter(); memory = PeakMemoryMonitor().start()
    with torch.no_grad():
        for traces, labels in loader:
            ranks.extend(update_ranks(cumulative, torch.log_softmax(model(traces), dim=1).numpy(), labels.numpy()))
    elapsed = time.perf_counter() - started
    result = {"ge_all": float(ranks[-1]), "inference_seconds": elapsed, "peak_memory_bytes": memory.stop(), "ge_100": float(ranks[99]) if len(ranks) >= 100 else None, "ge_500": float(ranks[499]) if len(ranks) >= 500 else None, "ge_1000": float(ranks[999]) if len(ranks) >= 1000 else None}
    return result

rows = []
for name, path, is_mps in (("CNN", args.cnn_checkpoint, False), ("CNN+MPS", args.mps_checkpoint, True)):
    model, checkpoint = load_model(path, is_mps)
    metrics = attack_metrics(model)
    log_path = Path("results/logs") / ("cnn_mps_training.csv" if is_mps else "cnn_training.csv")
    epoch_time = None; validation_accuracy = None
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        if records:
            epoch_time = float(records[-1]["epoch_time_seconds"]); validation_accuracy = float(records[-1]["validation_accuracy"])
    rows.append({"model": name, "parameters": count_parameters(model), "training_time_per_epoch_seconds": epoch_time, "peak_memory_bytes": metrics["peak_memory_bytes"], "validation_accuracy": validation_accuracy, **metrics})

Path("results/benchmarks").mkdir(parents=True, exist_ok=True); Path("results/tables").mkdir(parents=True, exist_ok=True); Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/benchmarks/model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
with Path("results/benchmarks/model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
with Path("results/tables/model_comparison.tex").open("w", encoding="utf-8") as handle:
    handle.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nModel & Parameters & Epoch time (s) & Val. acc. & GE@100 & GE@500 & GE@1000 \\\\\n\\midrule\n")
    for row in rows:
        values = [row["model"], row["parameters"], row["training_time_per_epoch_seconds"], row["validation_accuracy"], row["ge_100"], row["ge_500"], row["ge_1000"]]
        handle.write("{} & {} & {} & {} & {} & {} & {} \\\\\n".format(*["--" if value is None else f"{value:.4g}" if isinstance(value, float) else value for value in values]))
    handle.write("\\bottomrule\n\\end{tabular}\n")
plt.figure(figsize=(7, 4)); names = [row["model"] for row in rows]; ge = [row["ge_all"] for row in rows]; plt.bar(names, ge); plt.ylabel("GE at all attack traces"); plt.tight_layout(); plt.savefig("results/figures/model_comparison.png", dpi=160); plt.close()
print("Model comparison written to results/benchmarks, results/tables, and results/figures.")
