from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mojopqc_sca.datasets.torch_hdf5 import HDF5TraceDataset
from mojopqc_sca.evaluation.guessing_entropy import compute_ge_curve
from mojopqc_sca.models.cnn_mps import CNNWithMPS

parser = argparse.ArgumentParser(description="Evaluate attack traces with guessing entropy")
parser.add_argument("--input", default="data/processed/python_processed.h5")
parser.add_argument("--checkpoint", default="results/models/cnn_mps.pt")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--step", type=int, default=100)
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location="cpu")
model = CNNWithMPS(checkpoint["input_len"], checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state"])
model.eval()
dataset = HDF5TraceDataset(args.input, "attack")
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
all_log_probs, all_labels = [], []
with torch.no_grad():
    for traces, labels in loader:
        all_log_probs.append(torch.log_softmax(model(traces), dim=1).numpy())
        all_labels.append(labels.numpy())
log_probs = np.concatenate(all_log_probs)
labels = np.concatenate(all_labels)
curve = compute_ge_curve(log_probs, labels, args.step)
Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/benchmarks/ge_results.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
plt.figure(figsize=(7, 4))
plt.plot(list(curve), list(curve.values()), marker="o")
plt.xlabel("Attack traces"); plt.ylabel("Guessing entropy (rank)"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/ge_curve.png", dpi=160); plt.close()
for requested in (100, 500, 1000):
    available = [key for key in curve if key >= requested]
    if available:
        print(f"GE @ {requested} traces: {curve[available[0]]:.2f}")
print(f"GE @ all traces: {curve[max(curve)]:.2f}")
