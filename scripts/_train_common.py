from __future__ import annotations

import csv
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from mojopqc_sca.datasets.torch_hdf5 import HDF5TraceDataset
from mojopqc_sca.utils.benchmark import PeakMemoryMonitor


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_model(model, data_path: str, checkpoint: str, log_path: str, epochs: int, batch_size: int, learning_rate: float, validation_fraction: float = 0.2, seed: int = 2026):
    seed_everything(seed)
    memory = PeakMemoryMonitor().start()
    model.to("cpu")
    full = HDF5TraceDataset(data_path, "profiling")
    indices = np.random.default_rng(seed).permutation(len(full))
    split = max(1, int(len(indices) * (1 - validation_fraction)))
    train_set = HDF5TraceDataset(data_path, "profiling", indices[:split].tolist())
    valid_set = HDF5TraceDataset(data_path, "profiling", indices[split:].tolist())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    rows = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train(); train_loss = 0.0; train_correct = 0; train_count = 0
        for traces, labels in train_loader:
            optimizer.zero_grad(); logits = model(traces); loss = loss_fn(logits, labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * labels.size(0); train_correct += (logits.argmax(1) == labels).sum().item(); train_count += labels.size(0)
        model.eval(); valid_acc = 0.0; valid_count = 0
        with torch.no_grad():
            for traces, labels in valid_loader:
                logits = model(traces); valid_acc += (logits.argmax(1) == labels).sum().item(); valid_count += labels.size(0)
        row = {"epoch": epoch, "train_loss": train_loss / max(1, train_count), "train_accuracy": train_correct / max(1, train_count), "validation_accuracy": valid_acc / max(1, valid_count), "epoch_time_seconds": time.perf_counter() - epoch_started}
        rows.append(row); print(f"epoch {epoch}/{epochs}: loss={row['train_loss']:.4f} val_acc={row['validation_accuracy']:.4f}")
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True); torch.save({"model_state": model.state_dict(), "input_len": model.input_len, "num_classes": model.num_classes, "bond_dim": getattr(model, "bond_dim", None)}, checkpoint)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Training time: {time.perf_counter() - started:.3f}s")
    print(f"Peak RSS: {memory.stop() / (1024 ** 2):.1f} MiB")
    return rows[-1]
