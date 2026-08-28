from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mojopqc_sca.models.cnn import count_parameters
from mojopqc_sca.models.cnn_mps import CNNWithMPS
from scripts._train_common import train_model

parser = argparse.ArgumentParser(); parser.add_argument("--input", default="data/processed/python_processed.h5"); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--input-len", type=int, default=5000); parser.add_argument("--classes", type=int, default=256); parser.add_argument("--bond-dim", type=int, default=8)
args = parser.parse_args(); model = CNNWithMPS(args.input_len, args.classes, args.bond_dim)
print(f"Trainable parameters: {count_parameters(model)}")
assert count_parameters(model) < 200_000
train_model(model, args.input, "results/models/cnn_mps.pt", "results/logs/cnn_mps_training.csv", args.epochs, args.batch_size, 1e-3)
