from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mojopqc_sca.models.cnn import LightweightCNN, count_parameters
from scripts._train_common import train_model

parser = argparse.ArgumentParser(); parser.add_argument("--input", default="data/processed/python_processed.h5"); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--input-len", type=int, default=5000); parser.add_argument("--classes", type=int, default=256); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
args = parser.parse_args(); model = LightweightCNN(args.input_len, args.classes)
print(f"Trainable parameters: {count_parameters(model)}")
train_model(model, args.input, "results/models/cnn_baseline.pt", "results/logs/cnn_training.csv", args.epochs, args.batch_size, 1e-3, device=args.device)
