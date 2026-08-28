from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from mojopqc_sca.models.cnn_mps import CNNWithMPS

parser = argparse.ArgumentParser(description="Export CNN+MPS checkpoint to ONNX and int8 ONNX")
parser.add_argument("--checkpoint", default="results/models/cnn_mps.pt")
parser.add_argument("--onnx", default="results/models/cnn_mps.onnx")
parser.add_argument("--int8-onnx", default="results/models/cnn_mps_int8.onnx")
args = parser.parse_args()
checkpoint = torch.load(args.checkpoint, map_location="cpu")
model = CNNWithMPS(checkpoint["input_len"], checkpoint["num_classes"], checkpoint.get("bond_dim") or 8)
model.load_state_dict(checkpoint["model_state"]); model.eval()
Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)
example = torch.randn(1, 1, checkpoint["input_len"])
torch.onnx.export(model, example, args.onnx, input_names=["trace"], output_names=["logits"], dynamic_axes={"trace": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
with tempfile.TemporaryDirectory(prefix="mojopqc_onnx_") as temporary:
    # ONNX Runtime's quantizer currently rejects some exporter-provided
    # intermediate value_info shapes around the MPS reshape/einsum graph.
    # Removing only those optional annotations preserves the graph and lets
    # the quantizer infer them consistently.
    model_proto = onnx.load(args.onnx, load_external_data=True)
    del model_proto.graph.value_info[:]
    sanitized = str(Path(temporary) / "model.onnx")
    onnx.save(model_proto, sanitized)
    quantize_dynamic(sanitized, args.int8_onnx, weight_type=QuantType.QInt8)
print(f"Wrote {args.onnx}")
print(f"Wrote {args.int8_onnx}")
