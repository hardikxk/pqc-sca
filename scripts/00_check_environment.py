from __future__ import annotations
import importlib.util
import platform
import shutil
import sys


def check(name: str, module: str) -> None:
    print(f"{name}: {'available' if importlib.util.find_spec(module) else 'missing'}")


print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
for name, module in (("NumPy", "numpy"), ("SciPy", "scipy"), ("h5py", "h5py"), ("PyTorch", "torch"), ("ONNX Runtime", "onnxruntime"), ("PyYAML", "yaml")):
    check(name, module)
try:
    import torch
    print(f"CUDA: {'available' if torch.cuda.is_available() else 'unavailable (CPU mode)'}")
except ImportError:
    print("CUDA: unavailable (PyTorch missing)")
print(f"Mojo: {'available' if shutil.which('mojo') else 'unavailable (Numba/Python fallback)'}")
