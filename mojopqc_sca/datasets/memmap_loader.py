from __future__ import annotations

from pathlib import Path
import numpy as np


def open_memmap(path: str | Path, shape: tuple[int, ...], dtype: str | np.dtype = "float32", mode: str = "r") -> np.memmap:
    """Open a raw trace array without copying it into process memory."""
    return np.memmap(Path(path), dtype=dtype, mode=mode, shape=shape)
