from __future__ import annotations

from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset


class HDF5TraceDataset(Dataset):
    """Random-access PyTorch dataset backed by HDF5; samples are read on demand."""

    def __init__(self, path: str | Path, split: str = "profiling", indices: list[int] | None = None):
        self.path = str(path)
        self.split = split
        self.indices = indices
        self._file = None

    def _datasets(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file[f"traces/{self.split}"], self._file[f"labels/{self.split}"]

    def __len__(self) -> int:
        traces, _ = self._datasets()
        return traces.shape[0] if self.indices is None else len(self.indices)

    def __getitem__(self, index: int):
        traces, labels = self._datasets()
        source_index = index if self.indices is None else self.indices[index]
        trace = torch.from_numpy(traces[source_index]).float().unsqueeze(0)
        label = torch.tensor(int(labels[source_index]), dtype=torch.long)
        return trace, label

    def __del__(self):
        if self._file is not None:
            self._file.close()
