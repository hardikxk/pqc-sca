import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mojopqc_sca.datasets.synthetic_pqc import generate_dataset
from mojopqc_sca.datasets.hdf5_loader import iter_hdf5_split, validate_hdf5_layout
from mojopqc_sca.preprocessing.python_baseline import downsample_traces, preprocess_pipeline
from mojopqc_sca.utils.config import validate_config


class PreprocessingTests(unittest.TestCase):
    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({"dataset": {"profiling_traces": 1, "attack_traces": 1, "trace_length": 10, "chunk_size": 1, "classes": 256}, "preprocessing": {"filter_kernel_size": 2, "max_shift": 0, "target_length": 5}})

    def test_downsample_respects_requested_length(self):
        traces = np.arange(20, dtype=np.float32).reshape(2, 10)
        self.assertEqual(downsample_traces(traces, 5).shape, (2, 5))

    def test_synthetic_to_processed_hdf5(self):
        config = {"seed": 1, "dataset": {"profiling_traces": 3, "attack_traces": 2, "trace_length": 100, "chunk_size": 2, "classes": 4, "noise_std": 1.0, "leakage_amplitude": 0.1}, "preprocessing": {"filter_kernel_size": 3, "max_shift": 5, "target_length": 20}}
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.h5"; processed = Path(directory) / "processed.h5"
            generate_dataset(raw, config); stats = preprocess_pipeline(raw, processed, config)
            self.assertGreater(stats["execution_time_seconds"], 0)
            with h5py.File(processed, "r") as dataset:
                self.assertEqual(dataset["traces/profiling"].shape, (3, 20))
                self.assertEqual(dataset["traces/attack"].shape, (2, 20))

    def test_hdf5_contract_and_chunk_iterator(self):
        config = {"seed": 2, "dataset": {"profiling_traces": 5, "attack_traces": 2, "trace_length": 30, "chunk_size": 2, "classes": 4, "noise_std": 1.0, "leakage_amplitude": 0.1}, "preprocessing": {"filter_kernel_size": 3, "max_shift": 2, "target_length": 10}}
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.h5"; generate_dataset(raw, config)
            layout = validate_hdf5_layout(raw)
            self.assertEqual(layout["traces/profiling"], (5, 30))
            batches = list(iter_hdf5_split(raw, "profiling", chunk_size=2))
            self.assertEqual([batch[0].shape[0] for batch in batches], [2, 2, 1])
