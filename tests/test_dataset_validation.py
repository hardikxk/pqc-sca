import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mojopqc_sca.datasets.validation import summarize_project_hdf5


class DatasetValidationTests(unittest.TestCase):
    def test_streaming_summary_reports_layout_quality_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.h5"
            with h5py.File(path, "w") as dataset:
                dataset.create_dataset("traces/profiling", data=np.arange(12, dtype=np.float32).reshape(3, 4))
                dataset.create_dataset("labels/profiling", data=np.array([0, 1, 1], dtype=np.uint8))
                dataset.create_dataset("traces/attack", data=np.ones((2, 4), dtype=np.float32))
                dataset.create_dataset("labels/attack", data=np.array([1, 2], dtype=np.uint8))
                dataset.create_dataset("metadata/config", data='{"algorithm": "ML-KEM-512"}')
            report = summarize_project_hdf5(path, chunk_size=2)
            self.assertTrue(report["ready_for_pipeline"])
            self.assertTrue(report["metadata"]["json_valid"])
            self.assertEqual(report["splits"]["profiling"]["trace_count"], 3)
            self.assertEqual(report["splits"]["profiling"]["unique_labels"], [0, 1])

    def test_summary_flags_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.h5"
            with h5py.File(path, "w") as dataset:
                dataset.create_dataset("traces/profiling", data=np.array([[0.0, np.nan]], dtype=np.float32))
                dataset.create_dataset("labels/profiling", data=np.array([0], dtype=np.uint8))
                dataset.create_dataset("traces/attack", data=np.zeros((1, 2), dtype=np.float32))
                dataset.create_dataset("labels/attack", data=np.array([1], dtype=np.uint8))
            report = summarize_project_hdf5(path)
            self.assertFalse(report["ready_for_pipeline"])
            self.assertFalse(report["quality_checks"]["finite_traces"])
