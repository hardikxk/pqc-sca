import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mojopqc_sca.datasets.external_hdf5 import convert_zenodo_hdf5, discover_zenodo_layout, inspect_external_hdf5


class ExternalHDF5Tests(unittest.TestCase):
    def test_inspect_discover_and_convert_fixed_random_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "external.h5"; output_path = Path(directory) / "converted.h5"
            with h5py.File(source_path, "w") as source:
                source.create_dataset("fixed/traces", data=np.ones((4, 12), dtype=np.float64))
                source.create_dataset("fixed/inputs", data=np.array([[7, 1], [8, 1], [9, 1], [10, 1]], dtype=np.uint8))
                source.create_dataset("random/traces", data=np.zeros((3, 12), dtype=np.float64))
                source.create_dataset("random/input", data=np.array([[3, 2], [4, 2], [5, 2]], dtype=np.uint8))
            self.assertEqual(discover_zenodo_layout(source_path)["attack_inputs"], "random/input")
            self.assertEqual(len(inspect_external_hdf5(source_path)), 4)
            record = convert_zenodo_hdf5(source_path, output_path, label_byte=0, chunk_size=2)
            self.assertEqual(record["profiling_traces"], 4)
            self.assertGreaterEqual(record["execution_time_seconds"], 0.0)
            self.assertGreater(record["peak_memory_bytes"], 0)
            with h5py.File(output_path, "r") as converted:
                np.testing.assert_array_equal(converted["labels/profiling"][:], [7, 8, 9, 10])
                self.assertEqual(converted["traces/attack"].shape, (3, 12))
                self.assertEqual(json.loads(converted["metadata/config"][()].decode())["label_definition"]["byte_index"], 0)

    def test_explicit_paths_override_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "external.h5"; output_path = Path(directory) / "converted.h5"
            with h5py.File(source_path, "w") as source:
                source.create_dataset("custom/power_a", data=np.ones((2, 3), dtype=np.float32))
                source.create_dataset("custom/inputs_a", data=np.array([[5], [6]], dtype=np.uint8))
                source.create_dataset("custom/power_b", data=np.zeros((2, 3), dtype=np.float32))
                source.create_dataset("custom/inputs_b", data=np.array([[7], [8]], dtype=np.uint8))
            convert_zenodo_hdf5(
                source_path,
                output_path,
                label_byte=0,
                paths={
                    "profiling_traces": "custom/power_a",
                    "profiling_inputs": "custom/inputs_a",
                    "attack_traces": "custom/power_b",
                    "attack_inputs": "custom/inputs_b",
                },
            )
            with h5py.File(output_path, "r") as converted:
                np.testing.assert_array_equal(converted["labels/attack"][:], [7, 8])
