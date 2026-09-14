import hashlib
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mojopqc_sca.utils.reproducibility import build_run_manifest, sha256_file


class ReproducibilityTests(unittest.TestCase):
    def test_sha256_and_manifest_capture_dataset_and_artifact_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                "seed: 1\ndataset:\n  profiling_traces: 1\n  attack_traces: 1\n  trace_length: 2\n  chunk_size: 1\n  classes: 2\npreprocessing:\n  filter_kernel_size: 1\n  max_shift: 0\n  target_length: 2\n",
                encoding="utf-8",
            )
            dataset = root / "data.h5"
            with h5py.File(dataset, "w") as handle:
                handle.create_dataset("traces/profiling", data=np.zeros((1, 2), dtype=np.float32))
                handle.create_dataset("labels/profiling", data=np.array([0], dtype=np.uint8))
                handle.create_dataset("traces/attack", data=np.ones((1, 2), dtype=np.float32))
                handle.create_dataset("labels/attack", data=np.array([1], dtype=np.uint8))
            artifact = root / "artifact.txt"
            artifact.write_text("demo", encoding="utf-8")
            expected = hashlib.sha256(b"demo").hexdigest()
            self.assertEqual(sha256_file(artifact), expected)
            manifest = build_run_manifest(config, dataset, [artifact], project_root=root)
            self.assertTrue(manifest["dataset"]["validation"]["ready_for_pipeline"])
            self.assertTrue(manifest["dataset"]["consistency_checks"]["profiling_count_matches_config"])
            self.assertTrue(manifest["dataset"]["consistency_checks"]["processed_length_matches_target"])
            self.assertEqual(manifest["artifacts"][0]["sha256"], expected)
            self.assertIn("environment", manifest)
