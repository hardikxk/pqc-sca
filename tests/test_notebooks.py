"""
Unit tests to ensure all Jupyter / Colab notebooks are well-formed and valid.
"""
import glob
import json
import os
import unittest


class TestNotebooks(unittest.TestCase):
    def setUp(self):
        self.notebook_dir = os.path.join(os.path.dirname(__file__), "..", "notebooks")
        self.notebook_files = glob.glob(os.path.join(self.notebook_dir, "*.ipynb"))

    def test_notebooks_exist(self):
        self.assertGreaterEqual(
            len(self.notebook_files),
            3,
            f"Expected at least 3 notebooks in {self.notebook_dir}, found {len(self.notebook_files)}",
        )

    def test_notebook_json_validity(self):
        expected_names = {"colab_train.ipynb", "quick_demo.ipynb", "results_visualization.ipynb"}
        found_names = {os.path.basename(p) for p in self.notebook_files}
        self.assertTrue(
            expected_names.issubset(found_names),
            f"Missing expected notebooks: {expected_names - found_names}",
        )

        for nb_path in self.notebook_files:
            nb_name = os.path.basename(nb_path)
            with self.subTest(notebook=nb_name):
                with open(nb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.assertIn("cells", data, f"{nb_name} missing 'cells' key")
                self.assertIsInstance(data["cells"], list, f"{nb_name} cells should be a list")
                self.assertGreater(len(data["cells"]), 0, f"{nb_name} has no cells")

                # Verify cell types and source presence
                for idx, cell in enumerate(data["cells"]):
                    self.assertIn("cell_type", cell, f"{nb_name} cell #{idx} missing cell_type")
                    self.assertIn(cell["cell_type"], ("code", "markdown", "raw"), f"{nb_name} cell #{idx} unknown type")
                    self.assertIn("source", cell, f"{nb_name} cell #{idx} missing source")


if __name__ == "__main__":
    unittest.main()
