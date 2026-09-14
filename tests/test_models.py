import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional in the CPU preprocessing environment")
class ModelTests(unittest.TestCase):
    def test_parameter_budget_and_output_shapes(self):
        import torch
        from mojopqc_sca.models.cnn import LightweightCNN, count_parameters
        from mojopqc_sca.models.cnn_mps import CNNWithMPS

        for model in (LightweightCNN(), CNNWithMPS()):
            self.assertLess(count_parameters(model), 200_000)
            with torch.no_grad():
                self.assertEqual(model(torch.randn(2, 1, 5000)).shape, (2, 256))

    def test_training_device_auto_is_safe_without_cuda(self):
        from scripts._train_common import resolve_device

        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertIn(resolve_device("auto").type, {"cpu", "cuda"})
        with self.assertRaises(ValueError):
            resolve_device("tpu")
