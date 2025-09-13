import unittest
import os
import importlib


class TestMVLMEngine(unittest.TestCase):
    def test_factory_selects_mvlm_backend(self):
        # Ensure env selects MVLM before importing the module
        os.environ["NEURAL_ENGINE_BACKEND"] = "mvlm"

        # Reload module to pick up new env settings
        ne = importlib.import_module("mcp_server.neural_engine.neural_engine")
        importlib.reload(ne)

        engine = ne.NeuralEngine()
        out = engine.generate_text("Please write a comprehensive overview")
        # Should be a MVLM-style deterministic string
        self.assertIsInstance(out, str)
        self.assertIn("MVLM", out)


if __name__ == "__main__":
    unittest.main()

