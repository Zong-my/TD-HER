"""Data-free behavioral tests; no published experiment result is asserted."""
import unittest
import numpy as np
from experiments import exp_tdh_router as router
from tdher_paths import REPO_ROOT, external_output, data_path, DATA_ROOT
from tdher import _protect_sources

class RoutingTests(unittest.TestCase):
    def test_per_target_admission_and_boundary(self):
        truth = np.zeros((8, 2))
        preds = {"base": np.ones((8, 2)),
                 "LightGBM-Adapter": np.tile([0.8, 0.81], (8, 1))}
        admitted0, info = router.admitted_models_for_target(truth, preds, 0)
        admitted1, _ = router.admitted_models_for_target(truth, preds, 1)
        self.assertIn("LightGBM-Adapter", admitted0)
        self.assertNotIn("LightGBM-Adapter", admitted1)
        self.assertEqual(info["adapter_details"]["LightGBM-Adapter"]["required_MAE"], 0.8)

    def test_convex_combination_recovers_complementarity(self):
        x = np.arange(12, dtype=float)
        columns = np.column_stack([x - 1, x + 2])
        weights, loss = router.optimize_convex_weights(x, columns)
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1, places=10)
        self.assertLess(loss, 1e-5)
        self.assertLess(loss, min(np.abs(columns[:,i]-x).mean() for i in range(2)))

    def test_affine_bias_and_projection(self):
        pred = np.arange(8, dtype=float)
        slope, bias, _ = router.fit_restricted_affine(pred + 0.5, pred)
        self.assertGreaterEqual(slope, 0)
        self.assertLessEqual(slope, 3)
        self.assertLess(np.mean(np.abs(router.apply_affine(pred, slope, bias) - pred - 0.5)), 1e-5)
        values = np.array([[-2.0,-1.0],[1.0,2.0]])
        output = router.enforce_y2_nonnegative(values)
        np.testing.assert_equal(output, [[-2.0,0.0],[1.0,2.0]])
        np.testing.assert_equal(values, [[-2.0,-1.0],[1.0,2.0]])

    def test_source_write_guard(self):
        with self.assertRaises(ValueError):
            external_output(REPO_ROOT / "generated.csv")
        with self.assertRaises(ValueError):
            _protect_sources("open", (str(REPO_ROOT/"generated.csv"), "w", 0))
        _protect_sources("open", (str(REPO_ROOT/"README.md"), "r", 0))
        self.assertEqual(data_path("ieee39_v8_80_10_10/repA"), DATA_ROOT/"ieee39/repA")

if __name__ == "__main__":
    unittest.main()
