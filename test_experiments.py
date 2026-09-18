import unittest
import numpy as np
from experiments import run_suite, solve


class OptimizationTests(unittest.TestCase):
    def test_identity_converges_in_one_update(self):
        result = solve(np.eye(2), [3, -4])
        self.assertEqual(result["status"], "converged")
        self.assertEqual(len(result["trace"]), 2)
        self.assertEqual(result["trace"][-1]["objective"], 0)

    def test_fixed_matches_closed_form(self):
        result = solve(np.diag([1., 4.]), [3, -4], step=.1, max_steps=20)
        for row in result["trace"]:
            np.testing.assert_allclose(row["point"], np.array([3, -4]) * np.array([.9, .6])**row["iteration"], atol=1e-13)

    def test_armijo_is_monotone(self):
        result = solve(np.diag([1., 100.]), [3, -4], "armijo", step=1.)
        self.assertEqual(result["status"], "converged")
        self.assertTrue(np.all(np.diff([r["objective"] for r in result["trace"]]) <= 1e-12))

    def test_divergence_is_reported(self):
        self.assertEqual(solve(np.eye(2), [3, -4], step=3.)["status"], "diverged")

    def test_budget_is_not_convergence(self):
        self.assertEqual(solve(np.eye(2), [3, -4], max_steps=0)["status"], "budget_exhausted")

    def test_reproducible(self):
        self.assertEqual(run_suite(7), run_suite(7))

    def test_invalid_inputs(self):
        for a, x, kw in [(np.zeros((2, 2)), [1, 2], {}), (np.eye(2), [1, 2], {"step": -1}),
                         (np.eye(2), [np.nan, 2], {}), ([[1, 4], [0, 1]], [1, 2], {})]:
            with self.assertRaises(ValueError):
                solve(a, x, **kw)


if __name__ == "__main__":
    unittest.main()
