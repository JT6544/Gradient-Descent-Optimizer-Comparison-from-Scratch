from __future__ import annotations

import unittest

import numpy as np

from optimization_lab.objectives import default_objectives


class ObjectiveTests(unittest.TestCase):
    def test_known_minima_are_stationary(self) -> None:
        for objective in default_objectives():
            point = np.asarray(objective.minimizer)
            self.assertAlmostEqual(objective.value(point), objective.minimum, places=12)
            np.testing.assert_allclose(objective.gradient(point), 0.0, atol=1e-10)

    def test_analytical_gradients_match_finite_differences(self) -> None:
        step = 1e-6
        for objective in default_objectives():
            point = np.asarray(objective.initial_point, dtype=float)
            numerical = np.empty(objective.dimension)
            for index in range(objective.dimension):
                offset = np.zeros(objective.dimension)
                offset[index] = step
                numerical[index] = (
                    objective.value(point + offset) - objective.value(point - offset)
                ) / (2.0 * step)
            np.testing.assert_allclose(
                objective.gradient(point), numerical, rtol=1e-5, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
