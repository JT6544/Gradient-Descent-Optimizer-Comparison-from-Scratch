from __future__ import annotations

import unittest

import numpy as np

from optimization_lab.objectives import Objective
from optimization_lab.optimizers import OptimizerConfig, optimize


def quadratic(initial_point: tuple[float, ...]) -> Objective:
    return Objective(
        key="test_quadratic",
        display_name="Test quadratic",
        initial_point=initial_point,
        minimizer=tuple(0.0 for _ in initial_point),
        minimum=0.0,
        value_function=lambda point: float(0.5 * np.dot(point, point)),
        gradient_function=lambda point: point.copy(),
        plot_bounds=tuple((-2.0, 2.0) for _ in initial_point),
    )


class OptimizerTests(unittest.TestCase):
    def test_gradient_descent_first_update(self) -> None:
        result = optimize(
            quadratic((2.0, -4.0)),
            OptimizerConfig("gradient_descent", learning_rate=0.1, max_steps=1, tolerance=0.0),
        )
        np.testing.assert_allclose(result.points[1], [1.8, -3.6])

    def test_normalized_gradient_has_configured_step_length(self) -> None:
        result = optimize(
            quadratic((3.0, 4.0)),
            OptimizerConfig(
                "normalized_gradient_descent",
                learning_rate=0.2,
                max_steps=1,
                tolerance=0.0,
            ),
        )
        self.assertAlmostEqual(float(np.linalg.norm(result.points[1] - result.points[0])), 0.2)

    def test_rmsprop_first_update_uses_accumulator(self) -> None:
        config = OptimizerConfig(
            "rmsprop", learning_rate=0.1, max_steps=1, tolerance=0.0, rho=0.9
        )
        result = optimize(quadratic((2.0, -4.0)), config)
        gradient = np.asarray([2.0, -4.0])
        expected = gradient - 0.1 * gradient / (np.sqrt(0.1 * gradient**2) + config.epsilon)
        np.testing.assert_allclose(result.points[1], expected)

    def test_adam_first_update_is_bias_corrected(self) -> None:
        config = OptimizerConfig("adam", learning_rate=0.1, max_steps=1, tolerance=0.0)
        result = optimize(quadratic((2.0, -4.0)), config)
        np.testing.assert_allclose(result.points[1], [1.9, -3.9], rtol=1e-7, atol=1e-8)

    def test_all_histories_are_finite_and_aligned(self) -> None:
        for optimizer in (
            "gradient_descent",
            "normalized_gradient_descent",
            "rmsprop",
            "adam",
        ):
            result = optimize(
                quadratic((1.0, -0.5)),
                OptimizerConfig(optimizer, learning_rate=0.05, max_steps=25),
            )
            self.assertEqual(result.points.shape[0], result.values.size)
            self.assertEqual(result.values.size, result.gradient_norms.size)
            self.assertTrue(np.isfinite(result.points).all())
            self.assertTrue(np.isfinite(result.values).all())


if __name__ == "__main__":
    unittest.main()
