"""NumPy implementations of four first-order optimisation algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objectives import Objective


OPTIMIZER_NAMES = (
    "gradient_descent",
    "normalized_gradient_descent",
    "rmsprop",
    "adam",
)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    max_steps: int = 5_000
    tolerance: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999
    rho: float = 0.9
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.name not in OPTIMIZER_NAMES:
            raise ValueError(f"Unknown optimizer {self.name!r}.")
        if self.learning_rate <= 0 or self.max_steps <= 0:
            raise ValueError("Learning rate and maximum steps must be positive.")
        if self.tolerance < 0 or self.epsilon <= 0:
            raise ValueError("Tolerance cannot be negative and epsilon must be positive.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must lie in [0, 1).")
        if not 0 <= self.rho < 1:
            raise ValueError("RMSProp rho must lie in [0, 1).")


@dataclass(frozen=True)
class OptimizationResult:
    optimizer: str
    learning_rate: float
    points: np.ndarray
    values: np.ndarray
    gradient_norms: np.ndarray
    converged: bool
    converged_step: int | None
    stop_reason: str

    @property
    def steps_run(self) -> int:
        return int(self.values.size - 1)


def optimize(objective: Objective, config: OptimizerConfig) -> OptimizationResult:
    """Minimise an objective with the selected first-order algorithm."""

    config.validate()
    point = np.asarray(objective.initial_point, dtype=float)
    first_moment = np.zeros_like(point)
    second_moment = np.zeros_like(point)

    points = [point.copy()]
    values = [objective.value(point)]
    gradient = objective.gradient(point)
    gradient_norms = [float(np.linalg.norm(gradient))]

    if gradient_norms[-1] <= config.tolerance:
        return _result(config, points, values, gradient_norms, True, 0, "gradient_tolerance")

    for step in range(1, config.max_steps + 1):
        gradient = objective.gradient(point)

        if config.name == "gradient_descent":
            direction = gradient
        elif config.name == "normalized_gradient_descent":
            norm = float(np.linalg.norm(gradient))
            direction = gradient / max(norm, config.epsilon)
        elif config.name == "rmsprop":
            second_moment = (
                config.rho * second_moment + (1.0 - config.rho) * gradient**2
            )
            direction = gradient / (np.sqrt(second_moment) + config.epsilon)
        else:
            first_moment = config.beta1 * first_moment + (1.0 - config.beta1) * gradient
            second_moment = (
                config.beta2 * second_moment + (1.0 - config.beta2) * gradient**2
            )
            corrected_first = first_moment / (1.0 - config.beta1**step)
            corrected_second = second_moment / (1.0 - config.beta2**step)
            direction = corrected_first / (np.sqrt(corrected_second) + config.epsilon)

        point = point - config.learning_rate * direction
        if not np.isfinite(point).all():
            raise FloatingPointError(
                f"{config.name} produced a non-finite point on {objective.display_name}."
            )

        value = objective.value(point)
        gradient_norm = float(np.linalg.norm(objective.gradient(point)))
        points.append(point.copy())
        values.append(value)
        gradient_norms.append(gradient_norm)

        if gradient_norm <= config.tolerance:
            return _result(
                config,
                points,
                values,
                gradient_norms,
                True,
                step,
                "gradient_tolerance",
            )

    return _result(
        config,
        points,
        values,
        gradient_norms,
        False,
        None,
        "maximum_steps",
    )


def _result(
    config: OptimizerConfig,
    points: list[np.ndarray],
    values: list[float],
    gradient_norms: list[float],
    converged: bool,
    converged_step: int | None,
    stop_reason: str,
) -> OptimizationResult:
    return OptimizationResult(
        optimizer=config.name,
        learning_rate=config.learning_rate,
        points=np.asarray(points, dtype=float),
        values=np.asarray(values, dtype=float),
        gradient_norms=np.asarray(gradient_norms, dtype=float),
        converged=converged,
        converged_step=converged_step,
        stop_reason=stop_reason,
    )
