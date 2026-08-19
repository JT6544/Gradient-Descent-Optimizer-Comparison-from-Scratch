"""Deterministic scalar and two-dimensional optimisation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Objective:
    """An objective with an analytical gradient and known global minimiser."""

    key: str
    display_name: str
    initial_point: tuple[float, ...]
    minimizer: tuple[float, ...]
    minimum: float
    value_function: Callable[[Array], float]
    gradient_function: Callable[[Array], Array]
    plot_bounds: tuple[tuple[float, float], ...]

    @property
    def dimension(self) -> int:
        return len(self.initial_point)

    def value(self, point: Array) -> float:
        point = self._validate_point(point)
        value = float(self.value_function(point))
        if not np.isfinite(value):
            raise FloatingPointError(f"{self.display_name} returned a non-finite value.")
        return value

    def gradient(self, point: Array) -> Array:
        point = self._validate_point(point)
        gradient = np.asarray(self.gradient_function(point), dtype=float)
        if gradient.shape != point.shape:
            raise ValueError(
                f"{self.display_name} returned gradient shape {gradient.shape}; "
                f"expected {point.shape}."
            )
        if not np.isfinite(gradient).all():
            raise FloatingPointError(f"{self.display_name} returned a non-finite gradient.")
        return gradient

    def _validate_point(self, point: Array) -> Array:
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.size != self.dimension:
            raise ValueError(
                f"{self.display_name} expects {self.dimension} values; found {point.size}."
            )
        if not np.isfinite(point).all():
            raise ValueError("Objective points must contain only finite values.")
        return point


def _quartic_value(point: Array) -> float:
    return float(point[0] ** 4)


def _quartic_gradient(point: Array) -> Array:
    return np.asarray([4.0 * point[0] ** 3])


def _tilted_quartic_value(point: Array) -> float:
    value = point[0]
    return float((value**4 + value**2 + 10.0 * value) / 50.0)


def _tilted_quartic_gradient(point: Array) -> Array:
    value = point[0]
    return np.asarray([(4.0 * value**3 + 2.0 * value + 10.0) / 50.0])


def _tilted_quartic_minimizer() -> float:
    value = -1.25
    for _ in range(20):
        first = 4.0 * value**3 + 2.0 * value + 10.0
        second = 12.0 * value**2 + 2.0
        value -= first / second
    return float(value)


def _quadratic_value(point: Array) -> float:
    return float(0.5 * (point[0] ** 2 + 100.0 * point[1] ** 2))


def _quadratic_gradient(point: Array) -> Array:
    return np.asarray([point[0], 100.0 * point[1]])


def _rosenbrock_value(point: Array) -> float:
    x_value, y_value = point
    return float((1.0 - x_value) ** 2 + 100.0 * (y_value - x_value**2) ** 2)


def _rosenbrock_gradient(point: Array) -> Array:
    x_value, y_value = point
    return np.asarray(
        [
            2.0 * (x_value - 1.0) - 400.0 * x_value * (y_value - x_value**2),
            200.0 * (y_value - x_value**2),
        ]
    )


def default_objectives() -> tuple[Objective, ...]:
    """Return the four benchmark objectives used by the reference experiment."""

    tilted_minimizer = _tilted_quartic_minimizer()
    tilted_point = np.asarray([tilted_minimizer])
    return (
        Objective(
            key="quartic",
            display_name="Quartic",
            initial_point=(-1.5,),
            minimizer=(0.0,),
            minimum=0.0,
            value_function=_quartic_value,
            gradient_function=_quartic_gradient,
            plot_bounds=((-1.7, 1.0),),
        ),
        Objective(
            key="tilted_quartic",
            display_name="Tilted quartic",
            initial_point=(2.0,),
            minimizer=(tilted_minimizer,),
            minimum=_tilted_quartic_value(tilted_point),
            value_function=_tilted_quartic_value,
            gradient_function=_tilted_quartic_gradient,
            plot_bounds=((-1.8, 2.2),),
        ),
        Objective(
            key="ill_conditioned_quadratic",
            display_name="Ill-conditioned quadratic",
            initial_point=(-4.0, 3.0),
            minimizer=(0.0, 0.0),
            minimum=0.0,
            value_function=_quadratic_value,
            gradient_function=_quadratic_gradient,
            plot_bounds=((-4.5, 4.5), (-3.5, 3.5)),
        ),
        Objective(
            key="rosenbrock",
            display_name="Rosenbrock",
            initial_point=(-1.5, 1.5),
            minimizer=(1.0, 1.0),
            minimum=0.0,
            value_function=_rosenbrock_value,
            gradient_function=_rosenbrock_gradient,
            plot_bounds=((-2.0, 2.0), (-1.0, 3.0)),
        ),
    )
