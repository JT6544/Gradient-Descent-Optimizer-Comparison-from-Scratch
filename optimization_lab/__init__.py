"""Educational optimisation algorithms and deterministic benchmark objectives."""

from .objectives import Objective, default_objectives
from .optimizers import OptimizationResult, OptimizerConfig, optimize

__all__ = [
    "Objective",
    "OptimizationResult",
    "OptimizerConfig",
    "default_objectives",
    "optimize",
]
