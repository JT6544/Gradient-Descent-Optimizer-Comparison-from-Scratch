"""Run, record, and plot the optimizer benchmark suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .objectives import Objective, default_objectives
from .optimizers import OPTIMIZER_NAMES, OptimizationResult, OptimizerConfig, optimize
from .plotting import create_plots


DEFAULT_LEARNING_RATES: dict[str, dict[str, float]] = {
    "quartic": {
        "gradient_descent": 0.01,
        "normalized_gradient_descent": 0.01,
        "rmsprop": 0.01,
        "adam": 0.1,
    },
    "tilted_quartic": {
        "gradient_descent": 1.0,
        "normalized_gradient_descent": 0.01,
        "rmsprop": 0.02,
        "adam": 0.05,
    },
    "ill_conditioned_quadratic": {
        "gradient_descent": 0.01,
        "normalized_gradient_descent": 0.05,
        "rmsprop": 0.05,
        "adam": 0.1,
    },
    "rosenbrock": {
        "gradient_descent": 0.001,
        "normalized_gradient_descent": 0.001,
        "rmsprop": 0.001,
        "adam": 0.03,
    },
}


def run_experiment(
    output_dir: str | Path,
    max_steps: int = 5_000,
    tolerance: float = 1e-6,
    create_figures: bool = True,
) -> tuple[dict[str, object], dict[str, dict[str, OptimizationResult]]]:
    if max_steps <= 0:
        raise ValueError("Maximum steps must be positive.")
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")

    objectives = default_objectives()
    results: dict[str, dict[str, OptimizationResult]] = {}
    summary_rows: list[dict[str, object]] = []

    for objective in objectives:
        results[objective.key] = {}
        for optimizer_name in OPTIMIZER_NAMES:
            config = OptimizerConfig(
                name=optimizer_name,
                learning_rate=DEFAULT_LEARNING_RATES[objective.key][optimizer_name],
                max_steps=max_steps,
                tolerance=tolerance,
            )
            result = optimize(objective, config)
            results[objective.key][optimizer_name] = result
            final_point = result.points[-1]
            objective_gap = max(float(result.values[-1] - objective.minimum), 0.0)
            summary_rows.append(
                {
                    "objective": objective.key,
                    "objective_name": objective.display_name,
                    "dimension": objective.dimension,
                    "optimizer": optimizer_name,
                    "learning_rate": config.learning_rate,
                    "steps_run": result.steps_run,
                    "converged": result.converged,
                    "converged_step": result.converged_step,
                    "stop_reason": result.stop_reason,
                    "initial_value": float(result.values[0]),
                    "final_value": float(result.values[-1]),
                    "objective_gap": objective_gap,
                    "final_gradient_norm": float(result.gradient_norms[-1]),
                    "distance_to_minimizer": float(
                        np.linalg.norm(final_point - np.asarray(objective.minimizer))
                    ),
                    "final_point": final_point.tolist(),
                }
            )

    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": {
            "max_steps": max_steps,
            "gradient_tolerance": tolerance,
            "deterministic": True,
            "adam_bias_correction": True,
            "rmsprop_rho": 0.9,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "epsilon": 1e-8,
        },
        "objectives": [
            {
                "key": objective.key,
                "name": objective.display_name,
                "dimension": objective.dimension,
                "initial_point": list(objective.initial_point),
                "known_minimizer": list(objective.minimizer),
                "known_minimum": objective.minimum,
            }
            for objective in objectives
        ],
        "learning_rates": DEFAULT_LEARNING_RATES,
        "results": summary_rows,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "summary.json", payload)
    _write_summary_csv(output_path / "summary.csv", summary_rows)
    _write_history_csv(output_path / "convergence_history.csv", objectives, results)
    if create_figures:
        create_plots(objectives, results, output_path)
    return payload, results


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [key for key in rows[0] if key != "final_point"] + ["final_point"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["final_point"] = json.dumps(serializable["final_point"])
            writer.writerow(serializable)


def _write_history_csv(
    path: Path,
    objectives: tuple[Objective, ...],
    results: dict[str, dict[str, OptimizationResult]],
) -> None:
    fieldnames = ("objective", "optimizer", "step", "value", "objective_gap", "gradient_norm")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for objective in objectives:
            for optimizer_name, result in results[objective.key].items():
                for step, (value, gradient_norm) in enumerate(
                    zip(result.values, result.gradient_norms)
                ):
                    writer.writerow(
                        {
                            "objective": objective.key,
                            "optimizer": optimizer_name,
                            "step": step,
                            "value": float(value),
                            "objective_gap": max(float(value - objective.minimum), 0.0),
                            "gradient_norm": float(gradient_norm),
                        }
                    )
