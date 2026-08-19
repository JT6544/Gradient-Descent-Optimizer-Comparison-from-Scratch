"""Plotting helpers for the benchmark experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .objectives import Objective
from .optimizers import OptimizationResult


COLOURS = {
    "gradient_descent": "#2563EB",
    "normalized_gradient_descent": "#D97706",
    "rmsprop": "#059669",
    "adam": "#B91C1C",
}

DISPLAY_NAMES = {
    "gradient_descent": "Gradient descent",
    "normalized_gradient_descent": "Normalised gradient descent",
    "rmsprop": "RMSProp",
    "adam": "Adam",
}


def create_plots(
    objectives: tuple[Objective, ...],
    results: dict[str, dict[str, OptimizationResult]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_convergence(objectives, results, output_dir)
    _plot_trajectories(objectives, results, output_dir)
    _plot_final_performance(objectives, results, output_dir)


def _plot_convergence(
    objectives: tuple[Objective, ...],
    results: dict[str, dict[str, OptimizationResult]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for objective, axis in zip(objectives, axes.ravel()):
        for optimizer, result in results[objective.key].items():
            gap = np.maximum(result.values - objective.minimum, 1e-16)
            axis.semilogy(
                np.arange(gap.size),
                gap,
                color=COLOURS[optimizer],
                label=DISPLAY_NAMES[optimizer],
                linewidth=1.8,
            )
        axis.set(
            title=objective.display_name,
            xlabel="Update step",
            ylabel="Objective gap",
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Optimizer convergence across four benchmark objectives")
    figure.tight_layout()
    figure.savefig(output_dir / "convergence_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_trajectories(
    objectives: tuple[Objective, ...],
    results: dict[str, dict[str, OptimizationResult]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    two_dimensional = tuple(objective for objective in objectives if objective.dimension == 2)
    figure, axes = plt.subplots(1, len(two_dimensional), figsize=(12, 5.2))
    axes_array = np.atleast_1d(axes)

    for objective, axis in zip(two_dimensional, axes_array):
        x_bounds, y_bounds = objective.plot_bounds
        x_values = np.linspace(*x_bounds, 260)
        y_values = np.linspace(*y_bounds, 260)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        z_grid = np.empty_like(x_grid)
        for row in range(x_grid.shape[0]):
            for column in range(x_grid.shape[1]):
                z_grid[row, column] = objective.value(
                    np.asarray([x_grid[row, column], y_grid[row, column]])
                )
        positive = np.maximum(z_grid - objective.minimum, 1e-8)
        levels = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 22)
        axis.contour(x_grid, y_grid, positive, levels=levels, cmap="Greys", alpha=0.55)

        for optimizer, result in results[objective.key].items():
            stride = max(1, result.points.shape[0] // 400)
            path = result.points[::stride]
            if not np.array_equal(path[-1], result.points[-1]):
                path = np.vstack((path, result.points[-1]))
            axis.plot(
                path[:, 0],
                path[:, 1],
                color=COLOURS[optimizer],
                label=DISPLAY_NAMES[optimizer],
                linewidth=1.4,
                alpha=0.9,
            )
        axis.scatter(*objective.initial_point, marker="o", color="black", s=32, label="Start")
        axis.scatter(*objective.minimizer, marker="*", color="#7C3AED", s=95, label="Optimum")
        axis.set(
            title=objective.display_name,
            xlabel="$w_1$",
            ylabel="$w_2$",
            xlim=x_bounds,
            ylim=y_bounds,
        )
        axis.grid(alpha=0.15)
    axes_array[0].legend(fontsize=7, loc="best")
    figure.suptitle("Two-dimensional optimisation trajectories")
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_final_performance(
    objectives: tuple[Objective, ...],
    results: dict[str, dict[str, OptimizationResult]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    optimizers = tuple(COLOURS)
    gaps = np.asarray(
        [
            [
                max(results[objective.key][optimizer].values[-1] - objective.minimum, 1e-16)
                for optimizer in optimizers
            ]
            for objective in objectives
        ]
    )
    log_gaps = np.log10(gaps)
    figure, axis = plt.subplots(figsize=(9.5, 5.3))
    image = axis.imshow(log_gaps, cmap="viridis_r", aspect="auto")
    for row in range(gaps.shape[0]):
        for column in range(gaps.shape[1]):
            axis.text(
                column,
                row,
                f"{gaps[row, column]:.1e}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if log_gaps[row, column] > np.median(log_gaps) else "black",
            )
    heatmap_labels = (
        "Gradient\ndescent",
        "Normalised gradient\ndescent",
        "RMSProp",
        "Adam",
    )
    axis.set_xticks(np.arange(len(optimizers)), labels=heatmap_labels)
    axis.set_yticks(np.arange(len(objectives)), labels=[item.display_name for item in objectives])
    axis.set(
        title="Final objective gap after optimisation (lower is better)",
        xlabel="Optimizer",
        ylabel="Objective",
    )
    colourbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colourbar.set_label("log10(final objective gap)")
    figure.tight_layout()
    figure.savefig(output_dir / "final_performance_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
