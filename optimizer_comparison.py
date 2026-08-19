"""Compare four first-order optimizers on deterministic benchmark objectives."""

from __future__ import annotations

import argparse
from pathlib import Path

from optimization_lab.experiment import run_experiment


REPOSITORY_ROOT = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs",
        help="Directory for metrics, histories, and figures.",
    )
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    rows = payload["results"]
    assert isinstance(rows, list)
    headers = ("Objective", "Optimizer", "Steps", "Final gap", "Gradient norm", "Status")
    printable = []
    for row in rows:
        assert isinstance(row, dict)
        printable.append(
            (
                str(row["objective_name"]),
                str(row["optimizer"]),
                str(row["steps_run"]),
                f"{float(row['objective_gap']):.3e}",
                f"{float(row['final_gradient_norm']):.3e}",
                "converged" if row["converged"] else "max steps",
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in printable))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in printable:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> int:
    args = _build_parser().parse_args()
    payload, _results = run_experiment(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
        create_figures=not args.no_plots,
    )
    _print_summary(payload)
    print(f"\nResults written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
