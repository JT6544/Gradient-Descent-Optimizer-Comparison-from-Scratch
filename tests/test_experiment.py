from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from optimization_lab.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def test_experiment_writes_complete_machine_readable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            payload, results = run_experiment(
                output_dir, max_steps=20, tolerance=0.0, create_figures=False
            )
            self.assertEqual(len(payload["results"]), 16)
            self.assertEqual(len(results), 4)

            with (output_dir / "summary.json").open(encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["schema_version"], 1)
            self.assertTrue(saved["experiment"]["adam_bias_correction"])

            with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 16)

            with (output_dir / "convergence_history.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                history_rows = list(csv.DictReader(handle))
            self.assertEqual(len(history_rows), 16 * 21)


if __name__ == "__main__":
    unittest.main()
