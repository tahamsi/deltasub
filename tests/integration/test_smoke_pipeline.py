from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from deltasub.reporting.schema import REQUIRED_RUN_FILES
from deltasub.training.smoke_pipeline import run_smoke_pipeline


class SmokePipelineTests(unittest.TestCase):
    def test_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            first = run_smoke_pipeline(run, size=4, seed=0, device="cpu")
            second = run_smoke_pipeline(run, size=4, seed=0, device="cpu", resume=True)
            self.assertEqual(first["metrics"]["synthetic_only"], True)
            self.assertEqual(second["metrics"]["synthetic_only"], True)
            self.assertTrue(all((run / name).exists() for name in REQUIRED_RUN_FILES))
            self.assertEqual(json.loads((run / "metrics.json").read_text())["schema_version"], 1)
            self.assertEqual(len(pd.read_parquet(run / "selection_statistics.parquet")), 8)


if __name__ == "__main__":
    unittest.main()
