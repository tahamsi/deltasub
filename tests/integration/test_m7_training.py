from __future__ import annotations

import tempfile
import unittest
import torch

from deltasub.diagnostics.subvit.fixture import run_fixture
from deltasub.diagnostics.subvit.training import inspect_checkpoint


class M7TrainingIntegration(unittest.TestCase):
    def test_checkpoint_resume_and_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_fixture(directory)
            resumed = run_fixture(directory, resume=True)
            self.assertEqual(first["training"]["router_hash"], resumed["training"]["router_hash"])
            self.assertEqual(first["deterministic_hash"], resumed["deterministic_hash"])
            self.assertTrue(resumed["training"]["frozen_teacher_equal"])
            self.assertEqual(inspect_checkpoint(resumed["training"]["checkpoint"])["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
