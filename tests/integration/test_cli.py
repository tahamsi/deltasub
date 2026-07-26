from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from deltasub import cli
from deltasub.utils.checkpointing import load_checkpoint


class CLITests(unittest.TestCase):
    def invoke(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = cli.main(arguments)
            except SystemExit as error:
                code = error.code
        return int(code or 0), stdout.getvalue(), stderr.getvalue()

    def test_all_help_paths(self):
        paths = [
            ["--help"],
            ["doctor", "--help"],
            ["doctor", "memory", "--help"],
            ["smoke", "--help"],
            ["data", "--help"],
            ["data", "download", "--help"],
            ["data", "prepare", "--help"],
            ["data", "prepare", "cub", "--help"],
            ["data", "prepare", "aircraft", "--help"],
            ["data", "prepare", "cars", "--help"],
            ["data", "prepare", "cifar10", "--help"],
            ["data", "prepare", "imagenet100", "--help"],
            ["data", "validate", "--help"],
            ["data", "validate-all", "--help"],
            ["references", "--help"],
            ["references", "inspect", "--help"],
            ["backbone", "inspect", "--help"],
            ["selex", "verify-equivalence", "--help"],
            ["train", "baseline", "--help"],
            ["paper", "--help"],
            ["paper", "build-all", "--help"],
        ]
        for arguments in paths:
            with self.subTest(arguments=arguments):
                code, output, _ = self.invoke(arguments)
                self.assertEqual(code, 0)
                self.assertIn("usage:", output)

    def test_doctor_emits_json(self):
        code, output, _ = self.invoke(["doctor"])
        self.assertEqual(code, 0)
        value = json.loads(output)
        self.assertIn("python", value)
        self.assertIn("torch", value)
        self.assertIn("cuda_available", value)

    def test_selex_equivalence_cuda_flags_are_explicit(self):
        with mock.patch(
            "deltasub.cli.verify_equivalence", return_value={"status": "passed"}
        ) as operation:
            code, output, _ = self.invoke(
                ["selex", "verify-equivalence", "--cuda", "--bf16"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["status"], "passed")
        operation.assert_called_once_with(
            "artifacts/gates/selex_equivalence.json",
            include_cuda=True,
            include_bf16=True,
        )

    def test_memory_doctor_cpu_path_and_config_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            hardware = root / "hardware.yaml"
            config.write_text("{}\n")
            hardware.write_text("{}\n")
            output = root / "profile.yaml"
            with mock.patch("deltasub.cli.torch.cuda.is_available", return_value=False):
                code, text, _ = self.invoke(
                    [
                        "doctor",
                        "memory",
                        "--config",
                        str(config),
                        "--hardware",
                        str(hardware),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(text)["status"], "no_cuda")
            self.assertTrue(output.is_file())
            with self.assertRaises(FileNotFoundError):
                cli.main(["doctor", "memory", "--config", str(root / "missing.yaml")])

    def test_smoke_cli_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for resume in (False, True):
                arguments = [
                    "smoke",
                    "--output",
                    str(output),
                    "--size",
                    "4",
                    "--seed",
                    "7",
                    "--device",
                    "cpu",
                ]
                if resume:
                    arguments.append("--resume")
                code, text, _ = self.invoke(arguments)
                self.assertEqual(code, 0)
                self.assertTrue(json.loads(text)["metrics"]["synthetic_only"])
            self.assertEqual(load_checkpoint(output / "checkpoint_last.pt")["step"], 2)

    def test_download_dispatch_and_invalid_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "CUB.tgz"
            with mock.patch("deltasub.cli.download", return_value=archive) as operation:
                code, text, _ = self.invoke(["data", "download", "cub", "--root", directory])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(text)["archive"], str(archive))
            operation.assert_called_once_with("cub", directory)
            code, _, error = self.invoke(["data", "download", "imagenet100", "--root", directory])
            self.assertEqual(code, 2)
            self.assertIn("invalid choice", error)

    def test_references_inspect_and_data_failures(self):
        code, text, _ = self.invoke(["references", "inspect"])
        self.assertEqual(code, 0)
        self.assertEqual(
            {item["id"] for item in json.loads(text)["references"]},
            {"selex", "generalized_category_discovery"},
        )
        with tempfile.TemporaryDirectory() as directory:
            code, _, error = self.invoke(["data", "prepare", "cub", "--root", directory])
            self.assertEqual(code, 2)
            self.assertIn("source checksum is unavailable", error)
            code, _, error = self.invoke(
                ["data", "validate", "cub", "--root", directory]
            )
            self.assertEqual(code, 2)
            self.assertIn("manifest does not exist", error)

    def test_paper_build_all_writes_three_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, output = root / "runs", root / "paper"
            for seed, all_value in enumerate((60.0, 62.0, 64.0)):
                run = runs / f"method_seed{seed}"
                run.mkdir(parents=True)
                (run / "metrics.json").write_text(
                    json.dumps(
                        {
                            "method": "method",
                            "dataset": "cub",
                            "seed": seed,
                            "all": all_value,
                            "known": all_value + 1,
                            "novel": all_value - 1,
                        }
                    )
                )
            code, text, _ = self.invoke(
                ["paper", "build-all", "--runs", str(runs), "--output", str(output)]
            )
            self.assertEqual(code, 0)
            paths = [Path(item) for item in json.loads(text)["outputs"]]
            self.assertEqual({path.suffix for path in paths}, {".csv", ".md", ".tex"})
            self.assertTrue(all(path.is_file() for path in paths))
            summary = pd.read_csv(output / "generated_tables/results.csv")
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.loc[0, "seeds"], 3)
            self.assertEqual(summary.loc[0, "all_mean"], 62.0)

    def test_paper_build_rejects_synthetic_and_empty_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "synthetic"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({"synthetic_only": True}))
            code, _, error = self.invoke(
                ["paper", "build-all", "--runs", str(root / "runs"), "--output", str(root / "paper")]
            )
            self.assertNotEqual(code, 0)
            self.assertIn("no non-synthetic completed runs", error)


if __name__ == "__main__":
    unittest.main()
