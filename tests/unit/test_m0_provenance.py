"""Tests for the milestone-M0 documentation and provenance inventory."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "third_party" / "manifest.yaml"
SHA40 = re.compile(r"[0-9a-f]{40}")

REQUIRED_REFERENCE_FIELDS = {
    "id",
    "paper",
    "repository",
    "commit",
    "license",
    "code_status",
    "original_task",
    "original_backbone",
    "integration",
    "difficulty",
}

REQUIRED_BASELINE_NAMES = {
    "ViT / DINOv2 + SelEx",
    "TransFG-GCD-Port",
    "CFViT-GCD-Port",
    "LFViT-GCD-Port",
    "MSViT-GCD-Reimplementation",
    "DART-GCD-Port",
    "SATA-GCD-Port",
    "SpiralFovea-GCD-Reimplementation",
    "ARTA-Cls-Port",
    "SubViT-Reimplementation",
    "DeltaSub",
}


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()
        self.references = self.manifest["references"]

    def test_top_level_schema_is_exact(self) -> None:
        self.assertEqual(
            set(self.manifest),
            {"schema_version", "inspected_at", "policy", "references"},
        )
        self.assertEqual(self.manifest["schema_version"], 1)

    def test_reference_ids_are_unique(self) -> None:
        ids = [reference["id"] for reference in self.references]
        self.assertEqual(len(ids), 13)
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_fields_and_commit_shapes(self) -> None:
        for reference in self.references:
            with self.subTest(reference=reference.get("id")):
                self.assertFalse(REQUIRED_REFERENCE_FIELDS - reference.keys())
                commit = reference["commit"]
                if commit is not None:
                    self.assertRegex(commit, SHA40)
                    self.assertIsNotNone(reference["repository"])

    def test_unavailable_integrations_cannot_look_usable(self) -> None:
        for reference in self.references:
            if reference["integration"] == "unavailable":
                with self.subTest(reference=reference["id"]):
                    self.assertIn(
                        reference["code_status"],
                        {"unavailable", "official_but_unlicensed"},
                    )

    def test_known_provenance_risks_remain_explicit(self) -> None:
        by_id = {reference["id"]: reference for reference in self.references}
        self.assertEqual(by_id["sata"]["integration"], "unavailable")
        self.assertEqual(by_id["lf_vit"]["license"], "NOASSERTION")
        self.assertIn("README", by_id["dart"]["notes"])
        self.assertEqual(
            by_id["msvit"]["integration"], "clean_room_reimplementation"
        )


class DocumentationTests(unittest.TestCase):
    def test_required_m0_documents_exist_and_are_nonempty(self) -> None:
        for relative_path in (
            "AGENTS.md",
            "PLAN.md",
            "BASELINE_STATUS.md",
            "third_party/manifest.yaml",
        ):
            with self.subTest(path=relative_path):
                path = ROOT / relative_path
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_baseline_status_covers_every_required_name(self) -> None:
        text = (ROOT / "BASELINE_STATUS.md").read_text(encoding="utf-8")
        for name in REQUIRED_BASELINE_NAMES:
            with self.subTest(name=name):
                self.assertIn(name, text)

    def test_markdown_hygiene(self) -> None:
        for path in ROOT.glob("*.md"):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"[ \t]+$", text, re.MULTILINE))
                self.assertTrue(text.endswith("\n"))
                self.assertFalse(text.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
