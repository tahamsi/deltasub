"""Milestone-M1 tests using tiny, explicitly synthetic filesystem fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from deltasub.data.gcd_splits import load_split_definition
from deltasub.data.manifests import (
    MANIFEST_SCHEMA_VERSION,
    manifest_sha256,
    read_manifest,
    serialize_manifest,
    validate_manifest,
    write_manifest,
)
from deltasub.data.prepare import REFERENCE_COMMIT, prepare_dataset, validate_dataset


ZERO_SHA = "0" * 64


def record(sample_id: str = "fixture:1", **updates) -> dict:
    value = {
        "sample_id": sample_id,
        "dataset": "test-fixture",
        "image_path": "images/one.jpg",
        "original_class_id": 0,
        "original_class_name": "fixture class",
        "known_or_novel": "known",
        "labelled_or_unlabelled": "labelled",
        "train_or_test_split": "train",
        "bounding_box": None,
        "source_archive_checksum": ZERO_SHA,
        "split_source": "test-fixture-split.json",
        "split_revision": REFERENCE_COMMIT,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
    }
    value.update(updates)
    return value


class ManifestSchemaTests(unittest.TestCase):
    def test_ordering_serialization_and_hash_are_deterministic(self):
        first = [record("z", image_path="z.jpg"), record("a", image_path="a.jpg")]
        second = list(reversed(copy.deepcopy(first)))
        self.assertEqual(serialize_manifest(first), serialize_manifest(second))
        self.assertEqual(manifest_sha256(first), manifest_sha256(second))
        self.assertTrue(serialize_manifest(first).startswith(b'{"bounding_box":null'))

    def test_write_and_read_use_canonical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            checksum = write_manifest([record("b"), record("a")], path)
            self.assertEqual([item["sample_id"] for item in read_manifest(path)], ["a", "b"])
            self.assertEqual(path.with_suffix(".jsonl.sha256").read_text().strip(), checksum)

    def test_schema_version_and_required_fields(self):
        broken = record()
        del broken["dataset"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_manifest([broken])
        with self.assertRaisesRegex(ValueError, "unsupported manifest_schema_version"):
            validate_manifest([record(manifest_schema_version=999)])

    def test_duplicate_sample_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            validate_manifest([record(), record()])

    def test_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "manifest image is missing"):
                validate_manifest([record()], dataset_root=directory, check_images=True)

    def test_invalid_class_ids_and_assignments(self):
        for invalid in (-1, "1", True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "invalid original_class_id"):
                validate_manifest([record(original_class_id=invalid)])
        with self.assertRaisesRegex(ValueError, "incorrectly assigned known"):
            validate_manifest([record(original_class_id=1)], known_class_ids=[0], novel_class_ids=[1])
        with self.assertRaisesRegex(ValueError, "novel samples cannot be labelled"):
            validate_manifest([record(known_or_novel="novel")])
        with self.assertRaisesRegex(ValueError, "test samples cannot be labelled"):
            validate_manifest([record(train_or_test_split="test")])

    def test_invalid_statuses(self):
        with self.assertRaisesRegex(ValueError, "known_or_novel"):
            validate_manifest([record(known_or_novel="maybe")])
        with self.assertRaisesRegex(ValueError, "labelled_or_unlabelled"):
            validate_manifest([record(labelled_or_unlabelled="maybe")])

    def test_malformed_bounding_boxes(self):
        for box in ([1, 2, 3], [1, 2, 0, 4], [1, 2, "3", 4]):
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, "malformed_bounding_box|malformed bounding_box"):
                validate_manifest([record(bounding_box=box)])

    def test_malformed_manifest_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("{broken\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed manifest JSON"):
                read_manifest(path)


class SplitAndPreparationTests(unittest.TestCase):
    def load_fixture_split(self, payload: dict) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "split.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_split_definition(
            path,
            upstream_repository="upstream",
            pinned_commit=REFERENCE_COMMIT,
            expected_revision=REFERENCE_COMMIT,
            labelled_proportion=0.5,
        )

    def test_pinned_ssb_dictionary_split(self):
        report = self.load_fixture_split(
            {
                "known_classes": [3, 1],
                "unknown_classes": {
                    "Easy": [8, 7],
                    "Medium": [6, 5],
                    "Hard": [4, 2],
                },
                "closed_set_open_set_pairs": {"ignored": [999]},
            }
        )
        self.assertEqual(report["known_class_ids"], [1, 3])
        self.assertEqual(report["novel_class_ids"], [4, 2, 6, 5, 8, 7])

    def test_pinned_ssb_split_requires_all_difficulty_partitions(self):
        with self.assertRaisesRegex(ValueError, "exactly Easy, Medium, and Hard"):
            self.load_fixture_split(
                {
                    "known_classes": [0],
                    "unknown_classes": {"Easy": [1], "Medium": [2]},
                }
            )

    def test_pinned_ssb_split_rejects_malformed_partition(self):
        with self.assertRaisesRegex(ValueError, "difficulty partitions must be lists"):
            self.load_fixture_split(
                {
                    "known_classes": [0],
                    "unknown_classes": {"Easy": [1], "Medium": "2", "Hard": [3]},
                }
            )

    def test_pinned_ssb_split_rejects_overlap_and_duplicates(self):
        malformed_splits = (
            ({"Easy": [1], "Medium": [2], "Hard": [0]}, "overlap"),
            ({"Easy": [1], "Medium": [2], "Hard": [1]}, "duplicates"),
        )
        for unknown_classes, message in malformed_splits:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load_fixture_split(
                    {"known_classes": [0], "unknown_classes": unknown_classes}
                )

    def test_existing_split_schema_regression(self):
        report = self.load_fixture_split(
            {"train_classes": [2, 0], "unlabeled_classes": [3, 1]}
        )
        self.assertEqual(report["known_class_ids"], [0, 2])
        self.assertEqual(report["novel_class_ids"], [1, 3])

    def test_exact_split_hash_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            payload = b'{"known_class_ids":[2,0],"novel_class_ids":[3,1]}\n'
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            report = load_split_definition(
                path,
                upstream_repository="upstream",
                pinned_commit=REFERENCE_COMMIT,
                expected_revision=REFERENCE_COMMIT,
                labelled_proportion=0.5,
                expected_sha256=digest,
            )
            self.assertEqual(report["source_file_sha256"], digest)
            self.assertEqual(report["known_class_ids"], [0, 2])
            self.assertEqual(report["validation_outcome"], "passed")

    def test_split_hash_and_revision_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text('{"known":[0],"novel":[1]}\n', encoding="utf-8")
            common = dict(
                path=path,
                upstream_repository="upstream",
                pinned_commit=REFERENCE_COMMIT,
                labelled_proportion=0.5,
            )
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                load_split_definition(expected_revision="wrong", **common)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_split_definition(
                    expected_revision=REFERENCE_COMMIT, expected_sha256=ZERO_SHA, **common
                )

    def test_absent_split_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "split file"):
            load_split_definition(
                "/absent/test-fixture-split.json",
                upstream_repository="upstream",
                pinned_commit=REFERENCE_COMMIT,
                expected_revision=REFERENCE_COMMIT,
                labelled_proportion=0.5,
            )

    def test_archive_checksum_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.archive"
            archive.write_bytes(b"test fixture, not dataset data")
            with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
                prepare_dataset(
                    "cub",
                    root=root,
                    archive=archive,
                    archive_sha256=ZERO_SHA,
                )

    def test_cub_fixture_preparation_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "CUB_200_2011"
            (base / "images" / "class").mkdir(parents=True)
            (base / "images" / "class" / "one.jpg").write_bytes(b"test fixture image")
            (base / "images.txt").write_text("1 class/one.jpg\n")
            (base / "image_class_labels.txt").write_text("1 1\n")
            (base / "classes.txt").write_text("1 class\n")
            (base / "train_test_split.txt").write_text("1 1\n")
            (base / "bounding_boxes.txt").write_text("1 0 0 10 10\n")
            archive = root / "CUB.fixture"
            archive.write_bytes(b"test fixture archive")
            split_file = root / "fixture-split.json"
            split_file.write_text('{"known_class_ids":[0],"novel_class_ids":[]}\n')
            result = prepare_dataset(
                "cub",
                root=root,
                archive=archive,
                split_file=split_file,
                labelled_proportion=1.0,
            )
            self.assertEqual(result["samples"], 1)
            self.assertEqual(validate_dataset("cub", root)["status"], "passed")

    def test_legal_guards_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "explicit --source manual"):
                prepare_dataset("cars", root=directory)
            with self.assertRaisesRegex(ValueError, "--imagenet-root is required"):
                prepare_dataset("imagenet100", root=directory)


if __name__ == "__main__":
    unittest.main()
