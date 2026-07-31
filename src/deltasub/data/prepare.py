"""Local-only dataset preparation for milestone M1.

Preparation parses legally obtained data. It does not download ImageNet or Cars,
and it never creates image placeholders.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .gcd_splits import load_split_definition
from .manifests import MANIFEST_SCHEMA_VERSION, read_manifest, validate_manifest, write_manifest


REFERENCE_REPOSITORY = "https://github.com/sgvaze/generalized-category-discovery"
REFERENCE_COMMIT = "831a645c3d09a68ec4633a45741025765bacf7e0"
DATASET_ALIASES = {
    "cub": "cub",
    "aircraft": "aircraft",
    "cars": "cars",
    "cifar10": "cifar10",
    "imagenet100": "imagenet100",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_checksum(root: Path, archive: str | Path | None, expected: str | None) -> str:
    if archive is not None:
        archive_path = Path(archive)
        if not archive_path.is_file():
            raise FileNotFoundError(f"source archive does not exist: {archive_path}")
        observed = sha256_file(archive_path)
        if expected is not None and observed != expected:
            raise ValueError(
                f"archive checksum mismatch: expected {expected}, observed {observed}"
            )
        return observed
    checksum_file = root / "source_archive.sha256"
    if not checksum_file.is_file():
        raise FileNotFoundError(
            "source checksum is unavailable; supply --archive (and preferably "
            "--archive-sha256), or create source_archive.sha256 from the licensed archive"
        )
    checksum = checksum_file.read_text(encoding="utf-8").strip().split()[0]
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError(f"malformed source archive SHA256 in {checksum_file}")
    if expected is not None and checksum != expected:
        raise ValueError(f"archive checksum mismatch: expected {expected}, observed {checksum}")
    return checksum


def _resolve_split(
    dataset: str,
    root: Path,
    split_file: str | Path | None,
    split_revision: str,
    split_sha256: str | None,
    labelled_proportion: float,
) -> dict:
    source = Path(split_file) if split_file else root / "splits" / f"{dataset}.json"
    return load_split_definition(
        source,
        upstream_repository=REFERENCE_REPOSITORY,
        pinned_commit=REFERENCE_COMMIT,
        expected_revision=split_revision,
        labelled_proportion=labelled_proportion,
        expected_sha256=split_sha256,
    )


def _base_record(
    *,
    dataset: str,
    root: Path,
    image: Path,
    class_id: int,
    class_name: str | None,
    train_test: str,
    box: list[float] | None,
    checksum: str,
    split: dict,
    ordinal: str,
) -> dict:
    known = class_id in set(split["known_class_ids"])
    novel = class_id in set(split["novel_class_ids"])
    if not known and not novel:
        raise ValueError(f"class ID {class_id} is absent from the exact split definition")
    status = "known" if known else "novel"
    try:
        image_path = image.relative_to(root).as_posix()
    except ValueError:
        image_path = str(image.resolve())
    return {
        "sample_id": f"{dataset}:{ordinal}",
        "dataset": dataset,
        "image_path": image_path,
        "original_class_id": class_id,
        "original_class_name": class_name,
        "known_or_novel": status,
        "labelled_or_unlabelled": "unlabelled",
        "train_or_test_split": train_test,
        "bounding_box": box,
        "source_archive_checksum": checksum,
        "split_source": split["source_file_path"],
        "split_revision": split["pinned_commit"],
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
    }


def _assign_labels(records: list[dict], proportion: float) -> None:
    """Select a stable per-class prefix of known training samples as labelled."""
    groups: dict[int, list[dict]] = {}
    for record in records:
        if record["known_or_novel"] == "known" and record["train_or_test_split"] == "train":
            groups.setdefault(record["original_class_id"], []).append(record)
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item["sample_id"])
        count = round(len(ordered) * proportion)
        for record in ordered[:count]:
            record["labelled_or_unlabelled"] = "labelled"


def _read_index(path: Path, cast=str) -> dict[int, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required dataset metadata file is absent: {path}")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.strip().split(maxsplit=1)
        result[int(key)] = cast(value)
    return result


def _cub_records(root: Path, checksum: str, split: dict) -> list[dict]:
    base = root / "CUB_200_2011" if (root / "CUB_200_2011").is_dir() else root
    images = _read_index(base / "images.txt")
    classes = _read_index(base / "image_class_labels.txt", int)
    names = _read_index(base / "classes.txt")
    train = _read_index(base / "train_test_split.txt", int)
    boxes = _read_index(
        base / "bounding_boxes.txt",
        lambda value: [float(item) for item in value.split()],
    )
    records = []
    for index in sorted(images):
        class_id = int(classes[index]) - 1
        records.append(
            _base_record(
                dataset="cub",
                root=root,
                image=base / "images" / str(images[index]),
                class_id=class_id,
                class_name=str(names[int(classes[index])]),
                train_test="train" if train[index] else "test",
                box=boxes[index],
                checksum=checksum,
                split=split,
                ordinal=str(index),
            )
        )
    return records


def _aircraft_records(root: Path, checksum: str, split: dict) -> list[dict]:
    base = root / "fgvc-aircraft-2013b" / "data"
    if not base.is_dir():
        base = root / "data" if (root / "data").is_dir() else root
    variants_file = base / "variants.txt"
    if not variants_file.is_file():
        raise FileNotFoundError(f"required dataset metadata file is absent: {variants_file}")
    names = [line.strip() for line in variants_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_ids = {name: index for index, name in enumerate(names)}
    records = []
    for partition, output_split in (("train", "train"), ("val", "train"), ("test", "test")):
        annotation = base / f"images_variant_{partition}.txt"
        if not annotation.is_file():
            raise FileNotFoundError(f"required dataset metadata file is absent: {annotation}")
        for line in annotation.read_text(encoding="utf-8").splitlines():
            image_id, name = line.strip().split(maxsplit=1)
            records.append(
                _base_record(
                    dataset="aircraft",
                    root=root,
                    image=base / "images" / f"{image_id}.jpg",
                    class_id=class_ids[name],
                    class_name=name,
                    train_test=output_split,
                    box=None,
                    checksum=checksum,
                    split=split,
                    ordinal=image_id,
                )
            )
    return records


def _mat_string(value) -> str:
    while isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.flat[0]
        else:
            return "".join(str(item) for item in value.flat)
    return str(value)


def _cars_records(root: Path, checksum: str, split: dict) -> list[dict]:
    devkit = root / "devkit"
    meta_path = devkit / "cars_meta.mat"
    train_ann = devkit / "cars_train_annos.mat"
    test_ann = devkit / "cars_test_annos_withlabels.mat"
    for path in (meta_path, train_ann, test_ann):
        if not path.is_file():
            raise FileNotFoundError(
                f"required manually supplied official Cars metadata is absent: {path}"
            )
    names_raw = loadmat(meta_path, squeeze_me=True)["class_names"]
    names = [_mat_string(item) for item in np.atleast_1d(names_raw)]
    records = []
    for ann_path, directory, partition in (
        (train_ann, "cars_train", "train"),
        (test_ann, "cars_test", "test"),
    ):
        annotations = np.atleast_1d(loadmat(ann_path, squeeze_me=True, struct_as_record=False)["annotations"])
        for index, ann in enumerate(annotations):
            class_id = int(getattr(ann, "class")) - 1
            filename = _mat_string(ann.fname)
            records.append(
                _base_record(
                    dataset="cars",
                    root=root,
                    image=root / directory / filename,
                    class_id=class_id,
                    class_name=names[class_id],
                    train_test=partition,
                    box=[float(ann.bbox_x1), float(ann.bbox_y1), float(ann.bbox_x2 - ann.bbox_x1), float(ann.bbox_y2 - ann.bbox_y1)],
                    checksum=checksum,
                    split=split,
                    ordinal=f"{partition}:{filename}",
                )
            )
    return records


def _unpickle(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="bytes")


def _cifar_records(root: Path, checksum: str, split: dict) -> list[dict]:
    base = root / "cifar-10-batches-py"
    meta = _unpickle(base / "batches.meta")
    names = [item.decode() if isinstance(item, bytes) else str(item) for item in meta[b"label_names"]]
    records = []
    image_root = root / "prepared_images"
    image_root.mkdir(exist_ok=True)
    files = [(f"data_batch_{i}", "train") for i in range(1, 6)] + [("test_batch", "test")]
    for filename, partition in files:
        path = base / filename
        if not path.is_file():
            raise FileNotFoundError(f"required CIFAR-10 batch is absent: {path}")
        batch = _unpickle(path)
        filenames = batch[b"filenames"]
        pixels = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        for index, (label, _image_name) in enumerate(zip(batch[b"labels"], filenames)):
            # PPM keeps this extraction dependency-free and lossless. These bytes
            # are the real CIFAR batch sample, never a generated placeholder.
            output = image_root / f"{filename}-{index:05d}.ppm"
            interleaved = pixels[index].transpose(1, 2, 0).tobytes()
            output.write_bytes(b"P6\n32 32\n255\n" + interleaved)
            records.append(
                _base_record(
                    dataset="cifar10",
                    root=root,
                    image=output,
                    class_id=int(label),
                    class_name=names[int(label)],
                    train_test=partition,
                    box=None,
                    checksum=checksum,
                    split=split,
                    ordinal=f"{filename}:{index:05d}",
                )
            )
    return records


def _imagenet_records(root: Path, imagenet_root: Path, checksum: str, split: dict) -> list[dict]:
    if not imagenet_root.is_dir():
        raise FileNotFoundError(
            f"licensed ImageNet-1K root does not exist: {imagenet_root}; "
            "ImageNet is never downloaded automatically"
        )
    wnid_file = root / "splits" / "imagenet100_wnids.txt"
    if not wnid_file.is_file():
        raise FileNotFoundError(
            f"exact ImageNet-100 wnid file is absent: {wnid_file}; supply the pinned local split"
        )
    wnids = [line.strip() for line in wnid_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(wnids) != 100 or len(set(wnids)) != 100:
        raise ValueError("ImageNet-100 split must contain exactly 100 unique wnids")
    records = []
    for class_id, wnid in enumerate(wnids):
        for partition in ("train", "val"):
            directory = imagenet_root / partition / wnid
            if not directory.is_dir():
                raise FileNotFoundError(f"ImageNet class directory is absent: {directory}")
            for image in sorted(item for item in directory.iterdir() if item.is_file()):
                records.append(
                    _base_record(
                        dataset="imagenet100",
                        root=root,
                        image=image,
                        class_id=class_id,
                        class_name=wnid,
                        train_test="train" if partition == "train" else "test",
                        box=None,
                        checksum=checksum,
                        split=split,
                        ordinal=f"{partition}:{wnid}:{image.name}",
                    )
                )
    return records


def prepare_dataset(
    dataset: str,
    *,
    root: str | Path,
    split_file: str | Path | None = None,
    split_revision: str = REFERENCE_COMMIT,
    split_sha256: str | None = None,
    labelled_proportion: float = 0.5,
    archive: str | Path | None = None,
    archive_sha256: str | None = None,
    source: str | None = None,
    imagenet_root: str | Path | None = None,
) -> dict:
    dataset = DATASET_ALIASES.get(dataset, dataset)
    target_root = Path(root)
    if dataset == "cars" and source is None:
        raise ValueError("Stanford Cars requires explicit --source manual (or an explicitly selected source)")
    if dataset == "cars" and source != "manual":
        raise ValueError("M1 supports Stanford Cars only with --source manual")
    if dataset == "imagenet100" and imagenet_root is None:
        raise ValueError("--imagenet-root is required; ImageNet is never downloaded automatically")
    if not target_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {target_root}")
    checksum = _source_checksum(target_root, archive, archive_sha256)
    split = _resolve_split(
        dataset, target_root, split_file, split_revision, split_sha256, labelled_proportion
    )
    builders = {
        "cub": lambda: _cub_records(target_root, checksum, split),
        "aircraft": lambda: _aircraft_records(target_root, checksum, split),
        "cars": lambda: _cars_records(target_root, checksum, split),
        "cifar10": lambda: _cifar_records(target_root, checksum, split),
        "imagenet100": lambda: _imagenet_records(
            target_root, Path(imagenet_root), checksum, split
        ),
    }
    if dataset not in builders:
        raise ValueError(f"unsupported dataset: {dataset}")
    records = builders[dataset]()
    _assign_labels(records, labelled_proportion)
    validate_manifest(
        records,
        dataset_root=target_root,
        known_class_ids=split["known_class_ids"],
        novel_class_ids=split["novel_class_ids"],
        check_images=True,
    )
    manifest_path = target_root / "manifest.jsonl"
    manifest_hash = write_manifest(records, manifest_path)
    split_report = target_root / "split_validation.json"
    split_report.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "dataset": dataset,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "samples": len(records),
        "split_validation": str(split_report),
    }


def validate_dataset(dataset: str, root: str | Path) -> dict:
    target_root = Path(root)
    manifest = target_root / "manifest.jsonl"
    records = read_manifest(manifest, dataset_root=target_root, check_images=True)
    if any(record["dataset"] != dataset for record in records):
        raise ValueError(f"manifest dataset does not match requested dataset {dataset}")
    expected_path = manifest.with_suffix(manifest.suffix + ".sha256")
    if not expected_path.is_file():
        raise FileNotFoundError(f"manifest checksum file is absent: {expected_path}")
    expected = expected_path.read_text(encoding="utf-8").strip()
    observed = validate_manifest(records, dataset_root=target_root, check_images=True)
    if observed != expected:
        raise ValueError(f"manifest SHA256 mismatch: expected {expected}, observed {observed}")
    return {"dataset": dataset, "samples": len(records), "manifest_sha256": observed, "status": "passed"}


def validate_all(root: str | Path) -> dict:
    base = Path(root)
    results = {}
    for dataset in DATASET_ALIASES:
        dataset_root = base / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"dataset is absent: {dataset_root}")
        results[dataset] = validate_dataset(dataset, dataset_root)
    return {"status": "passed", "datasets": results}
