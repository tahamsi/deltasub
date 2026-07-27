from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from .data.download import OFFICIAL, download
from .data.prepare import DATASET_ALIASES, prepare_dataset, validate_all, validate_dataset
from .reporting.tables import summarize_runs, write_formats
from .training.smoke_pipeline import run_smoke_pipeline
from .training.baseline import run_baseline_training, validate_baseline
from .training.selex_equivalence import verify_equivalence
from .models.backbones.dinov2 import inspect_official_checkpoint
from .models.subtokens.diagnostic import run_fixture_diagnostic
from .gains.cache import inspect_cache, open_cache
from .gains.collector import collect as collect_gains
from .utils.hashing import sha256_file


def doctor() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, vram_bytes=prop.total_memory)
    return info


def memory_doctor(output: Path) -> dict:
    info = doctor()
    if not torch.cuda.is_available():
        result = {**info, "status": "no_cuda", "recommended_physical_batch_size": 1}
    else:
        prop = torch.cuda.get_device_properties(0)
        limit = int(prop.total_memory * 0.9)
        safe = 1
        for candidate in (1, 2, 4, 8, 16, 32):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                tensor = torch.empty(candidate, 3, 224, 224, device="cuda")
                model = torch.nn.Conv2d(3, 64, 14, 14).cuda()
                model(tensor).sum().backward()
                peak = torch.cuda.max_memory_allocated()
                if peak < limit:
                    safe = candidate
                del tensor, model
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        result = {**info, "status": "measured", "recommended_physical_batch_size": safe}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result), encoding="utf-8")
    return result


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"required file does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deltasub")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor_parser = sub.add_parser("doctor")
    doctor_sub = doctor_parser.add_subparsers(dest="doctor_command")
    memory = doctor_sub.add_parser("memory")
    memory.add_argument("--config")
    memory.add_argument("--hardware")
    memory.add_argument("--output", default="artifacts/hardware_profile.yaml")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", default="artifacts/runs/synthetic_smoke")
    smoke.add_argument("--size", type=int, default=8)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--resume", action="store_true")
    references = sub.add_parser("references")
    references_sub = references.add_subparsers(dest="references_command", required=True)
    inspect = references_sub.add_parser("inspect")
    inspect.add_argument("--manifest", default="third_party/manifest.yaml")
    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    download_parser = data_sub.add_parser("download")
    download_parser.add_argument("dataset", choices=sorted(OFFICIAL))
    download_parser.add_argument("--root", required=True)
    prepare = data_sub.add_parser("prepare")
    prepare_sub = prepare.add_subparsers(dest="dataset", required=True)
    for dataset in DATASET_ALIASES:
        dataset_parser = prepare_sub.add_parser(dataset)
        dataset_parser.add_argument("--root", required=True)
        dataset_parser.add_argument("--split-file")
        dataset_parser.add_argument(
            "--split-revision",
            default="831a645c3d09a68ec4633a45741025765bacf7e0",
        )
        dataset_parser.add_argument("--split-sha256")
        dataset_parser.add_argument("--labelled-proportion", type=float, default=0.5)
        dataset_parser.add_argument("--archive")
        dataset_parser.add_argument("--archive-sha256")
        if dataset == "cars":
            dataset_parser.add_argument("--source", choices=["manual"], required=True)
        if dataset == "imagenet100":
            dataset_parser.add_argument("--imagenet-root", required=True)
    validate = data_sub.add_parser("validate")
    validate.add_argument("dataset", choices=sorted(DATASET_ALIASES))
    validate.add_argument("--root", required=True)
    validate_all_parser = data_sub.add_parser("validate-all")
    validate_all_parser.add_argument("--root", required=True)
    backbone = sub.add_parser("backbone")
    backbone_sub = backbone.add_subparsers(dest="backbone_command", required=True)
    backbone_inspect = backbone_sub.add_parser("inspect")
    backbone_inspect.add_argument("--name", choices=["dinov2_vitb14"], default="dinov2_vitb14")
    backbone_inspect.add_argument("--checkpoint", required=True)
    backbone_inspect.add_argument("--expected-sha256", required=True)
    backbone_inspect.add_argument("--source-root", required=True)
    selex = sub.add_parser("selex")
    selex_sub = selex.add_subparsers(dest="selex_command", required=True)
    verify = selex_sub.add_parser("verify-equivalence")
    verify.add_argument("--output", default="artifacts/gates/selex_equivalence.json")
    verify.add_argument("--cuda", action="store_true", help="also verify CUDA FP32")
    verify.add_argument(
        "--bf16", action="store_true",
        help="also verify CUDA BF16 inputs with the FP32 distance/reduction policy (requires --cuda)",
    )
    subtokens = sub.add_parser("subtokens")
    subtokens_sub = subtokens.add_subparsers(dest="subtokens_command", required=True)
    subtokens_validate = subtokens_sub.add_parser("validate")
    subtokens_validate.add_argument("--config", default="configs/smoke/m3_subtokens.yaml")
    gains = sub.add_parser("gains")
    gains_sub = gains.add_subparsers(dest="gains_command", required=True)
    gains_collect = gains_sub.add_parser("collect")
    gains_collect.add_argument("--config", required=True)
    gains_collect.add_argument("--checkpoint")
    gains_collect.add_argument("--expected-sha256")
    gains_collect.add_argument("--source-root")
    gains_collect.add_argument("--resume", action="store_true")
    gains_collect.add_argument("--validate-only", action="store_true")
    gains_validate = gains_sub.add_parser("validate")
    gains_validate.add_argument("cache")
    gains_inspect = gains_sub.add_parser("inspect")
    gains_inspect.add_argument("cache")
    train = sub.add_parser("train")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    baseline = train_sub.add_parser("baseline")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--hardware")
    baseline.add_argument("--checkpoint")
    baseline.add_argument("--seed", type=int)
    baseline.add_argument("--resume", action="store_true")
    baseline.add_argument("--validate-only", "--dry-run", action="store_true", dest="validate_only")
    paper = sub.add_parser("paper")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    build = paper_sub.add_parser("build-all")
    build.add_argument("--runs", required=True)
    build.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        if args.doctor_command == "memory":
            if args.config:
                _existing_file(args.config)
            if args.hardware:
                _existing_file(args.hardware)
            result = memory_doctor(Path(args.output))
        else:
            result = doctor()
    elif args.command == "smoke":
        result = run_smoke_pipeline(args.output, size=args.size, seed=args.seed, device=args.device, resume=args.resume)
    elif args.command == "references":
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"reference manifest does not exist: {manifest_path}")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        selected = [
            reference
            for reference in manifest["references"]
            if reference["id"] in {"selex", "generalized_category_discovery"}
        ]
        result = {
            "manifest": str(manifest_path),
            "schema_version": manifest["schema_version"],
            "references": selected,
            "local_split_files": [],
            "split_status": "required_pinned_local_files_not_present",
        }
    elif args.command == "data":
        try:
            if args.data_command == "download":
                result = {"archive": str(download(args.dataset, args.root))}
            elif args.data_command == "prepare":
                result = prepare_dataset(
                    args.dataset,
                    root=args.root,
                    split_file=args.split_file,
                    split_revision=args.split_revision,
                    split_sha256=args.split_sha256,
                    labelled_proportion=args.labelled_proportion,
                    archive=args.archive,
                    archive_sha256=args.archive_sha256,
                    source=getattr(args, "source", None),
                    imagenet_root=getattr(args, "imagenet_root", None),
                )
            elif args.data_command == "validate":
                result = validate_dataset(args.dataset, args.root)
            elif args.data_command == "validate-all":
                result = validate_all(args.root)
            else:
                raise ValueError(f"unsupported data command: {args.data_command}")
        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"data error: {error}", file=sys.stderr)
            return 2
    elif args.command == "backbone":
        try:
            _, inspection = inspect_official_checkpoint(
                args.checkpoint, args.expected_sha256, source_root=args.source_root, model_name=args.name
            )
            result = vars(inspection)
            if not inspection.compatible:
                print(json.dumps(result, indent=2, default=str), file=sys.stderr)
                return 2
        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"backbone error: {error}", file=sys.stderr)
            return 2
    elif args.command == "subtokens":
        result = run_fixture_diagnostic(args.config)
    elif args.command == "gains":
        try:
            if args.gains_command == "collect":
                result = collect_gains(
                    args.config, checkpoint=args.checkpoint,
                    expected_sha256=args.expected_sha256, source_root=args.source_root,
                    resume=args.resume, validate_only=args.validate_only,
                )
            elif args.gains_command == "validate":
                result = open_cache(args.cache).validate()
            else:
                result = inspect_cache(args.cache)
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as error:
            print(f"gains error: {error}", file=sys.stderr)
            return 2
    elif args.command == "selex":
        try:
            result = verify_equivalence(
                args.output, include_cuda=args.cuda, include_bf16=args.bf16
            )
        except (ValueError, RuntimeError, OSError) as error:
            print(f"SelEx equivalence error: {error}", file=sys.stderr)
            return 2
    elif args.command == "train":
        try:
            result = (vars(validate_baseline(args.config, args.checkpoint)) if args.validate_only else
                      run_baseline_training(args.config, args.checkpoint, resume=args.resume, seed=args.seed))
        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"training error: {error}", file=sys.stderr)
            return 2
    elif args.command == "paper" and args.paper_command == "build-all":
        records = []
        for path in Path(args.runs).glob("*/metrics.json"):
            value = json.loads(path.read_text())
            if value.get("synthetic_only"):
                continue
            value.setdefault("method", path.parent.name)
            value.setdefault("dataset", "unknown")
            value.setdefault("seed", 0)
            records.append(value)
        if not records:
            print(
                "no non-synthetic completed runs found; synthetic smoke metrics are excluded",
                file=sys.stderr,
            )
            return 2
        summary = summarize_runs(pd.DataFrame(records))
        result = {"outputs": [str(p) for p in write_formats(summary, Path(args.output) / "generated_tables/results")]}
    else:
        raise SystemExit("unsupported command")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
