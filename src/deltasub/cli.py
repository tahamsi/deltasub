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
from .router.config import load_router_config
from .router.fixture import run_router_fixture
from .router.features import load_feature_cache
from .router.training import inspect_router_checkpoint, train_from_cache, validate_router_cache
from .utils.hashing import sha256_file
from .adaptive.config import load_adaptive_config
from .adaptive.training import inspect_adaptive_checkpoint, run_fixture_training
from .diagnostics.subvit.config import load_subvit_config
from .diagnostics.subvit.fixture import run_fixture as run_subvit_fixture
from .diagnostics.subvit.training import inspect_checkpoint as inspect_subvit_checkpoint
from .baselines.config import load_config as load_m8_config
from .baselines.registry import adapter_statuses, build_registry
from .baselines.comparison import compare_fixture
from .baselines.training import run_fixture_training as run_m8_training
from .diagnostic import preflight as m9_preflight, run_diagnostic as run_m9_diagnostic, summarize as summarize_m9
from .experiment.campaign import (preflight as experiment_preflight, run as experiment_run,
                                  status as experiment_status, aggregate as experiment_aggregate)


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
    router = sub.add_parser(
        "router", help="M5 pre-transformer gain router (no token selection or budgets)"
    )
    router_sub = router.add_subparsers(dest="router_command", required=True)
    router_train = router_sub.add_parser("train")
    router_train.add_argument("--config", required=True)
    router_train.add_argument("--resume", action="store_true")
    router_train.add_argument("--validate-only", action="store_true")
    router_validate = router_sub.add_parser("validate")
    router_validate.add_argument("--config", required=True)
    router_validate.add_argument("--checkpoint")
    router_inspect = router_sub.add_parser("inspect")
    router_inspect.add_argument("checkpoint")
    router_fixture = router_sub.add_parser("fixture")
    router_fixture.add_argument("--output", default="artifacts/router/m5_fixture")
    router_fixture.add_argument("--resume", action="store_true")
    adaptive = sub.add_parser(
        "adaptive", help="M6 deterministic adaptive budget execution (no M7 reproduction)"
    )
    adaptive_sub = adaptive.add_subparsers(dest="adaptive_command", required=True)
    adaptive_run = adaptive_sub.add_parser("run", help="generate plans and execute a fixture")
    adaptive_run.add_argument("--config", default="configs/smoke/m6_adaptive.yaml")
    adaptive_run.add_argument("--output", default="artifacts/adaptive/m6_fixture")
    adaptive_train = adaptive_sub.add_parser("train", help="train the M6 adaptive foundation")
    adaptive_train.add_argument("--config", default="configs/smoke/m6_adaptive.yaml")
    adaptive_train.add_argument("--output", default="artifacts/adaptive/m6_fixture")
    adaptive_train.add_argument("--resume", action="store_true")
    adaptive_validate = adaptive_sub.add_parser("validate", help="validate configuration/checkpoints only")
    adaptive_validate.add_argument("--config", default="configs/smoke/m6_adaptive.yaml")
    adaptive_validate.add_argument("--checkpoint")
    adaptive_inspect = adaptive_sub.add_parser("inspect", help="inspect an M6 checkpoint on CPU")
    adaptive_inspect.add_argument("checkpoint")
    adaptive_fixture = adaptive_sub.add_parser("fixture", help="run synthetic diagnostic fixture")
    adaptive_fixture.add_argument("--output", default="artifacts/adaptive/m6_fixture")
    adaptive_fixture.add_argument("--resume", action="store_true")
    subvit = sub.add_parser(
        "subvit", help="M7 clean-room SubViT synthetic diagnostic reference (non-reportable)"
    )
    subvit_sub = subvit.add_subparsers(dest="subvit_command", required=True)
    for name in ("validate", "extract", "degrade", "train-router", "compare"):
        command = subvit_sub.add_parser(name)
        command.add_argument("--config", default="configs/smoke/m7_subvit.yaml")
        command.add_argument("--output", default="artifacts/subvit/m7_fixture")
        command.add_argument("--resume", action="store_true")
    subvit_fixture = subvit_sub.add_parser("fixture")
    subvit_fixture.add_argument("--output", default="artifacts/subvit/m7_fixture")
    subvit_fixture.add_argument("--resume", action="store_true")
    subvit_inspect = subvit_sub.add_parser("inspect")
    subvit_inspect.add_argument("checkpoint")
    baselines = sub.add_parser("baselines", help="M8 common-protocol baseline adapters")
    baselines_sub = baselines.add_subparsers(dest="baselines_command", required=True)
    baselines_list = baselines_sub.add_parser("list", help="list evidence-derived adapter status")
    baselines_list.add_argument("--config", default="configs/smoke/m8_baselines.yaml")
    baselines_validate = baselines_sub.add_parser("validate", help="validate configuration and availability")
    baselines_validate.add_argument("--config", default="configs/smoke/m8_baselines.yaml")
    baselines_fixture = baselines_sub.add_parser("fixture", help="run synthetic non-reportable adapter fixture")
    baselines_fixture.add_argument("--config", default="configs/smoke/m8_baselines.yaml")
    baselines_fixture.add_argument("--output", default="artifacts/baselines/m8_fixture")
    baselines_fixture.add_argument("--resume", action="store_true")
    baselines_run = baselines_sub.add_parser("run", help="run an available adapter; production fails closed")
    baselines_run.add_argument("--config", default="configs/baselines/common_m8.yaml")
    baselines_run.add_argument("--method", choices=sorted({"vit_dinov2_selex","subvit_reimplementation","deltasub","msvit_gcd_reimplementation","dart_gcd_port"}))
    baselines_run.add_argument("--resume", action="store_true")
    baselines_compare = baselines_sub.add_parser("compare", help="deterministic matched-budget comparison")
    baselines_compare.add_argument("--config", default="configs/smoke/m8_baselines.yaml")
    baselines_compare.add_argument("--output")
    baselines_inspect = baselines_sub.add_parser("inspect", help="inspect adapter status or fixture checkpoint")
    baselines_inspect.add_argument("--config", default="configs/smoke/m8_baselines.yaml")
    baselines_inspect.add_argument("--method", choices=sorted({"vit_dinov2_selex","subvit_reimplementation","deltasub","msvit_gcd_reimplementation","dart_gcd_port"}))
    train = sub.add_parser("train")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    baseline = train_sub.add_parser("baseline")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--hardware")
    baseline.add_argument("--checkpoint")
    baseline.add_argument("--seed", type=int)
    baseline.add_argument("--resume", action="store_true")
    baseline.add_argument("--validate-only", "--dry-run", action="store_true", dest="validate_only")
    diagnostic = sub.add_parser("diagnostic", help="M9 strict seed-0 real-asset diagnostic gate")
    diagnostic_sub = diagnostic.add_subparsers(dest="diagnostic_command", required=True)
    diagnostic_preflight = diagnostic_sub.add_parser("preflight")
    diagnostic_preflight.add_argument("--config", required=True)
    diagnostic_run = diagnostic_sub.add_parser("run")
    diagnostic_run.add_argument("--config", required=True)
    diagnostic_run.add_argument("--resume", action="store_true")
    diagnostic_summary = diagnostic_sub.add_parser("summarize")
    diagnostic_summary.add_argument("--output-root", default="artifacts/diagnostic/m9")
    experiment = sub.add_parser("experiment", help="M9 production campaign runner")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    for name in ("preflight", "status", "summarize", "aggregate", "tables"):
        command = experiment_sub.add_parser(name); command.add_argument("--config", required=True)
    experiment_run_parser = experiment_sub.add_parser("run")
    experiment_run_parser.add_argument("--config", required=True)
    experiment_run_parser.add_argument("--dataset", choices=["cub", "aircraft"], required=True)
    experiment_run_parser.add_argument("--method", choices=["baseline", "deltasub"], required=True)
    experiment_run_parser.add_argument("--seed", type=int, required=True)
    experiment_run_parser.add_argument("--resume", action="store_true")
    experiment_run_parser.add_argument("--confirm-full", action="store_true")
    experiment_run_parser.add_argument("--ablation")
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
    elif args.command == "router":
        try:
            if args.router_command in {"train", "validate"}:
                config = load_router_config(args.config)
                cache, validation = validate_router_cache(
                    config["gain_cache_path"], config["expected_cache_id"] or None
                )
                result = {
                    "status": "validated", "cache_id": cache.cache_id,
                    "validation_hash": validation["deterministic_validation_sha256"],
                    "training_started": False,
                }
                checkpoint = getattr(args, "checkpoint", None)
                if checkpoint:
                    result["checkpoint"] = inspect_router_checkpoint(checkpoint)
                if args.router_command == "train" and not args.validate_only:
                    if not config["test_only"]:
                        result = train_from_cache(
                            args.config, load_feature_cache(config["feature_cache_path"]),
                            resume=args.resume,
                        )
                    else:
                        result = run_router_fixture(
                            Path(config["checkpoint_directory"]).parent,
                            resume=args.resume,
                        )
            elif args.router_command == "inspect":
                result = inspect_router_checkpoint(args.checkpoint)
            else:
                result = run_router_fixture(args.output, resume=args.resume)
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as error:
            print(f"router error: {error}", file=sys.stderr)
            return 2
    elif args.command == "adaptive":
        try:
            if args.adaptive_command == "inspect":
                result = inspect_adaptive_checkpoint(args.checkpoint, map_location="cpu")
            elif args.adaptive_command == "validate":
                config = load_adaptive_config(args.config)
                result = {
                    "status": "validated", "mode": config["mode"],
                    "training_started": False, "label": "SYNTHETIC DIAGNOSTIC NON-REPORTABLE"
                    if config["mode"] == "fixture" else "M6 PRODUCTION VALIDATION",
                }
                if args.checkpoint:
                    result["checkpoint"] = inspect_adaptive_checkpoint(args.checkpoint, "cpu")
            elif args.adaptive_command in {"run", "train"}:
                config = load_adaptive_config(args.config)
                if config["mode"] != "fixture":
                    raise ValueError("production adaptive execution requires supplied real checkpoints")
                result = run_fixture_training(
                    args.output, resume=getattr(args, "resume", False))
            else:
                result = run_fixture_training(args.output, resume=args.resume)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            print(f"adaptive error: {error}", file=sys.stderr)
            return 2
    elif args.command == "subvit":
        try:
            if args.subvit_command == "inspect":
                result = inspect_subvit_checkpoint(args.checkpoint)
            elif args.subvit_command == "fixture":
                result = run_subvit_fixture(args.output, resume=args.resume)
            else:
                config = load_subvit_config(args.config)
                if args.subvit_command == "validate":
                    result = {
                        "status": "validated", "training_started": False,
                        "mode": config["mode"],
                        "label": ("SYNTHETIC DIAGNOSTIC NON-REPORTABLE"
                                  if config["mode"] == "fixture" else
                                  "M7 SUBVIT DIAGNOSTIC CONFIGURATION"),
                    }
                elif config["mode"] != "fixture":
                    raise ValueError(
                        "production M7 execution requires an explicitly supplied integration dataset"
                    )
                else:
                    result = run_subvit_fixture(args.output, resume=args.resume)
                    result["requested_stage"] = args.subvit_command
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            print(f"subvit error: {error}", file=sys.stderr)
            return 2
    elif args.command == "baselines":
        try:
            config = load_m8_config(args.config)
            roots = config["source_roots"]
            if args.baselines_command == "list":
                result = {"schema_version": config["schema_version"], "adapters": adapter_statuses(roots)}
            elif args.baselines_command == "validate":
                statuses = adapter_statuses(roots)
                expected = config["expected_adapter_statuses"]
                mismatches = {x["method_id"]: {"expected": expected.get(x["method_id"]), "actual": x["status"]}
                              for x in statuses if expected.get(x["method_id"]) != x["status"]}
                if mismatches: raise ValueError(f"adapter status mismatch: {mismatches}")
                result = {"status": "validated", "mode": config["mode"], "training_started": False,
                          "config_hash": config["config_hash"], "adapters": statuses}
            elif args.baselines_command == "compare":
                if config["mode"] != "fixture": raise ValueError("production comparison requires validated real integrations")
                result = compare_fixture(seed=config["seed"], batch_size=config["batch_size"],
                    token_budget=config["common_protocol"]["token_budget"], execution_mode=config["execution_mode"],source_roots=roots)
                if args.output:
                    target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(result,indent=2,sort_keys=True))
            elif args.baselines_command == "fixture":
                if config["mode"] != "fixture": raise ValueError("fixture command requires fixture mode")
                result = run_m8_training(args.output, resume=args.resume, seed=config["seed"])
            elif args.baselines_command == "run":
                method=args.method or config["enabled_adapters"][0]; adapter=build_registry(roots)[method]
                adapter.require_available(config["mode"])
                if config["mode"] != "fixture": raise ValueError("production data execution is not available in M8 evidence")
                result=run_m8_training(config["output_directory"],method,resume=args.resume,seed=config["seed"])
            else:
                statuses=adapter_statuses(roots); result={"adapters":statuses}
                if args.method: result={"adapter":next(x for x in statuses if x["method_id"]==args.method)}
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            print(f"baselines error: {error}", file=sys.stderr); return 2
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
    elif args.command == "experiment":
        try:
            if args.experiment_command == "preflight": result = experiment_preflight(args.config)
            elif args.experiment_command == "run": result = experiment_run(args.config,args.dataset,args.method,args.seed,resume=args.resume,confirm_full=args.confirm_full,ablation=args.ablation)
            elif args.experiment_command == "status": result = experiment_status(args.config)
            else: result = experiment_aggregate(args.config)
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as error:
            print(f"experiment error: {error}", file=sys.stderr); return 2
    elif args.command == "diagnostic":
        try:
            if args.diagnostic_command == "preflight": result = m9_preflight(args.config)
            elif args.diagnostic_command == "run": result = run_m9_diagnostic(args.config, resume=args.resume)
            else: result = summarize_m9(args.output_root)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            print(f"diagnostic error: {error}", file=sys.stderr); return 2
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
