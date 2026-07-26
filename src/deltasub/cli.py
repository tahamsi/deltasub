from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import pandas as pd
import torch
import yaml

from .data.download import download
from .reporting.tables import summarize_runs, write_formats
from .training.smoke_pipeline import run_smoke_pipeline


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
    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    download_parser = data_sub.add_parser("download")
    download_parser.add_argument("dataset")
    download_parser.add_argument("--root", required=True)
    paper = sub.add_parser("paper")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    build = paper_sub.add_parser("build-all")
    build.add_argument("--runs", required=True)
    build.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = memory_doctor(Path(args.output)) if args.doctor_command == "memory" else doctor()
    elif args.command == "smoke":
        result = run_smoke_pipeline(args.output, size=args.size, seed=args.seed, device=args.device, resume=args.resume)
    elif args.command == "data":
        result = {"archive": str(download(args.dataset, args.root))}
    elif args.command == "paper" and args.paper_command == "build-all":
        records = []
        for path in Path(args.runs).glob("*/metrics.json"):
            value = json.loads(path.read_text())
            if value.get("synthetic_only"):
                continue
            value.update(method=path.parent.name, dataset="unknown", seed=0)
            records.append(value)
        if not records:
            raise SystemExit("no non-synthetic completed runs found; synthetic smoke metrics are excluded")
        summary = summarize_runs(pd.DataFrame(records))
        result = {"outputs": [str(p) for p in write_formats(summary, Path(args.output) / "generated_tables/results")]}
    else:
        raise SystemExit("unsupported command")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
