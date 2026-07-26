from pathlib import Path

from .manifests import read_manifest


def validate_paths(manifest_path: str | Path, allow_missing: bool = False) -> dict:
    records = read_manifest(manifest_path)
    missing = [r["image_path"] for r in records if not Path(r["image_path"]).is_file()]
    if missing and not allow_missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing")
    return {"samples": len(records), "missing": len(missing)}
