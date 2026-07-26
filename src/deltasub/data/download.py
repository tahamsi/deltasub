from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


OFFICIAL = {
    "cub": (
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        "97eceeb196236b17998738112f37df78",
    ),
    "aircraft": (
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
        None,
    ),
}


def download(dataset: str, root: str | Path) -> Path:
    if dataset not in OFFICIAL:
        raise ValueError(f"automatic download is not supported for {dataset}")
    url, expected_md5 = OFFICIAL[dataset]
    target = Path(root) / Path(url.split("?")[0]).name
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    if expected_md5:
        observed = hashlib.md5(target.read_bytes()).hexdigest()
        if observed != expected_md5:
            target.unlink()
            raise ValueError(f"checksum mismatch for {dataset}")
    return target
