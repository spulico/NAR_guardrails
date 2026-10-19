from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Iterable


def ensure_repo(raw_dir: Path, name: str, url: str, ref: str) -> Path:
    """Shallow-clone `url` at `ref` into raw_dir/name once; record the commit in .nar_commit."""
    path = Path(raw_dir) / name
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, url, str(path)], check=True)
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    (path / ".nar_commit").write_text(commit + "\n")
    return path


def hf_rows(dataset: str, config: str | None = None, split: str = "train", streaming: bool = False) -> Iterable[dict]:
    from datasets import load_dataset  # gated sets need `huggingface-cli login` and accepted terms

    ds = load_dataset(dataset, config, split=split, streaming=streaming)
    return (dict(r) for r in ds)


def sample(rows: list, n: int | None, seed: int) -> list:
    rows = list(rows)
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]
