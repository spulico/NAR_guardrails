from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()


def _matches_ref(path: Path, ref: str) -> bool:
    for args in (("rev-parse", "--abbrev-ref", "HEAD"), ("describe", "--tags", "--exact-match")):
        try:
            if _git(path, *args) == ref:
                return True
        except subprocess.CalledProcessError:
            pass
    try:
        return bool(ref) and _git(path, "rev-parse", "HEAD").startswith(ref)
    except subprocess.CalledProcessError:
        return False


def _clone(path: Path, url: str, ref: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, url, str(path)], check=True)


def ensure_repo(raw_dir: Path, name: str, url: str, ref: str) -> Path:
    """Shallow-clone `url` at `ref` into raw_dir/name; re-sync if the checkout differs; record the commit in .nar_commit."""
    path = Path(raw_dir) / name
    if path.exists() and not (path / ".git").exists():
        shutil.rmtree(path)
    if not path.exists():
        _clone(path, url, ref)
    elif not _matches_ref(path, ref):
        try:
            subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", ref], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
        except subprocess.CalledProcessError:
            shutil.rmtree(path)
            _clone(path, url, ref)
    (path / ".nar_commit").write_text(_git(path, "rev-parse", "HEAD") + "\n")
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
