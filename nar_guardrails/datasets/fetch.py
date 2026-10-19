from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()


def _clone(target: Path, url: str, ref: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, url, str(target)], check=True)


def _read_marker(path: Path) -> dict:
    try:
        d = json.loads((path / ".nar_ref").read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def ensure_repo(raw_dir: Path, name: str, url: str, ref: str) -> Path:
    """Shallow-clone `url` at `ref` into raw_dir/name; re-sync if the checkout differs.

    Records HEAD in .nar_commit and the requested ref + commit in .nar_ref. A clone whose marker
    matches the request and HEAD is used as-is with no network access; an existing clone is never deleted.
    """
    if not name or "/" in name or "\\" in name or ".." in name or name != Path(name).name:
        raise ValueError(f"invalid repo name {name!r}")
    path = Path(raw_dir) / name
    if path.exists() and not (path / ".git").exists():
        tmp = path.with_name(path.name + ".tmp-clone")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            _clone(tmp, url, ref)
        except subprocess.CalledProcessError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"clone of {url}@{ref} failed; left {path} untouched") from e
        shutil.rmtree(path)
        tmp.rename(path)
    elif not path.exists():
        _clone(path, url, ref)
    else:
        marker = _read_marker(path)
        head = _git(path, "rev-parse", "HEAD")
        if not (marker.get("ref") == ref and marker.get("commit") == head):
            try:
                subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", ref], check=True)
                subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"could not update {path} to ref {ref!r}; it stays at commit {head}") from e
    head = _git(path, "rev-parse", "HEAD")
    (path / ".nar_commit").write_text(head + "\n")
    (path / ".nar_ref").write_text(json.dumps({"ref": ref, "commit": head}))
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
