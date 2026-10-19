from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from nar_guardrails.device import pick_device

_PACKAGES = ("torch", "transformers", "datasets", "gliner2", "laya", "anthropic", "scikit-learn")


def file_sha256(path: Path) -> str | None:
    """sha256 of a file's bytes, or None if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return commit, dirty
    except Exception:
        return None, False


def collect_environment() -> dict:
    versions = {}
    for p in _PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    commit, dirty = git_state()
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "device": pick_device(), "packages": versions, "git_commit": commit, "git_dirty": dirty,
            "timestamp": datetime.now(timezone.utc).isoformat()}
