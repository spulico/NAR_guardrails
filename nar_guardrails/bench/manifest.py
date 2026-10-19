from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

from nar_guardrails.device import pick_device

_PACKAGES = ("torch", "transformers", "datasets", "gliner2", "laya", "anthropic", "scikit-learn")


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
