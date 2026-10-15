from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from nar_guardrails.detectors.base import REGISTRY, Detector, Prediction, register  # noqa: F401
from nar_guardrails.detectors.policy import Policy

_ADAPTER_MODULES = ("fake", "hf_classifier", "laya_adapter", "gliner2_adapter", "qwen3guard", "openjev", "claude_adapter")


def _load_adapters() -> None:
    for m in _ADAPTER_MODULES:
        try:
            importlib.import_module(f"nar_guardrails.detectors.{m}")
        except ModuleNotFoundError as e:
            if e.name != f"nar_guardrails.detectors.{m}":
                raise  # a missing transitive dependency must not be masked
            # adapter module not written yet


def load_models_cfg(path: Path = Path("configs/models.yaml")) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["detectors"]


def build_detector(key: str, policy: Policy, models_cfg: dict) -> Detector:
    _load_adapters()
    if key not in models_cfg:
        raise KeyError(f"unknown detector key {key!r}; known: {sorted(models_cfg)}")
    cfg = dict(models_cfg[key])
    adapter = cfg["adapter"]
    if adapter not in REGISTRY:
        raise KeyError(f"no adapter {adapter!r} registered for detector {key!r}")
    det = REGISTRY[adapter](cfg, policy)
    det.key = key
    return det
