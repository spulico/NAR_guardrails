"""Model revisions recorded by the HF-loading adapters, with every Hub/model call faked."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from nar_guardrails.detectors.base import hf_revision
from nar_guardrails.detectors.policy import Policy

POL = Policy("prompts", "shared", "q", "m", "b")
SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hub(monkeypatch):
    """Fake local HF cache: model ids in `cached` resolve to .../snapshots/<sha>, anything else is not cached."""
    cached: dict[str, str] = {}
    calls = []

    def snapshot_download(model_id, revision=None, local_files_only=False):
        calls.append((model_id, revision, local_files_only))
        if model_id not in cached:
            raise FileNotFoundError(model_id)
        return f"/cache/models--{model_id.replace('/', '--')}/snapshots/{cached[model_id]}"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    return SimpleNamespace(cached=cached, calls=calls)


def test_hf_revision_prefers_the_local_snapshot_then_the_config(hub):
    hub.cached["org/m"] = SHA
    assert hf_revision("org/m", "v1") == SHA and hub.calls[-1] == ("org/m", "v1", True)
    assert hf_revision("org/other", None, SimpleNamespace(_commit_hash="feed")) == "feed"
    assert hf_revision("org/other", None, SimpleNamespace(commit_hash="beef")) == "beef"
    assert hf_revision("org/other") is None and hf_revision(None) is None


class _Model:
    def __init__(self, config=None):
        self.config = config

    def to(self, device):
        return self

    def eval(self):
        return self


def _module(name, **attrs):
    m = ModuleType(name)
    m.__dict__.update(attrs)
    return m


def test_hf_classifier_records_revision(hub, monkeypatch):
    from nar_guardrails.detectors.hf_classifier import HFClassifier
    config = SimpleNamespace(id2label={0: "SAFE", 1: "INJECTION"}, _commit_hash="from-config")
    fake = _module("transformers",
                   AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
                   AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a, **k: _Model(config)))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    cfg = {"model_id": "org/clf", "positive_label": "INJECTION", "device": "cpu"}
    assert HFClassifier(cfg, POL).revision == "from-config"
    hub.cached["org/clf"] = SHA
    assert HFClassifier(cfg, POL).revision == SHA


def test_qwen3guard_records_revision(hub, monkeypatch):
    from nar_guardrails.detectors import qwen3guard
    config = SimpleNamespace(pad_token_id=None)
    fake = _module("transformers",
                   AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
                   AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
                   AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: _Model(config)))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(qwen3guard, "_ensure_default_rope", lambda: None)
    hub.cached["Qwen/Guard"] = SHA
    d = qwen3guard.Qwen3GuardDetector({"model_id": "Qwen/Guard", "revision": SHA, "device": "cpu"}, POL)
    assert d.revision == SHA and hub.calls[-1] == ("Qwen/Guard", SHA, True)


def test_laya_and_gliner2_record_revision(hub, monkeypatch):
    from nar_guardrails.detectors.gliner2_adapter import GLiNER2Detector
    from nar_guardrails.detectors.laya_adapter import LayaDetector
    monkeypatch.setitem(sys.modules, "laya", _module("laya", load=lambda *a, **k: SimpleNamespace(cfg={})))
    monkeypatch.setitem(sys.modules, "gliner2", _module(
        "gliner2", GLiNER2=SimpleNamespace(from_pretrained=lambda *a, **k: _Model())))
    hub.cached["convai/laya"] = SHA
    assert LayaDetector({"model_id": "convai/laya", "device": "cpu"}, POL).revision == SHA
    monkeypatch.setitem(sys.modules, "laya", _module("laya", load=lambda *a, **k: SimpleNamespace(cfg={}, revision="agent-sha")))
    assert LayaDetector({"model_id": "convai/laya", "device": "cpu"}, POL).revision == "agent-sha"
    assert GLiNER2Detector({"model_id": "fastino/not-cached", "device": "cpu"}, POL).revision is None
    hub.cached["fastino/decide"] = "abc"
    assert GLiNER2Detector({"model_id": "fastino/decide", "mode": "decide", "device": "cpu"}, POL).revision == "abc"
