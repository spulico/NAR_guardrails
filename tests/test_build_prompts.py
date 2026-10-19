import json
from pathlib import Path
import pytest
from nar_guardrails.schema import Item, read_items
from nar_guardrails.datasets.splits import assign_dev_test
from nar_guardrails.datasets.build_prompts import build_prompts, expected_ranges


def _items(source, n, label):
    return [Item(id=f"{source}:{k}", track="prompts", text=f"{source} {k} {label}", label=label, source=source) for k in range(n)]


def test_assign_dev_test_is_stratified_and_keeps_curated_in_test():
    items = _items("a", 100, 1) + _items("a", 100, 0) + [Item(id="curated:B1-001", track="prompts", text="c", label=0, source="curated")]
    out = assign_dev_test(items, 0.2, seed=0)
    dev = [i for i in out if i.split == "dev"]
    assert len(dev) == 40 and sum(i.label for i in dev) == 20
    assert next(i for i in out if i.source == "curated").split == "test"
    assert [i.split for i in assign_dev_test(items, 0.2, seed=0)] == [i.split for i in out]


def test_build_skips_inaccessible_source(tmp_path: Path, monkeypatch):
    def ok_loader(rows, cap, seed):
        return _items("ok", 10, 1) + _items("ok", 10, 0)

    def gated_loader(rows, cap, seed):
        raise PermissionError("gated")

    sources = {"ok": {"kind": "hf", "dataset": "x/ok", "config": None, "split": "train", "streaming": False, "cap": None,
                      "published": "2026-01", "license": "MIT", "gated": False, "loader": ok_loader},
               "gated": {"kind": "hf", "dataset": "x/gated", "config": None, "split": "train", "streaming": False, "cap": None,
                         "published": "2026-01", "license": "MIT", "gated": True, "loader": gated_loader}}
    monkeypatch.setattr("nar_guardrails.datasets.build_prompts.hf_rows", lambda *a, **k: iter([{}]))
    monkeypatch.setattr("nar_guardrails.datasets.build_prompts.load_curated_prompts", lambda: [Item(id="curated:B1-001", track="prompts", text="c", label=0, source="curated")])
    out = tmp_path / "prompts.jsonl"
    manifest = build_prompts(tmp_path / "raw", out, seed=0, sources=sources, strict=False)
    items = read_items(out)
    assert {i.source for i in items} == {"ok", "curated"}
    assert manifest["skipped"]["gated"].startswith("PermissionError")
    assert manifest["counts"]["ok"] == 20 and json.loads(out.with_suffix(".build.json").read_text())["counts"]["curated"] == 1


def test_strict_mode_fails_on_count_outside_expected_range(tmp_path: Path, monkeypatch):
    sources = {"agentshield": {"kind": "hf", "dataset": "x", "config": None, "split": "train", "streaming": False, "cap": None,
                               "published": "2026-03", "license": "Apache-2.0", "gated": False,
                               "loader": lambda rows, cap, seed: _items("agentshield", 3, 1)}}
    monkeypatch.setattr("nar_guardrails.datasets.build_prompts.hf_rows", lambda *a, **k: iter([{}]))
    monkeypatch.setattr("nar_guardrails.datasets.build_prompts.load_curated_prompts", lambda: [])
    with pytest.raises(SystemExit):
        build_prompts(tmp_path / "raw", tmp_path / "p.jsonl", sources=sources, strict=True)


def test_expected_ranges_file_lists_every_source():
    r = expected_ranges()
    from nar_guardrails.datasets.public_prompts import GIT_SOURCES, HF_SOURCES
    assert set(r["prompts"]) >= set(GIT_SOURCES) | set(HF_SOURCES) | {"curated"}
