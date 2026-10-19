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


def test_load_source_closes_hf_generator(monkeypatch):
    from nar_guardrails.datasets.build_prompts import _load_source
    state = {"closed": False}

    def gen():
        try:
            for k in range(100):
                yield {"k": k}
        finally:
            state["closed"] = True

    monkeypatch.setattr("nar_guardrails.datasets.build_prompts.hf_rows", lambda *a, **k: gen())
    spec = {"kind": "hf", "dataset": "x", "streaming": True, "stream_limit": 5, "cap": None,
            "loader": lambda rows, cap, seed: _items("s", len(list(rows)), 1)}
    assert len(_load_source("s", spec, Path("."), 0)) == 5
    assert state["closed"]


def test_assign_dev_test_is_stable_when_other_groups_are_added():
    ab = _items("a", 50, 1) + _items("b", 50, 0)
    before = {i.id: i.split for i in assign_dev_test(ab, 0.2, seed=3)}
    abc = _items("a", 50, 1) + _items("b", 50, 0) + _items("c", 50, 1)
    after = {i.id: i.split for i in assign_dev_test(abc, 0.2, seed=3)}
    assert all(after[k] == v for k, v in before.items())


def test_build_records_dedupe_accounting_and_label_conflicts(tmp_path: Path, monkeypatch):
    def l1(rows, cap, seed):
        return [Item(id="one:1", track="prompts", text="same text", label=1, source="one"),
                Item(id="one:2", track="prompts", text="other", label=1, source="one")]

    def l2(rows, cap, seed):
        return [Item(id="two:1", track="prompts", text="same text", label=0, source="two"),
                Item(id="two:2", track="prompts", text="third", label=0, source="two")]

    base = {"kind": "hf", "dataset": "x", "config": None, "split": "train", "streaming": False, "cap": None,
            "published": "2026-01", "license": "MIT", "gated": False}
    sources = {"one": {**base, "loader": l1}, "two": {**base, "loader": l2}}
    monkeypatch.setattr("nar_guardrails.datasets.build_prompts.hf_rows", lambda *a, **k: iter([{}]))
    monkeypatch.setattr("nar_guardrails.datasets.build_prompts.load_curated_prompts", lambda: [])
    m = build_prompts(tmp_path / "raw", tmp_path / "p.jsonl", sources=sources, strict=False)
    assert m["loaded"] == {"curated": 0, "one": 2, "two": 2}
    assert m["counts"] == {"curated": 0, "one": 2, "two": 1}
    assert m["dropped_duplicates"] == {"two": 1}
    assert m["label_conflicts"] == [{"kept": "one:1", "dropped": "two:1"}]
    assert m["strict"] is False and m["stream_limits"] == {}
