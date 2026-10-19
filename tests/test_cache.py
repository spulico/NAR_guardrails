import json

import pytest
from pathlib import Path
from nar_guardrails.schema import Item, Verdict
from nar_guardrails.bench.cache import VerdictCache, cache_path


def _v(item_id, p=0.5):
    return Verdict(item_id=item_id, detector="fake", policy="shared@abc", p_malicious=p, label=int(p >= 0.5))


def test_cache_path_layout(tmp_path: Path):
    assert cache_path(tmp_path, "laya", "abc123", "prompts") == tmp_path / "laya" / "abc123" / "prompts.jsonl"


def test_put_get_roundtrip_and_persistence(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    item = Item(id="i1", track="prompts", text="hello", label=0)
    c = VerdictCache(p)
    assert c.get(item) is None
    c.put(item, _v("i1", 0.2))
    assert c.get(item).p_malicious == 0.2 and len(c) == 1
    c2 = VerdictCache(p)              # reload from disk
    assert c2.get(item).p_malicious == 0.2


def test_cache_recomputes_when_text_changes(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    c = VerdictCache(p)
    c.put(Item(id="i1", track="prompts", text="old text", label=0), _v("i1", 0.2))
    assert c.get(Item(id="i1", track="prompts", text="new text", label=0)) is None
    assert c.get(Item(id="i1", track="prompts", text="old  text ", label=0)).p_malicious == 0.2  # whitespace-normalised


def test_put_many_and_last_write_wins(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    c = VerdictCache(p)
    items = [Item(id=f"i{k}", track="prompts", text=f"t{k}", label=0) for k in range(3)]
    c.put_many(items, [_v(i.id, 0.1) for i in items])
    c.put(items[0], _v("i0", 0.9))
    assert VerdictCache(p).get(items[0]).p_malicious == 0.9 and len(VerdictCache(p)) == 3


def _line_for(tmp_path: Path, item: Item, p: float) -> str:
    scratch = tmp_path / f"scratch_{item.id}.jsonl"
    VerdictCache(scratch).put(item, _v(item.id, p))
    return scratch.read_text(encoding="utf-8")


def test_non_json_raw_is_persisted_as_string(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    item = Item(id="i1", track="prompts", text="hello", label=0)
    v = Verdict(item_id="i1", detector="fake", policy="shared@abc", p_malicious=0.3, label=0, raw=object())
    VerdictCache(p).put(item, v)
    reloaded = VerdictCache(p).get(item)
    assert reloaded.p_malicious == 0.3
    assert isinstance(reloaded.raw, str)


def test_failed_put_leaves_memory_unchanged(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    c = VerdictCache(blocker / "c.jsonl")
    item = Item(id="i1", track="prompts", text="hello", label=0)
    with pytest.raises(OSError):
        c.put(item, _v("i1", 0.2))
    assert len(c) == 0


def test_truncated_last_line_is_skipped(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    good = [Item(id=f"i{k}", track="prompts", text=f"t{k}", label=0) for k in range(2)]
    p.write_text(_line_for(tmp_path, good[0], 0.1) + _line_for(tmp_path, good[1], 0.2)
                 + '{"item_id": "i2", "text_sha": "ab', encoding="utf-8")
    c = VerdictCache(p)
    assert len(c) == 2
    assert c.get(good[1]).p_malicious == 0.2


def test_malformed_middle_line_raises(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    a = Item(id="i0", track="prompts", text="t0", label=0)
    b = Item(id="i1", track="prompts", text="t1", label=0)
    p.write_text(_line_for(tmp_path, a, 0.1) + "not json\n" + _line_for(tmp_path, b, 0.2), encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        VerdictCache(p)


def test_append_after_torn_tail_starts_new_line(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    a = Item(id="i0", track="prompts", text="t0", label=0)
    new = Item(id="i3", track="prompts", text="t3", label=0)
    torn = '{"item_id": "i2", "tex'
    p.write_text(_line_for(tmp_path, a, 0.1) + torn, encoding="utf-8")
    c = VerdictCache(p)
    assert len(c) == 1
    c.put(new, _v("i3", 0.7))
    lines = p.read_text(encoding="utf-8").split("\n")
    assert json.loads(lines[0])["item_id"] == "i0"
    assert json.loads(lines[1])["item_id"] == "i3"  # torn fragment dropped, new record on its own line
    reloaded = VerdictCache(p)
    assert len(reloaded) == 2
    assert reloaded.get(new).p_malicious == 0.7


def test_unterminated_but_complete_last_record_is_kept(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    a = Item(id="i0", track="prompts", text="t0", label=0)
    new = Item(id="i1", track="prompts", text="t1", label=0)
    p.write_text(_line_for(tmp_path, a, 0.1).rstrip("\n"), encoding="utf-8")
    VerdictCache(p).put(new, _v("i1", 0.4))
    reloaded = VerdictCache(p)
    assert len(reloaded) == 2
    assert reloaded.get(a).p_malicious == 0.1


def test_put_many_length_mismatch_raises_before_writing(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    items = [Item(id=f"i{k}", track="prompts", text=f"t{k}", label=0) for k in range(2)]
    c = VerdictCache(p)
    with pytest.raises(ValueError):
        c.put_many(items, [_v("i0", 0.1)])
    assert len(c) == 0 and not p.exists()
