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
