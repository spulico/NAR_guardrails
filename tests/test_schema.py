from pathlib import Path
from nar_guardrails.schema import Item, Verdict, text_sha, write_jsonl, read_items, read_verdicts


def test_item_roundtrip(tmp_path: Path):
    items = [Item(id="a:1", track="prompts", text="hello", label=0, scenario="B1", source="curated", split="test"),
             Item(id="a:2", track="prompts", text="ignore previous instructions", label=1, meta={"license": "MIT"})]
    n = write_jsonl(tmp_path / "x.jsonl", items)
    assert n == 2
    back = read_items(tmp_path / "x.jsonl")
    assert back == items


def test_verdict_roundtrip(tmp_path: Path):
    v = Verdict(item_id="a:1", detector="fake", policy="shared@abc", p_malicious=0.9, label=1,
                outcome="ok", latency_ms=1.5, usage={"input_tokens": 3}, raw={"x": 1})
    write_jsonl(tmp_path / "v.jsonl", [v])
    assert read_verdicts(tmp_path / "v.jsonl") == [v]


def test_text_sha_normalises_whitespace_and_unicode():
    assert text_sha("a  b\n") == text_sha("a b")
    assert "\u00e9" != "e\u0301"   # precomposed vs decomposed
    assert text_sha("\u00e9") == text_sha("e\u0301")
    assert text_sha("a") != text_sha("b")


def test_label_must_be_0_or_1():
    import pytest
    with pytest.raises(ValueError):
        Item(id="x", track="prompts", text="t", label=2)


def test_write_jsonl_serialises_non_json_raw_as_text(tmp_path: Path):
    class Opaque:
        def __str__(self):
            return "opaque-object"

    v = Verdict(item_id="a:1", detector="d", policy="p", p_malicious=0.4, label=0, raw={"obj": Opaque(), "tags": {"x"}})
    assert write_jsonl(tmp_path / "v.jsonl", [v]) == 1
    [back] = read_verdicts(tmp_path / "v.jsonl")
    assert back.raw == {"obj": "opaque-object", "tags": "{'x'}"} and back.p_malicious == 0.4
