import json
import math

import pytest
from nar_guardrails.schema import Item, Verdict
from nar_guardrails.bench.thresholds import choose_thresholds, thresholds_for_run, write_thresholds, read_thresholds

Y = [1, 1, 1, 1, 0, 0, 0, 0]
S = [0.9, 0.8, 0.6, 0.2, 0.7, 0.3, 0.1, 0.05]


def test_choose_thresholds_f1_and_recall_targets():
    t = choose_thresholds(Y, S)
    assert t["f1"] == pytest.approx(0.2)        # 4 TP, 2 FP -> F1 0.8; at 0.6 it is 0.75
    assert t["recall95"] == pytest.approx(0.2)  # 4 of 4 positives needed (ceil(0.95*4)=4)
    assert t["recall99"] == pytest.approx(0.2)
    assert t["dev"]["f1"]["f1"] == pytest.approx(0.8)


def test_f1_tie_prefers_higher_threshold():
    t = choose_thresholds([1, 0], [0.9, 0.1])
    assert t["f1"] == pytest.approx(0.9)


def test_thresholds_for_run_and_roundtrip(tmp_path):
    items = [Item(id=f"i{k}", track="prompts", text="t", label=Y[k], split="dev") for k in range(8)]
    vs = {"fake": [Verdict(f"i{k}", "fake", "shared@abc", S[k], int(S[k] >= 0.5)) for k in range(8)]}
    d = thresholds_for_run(items, vs, "shared@abc", "prompts")
    assert d["fake"]["policy"] == "shared@abc" and d["fake"]["f1"] == pytest.approx(0.2)
    p = tmp_path / "thresholds.json"
    write_thresholds(p, d)
    assert read_thresholds(p)["fake"]["recall95"] == pytest.approx(0.2)


def test_f1_scan_skips_nan_candidates():
    t = choose_thresholds([1, 1, 0], [0.9, 0.8, 0.1])
    assert not math.isnan(t["f1"]) and not math.isnan(t["dev"]["f1"]["f1"])
    assert t["f1"] == pytest.approx(0.8)
