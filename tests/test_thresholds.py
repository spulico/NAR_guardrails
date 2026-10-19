import pytest
from nar_guardrails.schema import Item, Verdict
from nar_guardrails.bench import thresholds as mod
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


def test_f1_tie_is_exact_not_float():
    # F1 at 0.9 and at 0.5 are both exactly 1/3; float rounding would make 0.5 win.
    y = [1, 1] + [0] * 9
    s = [0.9, 0.5] + [0.9] * 3 + [0.5] * 5 + [0.01]
    assert choose_thresholds(y, s)["f1"] == 0.9


def test_choose_thresholds_rejects_single_class():
    with pytest.raises(ValueError, match="pos=0, neg=3"):
        choose_thresholds([0, 0, 0], [0.9, 0.5, 0.1])


def test_scan_skips_candidate_with_undefined_f1(monkeypatch):
    real = mod.basic_metrics

    def fake(y, s, thr):
        m = real(y, s, thr)
        if thr == 0.9:  # zero denominator: F1 undefined, must be skipped
            m.update(tp=0, fp=0, fn=0)
        return m

    monkeypatch.setattr(mod, "basic_metrics", fake)
    t = choose_thresholds([1, 0], [0.9, 0.1])
    assert t["f1"] == pytest.approx(0.1)


def test_thresholds_for_run_and_roundtrip(tmp_path):
    items = [Item(id=f"i{k}", track="prompts", text="t", label=Y[k], split="dev") for k in range(8)]
    vs = {"fake": [Verdict(f"i{k}", "fake", "shared@abc", S[k], int(S[k] >= 0.5)) for k in range(8)]}
    d = thresholds_for_run(items, vs, "shared@abc", "prompts")
    assert d["fake"]["policy"] == "shared@abc" and d["fake"]["f1"] == pytest.approx(0.2)
    p = tmp_path / "thresholds.json"
    write_thresholds(p, d)
    assert read_thresholds(p) == d


def test_thresholds_for_run_single_class_detector_gets_error_others_ok():
    items = [Item(id=f"i{k}", track="prompts", text="t", label=Y[k], split="dev") for k in range(8)]
    vs = {
        "good": [Verdict(f"i{k}", "good", "shared@abc", S[k], None) for k in range(8)],
        "single": [Verdict(f"i{k}", "single", "shared@abc", S[k], None) for k in range(4, 8)],
    }
    d = thresholds_for_run(items, vs, "shared@abc", "prompts")
    assert "error" in d["single"] and "f1" not in d["single"]
    assert "pos=0, neg=4" in d["single"]["error"]
    assert d["good"]["f1"] == pytest.approx(0.2)


def test_write_thresholds_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        write_thresholds(tmp_path / "x.json", {"a": {"f1": float("nan")}})
