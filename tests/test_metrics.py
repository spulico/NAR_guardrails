import math
import pytest
from nar_guardrails.schema import Item, Verdict
from nar_guardrails.bench.metrics import usable, basic_metrics, auroc, fpr_at_recall, precision_at_prevalence, bootstrap_ci

Y = [1, 1, 1, 1, 0, 0, 0, 0]
S = [0.9, 0.8, 0.6, 0.2, 0.7, 0.3, 0.1, 0.05]


def test_basic_metrics_hand_computed():
    m = basic_metrics(Y, S, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 3, 1)
    assert m["recall"] == pytest.approx(0.75) and m["fpr"] == pytest.approx(0.25)
    assert m["precision"] == pytest.approx(0.75) and m["f1"] == pytest.approx(0.75)


def test_auroc_hand_computed_and_degenerate():
    assert auroc(Y, S) == pytest.approx(13 / 16)   # 13 of 16 pos/neg pairs ranked correctly
    assert auroc([1, 1], [0.1, 0.9]) is None


def test_fpr_at_recall():
    fpr, thr = fpr_at_recall(Y, S, 0.75)
    assert fpr == pytest.approx(0.25) and thr == pytest.approx(0.6)
    fpr, thr = fpr_at_recall(Y, S, 1.0)
    assert fpr == pytest.approx(0.5) and thr == pytest.approx(0.2)   # negatives >= 0.2: 0.7 and 0.3


def test_precision_at_prevalence():
    assert precision_at_prevalence(0.9, 0.1, 0.5) == pytest.approx(0.9)
    assert precision_at_prevalence(0.9, 0.1, 0.01) == pytest.approx(0.009 / (0.009 + 0.099))


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    stat = lambda y, s: basic_metrics(y, s, 0.5)["recall"]
    lo, hi = bootstrap_ci(Y, S, stat, n_boot=200, seed=1)
    assert lo <= 0.75 <= hi and (lo, hi) == bootstrap_ci(Y, S, stat, n_boot=200, seed=1)


def test_usable_filters_errors_and_falls_back_to_label():
    items = [Item(id=f"i{k}", track="prompts", text="t", label=k % 2) for k in range(4)]
    vs = [Verdict("i0", "d", "p", 0.2, 0), Verdict("i1", "d", "p", None, 1, "ok"),
          Verdict("i2", "d", "p", None, None, "error"), Verdict("i3", "d", "p", 1.0, 1, "refused")]
    y, s, ids = usable(items, vs)
    assert ids == ["i0", "i1", "i3"] and s == [0.2, 1.0, 1.0] and y == [0, 1, 1]


def test_undefined_metrics_are_nan_not_zero():
    m = basic_metrics([0, 0], [0.1, 0.2], 0.5)   # no positives: recall undefined
    assert math.isnan(m["recall"]) and math.isnan(m["precision"]) and math.isnan(m["f1"])
    m = basic_metrics([1, 1], [0.1, 0.2], 0.5)   # positives exist, none flagged
    assert m["recall"] == 0.0 and math.isnan(m["precision"]) and m["f1"] == 0.0
    assert math.isnan(precision_at_prevalence(float("nan"), 0.1, 0.5))


def test_bootstrap_ci_ignores_single_class_resamples():
    y = [1] + [0] * 9   # one positive in ten: many resamples contain no positive
    s = [0.9, 0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35, 0.45]
    stat = lambda yy, ss: basic_metrics(yy, ss, 0.5)["f1"]
    lo, hi = bootstrap_ci(y, s, stat, n_boot=300, seed=0)
    assert not (math.isnan(lo) and math.isnan(hi))
    assert math.isfinite(lo) and math.isfinite(hi)


def test_fpr_at_recall_rejects_out_of_range_target():
    with pytest.raises(ValueError):
        fpr_at_recall(Y, S, 0)
    with pytest.raises(ValueError):
        fpr_at_recall(Y, S, 1.5)


def test_tied_negative_at_threshold_counts_as_false_positive():
    m = basic_metrics([1, 0], [0.6, 0.6], 0.6)
    assert m["fpr"] == pytest.approx(1.0) and m["recall"] == pytest.approx(1.0)
