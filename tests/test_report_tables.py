import pytest
from nar_guardrails.schema import Item, Verdict
from nar_guardrails.report.tables import leaderboard, scenario_table, source_table, headline, default_label_table, split_seed, seed_aggregate

ITEMS = [Item(id=f"i{k}", track="prompts", text="t", label=1 if k < 4 else 0, scenario="D1" if k < 2 else ("J1" if k < 4 else "B1"),
              source="a" if k % 2 else "b") for k in range(8)]
S = [0.9, 0.8, 0.6, 0.2, 0.7, 0.3, 0.1, 0.05]
V = {"fake": [Verdict(f"i{k}", "fake", "p", S[k], int(S[k] >= 0.5), latency_ms=10 + k) for k in range(8)],
     "claude-haiku-5-5": [Verdict(f"i{k}", "claude-haiku-5-5", "p", S[k], int(S[k] >= 0.5), "refused" if k == 0 else "ok",
                                  latency_ms=500, usage={"input_tokens": 400, "output_tokens": 15}) for k in range(8)]}
THR = {"fake": {"f1": 0.6, "recall95": 0.2, "recall99": 0.2}, "claude-haiku-5-5": {"f1": 0.5, "recall95": 0.2, "recall99": 0.2}}
PRICES = {"claude-haiku-5-5": {"input": 0.10, "output": 0.50, "batch_discount": 0.5}}
MODELS = {"fake": {"adapter": "fake"}, "claude-haiku-5-5": {"adapter": "claude", "model_id": "claude-haiku-5-5"}}


def test_leaderboard_rows_and_values():
    df = leaderboard(ITEMS, V, THR, PRICES, MODELS, n_boot=50).set_index("detector")
    f = df.loc["fake"]
    assert f["kind"] == "local" and f["n"] == 8 and f["recall"] == pytest.approx(0.75) and f["fpr"] == pytest.approx(0.25)
    assert f["fpr_at_95_oracle"] == pytest.approx(0.5) and f["fpr_at_r95"] == pytest.approx(0.5) and f["recall_at_r95"] == pytest.approx(1.0) and f["per_1k_usd"] == 0.0 and f["p50_ms"] == pytest.approx(13.5)
    assert f["recall_lo"] <= 0.75 <= f["recall_hi"]
    assert not f["thr_default"]
    c = df.loc["claude-haiku-5-5"]
    assert c["kind"] == "remote" and c["refused"] == 1 and c["per_1k_usd"] > 0


def test_leaderboard_threshold_error_entry_falls_back():
    thr = {"fake": {"policy": "p", "track": "prompts", "error": "single-class dev data"}, "claude-haiku-5-5": THR["claude-haiku-5-5"]}
    df = leaderboard(ITEMS, V, thr, PRICES, MODELS, n_boot=0).set_index("detector")
    assert df.loc["fake", "thr_f1"] == 0.5 and bool(df.loc["fake", "thr_default"])
    assert not df.loc["claude-haiku-5-5", "thr_default"]


def test_scenario_and_source_tables():
    sc = scenario_table(ITEMS, V, THR)
    assert sc.loc["D1", "fake"] == pytest.approx(1.0) and sc.loc["D1", "metric"] == "recall"
    assert sc.loc["B1", "fake"] == pytest.approx(0.25) and sc.loc["B1", "metric"] == "fpr"
    so = source_table(ITEMS, V, THR)
    assert set(so.index) == {"a", "b"} and "fake_recall" in so.columns and so.loc["a", "n"] == 4


def test_headline_and_default_labels():
    h = headline(ITEMS, V, THR, PRICES, MODELS).set_index("detector")
    assert h.loc["fake", "fpr_at_r95"] == pytest.approx(0.5) and h.loc["fake", "recall_at_r95"] == pytest.approx(1.0)
    d = default_label_table(ITEMS, V).set_index("detector")
    assert d.loc["fake", "recall"] == pytest.approx(0.75)


def test_split_seed_and_aggregate():
    assert split_seed("laya-ft-seed2") == ("laya-ft", 2) and split_seed("laya") == ("laya", None)
    import pandas as pd
    df = pd.DataFrame({"detector": ["m-ft-seed1", "m-ft-seed2", "zs"], "recall": [0.9, 0.7, 0.5], "fpr": [0.1, 0.1, 0.2]})
    agg = seed_aggregate(df, ["recall", "fpr"]).set_index("detector")
    assert agg.loc["m-ft", "recall"] == "0.800 ± 0.141" and agg.loc["zs", "recall"] == "0.500"


def test_restrict_ignores_extra_ids_and_unusable_outcomes():
    extra = [Verdict("zzz", "fake", "p", 0.99, 1, latency_ms=1), Verdict("i7", "fake", "p", 0.99, 1, "error", latency_ms=1)]
    base = V["fake"][:7]  # i7 absent
    vs = {"fake": base + extra}
    lb = leaderboard(ITEMS, vs, THR, PRICES, MODELS, n_boot=0).set_index("detector")
    assert lb.loc["fake", "n"] == 7 and lb.loc["fake", "errors"] == 1
    clean = leaderboard(ITEMS, {"fake": base}, THR, PRICES, MODELS, n_boot=0).set_index("detector")
    assert lb.loc["fake", "fpr"] == clean.loc["fake", "fpr"] == pytest.approx(1 / 3)
    d = default_label_table(ITEMS, vs).set_index("detector")
    assert d.loc["fake", "fpr"] == pytest.approx(1 / 3)


def test_mixed_groups():
    items = [Item(id=f"m{k}", track="prompts", text="t", label=k % 2, scenario="M", source="s") for k in range(4)]
    vs = {"fake": [Verdict(f"m{k}", "fake", "p", [0.9, 0.9, 0.1, 0.1][k], 0) for k in range(4)]}
    thr = {"fake": {"f1": 0.5}}
    sc = scenario_table(items, vs, thr)
    assert sc.loc["M", "metric"] == "mixed" and sc.loc["M", "fake"] == pytest.approx(0.5) and sc.loc["M", "fake_fpr"] == pytest.approx(0.5)
    so = source_table(items, vs, thr)
    assert so.loc["s", "fake_recall"] == pytest.approx(0.5) and so.loc["s", "fake_fpr"] == pytest.approx(0.5)


def test_bootstrap_deterministic_and_null_threshold():
    a = leaderboard(ITEMS, V, THR, PRICES, MODELS, n_boot=50, seed=3)
    b = leaderboard(ITEMS, V, THR, PRICES, MODELS, n_boot=50, seed=3)
    cols = ["recall_lo", "recall_hi", "fpr_lo", "fpr_hi"]
    assert a[cols].equals(b[cols])
    thr = {"fake": {"f1": None, "recall95": None}, "claude-haiku-5-5": THR["claude-haiku-5-5"]}
    df = leaderboard(ITEMS, V, thr, PRICES, MODELS, n_boot=0).set_index("detector")
    assert df.loc["fake", "thr_f1"] == 0.5 and bool(df.loc["fake", "thr_default"]) and df["fpr_at_r95"].isna()["fake"]
