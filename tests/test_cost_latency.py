import pytest
from nar_guardrails.schema import Verdict
from nar_guardrails.bench.cost import cost_usd, summarize_cost, load_prices
from nar_guardrails.bench.latency import latency_summary

PRICE = {"input": 0.10, "output": 0.50, "batch_discount": 0.5}


def test_cost_usd_per_million():
    assert cost_usd({"input_tokens": 1_000_000, "output_tokens": 0}, PRICE) == pytest.approx(0.10)
    assert cost_usd({"input_tokens": 500, "output_tokens": 15}, PRICE) == pytest.approx(0.00005 + 0.0000075)


def test_summarize_cost_and_local_zero():
    vs = [Verdict("a", "c", "p", 0.1, 0, usage={"input_tokens": 500, "output_tokens": 15}),
          Verdict("b", "c", "p", 0.9, 1, "refused", usage={"input_tokens": 500, "output_tokens": 0}),
          Verdict("c", "c", "p", None, None, "error", usage=None)]
    s = summarize_cost(vs, PRICE)
    assert s["n_billed"] == 2 and s["input_tokens"] == 1000 and s["output_tokens"] == 15
    assert s["per_1k_usd"] == pytest.approx(s["total_usd"] / 2 * 1000)
    assert s["batch_per_1k_usd"] == pytest.approx(s["per_1k_usd"] * 0.5)
    local = summarize_cost([Verdict("a", "l", "p", 0.1, 0)], None)
    assert local["total_usd"] == 0.0 and local["per_1k_usd"] == 0.0 and local["n_billed"] == 0


def test_latency_summary_ignores_errors():
    vs = [Verdict("a", "d", "p", 0.1, 0, latency_ms=10), Verdict("b", "d", "p", 0.1, 0, latency_ms=30),
          Verdict("c", "d", "p", None, None, "error", latency_ms=9999), Verdict("d", "d", "p", 1.0, 1, "refused", latency_ms=20)]
    s = latency_summary(vs)
    assert s["n"] == 3 and s["p50_ms"] == pytest.approx(20) and s["mean_ms"] == pytest.approx(20)
    assert 20 <= s["p95_ms"] <= 30


def test_real_prices_file_loads():
    p = load_prices()
    assert p["claude-haiku-5-5"]["input"] == 0.10 and p["claude-sonnet-5-5"]["output"] == 10.0
