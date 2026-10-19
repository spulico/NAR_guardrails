from __future__ import annotations

from pathlib import Path

import yaml

from nar_guardrails.schema import Verdict

BILLED_OUTCOMES = ("ok", "refused", "truncated")


def load_prices(path: Path = Path("configs/prices.yaml")) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cost_usd(usage: dict, price: dict) -> float:
    inp = float(usage.get("input_tokens", 0)) * float(price["input"]) / 1e6
    out = float(usage.get("output_tokens", 0)) * float(price["output"]) / 1e6
    cache_read = float(usage.get("cache_read_input_tokens", 0)) * float(price.get("cache_read", price["input"])) / 1e6
    return inp + out + cache_read


def summarize_cost(verdicts: list[Verdict], price: dict | None) -> dict:
    if price is None:
        return {"total_usd": 0.0, "per_1k_usd": 0.0, "batch_per_1k_usd": 0.0,
                "input_tokens": 0, "output_tokens": 0, "n_billed": 0}
    billed = [v for v in verdicts if v.usage and v.outcome in BILLED_OUTCOMES]
    total = sum(cost_usd(v.usage, price) for v in billed)
    per_1k = (total / len(billed) * 1000.0) if billed else 0.0
    return {"total_usd": total, "per_1k_usd": per_1k,
            "batch_per_1k_usd": per_1k * float(price.get("batch_discount", 1.0)),
            "input_tokens": sum(int(v.usage.get("input_tokens", 0)) for v in billed),
            "output_tokens": sum(int(v.usage.get("output_tokens", 0)) for v in billed),
            "n_billed": len(billed)}
