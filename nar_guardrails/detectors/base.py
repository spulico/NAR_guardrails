from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from nar_guardrails.detectors.policy import Policy
from nar_guardrails.schema import Item, Verdict


@dataclass
class Prediction:
    p_malicious: float | None
    label: int | None
    outcome: str = "ok"
    usage: dict | None = None
    raw: Any = None


def _sanitise(p: Prediction) -> Prediction:
    prob = p.p_malicious
    if prob is None:
        return Prediction(None, p.label if p.label in (0, 1) else None,
                          "error" if p.outcome == "ok" else p.outcome, p.usage, p.raw)
    try:
        prob = float(prob)
    except (TypeError, ValueError):
        return Prediction(None, None, "error", p.usage, {"bad_probability": repr(p.p_malicious), "raw": p.raw})
    if math.isnan(prob) or math.isinf(prob):
        return Prediction(None, None, "error", p.usage, {"bad_probability": repr(p.p_malicious), "raw": p.raw})
    prob = min(1.0, max(0.0, prob))
    label = p.label if p.label in (0, 1) else int(prob >= 0.5)
    return Prediction(prob, label, p.outcome, p.usage, p.raw)


class Detector(ABC):
    key: str = "detector"

    def __init__(self, cfg: dict, policy: Policy):
        self.cfg = cfg
        self.policy = policy

    def warmup(self) -> None:
        self.predict_texts(["warm up"])

    @abstractmethod
    def predict_texts(self, texts: list[str]) -> list[Prediction]:
        ...

    def predict(self, items: list[Item], batch_size: int = 1) -> list[Verdict]:
        out: list[Verdict] = []
        for start in range(0, len(items), max(1, batch_size)):
            chunk = items[start:start + max(1, batch_size)]
            t0 = perf_counter()
            try:
                preds = self.predict_texts([i.text for i in chunk])
                if len(preds) != len(chunk):
                    raise RuntimeError(f"{self.key} returned {len(preds)} predictions for {len(chunk)} texts")
            except Exception as e:  # one bad batch must not kill the run
                preds = [Prediction(None, None, "error", None, f"{type(e).__name__}: {e}") for _ in chunk]
            per_item_ms = (perf_counter() - t0) * 1000.0 / len(chunk)
            for item, p in zip(chunk, preds):
                p = _sanitise(p)
                out.append(Verdict(item_id=item.id, detector=self.key, policy=self.policy.ref,
                                   p_malicious=p.p_malicious, label=p.label, outcome=p.outcome,
                                   latency_ms=per_item_ms, usage=p.usage, raw=p.raw))
        return out


REGISTRY: dict[str, type[Detector]] = {}


def register(adapter: str):
    def deco(cls: type[Detector]):
        REGISTRY[adapter] = cls
        return cls
    return deco
