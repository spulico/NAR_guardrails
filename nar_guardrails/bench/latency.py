from __future__ import annotations

import numpy as np

from nar_guardrails.schema import Verdict

USABLE = ("ok", "refused", "truncated")


def latency_summary(verdicts: list[Verdict]) -> dict:
    xs = [v.latency_ms for v in verdicts if v.outcome in USABLE]
    if not xs:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None, "n": 0}
    arr = np.asarray(xs, dtype=float)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()), "n": int(arr.size)}
