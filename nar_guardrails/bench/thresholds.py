from __future__ import annotations

import json
import math
from pathlib import Path

from nar_guardrails.bench.metrics import basic_metrics, fpr_at_recall, usable
from nar_guardrails.schema import Item, Verdict


def choose_thresholds(y: list[int], scores: list[float]) -> dict:
    candidates = sorted(set(scores), reverse=True)
    best_thr, best_f1 = 1.0, -1.0
    for thr in candidates:  # descending, so ties keep the higher threshold
        f1 = basic_metrics(y, scores, thr)["f1"]
        if math.isnan(f1):  # undefined (e.g. no positives in dev): not comparable
            continue
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    _, t95 = fpr_at_recall(y, scores, 0.95)
    _, t99 = fpr_at_recall(y, scores, 0.99)
    out = {"f1": best_thr, "recall95": t95, "recall99": t99}
    out["dev"] = {name: basic_metrics(y, scores, thr) for name, thr in out.items()}
    return out


def thresholds_for_run(items_dev: list[Item], verdicts_by_detector: dict[str, list[Verdict]],
                       policy_ref: str, track: str) -> dict:
    result = {}
    for key, verdicts in verdicts_by_detector.items():
        y, s, _ = usable(items_dev, verdicts)
        if not y or len(set(y)) < 2:
            result[key] = {"policy": policy_ref, "track": track, "error": "dev split has fewer than two classes"}
            continue
        result[key] = {"policy": policy_ref, "track": track, **choose_thresholds(y, s)}
    return result


def write_thresholds(path: Path, d: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")


def read_thresholds(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
