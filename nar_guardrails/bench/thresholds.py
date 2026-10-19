from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from nar_guardrails.bench.metrics import basic_metrics, fpr_at_recall, usable
from nar_guardrails.schema import Item, Verdict


def _class_counts(y: list[int]) -> tuple[int, int]:
    n_pos = sum(1 for v in y if v == 1)
    return n_pos, len(y) - n_pos


def choose_thresholds(y: list[int], scores: list[float]) -> dict:
    n_pos, n_neg = _class_counts(y)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"choose_thresholds needs both classes in dev; got {len(y)} items "
                         f"(pos={n_pos}, neg={n_neg})")
    candidates = sorted(set(scores), reverse=True)
    best_thr, best_f1 = None, None
    for thr in candidates:  # descending; strict > keeps the higher threshold on exact ties
        m = basic_metrics(y, scores, thr)
        denom = 2 * m["tp"] + m["fp"] + m["fn"]
        if denom == 0:  # F1 undefined at this threshold
            continue
        f1 = Fraction(2 * m["tp"], denom)  # exact, so ties are real ties
        if best_f1 is None or f1 > best_f1:
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
        n_pos, n_neg = _class_counts(y)
        if n_pos == 0 or n_neg == 0:
            if not y:
                msg = "dev split has 0 usable verdicts"
            else:
                msg = (f"dev split has {len(y)} usable verdicts but only one class "
                       f"(pos={n_pos}, neg={n_neg})")
            result[key] = {"policy": policy_ref, "track": track, "error": msg}
            continue
        result[key] = {"policy": policy_ref, "track": track, **choose_thresholds(y, s)}
    return result


def write_thresholds(path: Path, d: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2, allow_nan=False), encoding="utf-8")


def read_thresholds(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
