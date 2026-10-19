from __future__ import annotations

import math
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from nar_guardrails.schema import Item, Verdict

USABLE_OUTCOMES = ("ok", "refused", "truncated")


def usable(items: list[Item], verdicts: list[Verdict]) -> tuple[list[int], list[float], list[str]]:
    by_id = {v.item_id: v for v in verdicts}
    y, s, ids = [], [], []
    for it in items:
        v = by_id.get(it.id)
        if v is None or v.outcome not in USABLE_OUTCOMES:
            continue
        score = v.p_malicious if v.p_malicious is not None else (float(v.label) if v.label in (0, 1) else None)
        if score is None:
            continue
        y.append(it.label)
        s.append(float(score))
        ids.append(it.id)
    return y, s, ids


def basic_metrics(y: list[int], scores: list[float], threshold: float) -> dict:
    yhat = [int(sc >= threshold) for sc in scores]
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 0)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    n_pos, n_neg = tp + fn, fp + tn
    recall = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    if math.isnan(recall):
        f1 = float("nan")
    elif recall == 0.0:
        f1 = 0.0
    elif not math.isnan(precision) and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"threshold": threshold, "recall": recall, "fpr": fpr, "precision": precision, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_pos": n_pos, "n_neg": n_neg}


def auroc(y: list[int], scores: list[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def fpr_at_recall(y: list[int], scores: list[float], target: float) -> tuple[float, float]:
    if not (0 < target <= 1):
        raise ValueError(f"target recall must satisfy 0 < target <= 1, got {target}")
    pos = sorted((s for s, l in zip(scores, y) if l == 1), reverse=True)
    if not pos:
        return float("nan"), float("nan")
    k = max(1, math.ceil(target * len(pos)))
    thr = pos[min(k, len(pos)) - 1]
    return basic_metrics(y, scores, thr)["fpr"], thr


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Precision at a given malicious prevalence from recall and FPR; NaN when nothing would be flagged."""
    if math.isnan(recall) or math.isnan(fpr):
        return float("nan")
    tp = recall * prevalence
    fp = fpr * (1.0 - prevalence)
    return tp / (tp + fp) if (tp + fp) > 0 else float("nan")


def bootstrap_ci(y: list[int], scores: list[float], stat: Callable[[list[int], list[float]], float],
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_arr, s_arr = np.asarray(y), np.asarray(scores)
    n = len(y_arr)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat(y_arr[idx].tolist(), s_arr[idx].tolist())
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
