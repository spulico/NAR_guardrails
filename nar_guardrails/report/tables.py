from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nar_guardrails.bench.cost import summarize_cost
from nar_guardrails.bench.latency import latency_summary
from nar_guardrails.bench.metrics import USABLE_OUTCOMES, auroc, basic_metrics, bootstrap_ci, fpr_at_recall, usable
from nar_guardrails.schema import Item, Verdict, read_verdicts

_SEED = re.compile(r"^(.*)-seed(\d+)$")


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    verdicts = {f.stem.removeprefix("verdicts_"): read_verdicts(f) for f in sorted(run_dir.glob("verdicts_*.jsonl"))}
    thr_path = run_dir / "thresholds.json"
    thresholds = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.exists() else {}
    return {"manifest": manifest, "verdicts": verdicts, "thresholds": thresholds}


def split_seed(key: str) -> tuple[str, int | None]:
    m = _SEED.match(key)
    return (m.group(1), int(m.group(2))) if m else (key, None)


def _thr(thresholds: dict, key: str, which: str) -> float:
    val = (thresholds.get(key) or {}).get(which)
    return 0.5 if val is None else float(val)


def _has_thr(thresholds: dict, key: str, which: str) -> bool:
    return (thresholds.get(key) or {}).get(which) is not None


def _restrict(items: list[Item], verdicts: list[Verdict]) -> list[Verdict]:
    ids = {i.id for i in items}
    return [v for v in verdicts if v.item_id in ids]


def _boot_pair(y: list[int], scores: list[float], thr: float, n_boot: int, seed: int, alpha: float = 0.05):
    nan = float("nan")
    rng = np.random.default_rng(seed)
    y_arr, s_arr = np.asarray(y), np.asarray(scores)
    n = len(y_arr)
    rec, fpr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = basic_metrics(y_arr[idx].tolist(), s_arr[idx].tolist(), thr)
        if not math.isnan(m["recall"]):
            rec.append(m["recall"])
        if not math.isnan(m["fpr"]):
            fpr.append(m["fpr"])

    def ci(vals):
        if not vals:
            return nan, nan
        return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
    return ci(rec), ci(fpr)


def _kind(models_cfg: dict, key: str) -> str:
    base, _ = split_seed(key)
    return "remote" if models_cfg.get(base, models_cfg.get(key, {})).get("adapter") == "claude" else "local"


def leaderboard(items: list[Item], verdicts_by_det: dict[str, list[Verdict]], thresholds: dict, prices: dict,
                models_cfg: dict, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    nan = float("nan")
    rows = []
    for key, all_vs in verdicts_by_det.items():
        vs = _restrict(items, all_vs)
        y, s, _ = usable(items, vs)
        thr = _thr(thresholds, key, "f1")
        thr_default = not _has_thr(thresholds, key, "f1")
        m = basic_metrics(y, s, thr) if y else {}
        rec_ci, fpr_ci = _boot_pair(y, s, thr, n_boot, seed) if y else ((nan, nan), (nan, nan))
        op = {}
        for tag, which in (("r95", "recall95"), ("r99", "recall99")):
            if y and _has_thr(thresholds, key, which):
                om = basic_metrics(y, s, _thr(thresholds, key, which))
                op[f"recall_at_{tag}"], op[f"fpr_at_{tag}"] = om["recall"], om["fpr"]
            else:
                op[f"recall_at_{tag}"], op[f"fpr_at_{tag}"] = nan, nan
        lat = latency_summary(vs)
        base, _ = split_seed(key)
        cost = summarize_cost(vs, prices.get(models_cfg.get(base, {}).get("model_id")))
        rows.append({"detector": key, "kind": _kind(models_cfg, key), "n": len(y), "thr_f1": thr, "thr_default": thr_default,
                     "recall": m.get("recall"), "recall_lo": rec_ci[0], "recall_hi": rec_ci[1],
                     "fpr": m.get("fpr"), "fpr_lo": fpr_ci[0], "fpr_hi": fpr_ci[1],
                     "precision": m.get("precision"), "f1": m.get("f1"), "auroc": auroc(y, s) if y else None,
                     **op,
                     "fpr_at_95_oracle": fpr_at_recall(y, s, 0.95)[0] if y else None,
                     "fpr_at_99_oracle": fpr_at_recall(y, s, 0.99)[0] if y else None,
                     "p50_ms": lat["p50_ms"], "p95_ms": lat["p95_ms"], "per_1k_usd": cost["per_1k_usd"],
                     "refused": sum(v.outcome == "refused" for v in vs), "errors": sum(v.outcome == "error" for v in vs),
                     "truncated": sum(v.outcome == "truncated" for v in vs)})
    return pd.DataFrame(rows)


def _group_table(items: list[Item], verdicts_by_det: dict[str, list[Verdict]], thresholds: dict, group_of) -> tuple[list, dict]:
    groups = sorted({group_of(i) for i in items})
    res = {}
    for key, all_vs in verdicts_by_det.items():
        vs = _restrict(items, all_vs)
        thr = _thr(thresholds, key, "f1")
        for g in groups:
            sub = [i for i in items if group_of(i) == g]
            y, s, _ = usable(sub, vs)
            if y:
                m = basic_metrics(y, s, thr)
                res[(key, g)] = (m["recall"], m["fpr"], auroc(y, s))
            else:
                res[(key, g)] = (float("nan"), float("nan"), None)
    return groups, res


def scenario_table(items: list[Item], verdicts_by_det: dict[str, list[Verdict]], thresholds: dict) -> pd.DataFrame:
    groups, res = _group_table(items, verdicts_by_det, thresholds, lambda i: i.scenario)
    df = pd.DataFrame(index=groups)
    df["n"] = [sum(i.scenario == g for i in items) for g in groups]
    labels = [{i.label for i in items if i.scenario == g} for g in groups]
    metric = ["mixed" if len(ls) > 1 else ("recall" if ls == {1} else "fpr") for ls in labels]
    df["metric"] = metric
    for key in verdicts_by_det:
        df[key] = [res[(key, g)][1] if mt == "fpr" else res[(key, g)][0] for g, mt in zip(groups, metric)]
        if "mixed" in metric:
            df[f"{key}_fpr"] = [res[(key, g)][1] if mt == "mixed" else float("nan") for g, mt in zip(groups, metric)]
    return df


def source_table(items: list[Item], verdicts_by_det: dict[str, list[Verdict]], thresholds: dict) -> pd.DataFrame:
    groups, res = _group_table(items, verdicts_by_det, thresholds, lambda i: i.source)
    df = pd.DataFrame(index=groups)
    df["n"] = [sum(i.source == g for i in items) for g in groups]
    for key in verdicts_by_det:
        df[f"{key}_recall"] = [res[(key, g)][0] for g in groups]
        df[f"{key}_fpr"] = [res[(key, g)][1] for g in groups]
        df[f"{key}_auroc"] = [res[(key, g)][2] for g in groups]
    return df


def headline(items, verdicts_by_det, thresholds, prices, models_cfg) -> pd.DataFrame:
    lb = leaderboard(items, verdicts_by_det, thresholds, prices, models_cfg, n_boot=0)
    cols = ["detector", "kind", "recall_at_r95", "fpr_at_r95", "p50_ms", "per_1k_usd", "auroc"]
    return lb[cols].sort_values("fpr_at_r95", na_position="last").reset_index(drop=True)


def default_label_table(items, verdicts_by_det) -> pd.DataFrame:
    rows = []
    for key, all_vs in verdicts_by_det.items():
        by = {v.item_id: v for v in _restrict(items, all_vs) if v.outcome in USABLE_OUTCOMES}
        pairs = [(i.label, by[i.id].label) for i in items if i.id in by and by[i.id].label in (0, 1)]
        if not pairs:
            rows.append({"detector": key, "n": 0}); continue
        y = [p[0] for p in pairs]; yhat = [float(p[1]) for p in pairs]
        rows.append({"detector": key, **{k: v for k, v in basic_metrics(y, yhat, 0.5).items() if k != "threshold"}})
    return pd.DataFrame(rows)


SCRIPTS_SUFFIX = "#scripts"


def variant_table(text_items: list[Item], text_verdicts: dict[str, list[Verdict]], script_items: list[Item],
                  script_verdicts: dict[str, list[Verdict]], thresholds: dict) -> pd.DataFrame:
    """Recall on malicious skills that exist in both variants, paired by base id (`<id>` vs `<id>#scripts`).

    Only pairs with a usable verdict under both variants count; each detector uses its F1 threshold.
    """
    nan = float("nan")
    scripted = {i.id.removesuffix(SCRIPTS_SUFFIX): i for i in script_items}
    pairs = [(t, scripted[t.id]) for t in text_items if t.id in scripted and t.label == 1]
    keys = list(text_verdicts) + [k for k in script_verdicts if k not in text_verdicts]
    rows = []
    for key in keys:
        thr = _thr(thresholds, key, "f1")
        _, s_text, ids_text = usable([t for t, _ in pairs], text_verdicts.get(key, []))
        _, s_scr, ids_scr = usable([s for _, s in pairs], script_verdicts.get(key, []))
        by_text = dict(zip(ids_text, s_text))
        by_scr = {i.removesuffix(SCRIPTS_SUFFIX): sc for i, sc in zip(ids_scr, s_scr)}
        common = [b for b in by_text if b in by_scr]
        rec_text = sum(by_text[b] >= thr for b in common) / len(common) if common else nan
        rec_scr = sum(by_scr[b] >= thr for b in common) / len(common) if common else nan
        rows.append({"detector": key, "thr_f1": thr, "n_malicious_scripted": len(common),
                     "recall_text": rec_text, "recall_text+scripts": rec_scr})
    return pd.DataFrame(rows)


def seed_aggregate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["base"] = [split_seed(k)[0] for k in df["detector"]]
    rows = []
    for base, g in df.groupby("base", sort=False):
        row = {"detector": base, "seeds": len(g)}
        for c in cols:
            vals = g[c].dropna().astype(float)
            row[c] = f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}" if len(vals) > 1 else (f"{vals.iloc[0]:.3f}" if len(vals) else None)
        rows.append(row)
    return pd.DataFrame(rows)
