from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from nar_guardrails.bench.cost import summarize_cost
from nar_guardrails.bench.latency import latency_summary
from nar_guardrails.bench.metrics import auroc, basic_metrics, bootstrap_ci, fpr_at_recall, usable
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
    return float(thresholds.get(key, {}).get(which, 0.5))


def _kind(models_cfg: dict, key: str) -> str:
    base, _ = split_seed(key)
    return "remote" if models_cfg.get(base, models_cfg.get(key, {})).get("adapter") == "claude" else "local"


def leaderboard(items: list[Item], verdicts_by_det: dict[str, list[Verdict]], thresholds: dict, prices: dict,
                models_cfg: dict, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for key, vs in verdicts_by_det.items():
        y, s, _ = usable(items, vs)
        thr = _thr(thresholds, key, "f1")
        thr_default = "f1" not in thresholds.get(key, {})
        m = basic_metrics(y, s, thr) if y else {}
        rec_ci = bootstrap_ci(y, s, lambda a, b: basic_metrics(a, b, thr)["recall"], n_boot, seed) if y else (None, None)
        fpr_ci = bootstrap_ci(y, s, lambda a, b: basic_metrics(a, b, thr)["fpr"], n_boot, seed) if y else (None, None)
        lat = latency_summary(vs)
        base, _ = split_seed(key)
        cost = summarize_cost(vs, prices.get(models_cfg.get(base, {}).get("model_id")))
        rows.append({"detector": key, "kind": _kind(models_cfg, key), "n": len(y), "thr_f1": thr, "thr_default": thr_default,
                     "recall": m.get("recall"), "recall_lo": rec_ci[0], "recall_hi": rec_ci[1],
                     "fpr": m.get("fpr"), "fpr_lo": fpr_ci[0], "fpr_hi": fpr_ci[1],
                     "precision": m.get("precision"), "f1": m.get("f1"), "auroc": auroc(y, s) if y else None,
                     "fpr_at_95": fpr_at_recall(y, s, 0.95)[0] if y else None,
                     "fpr_at_99": fpr_at_recall(y, s, 0.99)[0] if y else None,
                     "p50_ms": lat["p50_ms"], "p95_ms": lat["p95_ms"], "per_1k_usd": cost["per_1k_usd"],
                     "refused": sum(v.outcome == "refused" for v in vs), "errors": sum(v.outcome == "error" for v in vs),
                     "truncated": sum(v.outcome == "truncated" for v in vs)})
    return pd.DataFrame(rows)


def _group_metric(items: list[Item], vs: list[Verdict], thr: float, group_of) -> dict:
    out = {}
    groups = sorted({group_of(i) for i in items})
    for g in groups:
        sub = [i for i in items if group_of(i) == g]
        y, s, _ = usable(sub, vs)
        if not y:
            out[g] = (None, len(sub), None)
            continue
        m = basic_metrics(y, s, thr)
        out[g] = ((m["recall"] if sub[0].label == 1 else m["fpr"]), len(sub), auroc(y, s))
    return out


def scenario_table(items: list[Item], verdicts_by_det: dict[str, list[Verdict]], thresholds: dict) -> pd.DataFrame:
    scenarios = sorted({i.scenario for i in items})
    df = pd.DataFrame(index=scenarios)
    df["n"] = [sum(i.scenario == sc for i in items) for sc in scenarios]
    df["metric"] = ["recall" if next(i for i in items if i.scenario == sc).label == 1 else "fpr" for sc in scenarios]
    for key, vs in verdicts_by_det.items():
        g = _group_metric(items, vs, _thr(thresholds, key, "f1"), lambda i: i.scenario)
        df[key] = [g[sc][0] for sc in scenarios]
    return df


def source_table(items: list[Item], verdicts_by_det: dict[str, list[Verdict]], thresholds: dict) -> pd.DataFrame:
    sources = sorted({i.source for i in items})
    df = pd.DataFrame(index=sources)
    df["n"] = [sum(i.source == so for i in items) for so in sources]
    for key, vs in verdicts_by_det.items():
        thr = _thr(thresholds, key, "f1")
        rec, fpr, auc = [], [], []
        for so in sources:
            sub = [i for i in items if i.source == so]
            y, s, _ = usable(sub, vs)
            m = basic_metrics(y, s, thr) if y else {}
            rec.append(m.get("recall")); fpr.append(m.get("fpr")); auc.append(auroc(y, s) if y else None)
        df[f"{key}_recall"], df[f"{key}_fpr"], df[f"{key}_auroc"] = rec, fpr, auc
    return df


def headline(items, verdicts_by_det, thresholds, prices, models_cfg) -> pd.DataFrame:
    lb = leaderboard(items, verdicts_by_det, thresholds, prices, models_cfg, n_boot=0)
    return lb[["detector", "kind", "fpr_at_95", "p50_ms", "per_1k_usd", "auroc"]].sort_values("fpr_at_95")


def default_label_table(items, verdicts_by_det) -> pd.DataFrame:
    rows = []
    for key, vs in verdicts_by_det.items():
        by = {v.item_id: v for v in vs}
        pairs = [(i.label, by[i.id].label) for i in items if i.id in by and by[i.id].label in (0, 1)]
        if not pairs:
            rows.append({"detector": key, "n": 0}); continue
        y = [p[0] for p in pairs]; yhat = [float(p[1]) for p in pairs]
        rows.append({"detector": key, **{k: v for k, v in basic_metrics(y, yhat, 0.5).items() if k != "threshold"}})
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
