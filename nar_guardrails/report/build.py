from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nar_guardrails.bench.cost import load_prices
from nar_guardrails.bench.thresholds import thresholds_for_run, write_thresholds
from nar_guardrails.detectors import load_models_cfg
from nar_guardrails.report.charts import scatter_cost_latency, scenario_heatmap
from nar_guardrails.report.tables import (default_label_table, headline, leaderboard, load_run, scenario_table,
                                          seed_aggregate, source_table)
from nar_guardrails.schema import read_items


def _resolve_run(results_dir: Path, run_id: str) -> Path:
    if run_id == "latest":
        run_id = (Path(results_dir) / "LATEST").read_text(encoding="utf-8").strip()
    return Path(results_dir) / run_id


def find_dev_run(results_dir: Path, track: str, policy_ref: str, exclude: str, protocol: str | None = None) -> Path | None:
    best = None
    for m in Path(results_dir).glob("*/manifest.json"):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        ts = (d.get("environment") or {}).get("timestamp") if isinstance(d.get("environment"), dict) else None
        if any(d.get(k) is None for k in ("run_id", "track", "policy", "split")) or ts is None:
            continue
        if (d["run_id"] == exclude or d["track"] != track or d["policy"] != policy_ref
                or d["split"] not in ("dev", "val") or d.get("protocol") != protocol):
            continue
        if best is None or ts > best[0]:
            best = (ts, m.parent)
    return best[1] if best else None


def _items_for(run: dict, data_dir: Path):
    m = run["manifest"]
    items = read_items(Path(data_dir) / f"{m['track']}.jsonl")
    if m.get("protocol"):
        return [i for i in items if i.meta.get("splits", {}).get(m["protocol"]) == m["split"]]
    return [i for i in items if i.split == m["split"]]


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_(no data)_"
    return df.to_markdown(index=not isinstance(df.index, pd.RangeIndex), floatfmt=floatfmt, missingval="–")


def build_report(results_dir: Path, run_id: str = "latest", dev_run_id: str | None = None,
                 data_dir: Path = Path("data/processed"), prices_path: Path = Path("configs/prices.yaml"),
                 models_path: Path | None = Path("configs/models.yaml"), models_cfg: dict | None = None) -> Path:
    run_dir = _resolve_run(results_dir, run_id)
    run = load_run(run_dir)
    models_cfg = models_cfg if models_cfg is not None else load_models_cfg(models_path)
    prices = load_prices(prices_path)
    items = _items_for(run, data_dir)
    m = run["manifest"]
    verdicts = {k: v for k, v in run["verdicts"].items() if v}
    dev_dir = (Path(results_dir) / dev_run_id) if dev_run_id else find_dev_run(results_dir, m["track"], m["policy"], m["run_id"], m.get("protocol"))
    notes = []
    thresholds: dict = {}
    if dev_dir is not None:
        dev = load_run(dev_dir)
        dev_items = _items_for(dev, data_dir)
        thresholds = thresholds_for_run(dev_items, dev["verdicts"], m["policy"], m["track"])
        proto = f", protocol {m['protocol']}" if m.get("protocol") else ""
        notes.append(f"Thresholds chosen on dev run `{dev_dir.name}`{proto} ({len(dev_items)} items).")
    else:
        notes.append("No dev run found for this track and policy: thresholds default to 0.5 for every detector.")
    write_thresholds(run_dir / "thresholds.json", thresholds)

    title = f"# NAR guardrails report: {m['track']} / {m['split']}" + (f" / {m['protocol']}" if m.get("protocol") else "")
    env = m.get("environment", {})
    head = [title, f"Run `{m['run_id']}`, policy `{m['policy']}`, {len(items)} items, device {env.get('device')}, "
                   f"commit {env.get('git_commit')}.", "", *notes, ""]
    skipped = {k: v.get("reason") for k, v in m["detectors"].items() if v.get("status") == "skipped"}
    skipped_md = ["## Skipped detectors", *[f"- `{k}`: {v}" for k, v in skipped.items()], ""] if skipped else []

    if not verdicts:
        parts = head + ["No detector produced verdicts in this run, so there are no tables or charts.", "", *skipped_md]
        (run_dir / "report.md").write_text("\n".join(parts), encoding="utf-8")
        return run_dir

    lb = leaderboard(items, verdicts, thresholds, prices, models_cfg)
    sc = scenario_table(items, verdicts, thresholds)
    so = source_table(items, verdicts, thresholds)
    hl = headline(items, verdicts, thresholds, prices, models_cfg)
    dl = default_label_table(items, verdicts)
    agg = seed_aggregate(lb, ["recall", "fpr", "auroc", "fpr_at_r95"])
    lb.to_csv(run_dir / "leaderboard.csv", index=False)
    sc.to_csv(run_dir / "scenario.csv")
    so.to_csv(run_dir / "source.csv")
    hl.to_csv(run_dir / "headline.csv", index=False)
    scatter_cost_latency(lb, run_dir / "cost_latency.png")
    scenario_heatmap(sc, run_dir / "scenario_heatmap.png")

    defaulted = [k for k, d in zip(lb["detector"], lb["thr_default"]) if d]
    if defaulted and dev_dir is not None:
        errs = {k: (thresholds.get(k) or {}).get("error") for k in defaulted}
        head += ["## Thresholds defaulted to 0.5",
                 *[f"- `{k}`" + (f": {e}" if e else "") for k, e in errs.items()], ""]
    parts = head + ["## Headline (dev-chosen threshold at 95% recall)", _md(hl), "",
                    "## Leaderboard (threshold = best F1 on dev; 95% bootstrap CIs)", _md(lb), "",
                    "## Seed-aggregated (mean ± std over -seedN runs; single-run detectors shown as is)", _md(agg), "",
                    "## Scenario breakdown (recall for malicious codes, FPR for benign codes)", _md(sc), "",
                    "## Per-source", _md(so), "",
                    "## Models' own default labels", _md(dl), "",
                    "## Charts", "![cost/latency](cost_latency.png)", "![scenarios](scenario_heatmap.png)", "",
                    *skipped_md]
    (run_dir / "report.md").write_text("\n".join(parts), encoding="utf-8")
    return run_dir
