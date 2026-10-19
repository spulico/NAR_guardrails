from __future__ import annotations

import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from nar_guardrails.bench.latency import latency_summary
from nar_guardrails.bench.runner import RunConfig, resolve_detectors, run
from nar_guardrails.datasets.curated import load_curated_prompts, load_curated_skills
from nar_guardrails.detectors import load_models_cfg
from nar_guardrails.report.tables import load_run
from nar_guardrails.schema import Item, write_jsonl


def _subset(items: list[Item], n: int) -> list[Item]:
    pos = sorted((i for i in items if i.label == 1 and i.variant == "text"), key=lambda i: i.id)[: n // 2]
    neg = sorted((i for i in items if i.label == 0 and i.variant == "text"), key=lambda i: i.id)[: n - n // 2]
    out = pos + neg
    for i in out:
        i.split = "smoke"
    return out


def _error_reason(vs: list) -> str | None:
    texts = [str(v.raw) for v in vs if v.outcome == "error"]
    if not texts:
        return None
    return Counter(texts).most_common(1)[0][0][:80]


def smoke(models_cfg: dict | None = None, results_dir: Path = Path("results"), n: int = 20,
          detectors: list[str] | None = None) -> pd.DataFrame:
    """Run detectors on a small subset of each track. Uses its own cache and never updates results/LATEST."""
    models_cfg = models_cfg if models_cfg is not None else load_models_cfg()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    requested = [d.strip() for d in (detectors or ["all", "fake"]) if d.strip()]
    if not requested:
        raise ValueError("empty detector spec: name detector keys or 'all'")
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        subsets = {"prompts": _subset(load_curated_prompts(), n), "skills": _subset(load_curated_skills(), n)}
        for track, subset in subsets.items():
            write_jsonl(data_dir / f"{track}.jsonl", subset)
            # a named detector that does not list this track sits this track out (unknown names still raise)
            spec = [d for d in requested if d == "all" or d not in models_cfg or track in models_cfg[d].get("tracks", [])]
            if not spec:
                continue
            keys = resolve_detectors(spec, models_cfg, track)
            run_id = f"smoke-{stamp}-{track}"
            run_dir = run(RunConfig(track=track, split="smoke", detectors=keys, allow_dirty=True, data_dir=data_dir,
                                    results_dir=Path(results_dir), models_cfg=models_cfg, run_id=run_id,
                                    cache_dir=Path(results_dir) / run_id / "cache", update_latest=False))
            res = load_run(run_dir)
            items = {i.id: i for i in subset}
            n_items = res["manifest"].get("n_items", len(subset))
            for key in keys:
                entry = res["manifest"]["detectors"].get(key, {})
                vs = res["verdicts"].get(key, [])
                ok = [v for v in vs if v.outcome in ("ok", "refused", "truncated") and v.p_malicious is not None]
                mal = [v.p_malicious for v in ok if items[v.item_id].label == 1]
                ben = [v.p_malicious for v in ok if items[v.item_id].label == 0]
                status, reason = entry.get("status", "missing"), entry.get("reason")
                if status == "ok" and len(ok) < n_items:
                    status, reason = "degraded", _error_reason(vs) or reason
                mp = (sum(mal) / len(mal)) if mal else None
                mb = (sum(ben) / len(ben)) if ben else None
                rows.append({"track": track, "detector": key, "status": status, "n_ok": len(ok),
                             "mean_p_malicious": mp, "mean_p_benign": mb,
                             "separates": (mp > mb) if mp is not None and mb is not None else None,
                             "p50_ms": latency_summary(vs)["p50_ms"], "reason": reason})
    return pd.DataFrame(rows)
