from __future__ import annotations

import tempfile
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


def smoke(models_cfg: dict | None = None, results_dir: Path = Path("results"), n: int = 20,
          detectors: list[str] | None = None) -> pd.DataFrame:
    models_cfg = models_cfg if models_cfg is not None else load_models_cfg()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        write_jsonl(data_dir / "prompts.jsonl", _subset(load_curated_prompts(), n))
        write_jsonl(data_dir / "skills.jsonl", _subset(load_curated_skills(), n))
        for track in ("prompts", "skills"):
            keys = detectors or (resolve_detectors(["all"], models_cfg, track) + ["fake"])
            keys = [k for k in keys if track in models_cfg[k].get("tracks", [])]
            run_dir = run(RunConfig(track=track, split="smoke", detectors=keys, allow_dirty=True, data_dir=data_dir,
                                    results_dir=Path(results_dir), models_cfg=models_cfg, run_id=f"smoke-{stamp}-{track}"))
            res = load_run(run_dir)
            items = {i.id: i for i in _subset(load_curated_prompts() if track == "prompts" else load_curated_skills(), n)}
            for key in keys:
                entry = res["manifest"]["detectors"].get(key, {})
                vs = res["verdicts"].get(key, [])
                ok = [v for v in vs if v.outcome in ("ok", "refused", "truncated") and v.p_malicious is not None]
                mal = [v.p_malicious for v in ok if items[v.item_id].label == 1]
                ben = [v.p_malicious for v in ok if items[v.item_id].label == 0]
                rows.append({"track": track, "detector": key, "status": entry.get("status", "missing"), "n_ok": len(ok),
                             "mean_p_malicious": (sum(mal) / len(mal)) if mal else None,
                             "mean_p_benign": (sum(ben) / len(ben)) if ben else None,
                             "p50_ms": latency_summary(vs)["p50_ms"], "reason": entry.get("reason")})
    return pd.DataFrame(rows)
