from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

from nar_guardrails.bench.cache import VerdictCache, cache_path
from nar_guardrails.bench.manifest import collect_environment
from nar_guardrails.detectors import build_detector, load_models_cfg
from nar_guardrails.detectors.policy import load_policy, pick_examples
from nar_guardrails.schema import Item, read_items, write_jsonl


@dataclass
class RunConfig:
    track: str
    split: str
    detectors: list[str]
    policy: str = "shared"
    protocol: str | None = None
    limit: int | None = None
    run_id: str | None = None
    allow_dirty: bool = False
    throughput: bool = False
    data_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")
    models_cfg: dict | None = None
    seed: int = 0


def load_items(data_dir: Path, track: str, split: str, protocol: str | None = None, limit: int | None = None) -> list[Item]:
    items = read_items(Path(data_dir) / f"{track}.jsonl")
    if protocol:
        items = [i for i in items if i.meta.get("splits", {}).get(protocol) == split]
    else:
        items = [i for i in items if i.split == split]
    items.sort(key=lambda i: i.id)
    return items[:limit] if limit else items


def resolve_detectors(spec: list[str], models_cfg: dict, track: str) -> list[str]:
    if spec == ["all"]:
        return [k for k, c in models_cfg.items() if k != "fake" and track in c.get("tracks", [])]
    for k in spec:
        if k not in models_cfg:
            raise KeyError(f"unknown detector {k!r}")
    return list(spec)


def measure_throughput(det, items: list[Item], batch_size: int, n: int = 256) -> dict:
    sample = items[:n]
    t0 = perf_counter()
    det.predict(sample, batch_size=batch_size)
    secs = perf_counter() - t0
    return {"n": len(sample), "batch_size": batch_size, "seconds": secs,
            "items_per_s": (len(sample) / secs) if secs > 0 else None}


def _policy_for(cfg: RunConfig, track: str):
    if cfg.policy == "8shot":
        pool_split = "train" if track == "skills" else "dev"
        pool = load_items(cfg.data_dir, track, pool_split, protocol=cfg.protocol)
        pol = load_policy(track, "shared", examples=pick_examples(pool, 8, cfg.seed))
        return replace(pol, name="8shot")
    return load_policy(track, cfg.policy)


def run(cfg: RunConfig) -> Path:
    env = collect_environment()
    if env["git_dirty"] and not cfg.allow_dirty:
        raise SystemExit("working tree is dirty; commit or pass --allow-dirty")
    models_cfg = cfg.models_cfg or load_models_cfg()
    keys = resolve_detectors(cfg.detectors, models_cfg, cfg.track)
    items = load_items(cfg.data_dir, cfg.track, cfg.split, cfg.protocol, cfg.limit)
    policy = _policy_for(cfg, cfg.track)
    run_id = cfg.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "track": cfg.track, "split": cfg.split, "protocol": cfg.protocol,
                "policy": policy.ref, "n_items": len(items), "limit": cfg.limit, "seed": cfg.seed,
                "environment": env, "detectors": {}}
    for key in keys:
        entry: dict = {"model_id": models_cfg[key].get("model_id")}
        t0 = perf_counter()
        try:
            det = build_detector(key, policy, models_cfg)
        except Exception as e:  # optional backends, gated downloads, missing deps
            entry.update(status="skipped", reason=f"{type(e).__name__}: {e}")
            manifest["detectors"][key] = entry
            continue
        cache = VerdictCache(cache_path(Path(cfg.results_dir) / "cache", key, policy.hash, cfg.track))
        todo = [i for i in items if cache.get(i) is None]
        if todo:
            det.warmup()
            cache.put_many(todo, det.predict(todo, batch_size=1))
        verdicts = [cache.get(i) for i in items]
        write_jsonl(run_dir / f"verdicts_{key}.jsonl", verdicts)
        entry.update(status="ok", computed=len(todo), cached=len(items) - len(todo),
                     seconds=perf_counter() - t0)
        if cfg.throughput and models_cfg[key].get("adapter") != "claude":
            entry["throughput"] = measure_throughput(det, items, int(models_cfg[key].get("batch_size", 16)))
        manifest["detectors"][key] = entry
        del det  # free the model before loading the next one
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (Path(cfg.results_dir) / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    return run_dir
