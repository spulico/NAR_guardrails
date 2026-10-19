from __future__ import annotations

import gc
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

from nar_guardrails.bench.cache import VerdictCache, cache_path
from nar_guardrails.bench.manifest import collect_environment
from nar_guardrails.detectors import build_detector, load_models_cfg
from nar_guardrails.detectors.policy import load_policy, pick_examples
from nar_guardrails.schema import Item, read_items, write_jsonl

CACHE_CHUNK = 64


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
    cache_dir: Path | None = None
    update_latest: bool = True
    variant: str = "text"


VARIANTS = ("text", "text+scripts")


def load_items(data_dir: Path, track: str, split: str, protocol: str | None = None, limit: int | None = None,
               variant: str = "text") -> list[Item]:
    """Items of one track/split (or protocol split) and one variant; variants are never mixed."""
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be a positive integer or None, got {limit}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    items = [i for i in read_items(Path(data_dir) / f"{track}.jsonl") if i.variant == variant]
    if protocol:
        items = [i for i in items if i.meta.get("splits", {}).get(protocol) == split]
    else:
        items = [i for i in items if i.split == split]
    items.sort(key=lambda i: i.id)
    return items if limit is None else items[:limit]


def resolve_detectors(spec: list[str], models_cfg: dict, track: str) -> list[str]:
    keys = [k.strip() for k in spec if k.strip()]
    out: list[str] = []
    for k in keys:
        if k == "all":
            names = [n for n, c in models_cfg.items() if n != "fake" and track in c.get("tracks", [])]
        elif k in models_cfg:
            names = [k]
        else:
            raise KeyError(f"unknown detector {k!r}")
        out.extend(n for n in names if n not in out)
    return out


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
        examples = pick_examples(pool, 8, cfg.seed)
        if len(examples) < 8:
            raise ValueError(f"8shot needs 8 examples but track={track!r} split={pool_split!r} "
                             f"pool has {len(pool)} items ({len(examples)} usable)")
        pol = load_policy(track, "shared", examples=examples)
        return replace(pol, name="8shot")
    return load_policy(track, cfg.policy)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _run_detector(key, det, items, cache, run_dir, cfg, models_cfg, entry) -> None:
    todo = [i for i in items if cache.get(i) is None]
    fresh: dict[str, object] = {}
    warmed = False
    if todo:
        det.warmup()
        warmed = True
        for start in range(0, len(todo), CACHE_CHUNK):
            chunk = todo[start:start + CACHE_CHUNK]
            verdicts = det.predict(chunk, batch_size=1)
            fresh.update((i.id, v) for i, v in zip(chunk, verdicts))
            good = [(i, v) for i, v in zip(chunk, verdicts) if v.outcome != "error"]
            if good:
                cache.put_many([i for i, _ in good], [v for _, v in good])
    out = [cache.get(i) or fresh[i.id] for i in items]
    write_jsonl(run_dir / f"verdicts_{key}.jsonl", out)
    entry.update(status="ok", computed=len(todo), cached=len(items) - len(todo),
                 errors=sum(v.outcome == "error" for v in out))
    if fresh and all(v.outcome == "error" for v in fresh.values()):
        top = Counter(str(v.raw) for v in fresh.values()).most_common(1)[0][0]
        entry.update(status="failed", reason=f"all verdicts are errors: {top}")
        return
    if cfg.throughput and models_cfg[key].get("adapter") != "claude":
        if not warmed:
            det.warmup()
        entry["throughput"] = measure_throughput(det, items, int(models_cfg[key].get("batch_size", 16)))


def run(cfg: RunConfig) -> Path:
    env = collect_environment()
    if env["git_dirty"] and not cfg.allow_dirty:
        raise SystemExit("working tree is dirty; commit or pass --allow-dirty")
    models_cfg = cfg.models_cfg or load_models_cfg()
    keys = resolve_detectors(cfg.detectors, models_cfg, cfg.track)
    items = load_items(cfg.data_dir, cfg.track, cfg.split, cfg.protocol, cfg.limit, cfg.variant)
    policy = _policy_for(cfg, cfg.track)
    run_id = cfg.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in run_dir.glob("verdicts_*.jsonl"):  # a reused run id keeps only this run's verdict files
        stale.unlink()
    manifest = {"run_id": run_id, "track": cfg.track, "split": cfg.split, "protocol": cfg.protocol, "variant": cfg.variant,
                "policy": policy.ref, "n_examples": len(policy.examples), "n_items": len(items), "limit": cfg.limit, "seed": cfg.seed,
                "environment": env, "detectors": {}}
    manifest_path = run_dir / "manifest.json"

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    for key in keys:
        entry: dict = {"model_id": models_cfg[key].get("model_id")}
        t0 = perf_counter()
        manifest["detectors"][key] = entry
        det = None
        try:
            det = build_detector(key, policy, models_cfg)
        except Exception as e:  # optional backends, gated downloads, missing deps
            entry.update(status="skipped", reason=f"{type(e).__name__}: {e}")
        else:
            cache = VerdictCache(cache_path(cfg.cache_dir or Path(cfg.results_dir) / "cache", key, policy.hash, cfg.track))
            try:
                _run_detector(key, det, items, cache, run_dir, cfg, models_cfg, entry)
            except Exception as e:  # one broken detector must not abort the run
                for k in ("computed", "cached", "errors", "throughput"):
                    entry.pop(k, None)
                entry.update(status="failed", reason=f"{type(e).__name__}: {e}")
            entry["seconds"] = perf_counter() - t0
        del det  # free the model before loading the next one
        _free_memory()
        save()
    save()
    if cfg.update_latest:
        (Path(cfg.results_dir) / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    return run_dir
