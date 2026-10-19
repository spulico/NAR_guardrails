from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

from nar_guardrails.datasets.curated import load_curated_prompts
from nar_guardrails.datasets.fetch import ensure_repo, hf_rows
from nar_guardrails.datasets.public_prompts import GIT_SOURCES, HF_SOURCES
from nar_guardrails.datasets.splits import assign_dev_test
from nar_guardrails.schema import Item, text_sha, write_jsonl

STREAM_LIMIT = 50_000


def expected_ranges(path: Path = Path("configs/datasets.yaml")) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def default_sources() -> dict:
    out = {k: {"kind": "git", **v} for k, v in GIT_SOURCES.items()}
    out.update({k: {"kind": "hf", **v} for k, v in HF_SOURCES.items()})
    return out


def _load_source(name: str, spec: dict, raw_dir: Path, seed: int) -> list[Item]:
    if spec["kind"] == "git":
        repo = ensure_repo(raw_dir, name, spec["url"], spec["ref"])
        loader = spec["loader"]
        return loader(repo) if name != "mcptox" else loader(repo, cap=500, seed=seed)
    rows = hf_rows(spec["dataset"], spec.get("config"), spec.get("split", "train"), spec.get("streaming", False))
    try:
        stream = itertools.islice(rows, spec.get("stream_limit", STREAM_LIMIT)) if spec.get("streaming") else rows
        return spec["loader"](stream, spec.get("cap"), seed)
    finally:
        close = getattr(rows, "close", None)
        if close:
            close()


def git_commit(repo: Path) -> str | None:
    """The commit recorded by ensure_repo in <repo>/.nar_commit, or None."""
    try:
        return (Path(repo) / ".nar_commit").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def source_provenance(name: str, spec: dict, raw_dir: Path) -> dict:
    if spec["kind"] == "git":
        return {"kind": "git", "url": spec.get("url"), "ref": spec.get("ref"), "commit": git_commit(Path(raw_dir) / name)}
    return {"kind": "hf", "dataset": spec.get("dataset"), "config": spec.get("config"), "split": spec.get("split", "train")}


def _dedupe(items: list[Item]) -> tuple[list[Item], dict, list[dict]]:
    seen: dict[str, Item] = {}
    out, dropped, conflicts = [], {}, []
    for it in items:
        h = text_sha(it.text)
        first = seen.get(h)
        if first is not None:
            dropped[it.source] = dropped.get(it.source, 0) + 1
            if first.label != it.label:
                conflicts.append({"kept": first.id, "dropped": it.id})
            continue
        seen[h] = it
        out.append(it)
    return out, dropped, conflicts


def build_prompts(raw_dir: Path, out_path: Path, seed: int = 0, sources: dict | None = None, strict: bool = True) -> dict:
    sources = sources if sources is not None else default_sources()
    ranges = expected_ranges().get("prompts", {})
    items: list[Item] = list(load_curated_prompts())
    loaded_counts, skipped, licenses = {"curated": len(items)}, {}, {"curated": "CC-BY-4.0"}
    stream_limits, provenance = {}, {}
    for name, spec in sources.items():
        try:
            loaded = _load_source(name, spec, Path(raw_dir), seed)
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {e}"
            continue
        loaded_counts[name] = len(loaded)
        licenses[name] = spec.get("license")
        provenance[name] = source_provenance(name, spec, Path(raw_dir))
        if spec.get("streaming"):
            stream_limits[name] = spec.get("stream_limit", STREAM_LIMIT)
        items.extend(loaded)
    items, dropped, conflicts = _dedupe(items)
    kept = {n: 0 for n in loaded_counts}
    for it in items:
        kept[it.source] = kept.get(it.source, 0) + 1
    problems = [f"{n}: {c} not in [{ranges[n]['min']}, {ranges[n]['max']}]"
                for n, c in kept.items() if n in ranges and not (ranges[n]["min"] <= c <= ranges[n]["max"])]
    manifest = {"seed": seed, "strict": strict, "stream_limits": stream_limits, "loaded": loaded_counts, "counts": kept,
                "dropped_duplicates": dropped, "label_conflicts": conflicts, "skipped": skipped, "licenses": licenses,
                "sources": provenance,
                "duplicates_dropped": sum(dropped.values()), "range_problems": problems, "n_items": len(items)}
    if strict and problems:
        raise SystemExit("count validation failed:\n" + "\n".join(problems))
    items = assign_dev_test(items, 0.2, seed)
    write_jsonl(out_path, items)
    Path(out_path).with_suffix(".build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
