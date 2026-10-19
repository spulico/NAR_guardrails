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
    if spec.get("streaming"):
        rows = itertools.islice(rows, STREAM_LIMIT)
    return spec["loader"](rows, spec.get("cap"), seed)


def _dedupe(items: list[Item]) -> tuple[list[Item], int]:
    seen, out, dropped = set(), [], 0
    for it in items:
        h = text_sha(it.text)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(it)
    return out, dropped


def build_prompts(raw_dir: Path, out_path: Path, seed: int = 0, sources: dict | None = None, strict: bool = True) -> dict:
    sources = sources if sources is not None else default_sources()
    ranges = expected_ranges().get("prompts", {})
    items: list[Item] = list(load_curated_prompts())
    counts, skipped, licenses = {"curated": len(items)}, {}, {"curated": "CC-BY-4.0"}
    for name, spec in sources.items():
        try:
            loaded = _load_source(name, spec, Path(raw_dir), seed)
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {e}"
            continue
        counts[name] = len(loaded)
        licenses[name] = spec.get("license")
        items.extend(loaded)
    items, dropped = _dedupe(items)
    problems = [f"{n}: {c} not in [{ranges[n]['min']}, {ranges[n]['max']}]"
                for n, c in counts.items() if n in ranges and not (ranges[n]["min"] <= c <= ranges[n]["max"])]
    manifest = {"seed": seed, "counts": counts, "skipped": skipped, "licenses": licenses, "duplicates_dropped": dropped,
                "range_problems": problems, "n_items": len(items)}
    if strict and problems:
        raise SystemExit("count validation failed:\n" + "\n".join(problems))
    items = assign_dev_test(items, 0.2, seed)
    write_jsonl(out_path, items)
    Path(out_path).with_suffix(".build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
