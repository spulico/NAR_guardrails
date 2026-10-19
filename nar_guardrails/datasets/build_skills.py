from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nar_guardrails.datasets.build_prompts import expected_ranges
from nar_guardrails.datasets.curated import assert_no_overlap, load_curated_skills, load_skillspector_fixtures
from nar_guardrails.datasets.fetch import ensure_repo, hf_rows
from nar_guardrails.datasets.skills_bench import REPO_ID, fetch_split_tables, load_malicious_skill_bench
from nar_guardrails.schema import Item, write_jsonl

SKILLSPECTOR_URL = "https://github.com/NVIDIA/SkillSpector"


def build_skills(raw_dir: Path, out_path: Path, seed: int = 0, strict: bool = True, rows=None, split_tables=None,
                 fixtures_repo: Path | None = None) -> dict:
    rows = rows if rows is not None else hf_rows(REPO_ID, "primary", "train")
    split_tables = split_tables if split_tables is not None else fetch_split_tables()
    msb = load_malicious_skill_bench(rows, split_tables)
    fixtures_repo = fixtures_repo or ensure_repo(Path(raw_dir), "skillspector", SKILLSPECTOR_URL, "main")
    fixtures = load_skillspector_fixtures(fixtures_repo)
    curated = load_curated_skills()
    train_items = [i for i in msb if "train" in (i.meta["splits"].get("MSD"), i.meta["splits"].get("SD"))]
    assert_no_overlap(curated + fixtures, train_items)
    items: list[Item] = msb + fixtures + curated
    counts = {"malicious_skill_bench": len(msb), "skillspector_fixtures": len(fixtures), "curated_skills": len(curated)}
    # nothing is excluded from the output: "excluded" SD rows simply never match a split and get their own bucket
    protocol_counts = {p: dict(Counter(i.meta["splits"].get(p) or "unassigned" for i in msb)) for p in ("MSD", "SD")}
    ranges = expected_ranges().get("skills", {})
    problems = [f"{n}: {c} not in [{ranges[n]['min']}, {ranges[n]['max']}]"
                for n, c in counts.items() if n in ranges and not (ranges[n]["min"] <= c <= ranges[n]["max"])]
    manifest = {"seed": seed, "counts": counts, "protocol_counts": protocol_counts, "range_problems": problems,
                "n_items": len(items), "licenses": {"malicious_skill_bench": "CC-BY-4.0 (metadata); upstream terms per skill",
                                                   "skillspector_fixtures": "Apache-2.0", "curated_skills": "CC-BY-4.0"}}
    if strict and problems:
        raise SystemExit("count validation failed:\n" + "\n".join(problems))
    write_jsonl(out_path, items)
    Path(out_path).with_suffix(".build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
