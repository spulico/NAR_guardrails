from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from nar_guardrails.schema import Item

MSB_META = {"license": "CC-BY-4.0", "published": "2026-08"}
REPO_ID = "ProtectSkills/MaliciousSkillBench"
# Hard cap on skill text length at load time. One upstream item (msb:ASB04_002434) is 21M
# characters; everything else is under 30k. Capped items keep their label and record the
# original length in meta["truncated_chars"].
MAX_SKILL_CHARS = 100_000


def _protocol_of_filename(name: str) -> str | None:
    low = Path(name).as_posix().lower()
    if "msd" in low or "structural" in low:  # real repo: splits/m_structural_disjoint.*
        return "MSD"
    if "balanced" in low or "random" in low:  # real repo: random / source_balanced_random are not SD
        return None
    if re.search(r"(^|[^a-z])sd([^a-z]|$)", low) or "source" in low:
        return "SD"
    return None


def _split_map(rows: Iterable[dict]) -> dict[str, str]:
    out = {}
    for r in rows:
        bid = r.get("benchmark_id") or r.get("id")
        sp = str(r.get("split", "")).lower()
        if bid and sp:
            out[str(bid)] = {"validation": "val", "dev": "val"}.get(sp, sp)
    return out


def _category(row: dict) -> str:
    if int(row.get("label", 0)) == 0:
        return "benign"
    for k in ("attack_category", "attack_category_name", "harmonized_attack_category", "attack_category_code"):
        v = row.get(k)
        if v:
            return str(v)
    for k in ("attack_categories", "attack_category_codes"):  # real parquet: list-valued columns
        v = row.get(k)
        if v is not None and not isinstance(v, str):
            v = [str(x) for x in list(v) if x]
            if v:
                return v[0]
        elif v:
            return str(v)
    return "unknown"


def load_malicious_skill_bench(rows: Iterable[dict], split_tables: dict[str, Iterable[dict]]) -> list[Item]:
    maps = {p: _split_map(t) for p, t in split_tables.items()}
    rows = list(rows)
    seen: set[str] = set()
    dups = sorted({str(r["benchmark_id"]) for r in rows if str(r["benchmark_id"]) in seen or seen.add(str(r["benchmark_id"]))})
    if dups:
        raise ValueError(f"duplicate benchmark_id values: {dups[:20]}")
    items = []
    for r in rows:
        text = r.get("skill_text") or r.get("public_skill_text")
        if not text:
            continue
        bid = str(r["benchmark_id"])
        splits = {"MSD": maps.get("MSD", {}).get(bid), "SD": maps.get("SD", {}).get(bid)}
        meta = {**MSB_META, "splits": splits, "source_id": r.get("source_id"),
                "redacted": bool(r.get("text_redacted")), "structural_family": r.get("structural_family_id")}
        if len(text) > MAX_SKILL_CHARS:
            meta["truncated_chars"] = len(text)
            text = text[:MAX_SKILL_CHARS]
        items.append(Item(id=f"msb:{bid}", track="skills", text=text, label=int(r["label"]), scenario=_category(r),
                          source="malicious_skill_bench", split=splits["MSD"] or "unassigned", meta=meta))
    return items


def fetch_split_tables(repo_id: str = REPO_ID) -> dict[str, list[dict]]:
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset") if f.startswith("splits/")]
    # the repo ships each manifest as both .csv and .parquet; read one copy only (parquet preferred)
    by_stem: dict[str, str] = {}
    for f in sorted(files):
        stem = str(Path(f).with_suffix(""))
        if stem not in by_stem or f.endswith(".parquet"):
            by_stem[stem] = f
    tables: dict[str, list[dict]] = {"MSD": [], "SD": []}
    for f in by_stem.values():
        proto = _protocol_of_filename(f)
        if proto is None:
            continue
        local = hf_hub_download(repo_id, f, repo_type="dataset")
        if local.endswith(".csv"):
            df = pd.read_csv(local)
        elif local.endswith(".parquet"):
            df = pd.read_parquet(local)
        elif local.endswith(".json") or local.endswith(".jsonl"):
            df = pd.read_json(local, lines=local.endswith(".jsonl"))
        else:
            continue
        if "split" not in df.columns:  # one file per split, e.g. splits/msd/test.csv
            df["split"] = Path(f).stem.split("_")[-1]
        tables[proto].extend(df.to_dict("records"))
    if not tables["MSD"] or not tables["SD"]:
        raise RuntimeError(f"could not find MSD and SD split manifests under splits/ in {repo_id}: {files}")
    return tables
