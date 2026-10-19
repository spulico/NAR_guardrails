from __future__ import annotations

import json
from pathlib import Path

from nar_guardrails.schema import Item

CURATED_PROMPT_COUNTS = {"B1": 35, "B2": 35, "B3": 30, "J1": 15, "J2": 10, "D1": 15, "D2": 10,
                         "I1": 15, "I2": 15, "I3": 10, "I4": 10}
SCENARIO_LABEL = {"B1": 0, "B2": 0, "B3": 0, "J1": 1, "J2": 1, "D1": 1, "D2": 1, "I1": 1, "I2": 1, "I3": 1, "I4": 1}
CURATED_META = {"license": "CC-BY-4.0", "published": "2026-10"}


def load_curated_prompts(path: Path = Path("data/curated/prompts.jsonl")) -> list[Item]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            items.append(Item(id=d["id"], track="prompts", text=d["text"], label=SCENARIO_LABEL[d["scenario"]],
                              scenario=d["scenario"], source="curated", split="test",
                              meta={**CURATED_META, "note": d.get("note", "")}))
    return items


from nar_guardrails.schema import text_sha  # noqa: E402

SKILL_CATEGORIES = ("data-exfiltration", "credential-harvesting", "anti-refusal", "hidden-instructions",
                    "download-and-execute", "persistence", "mcp-tool-poisoning", "privilege-escalation",
                    "typosquatted-dependency", "memory-poisoning")
SKILLS_META = {"license": "CC-BY-4.0", "published": "2026-10", "splits": {"MSD": "test", "SD": "test"}}
SCRIPT_SEP = "\n\n--- scripts ---\n"


PAYLOAD_IN = ("text", "scripts")


def _find_manifest(folder: Path) -> Path | None:
    """Return the exact SKILL.md / skill.md entry listed in folder (safe on case-insensitive filesystems)."""
    if not folder.is_dir():
        return None
    entries = {p.name: p for p in folder.iterdir() if p.is_file()}
    return entries.get("SKILL.md") or entries.get("skill.md")


def _skill_meta(meta: dict, d: dict) -> dict:
    payload_in = d.get("payload_in", "text")
    if payload_in not in PAYLOAD_IN:
        raise ValueError(f"payload_in must be one of {PAYLOAD_IN}, got {payload_in!r} for {d}")
    return {**meta, "payload_in": payload_in}


def _skill_dir_items(folder: Path, item_id: str, label: int, scenario: str, source: str, meta: dict) -> list[Item]:
    md = _find_manifest(folder)
    if md is None:
        raise FileNotFoundError(f"no SKILL.md in {folder}")
    text = md.read_text(encoding="utf-8")
    items = [Item(id=item_id, track="skills", text=text, label=label, scenario=scenario, source=source, split="test",
                  variant="text", meta=dict(meta))]
    extras = sorted(p for p in folder.rglob("*") if p.is_file() and p != md and p.suffix in
                    {".sh", ".py", ".js", ".ts", ".rb", ".ps1", ".yaml", ".yml", ".json", ".toml", ".txt", ".md"})
    if extras:
        joined = text + SCRIPT_SEP + "\n".join(f"## {p.relative_to(folder)}\n{p.read_text(encoding='utf-8', errors='replace')}" for p in extras)
        items.append(Item(id=item_id + "#scripts", track="skills", text=joined, label=label, scenario=scenario,
                          source=source, split="test", variant="text+scripts", meta=dict(meta)))
    return items


def load_curated_skills(root: Path = Path("data/curated/skills")) -> list[Item]:
    labels = [json.loads(l) for l in (Path(root) / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for d in labels:
        items.extend(_skill_dir_items(Path(root) / d["slug"], f"curated_skill:{d['slug']}", int(d["label"]), d["category"],
                                      "curated_skills", _skill_meta(SKILLS_META, d)))
    return items


def load_skillspector_fixtures(repo: Path, labels: Path = Path("data/curated/skillspector_labels.jsonl")) -> list[Item]:
    meta = {"license": "Apache-2.0", "published": "2026", "splits": {"MSD": "test", "SD": "test"}}
    fixtures = Path(repo) / "tests" / "fixtures"
    if not fixtures.is_dir():
        raise FileNotFoundError(f"SkillSpector fixtures not found at {fixtures}")
    rows = [json.loads(l) for l in Path(labels).read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [d["name"] for d in rows if _find_manifest(fixtures / d["name"]) is None]
    if missing:
        raise ValueError(f"labelled SkillSpector fixtures have no SKILL.md: {missing}")
    items = []
    for d in rows:
        items.extend(_skill_dir_items(fixtures / d["name"], f"skillspector:{d['name']}", int(d["label"]), d["category"],
                                      "skillspector_fixtures", _skill_meta(meta, d)))
    return items


def assert_no_overlap(held_out: list[Item], train: list[Item]) -> None:
    train_hashes = {text_sha(i.text) for i in train}
    clashes = [i.id for i in held_out if text_sha(i.text) in train_hashes]
    if clashes:
        raise ValueError(f"held-out skills appear in training data: {clashes}")
