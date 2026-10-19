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
