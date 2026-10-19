from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from nar_guardrails.schema import Item, Verdict, text_sha


def cache_path(root: Path, detector_key: str, policy_hash: str, track: str) -> Path:
    return Path(root) / detector_key / policy_hash / f"{track}.jsonl"


class VerdictCache:
    """Append-only JSONL keyed by item id; an entry is valid only if the item's text hash still matches."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, tuple[str, Verdict]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self._entries[d["item_id"]] = (d["text_sha"], Verdict(**d["verdict"]))

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, item: Item) -> Verdict | None:
        hit = self._entries.get(item.id)
        if hit is None or hit[0] != text_sha(item.text):
            return None
        return hit[1]

    def put(self, item: Item, verdict: Verdict) -> None:
        sha = text_sha(item.text)
        self._entries[item.id] = (sha, verdict)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"item_id": item.id, "text_sha": sha, "verdict": asdict(verdict)},
                               ensure_ascii=False) + "\n")

    def put_many(self, items: list[Item], verdicts: list[Verdict]) -> None:
        for i, v in zip(items, verdicts):
            self.put(i, v)
