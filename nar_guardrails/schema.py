from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

OUTCOMES = ("ok", "refused", "error", "truncated")


@dataclass
class Item:
    id: str
    track: str
    text: str
    label: int
    scenario: str = "unknown"
    source: str = "unknown"
    split: str = "test"
    variant: str = "text"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r} for item {self.id}")
        if self.track not in ("prompts", "skills"):
            raise ValueError(f"track must be prompts or skills, got {self.track!r}")


@dataclass
class Verdict:
    item_id: str
    detector: str
    policy: str
    p_malicious: float | None
    label: int | None
    outcome: str = "ok"
    latency_ms: float = 0.0
    usage: dict | None = None
    raw: Any = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")


_WS = re.compile(r"\s+")


def text_sha(text: str) -> str:
    norm = _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[Item | Verdict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def _read(path: str | Path, cls):
    names = {f.name for f in fields(cls)}
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(cls(**{k: v for k, v in d.items() if k in names}))
    return out


def read_items(path: str | Path) -> list[Item]:
    return _read(path, Item)


def read_verdicts(path: str | Path) -> list[Verdict]:
    return _read(path, Verdict)
