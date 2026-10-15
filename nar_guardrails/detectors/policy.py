from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from nar_guardrails.schema import Item

DEFAULT_POLICY_DIR = Path("configs/policies")


@dataclass(frozen=True)
class Policy:
    track: str
    name: str
    question: str
    malicious: str
    benign: str
    examples: tuple[tuple[str, int], ...] = ()

    @property
    def hash(self) -> str:
        payload = json.dumps([self.question, self.malicious, self.benign, list(self.examples)],
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.hash}"


def load_policy(track: str, name: str = "shared", examples=(), policy_dir: Path = DEFAULT_POLICY_DIR) -> Policy:
    path = Path(policy_dir) / track / f"{name}.yaml"
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Policy(track=track, name=name, question=d["question"].strip(), malicious=d["malicious"].strip(),
                  benign=d["benign"].strip(), examples=tuple((t, int(l)) for t, l in examples))


def pick_examples(items: list[Item], n: int = 8, seed: int = 0) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    pos = [i for i in items if i.label == 1]
    neg = [i for i in items if i.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    chosen = pos[:half] + neg[: n - half]
    rng.shuffle(chosen)
    return [(i.text, i.label) for i in chosen]
