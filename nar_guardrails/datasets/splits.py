from __future__ import annotations

import random
from collections import defaultdict

from nar_guardrails.schema import Item


def assign_dev_test(items: list[Item], dev_frac: float = 0.2, seed: int = 0) -> list[Item]:
    groups: dict[tuple[str, int], list[Item]] = defaultdict(list)
    for it in items:
        groups[(it.source, it.label)].append(it)
    for (source, label), members in sorted(groups.items()):
        members.sort(key=lambda i: i.id)
        if source == "curated":
            for m in members:
                m.split = "test"
            continue
        rng = random.Random(f"{seed}:{source}:{label}")  # per-group stream: other groups never shift this one
        n_dev = int(round(dev_frac * len(members)))
        dev_ids = set(rng.sample([m.id for m in members], n_dev))
        for m in members:
            m.split = "dev" if m.id in dev_ids else "test"
    return items
