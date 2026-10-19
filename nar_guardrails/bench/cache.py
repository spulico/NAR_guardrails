from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from nar_guardrails.schema import Item, Verdict, text_sha


def cache_path(root: Path, detector_key: str, policy_hash: str, track: str) -> Path:
    return Path(root) / detector_key / policy_hash / f"{track}.jsonl"


def _parse(line: str) -> tuple[str, tuple[str, Verdict]]:
    d = json.loads(line)
    return d["item_id"], (d["text_sha"], Verdict(**d["verdict"]))


class VerdictCache:
    """Append-only JSONL keyed by item id; an entry is valid only if the item's text hash still matches.

    A malformed final line (a torn write) is ignored on load and removed before the next append;
    a malformed line anywhere else raises. Memory is updated only after the line has been written.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, tuple[str, Verdict]] = {}
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            # split on "\n" only: json.dumps(ensure_ascii=False) may leave U+2028 etc. inside a line
            lines = f.read().split("\n")
        last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                item_id, entry = _parse(line)
            except (ValueError, KeyError, TypeError) as exc:
                if i == last:
                    break  # torn trailing write from an interrupted run
                raise ValueError(f"{self.path}: malformed cache line {i + 1}: {exc}") from exc
            self._entries[item_id] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, item: Item) -> Verdict | None:
        hit = self._entries.get(item.id)
        if hit is None or hit[0] != text_sha(item.text):
            return None
        return hit[1]

    def put(self, item: Item, verdict: Verdict) -> None:
        self._append([(item, verdict)])

    def put_many(self, items: list[Item], verdicts: list[Verdict]) -> None:
        rows = [(i, v) for i, v in zip(items, verdicts, strict=True)]
        self._append(rows)

    def _repair_unterminated_tail(self) -> None:
        """If the file ends without a newline, either terminate a complete record or drop a torn one."""
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        fragment = data[cut:].decode("utf-8", errors="replace")
        try:
            _parse(fragment)
        except (ValueError, KeyError, TypeError):
            with self.path.open("r+b") as f:
                f.truncate(cut)
        else:
            with self.path.open("ab") as f:
                f.write(b"\n")

    def _append(self, rows: list[tuple[Item, Verdict]]) -> None:
        # serialise everything before touching the file; default=str keeps non-JSON raw values writable
        records = []
        for item, verdict in rows:
            sha = text_sha(item.text)
            line = json.dumps({"item_id": item.id, "text_sha": sha, "verdict": asdict(verdict)},
                              ensure_ascii=False, default=str) + "\n"
            records.append((item.id, sha, verdict, line))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._repair_unterminated_tail()
        with self.path.open("a", encoding="utf-8") as f:
            for item_id, sha, verdict, line in records:
                f.write(line)
                f.flush()
                self._entries[item_id] = (sha, verdict)  # memory moves only after the write succeeded
