from __future__ import annotations

import csv
import json
from pathlib import Path

from nar_guardrails.datasets.fetch import sample
from nar_guardrails.schema import Item

TEXT_COLS = ("text", "prompt", "input", "input_text", "content")
LABEL_COLS = ("label", "is_injection", "is_malicious", "y", "target")
POSITIVE_WORDS = {"1", "injection", "malicious", "attack", "jailbreak", "unsafe", "true"}


def _label(v) -> int:
    s = str(v).strip().lower()
    try:
        f = float(s)
    except ValueError:
        pass
    else:
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
        raise ValueError(f"unrecognised label value {v!r}")
    if s in POSITIVE_WORDS:
        return 1
    if s in {"0", "benign", "safe", "clean", "false"}:
        return 0
    raise ValueError(f"unrecognised label value {v!r}")


def _ensure_unique_ids(items: list[Item], source: str) -> list[Item]:
    seen, dups = set(), []
    for i in items:
        if i.id in seen and i.id not in dups:
            dups.append(i.id)
        seen.add(i.id)
    if dups:
        raise ValueError(f"{source}: duplicate item ids {dups}")
    return items


def _pick(row: dict, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"none of {candidates} in columns {sorted(row)}")


# ---------------- AgentShield (github.com/doronp/agentshield-benchmark) ----------------
_AS_META = {"license": "Apache-2.0", "published": "2026-03"}
_AS_SCENARIO = {"prompt-injection": "D1", "jailbreak": "J1", "data-exfiltration": "I2", "tool-abuse": "I2",
                "multi-agent": "I4", "over-refusal": "B2", "provenance-audit": "I4"}


def load_agentshield(repo: Path) -> list[Item]:
    items = []
    for f in sorted((Path(repo) / "corpus").glob("*/tests.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            label = 0 if str(d.get("expected_behavior", "")).lower() == "allow" else 1
            scenario = _AS_SCENARIO.get(d["category"], "unknown")
            if d["category"] == "prompt-injection" and "system" in str(d.get("subcategory", "")).lower():
                scenario = "D2"
            if label == 0 and scenario not in ("B2",):
                scenario = "B3"
            items.append(Item(id=f"agentshield:{d['id']}", track="prompts", text=d["input_text"], label=label,
                              scenario=scenario, source="agentshield", split="test",
                              meta={**_AS_META, "subcategory": d.get("subcategory"), "severity": d.get("severity")}))
    return _ensure_unique_ids(items, "agentshield")


# ---------------- PIDS-Bench eval subsets (github.com/ShirePyDev/Prompt-Injection-Detection-System) ----------------
_PIDS_META = {"license": "MIT", "published": "2026-09"}
_PIDS_FILES = {"eval_subsets/hard_benign_test.csv": "B2", "eval_subsets/obfuscated_attacks.csv": "J2",
               "ood/domain_ood.csv": "shift-domain", "ood/structural_ood.csv": "shift-structural"}


def load_pids_bench(repo: Path) -> list[Item]:
    items = []
    base = Path(repo) / "data" / "pids_bench_v3"
    for rel, scenario in _PIDS_FILES.items():
        stem = Path(rel).stem
        with (base / rel).open(encoding="utf-8-sig", newline="") as f:
            for n, row in enumerate(csv.DictReader(f)):
                text = row[_pick(row, TEXT_COLS)]
                label = _label(row[_pick(row, LABEL_COLS)])
                if scenario == "B2" and label != 0:
                    raise ValueError(f"hard_benign_test row {n} is not benign")
                items.append(Item(id=f"pids_bench:{stem}:{n}", track="prompts", text=text, label=label,
                                  scenario=scenario, source="pids_bench", split="test",
                                  meta={**_PIDS_META, "file": stem, "domain": row.get("domain")}))
    return _ensure_unique_ids(items, "pids_bench")


# ---------------- mcp-guardbench (github.com/getmcpm/mcp-guardbench) ----------------
_MGB_META = {"license": "MIT", "published": "2026-10"}


_MGB_BUCKETS = ("attacks", "benign", "warn")


def load_mcp_guardbench(repo: Path) -> list[Item]:
    items = []
    files = [f for b in _MGB_BUCKETS for f in sorted((Path(repo) / "cases" / b).glob("*.json"))]
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "id" not in d or "message" not in d:
            raise ValueError(f"{f}: case lacks 'id' or 'message'")
        bucket = d.get("bucket") or f.parent.name
        if bucket not in _MGB_BUCKETS:
            raise ValueError(f"{f}: unknown bucket {bucket!r}")
        label = 0 if bucket == "benign" else 1
        text = f"MCP frame ({d.get('category', 'unknown')}):\n" + json.dumps(d["message"], ensure_ascii=False)
        items.append(Item(id=f"mcp_guardbench:{d['id']}", track="prompts", text=text, label=label,
                          scenario="I3" if label else "B3", source="mcp_guardbench", split="test",
                          meta={**_MGB_META, "bucket": bucket, "expected": d.get("expected"), "case_source": d.get("source")}))
    return _ensure_unique_ids(items, "mcp_guardbench")


# ---------------- MCPTox (github.com/zhiqiangwang4/MCPTox-Benchmark) ----------------
_MCPTOX_META = {"license": "none-stated-cite-only", "published": "2025-08"}


def load_mcptox(repo: Path, cap: int = 500, seed: int = 0) -> list[Item]:
    raw = json.loads((Path(repo) / "pure_tool.json").read_text(encoding="utf-8"))
    rows = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        for key, d in entry.items():
            if isinstance(d, dict):
                rows.append((key, d))
    if not rows:
        raise ValueError("mcptox: no usable entries in pure_tool.json")
    items = []
    for key, d in sample(rows, cap, seed):
        g = {k: str(d.get(k) or "") for k in ("tool_name", "server_name", "tool_content", "query")}
        text = (f"MCP tool description for `{g['tool_name']}` (server: {g['server_name']}):\n"
                f"{g['tool_content'].strip()}\n\nUser query: {g['query'].strip()}")
        items.append(Item(id=f"mcptox:{key}", track="prompts", text=text, label=1, scenario="I3",
                          source="mcptox", split="test", meta={**_MCPTOX_META, "server": g["server_name"]}))
    return _ensure_unique_ids(items, "mcptox")


GIT_SOURCES = {
    "agentshield": {"url": "https://github.com/doronp/agentshield-benchmark", "ref": "main",
                    "published": "2026-03", "license": "Apache-2.0", "loader": load_agentshield},
    "pids_bench": {"url": "https://github.com/ShirePyDev/Prompt-Injection-Detection-System", "ref": "v1.0-pids-bench",
                   "published": "2026-09", "license": "MIT", "loader": load_pids_bench},
    "mcp_guardbench": {"url": "https://github.com/getmcpm/mcp-guardbench", "ref": "main",
                       "published": "2026-10", "license": "MIT", "loader": load_mcp_guardbench},
    "mcptox": {"url": "https://github.com/zhiqiangwang4/MCPTox-Benchmark", "ref": "main",
               "published": "2025-08", "license": "none-stated-cite-only", "loader": load_mcptox},
}
