from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from nar_guardrails.datasets.fetch import sample
from nar_guardrails.schema import Item, text_sha

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


def _dedupe(rows, text_of):
    """Drop rows whose text hash was already seen; return [(row, text, hash)] in order."""
    seen, out = set(), []
    for r in rows:
        text = text_of(r)
        h = text_sha(text)[:12]
        if h in seen:
            continue
        seen.add(h)
        out.append((r, text, h))
    return out


# ---------------- neuralchemy/Prompt-injection-dataset (core, test) ----------------
_NC_META = {"license": "Apache-2.0", "published": "2026-01"}
_NC_OLD = re.compile(r"harmbench|wildguard|hackaprompt|judge", re.I)
_NC_SCENARIO = {"direct_injection": "D1", "system_extraction": "D2", "prompt_leaking": "D2", "indirect_injection": "I1",
                "jailbreak": "J1", "persona_replacement": "J1", "many_shot": "J1", "crescendo": "J1",
                "encoding_obfuscation": "J2", "token_smuggling": "J2", "context_overflow": "J2"}


def load_neuralchemy(rows, cap, seed) -> list[Item]:
    kept = [r for r in rows if not _NC_OLD.search(str(r.get("source", "")))]
    items = []
    for r, text, h in sample(_dedupe(kept, lambda r: r["text"]), cap, seed):
        label = int(r["label"])
        if label == 0:
            scenario = "B2" if any("hard" in str(t).lower() for t in (r.get("tags") or [])) else "B1"
        else:
            scenario = _NC_SCENARIO.get(str(r.get("category")), "unknown")
        items.append(Item(id=f"neuralchemy:{h}", track="prompts", text=text, label=label, scenario=scenario,
                          source="neuralchemy", split="test", meta={**_NC_META, "category": r.get("category"), "orig_source": r.get("source")}))
    return _ensure_unique_ids(items, "neuralchemy")


# ---------------- prodnull/prompt-injection-repo-dataset ----------------
_PN_META = {"license": "Apache-2.0", "published": "2026-03", "gated": True}


def load_prodnull(rows, cap, seed) -> list[Item]:
    ded = _dedupe(rows, lambda r: r["text"])
    pos = [t for t in ded if int(t[0]["label"]) == 1]
    neg = [t for t in ded if int(t[0]["label"]) == 0]
    half = None if cap is None else cap // 2
    chosen = sample(pos, half, seed) + sample(neg, None if cap is None else cap - (half or 0), seed + 1)
    return _ensure_unique_ids([Item(id=f"prodnull:{h}", track="prompts", text=text, label=int(r["label"]),
                                    scenario="I5" if int(r["label"]) else "B3", source="prodnull", split="test", meta=dict(_PN_META))
                               for r, text, h in chosen], "prodnull")


# ---------------- sureheremarv/ipi_arena_attacks ----------------
_IPI_META = {"license": "MIT", "published": "2026-03"}


def load_ipi_arena(rows, cap, seed) -> list[Item]:
    return _ensure_unique_ids([Item(id=f"ipi_arena:{h}", track="prompts", text=text, label=1, scenario="I1",
                                    source="ipi_arena", split="test", meta={**_IPI_META, "behavior": r["behavior_id"]})
                               for r, text, h in sample(_dedupe(rows, lambda r: r["attack"]), cap, seed)], "ipi_arena")


# ---------------- centrepourlasecuriteia/jailbreak-dataset (BELLS) ----------------
_BELLS_META = {"license": "CC-BY-NC-SA-4.0", "published": "2026-02", "gated": True}
_BELLS_J1 = ("cognitive", "psycholog", "few_shot", "fsh", "distract", "dap")


def _bells_scenario(r) -> str:
    if str(r.get("category", "")).lower() == "benign":
        return "B2"
    tt = str(r.get("technique_type", "")).lower()
    return "J1" if any(k in tt for k in _BELLS_J1) else "J2"


def load_bells(rows, cap, seed) -> list[Item]:
    """Harmful and benign-origin rows are each sampled to at most `cap`.

    The dataset `id` is the hash of the base prompt shared by its transformed variants,
    so item ids are `bells:<id>:<technique>` (plus `:<row n>` if that still collides).
    """
    rows = list(rows)
    benign = [r for r in rows if str(r.get("category", "")).lower() == "benign"]
    harmful = [r for r in rows if str(r.get("category", "")).lower() != "benign"]
    chosen = sample(harmful, cap, seed) + sample(benign, cap, seed + 1)
    items, seen = [], set()
    for n, r in enumerate(chosen):
        iid = f"bells:{r['id']}:{r.get('technique') or n}"
        if iid in seen:
            iid = f"{iid}:{n}"
        seen.add(iid)
        benign_row = str(r.get("category", "")).lower() == "benign"
        items.append(Item(id=iid, track="prompts", text=r["jailbreak_prompt"], label=0 if benign_row else 1,
                          scenario=_bells_scenario(r), source="bells", split="test",
                          meta={**_BELLS_META, "technique": r.get("technique"), "technique_type": r.get("technique_type"), "category": r.get("category")}))
    return _ensure_unique_ids(items, "bells")


# ---------------- microsoft/llmail-inject-challenge (Phase2) ----------------
_LLMAIL_META = {"license": "MIT", "published": "2025-05"}


def _llmail_success(output) -> bool:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return False
    if not isinstance(output, dict):
        return False
    return bool(output.get("defense.undetected")) or bool(output.get("exfil.sent"))


def load_llmail(rows, cap, seed) -> list[Item]:
    good = [r for r in rows if _llmail_success(r.get("output"))]
    ded = _dedupe(good, lambda r: f"Email subject: {r.get('subject', '')}\n\n{r.get('body', '')}")
    return _ensure_unique_ids([Item(id=f"llmail:{h}", track="prompts", text=text, label=1, scenario="I2", source="llmail",
                                    split="test", meta={**_LLMAIL_META, "scenario": r.get("scenario"), "level": r.get("level")})
                               for r, text, h in sample(ded, cap, seed)], "llmail")


# ---------------- hackaprompt/Pliny_HackAPrompt_Dataset ----------------
_PLINY_META = {"license": "CC-BY-4.0", "published": "2025-07", "gated": True}
_PLINY_IMAGE = {"pliny_6", "pliny_7", "pliny_8"}
_PLINY_EVASION = {"pliny_1", "pliny_9", "pliny_10", "pliny_11", "pliny_12"}


def load_pliny(rows, cap, seed) -> list[Item]:
    good = []
    for r in rows:
        slug = str(r.get("challenge_slug", "")).replace("_challenge", "")
        if slug in _PLINY_IMAGE or not r.get("passed"):
            continue
        user_turns = [m["content"] for m in (r.get("messages") or []) if m.get("role") == "user" and isinstance(m.get("content"), str)]
        if user_turns:
            good.append((slug, "\n\n".join(user_turns), r.get("submission_id")))
    items = []
    for slug, text, sid in sample(good, cap, seed):
        evasion = slug in _PLINY_EVASION
        items.append(Item(id=f"pliny:{sid}", track="prompts", text=text, label=1, scenario="J2" if evasion else "J1",
                          source="pliny", split="test",
                          meta={**_PLINY_META, "challenge": slug, "slice": "promptguard-evasion" if slug in ("pliny_9", "pliny_10") else slug}))
    return _ensure_unique_ids(items, "pliny")


# ---------------- Simsonsun/JailbreakPrompts ----------------
_SIM_META = {"license": "MIT", "published": "2025-06"}


def load_simsonsun(rows, cap, seed) -> list[Item]:
    return _ensure_unique_ids([Item(id=f"simsonsun:{h}", track="prompts", text=text, label=1, scenario="J1", source="simsonsun",
                                    split="test", meta={**_SIM_META, "orig_source": r.get("Source")})
                               for r, text, h in sample(_dedupe(rows, lambda r: r["Prompt"]), cap, seed)], "simsonsun")


# ---------------- Mindgard/evaded-prompt-injection-and-jailbreak-samples ----------------
_MG_META = {"license": "CC-BY-NC-4.0", "published": "2025-04", "gated": True}
_MG_EXACT = ("modified_prompt", "modified", "transformed_prompt", "evaded_prompt", "adversarial_prompt", "attack_prompt")


def _mg_column(r: dict) -> str:
    for k in _MG_EXACT:
        if k in r:
            return k
    for k in r:
        if "original" not in k.lower() and re.search(r"modif|transform|alter|evad", k, re.I):
            return k
    raise KeyError(f"mindgard row has no modified-prompt column; keys: {sorted(r)}")


def load_mindgard(rows, cap, seed) -> list[Item]:
    prepared = [(r, str(r[_mg_column(r)])) for r in rows]
    items = []
    for (r, _), text, h in _dedupe(prepared, lambda p: p[1]):
        name = str(r.get("attack_name") or "")
        encoded = any(k in name.lower() for k in ("emoji", "base64"))
        meta = {**_MG_META, "attack_name": r.get("attack_name")}
        if encoded:
            meta["encoded"] = True
        items.append(Item(id=f"mindgard:{h}", track="prompts", text=text, label=1, scenario="J2", source="mindgard",
                          split="test", meta=meta))
    return _ensure_unique_ids(sample(items, cap, seed), "mindgard")


# ---------------- allenai/WildChat-4.8M (benign user turns) ----------------
_WC_META = {"license": "ODC-BY", "published": "2025-08"}


def load_wildchat(rows, cap, seed) -> list[Item]:
    """Benign first user turns. The candidate scan stops at max(cap or 1000, 1000) * 5 rows,
    so when `rows` is a stream the sample is drawn from the head of the stream."""
    limit = max(cap or 1000, 1000) * 5
    good = []
    for r in rows:
        if str(r.get("language", "")).lower() != "english" or r.get("toxic"):
            continue
        conv = r.get("conversation") or []
        first = next((m.get("content") for m in conv if m.get("role") == "user"), None)
        if isinstance(first, str) and 20 <= len(first) <= 2000:
            good.append(first)
        if len(good) >= limit:
            break
    return _ensure_unique_ids([Item(id=f"wildchat:{h}", track="prompts", text=t, label=0, scenario="B1", source="wildchat",
                                    split="test", meta=dict(_WC_META))
                               for _r, t, h in sample(_dedupe(good, lambda t: t), cap, seed)], "wildchat")


HF_SOURCES = {
    "neuralchemy": {"dataset": "neuralchemy/Prompt-injection-dataset", "config": "core", "split": "test", "streaming": False,
                    "cap": 500, "published": "2026-01", "license": "Apache-2.0", "gated": False, "loader": load_neuralchemy},
    "prodnull": {"dataset": "prodnull/prompt-injection-repo-dataset", "config": None, "split": "train", "streaming": False,
                 "cap": 500, "published": "2026-03", "license": "Apache-2.0", "gated": True, "loader": load_prodnull},
    "ipi_arena": {"dataset": "sureheremarv/ipi_arena_attacks", "config": None, "split": "train", "streaming": False,
                  "cap": None, "published": "2026-03", "license": "MIT", "gated": False, "loader": load_ipi_arena},
    "bells": {"dataset": "centrepourlasecuriteia/jailbreak-dataset", "config": None, "split": "train", "streaming": False,
              "cap": 500, "published": "2026-02", "license": "CC-BY-NC-SA-4.0", "gated": True, "loader": load_bells},
    "llmail": {"dataset": "microsoft/llmail-inject-challenge", "config": None, "split": "Phase2", "streaming": True,
               "cap": 500, "published": "2025-05", "license": "MIT", "gated": False, "loader": load_llmail},
    "pliny": {"dataset": "hackaprompt/Pliny_HackAPrompt_Dataset", "config": None, "split": "train", "streaming": False,
              "cap": 500, "published": "2025-07", "license": "CC-BY-4.0", "gated": True, "loader": load_pliny},
    "simsonsun": {"dataset": "Simsonsun/JailbreakPrompts", "config": None, "split": "Dataset_2", "streaming": False,
                  "cap": 500, "published": "2025-06", "license": "MIT", "gated": False, "loader": load_simsonsun},
    "mindgard": {"dataset": "Mindgard/evaded-prompt-injection-and-jailbreak-samples", "config": None, "split": "train", "streaming": False,
                 "cap": 500, "published": "2025-04", "license": "CC-BY-NC-4.0", "gated": True, "loader": load_mindgard},
    "wildchat": {"dataset": "allenai/WildChat-4.8M", "config": None, "split": "train", "streaming": True,
                 "cap": 1000, "published": "2025-08", "license": "ODC-BY", "gated": False, "loader": load_wildchat},
}
