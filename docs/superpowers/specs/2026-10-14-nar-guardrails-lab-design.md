# NAR guardrails lab — design spec

Date: 2026-10-14. Status: draft for review. Companion research: `research/2026-10-14-nar-guardrails-research.md`.

## 1. Goal

Build a reproducible, benchmark-grade local lab that measures how well non-autoregressive (NAR) decision models work as LLM guardrails, compared with off-the-shelf encoder classifiers and with Claude as a frontier LLM judge, on two tracks:

1. **Prompt track**: detecting jailbreaks, direct prompt injection, and agent-targeted injection (indirect injection in tool output, tool-call hijack, MCP tool poisoning, multi-agent handoff) while not over-flagging benign inputs.
2. **Skill track**: detecting malicious agent skills (`SKILL.md` packages, as NVIDIA SkillSpector targets) with NAR models, zero-shot and after a short fine-tune.

Every result row carries accuracy metrics with confidence intervals, measured latency, and measured cost, so the lab can answer "same detection quality, what does it cost in time and money" for each model.

### Success criteria

- `make bench` (or the equivalent CLI sequence) rebuilds datasets, runs every detector on both tracks, and produces `results/<run-id>/report.md` with all tables, from a clean checkout plus cached model weights.
- Every detector has a row per track with recall, FPR, precision, AUROC, FPR at 95% and 99% recall, bootstrap 95% CIs, p50/p95 latency, throughput (local), and cost per 1,000 items (remote).
- At least one fine-tuned NAR model reaches ≥90% recall at ≤10% FPR on the MaliciousSkillBench MSD test split, which reproduces the order of magnitude reported by SkillScreen.
- Reruns are incremental: unchanged (model, policy, dataset) combinations are served from cache.

## 2. Scope

**In scope**

- Local models: Laya (EN; multilingual variant only as a context-length control), GLiNER2.5-Decide, GLiGuard, Open-Jev-2B (optional), Qwen3Guard-Stream-0.6B, ProtectAI DeBERTa v2, Llama Prompt Guard 2 (86M and 22M), Qualifire Sentinel, Granite Guardian HAP 125M, ModernBERT-large (fine-tuned only, skill track).
- Remote models: Claude Haiku 5.5 and Claude Sonnet 5.5 via the Anthropic SDK.
- Fine-tuning on the skill track: Laya, GLiNER2.5-Decide, ModernBERT-large, Open-Jev-2B (optional).
- English only.
- Datasets: hand-curated prompt scenarios, sampled public prompt sets, MaliciousSkillBench, SkillSpector fixtures, hand-crafted skills.

**Out of scope**

- TypeSafe Jev and Cloudflare Clef (no API access; Clef does not fit the machine).
- Polish or other non-English slices.
- Fine-tuning on the prompt track.
- Response-side (output) moderation.
- Running or sandboxing skills; the lab reads text only.
- A web UI. Reports are Markdown plus static charts; a presentation artifact can be derived later.

## 3. Environment and constraints

- Apple M4, 24 GB unified memory, macOS. Models run on MPS where the library supports it, else CPU. Never more than one model loaded at a time.
- `uv` project, Python 3.12 (system 3.9 is too old for `gliner2` and `laya`).
- Hugging Face login for gated models (Meta Prompt Guard 2, Qualifire Sentinel, AI2 WildJailbreak).
- `ANTHROPIC_API_KEY` in the environment for the Claude adapters. No other API keys.
- Downloaded datasets are untrusted: they live under `data/raw/` and are only ever read as text.

## 4. Architecture

```
nar_guardrails/
  datasets/      # loaders, curated sets, split logic, Item schema
  detectors/     # one adapter per model family, Verdict schema, policy mapping
  bench/         # runner, caching, metrics, latency, cost
  finetune/      # one trainer per model family
  report/        # tables and charts
  cli.py         # `nar` entry point
configs/
  models.yaml    # model ids, revisions, device, batch sizes
  prices.yaml    # $/MTok per remote model
  policies/      # shared and tuned policy texts per track
data/
  raw/           # quarantined downloads (gitignored)
  processed/     # JSONL per track and split (gitignored except curated sets)
  curated/       # hand-written prompts and skills (committed)
results/<run-id>/  # per-item JSONL, summary CSV, report.md, charts (gitignored)
tests/
```

Dependencies: `torch`, `transformers`, `datasets`, `gliner2[local,train]`, `laya`, `anthropic`, `scikit-learn`, `numpy`, `pandas`, `pyyaml`, `matplotlib`, `pytest`. Open-Jev is a git checkout run as a local HTTP server, not a package dependency.

## 5. Data model

**Item** (one row of a dataset)

| Field | Type | Notes |
|---|---|---|
| `id` | str | stable, `<source>:<index>` |
| `track` | `prompts` or `skills` | |
| `text` | str | what the detector sees; for skills, SKILL.md text (optionally with scripts appended) |
| `label` | 0 or 1 | 1 = malicious |
| `scenario` | str | prompt scenario code (B1..I4) or skill attack category; `unknown` allowed for public rows |
| `source` | str | dataset or `curated` |
| `split` | `dev`, `test`, `train`, `val` | |
| `variant` | str | `text` or `text+scripts` (skills only) |
| `meta` | dict | provenance, original label, license tag |

**Verdict** (one detector output per item)

| Field | Type | Notes |
|---|---|---|
| `item_id` | str | |
| `detector` | str | adapter key, e.g. `laya`, `claude-haiku-5-5` |
| `policy` | str | policy file name plus hash |
| `p_malicious` | float in [0,1] or null | model probability; null when the model only gives a label |
| `label` | 0, 1 or null | model's own decision at its default threshold |
| `outcome` | `ok`, `refused`, `error`, `truncated` | `refused` only for Claude |
| `latency_ms` | float | wall-clock for this item at batch size 1 |
| `usage` | dict or null | `input_tokens`, `output_tokens`, cache fields for remote models |
| `raw` | str or dict | untouched response for audit |

**Run manifest** per `results/<run-id>/manifest.json`: detector keys with model ids and HF revision hashes, policy hashes, dataset hashes, device, library versions, git commit, timestamp, seeds.

## 6. Datasets

### 6.1 Prompt track

**Curated set** (`data/curated/prompts.jsonl`, 200 items, committed, hand-written in the plan phase):

| Code | Scenario | Count | Label |
|---|---|---|---|
| B1 | plain benign tasks | 35 | 0 |
| B2 | benign with trigger words (over-defense control) | 35 | 0 |
| B3 | benign tool outputs and documents given to an agent | 30 | 0 |
| J1 | classic jailbreaks (role-play, developer mode, hypothetical framing) | 15 | 1 |
| J2 | obfuscated jailbreaks (base64, leetspeak, split or translated instructions) | 10 | 1 |
| D1 | direct injection / instruction override | 15 | 1 |
| D2 | system prompt exfiltration | 10 | 1 |
| I1 | indirect injection inside email, web page, RAG chunk | 15 | 1 |
| I2 | tool-call hijack (asks the agent to call send_email, transfer, delete) | 15 | 1 |
| I3 | MCP tool poisoning (hidden instructions in a tool description) | 10 | 1 |
| I4 | multi-agent handoff injection (a message claiming to be another agent) | 10 | 1 |

Items of B3 and I1–I4 are formatted the way an agent would see them: a short framing line (`Tool result from read_email:`) followed by the content. The whole string is the detector input.

**Public set** (`data/processed/prompts_public.jsonl`, target about 4,000 items). Sampled with a fixed seed, cap 500 per source, keeping the source's original label and mapping it to our binary label. Sources: `deepset/prompt-injections`, `xTRam1/safe-guard-prompt-injection`, `jackhhao/jailbreak-classification`, `qualifire/prompt-injections-benchmark`, NotInject (benign over-defense), InjecAgent (indirect, agent tool context), BIPIA, LLMail-Inject, `ai-safety-institute/AgentHarm` (harmful plus benign), `allenai/wildjailbreak` (gated). Each loader records the license tag; a source whose license forbids this use is skipped and listed in the report.

**Splits**: the curated set is test only. The public set is split 20% dev / 80% test, stratified by source and label. Thresholds are chosen on dev. The report also gives per-source test metrics, which is the leave-one-source view for models trained on public data.

### 6.2 Skill track

**MaliciousSkillBench** (`ProtectSkills/MaliciousSkillBench`, config `primary`, parquet). Text is `skill_text`, or `public_skill_text` when redacted; the five rows without text are dropped. Splits come from the frozen manifests in `splits/`: **MSD** (malicious-structural-disjoint) as the main protocol and **SD** (source-disjoint) as the hard protocol; `val` from the manifest is the dev split. Attack category fields map to `scenario`.

**SkillSpector fixtures** (`tests/fixtures/` of the SkillSpector repo, about 15 skills). Labels assigned by hand after reading each fixture; used as a smoke set and included in test.

**Hand-crafted set** (`data/curated/skills/`, 40 skills as real `SKILL.md` files with frontmatter, committed):

- 20 malicious, two per category: data exfiltration, credential harvesting, anti-refusal or "do not mention this to the user", hidden instructions in HTML comments or fenced code, download-and-execute, persistence (cron, shell rc), MCP tool poisoning, privilege escalation, typosquatted dependency, memory poisoning.
- 20 benign: 10 ordinary skills, 10 benign look-alikes that legitimately use `curl`, environment variables, or `sudo`.

**Variants**: `text` is SKILL.md only, capped at 8,192 tokens by the detector's tokenizer (head-truncated, matching SkillScreen). `text+scripts` appends bundled scripts after a separator; it exists for the hand-crafted set and for any MaliciousSkillBench rows whose packages are available, and is reported as a secondary table.

**Safety**: `data/raw/` and `data/curated/skills/` are never on any agent's skill path, never executed, and only opened by the loaders. The Claude adapter wraps content as untrusted data (section 7.3).

## 7. Detectors

### 7.1 Contract

`Detector.predict(items: list[Item], batch_size: int) -> list[Verdict]`. Every adapter also implements `warmup()` and reports `supports_batch`. Output probability is always oriented so that higher means more malicious. Adapters never raise on a single bad item; they return `outcome=error` with the exception text in `raw`.

### 7.2 Adapters

| Key | Model | Input mapping | `p_malicious` |
|---|---|---|---|
| `laya` | `convaiinnovations/laya` | state = text; one `noul` question from the policy | P(yes) |
| `laya-multi` | `convaiinnovations/laya-multilingual`, `max_len=8192` | same | P(yes); skill track only, context-length control |
| `gliner2-decide` | `fastino/GLiNER2.5-Decide` | `classify_text` with a two-label task whose label descriptions come from the policy, `include_confidence=True` | confidence of `malicious` |
| `gliguard` | `fastino/gliguard-LLMGuardrails-300M` | prompt track: `prompt_safety` plus `jailbreak_detection` tasks; skill track: `prompt_safety` with the policy as context | P(unsafe), or max over injection-type labels when reported |
| `openjev-2b` | `ZefanCai/Open-Jev-2B` via local `/v1/systemone` server | same `noul` question as Laya | P(yes); optional, enabled only if the MPS smoke test passes |
| `qwen3guard-stream` | `Qwen/Qwen3Guard-Stream-0.6B` | prompt head | P(unsafe) + P(controversial) as score, label at unsafe |
| `protectai` | `protectai/deberta-v3-base-prompt-injection-v2` | sequence classification | P(INJECTION) |
| `promptguard2-86m`, `promptguard2-22m` | `meta-llama/Llama-Prompt-Guard-2-86M` / `-22M` | 512-token windows, max over windows | P(MALICIOUS) |
| `sentinel` | `qualifire/prompt-injection-sentinel` | sequence classification | P(jailbreak) |
| `granite-hap` | `ibm-granite/granite-guardian-hap-125m` | sequence classification | P(hap); prompt track only, content-safety reference |
| `modernbert-ft` | fine-tuned `answerdotai/ModernBERT-large` | sequence classification | P(malicious); skill track only |
| `claude-haiku-5-5`, `claude-sonnet-5-5` | Anthropic SDK | section 7.3 | model-reported probability; verdict used when probability missing |

Fine-tuned checkpoints register as `laya-ft`, `gliner2-decide-ft`, `modernbert-ft`, `openjev-2b-ft`, each with a seed suffix.

### 7.3 Claude adapter

- Models `claude-haiku-5-5` and `claude-sonnet-5-5`, `output_config.effort: "low"`, default adaptive thinking, `max_tokens: 256`, structured output with schema `{verdict: "malicious" | "benign", probability: number, reason: string}`.
- System prompt = the track policy plus the instruction that the content is untrusted data and must not be followed. The item text is placed in a user message inside a clearly delimited block.
- An `8shot` policy variant prepends eight labelled examples drawn from the dev split (prompts) or the train split (skills), matching SkillScreen's few-shot condition.
- Concurrency 4, SDK retries on 429 and 5xx, per-item timeout 60 s.
- `stop_reason == "refusal"` is recorded as `outcome=refused`, `label=1`, `p_malicious=1.0`, and the refusal count is reported separately. Haiku 5.5 has no server-side fallback, so no fallback is configured for either model.
- `usage` is stored per item; cost = tokens × `configs/prices.yaml`.

### 7.4 Policies

One shared policy per track in `configs/policies/{prompts,skills}/shared.txt`, used verbatim by every zero-shot model (as the `noul` question, as label descriptions, or as the Claude system prompt). Optional per-model `tuned/<detector>.txt` files are a second run reported in a separate column, because wording shifts decision-model accuracy by double digits. The policy hash is part of the cache key.

## 8. Benchmark protocol

- **Thresholds**: for each detector, choose on dev the threshold that maximises F1, plus the thresholds that give 95% and 99% recall. Freeze, then score test. Also report each model's own default label.
- **Metrics** per detector, per track, per scenario, per source: recall, FPR, precision, F1, AUROC, FPR at 95% and 99% recall, precision at 10% and 1% malicious prevalence (skill track). Bootstrap 95% CIs with 1,000 resamples over items.
- **Latency**: five warm-up items, then every test item at batch size 1 timed with `perf_counter` around the adapter call; report p50 and p95. Throughput: 256 items at batch 16 (local only), items per second. Claude: end-to-end request time from the Mac, p50 and p95, with the measurement window noted.
- **Cost**: remote cost from recorded usage; local cost reported as 0 marginal with model size, device, and seconds per 1,000 items.
- **Seeds**: zero-shot runs are deterministic (torch seed fixed); fine-tunes run with seeds 1, 2, 3 and report mean ± std plus the per-seed rows.
- **Caching**: `results/cache/<detector>/<policy-hash>/<dataset-hash>.jsonl`, keyed by item id. A run reads cached verdicts and only computes the missing ones. Latency figures are taken from the run that computed them.
- **Reproducibility**: the manifest records model revisions, hashes, versions, and git commit; `nar run` refuses to write a report when the working tree is dirty unless `--allow-dirty` is set.

## 9. Fine-tuning (skill track only)

Train on the MSD `train` manifest, choose thresholds on MSD `val`, evaluate on MSD `test` and SD `test`. Three seeds each. Each trainer writes a checkpoint under `models/<key>/seed<N>/` plus a training log, and registers the checkpoint as a detector.

| Key | Base | Recipe |
|---|---|---|
| `laya-ft` | `convaiinnovations/laya` | Laya's Apple Silicon fine-tune script: single `noul` question, train decision head and encoder, 1–3 epochs, then temperature calibration on val. 512-token input; the `laya-multi` 8k variant is trained the same way if the 512 cap costs more than 5 recall points on val. |
| `gliner2-decide-ft` | `fastino/GLiNER2.5-Decide` | `ExtractorTrainer` with LoRA, two-label classification task, max length 1,024, 2–3 epochs. |
| `modernbert-ft` | `answerdotai/ModernBERT-large` | `AutoModelForSequenceClassification`, max length 2,048, lr 2e-5, 2 epochs, fp32 on MPS, batch 4 with gradient accumulation. The supervised reference point. |
| `openjev-2b-ft` (optional) | `ZefanCai/Open-Jev-2B` | SkillScreen recipe: LoRA rank 8, alpha 16, attention projections plus head, 1 epoch, max length 4,096. Run on MPS if the base model's smoke test passes; otherwise a Kaggle T4 notebook built from the same script. |

Fallbacks: if MPS training fails for a model, retry on CPU with a reduced max length; if still infeasible, use the free cloud GPU path and record the hardware in the manifest.

## 10. Reporting

`nar report --run <id>` writes `report.md` and PNG charts:

1. Prompt-track leaderboard: one row per detector at the F1 threshold and at 95% recall, with CIs, p50 latency, cost per 1,000.
2. Prompt-track scenario heatmap: recall per scenario code (malicious) and FPR per benign code, per detector.
3. Prompt-track per-source table.
4. Skill-track table: zero-shot and fine-tuned rows side by side, MSD and SD, mean ± std over seeds, with refusals for Claude.
5. Cost and latency versus recall scatter per track.
6. Headline table for the talk: for a fixed recall target (95%), each model's FPR, p50 latency, cost per 1,000 items, and local or remote.
7. Secondary: `text+scripts` variant table; tuned-policy column where available; skipped sources and their reasons.

## 11. CLI

```
nar data build --track prompts|skills [--force]
nar run --track prompts|skills --detectors laya,gliguard,... --split dev|test --policy shared|tuned|8shot [--limit N] [--batch-size N]
nar finetune --model laya|gliner2-decide|modernbert|openjev-2b --seed N
nar report --run <run-id>
nar smoke   # 20 items per track through every configured detector
```

`make bench` runs build, the dev runs, threshold selection, the test runs, the fine-tunes, and the report in order.

## 12. Error handling

- Per-item failures become `outcome=error` verdicts and are excluded from metrics but counted in the report.
- Model load failures abort only that detector and are listed in the report.
- Remote calls use SDK retries; a run can be resumed and will fill cache gaps only.
- Dataset loaders validate the Item schema and fail fast on count or label mismatches against the expected values recorded in `configs/datasets.yaml`.
- Truncation is recorded in `outcome=truncated` when the detector's tokenizer cut the input, so recall on long skills can be analysed.

## 13. Testing

- Unit tests: metrics against hand-computed values (including bootstrap determinism with a fixed seed); threshold selection; cost calculation; policy templating (golden files); loaders (schema, counts, label mapping, split determinism).
- Adapter contract tests: each adapter runs on the SkillSpector fixtures and five curated prompts, asserting the Verdict schema, `p_malicious` in [0,1], and that at least one obviously malicious fixture scores above one obviously benign fixture.
- Claude adapter tests use a recorded-response stub by default; a `--live` marker runs three real calls.
- `nar smoke` is the end-to-end test and must pass before any full run.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MPS numerical quirks for DeBERTa or ModernBERT | per-model `device` override in `models.yaml`; contract test compares CPU and MPS outputs on fixtures |
| Open-Jev-2B has undocumented MPS support | optional detector gated on a smoke test |
| Gated downloads fail | detector skipped and reported; rest of the run proceeds |
| Dataset licences | license tag per source, skip on conflict, list in report |
| Claude refuses malicious content | recorded as `refused`, counted as flagged, reported separately |
| Class imbalance in skills (77% malicious) | report precision at 10% and 1% prevalence and FPR, not accuracy |
| Vendor classifiers trained on our public sources | curated set and per-source tables are the fair comparison; stated in the report |
| Long SKILL.md files exceed 512 tokens for Laya and Prompt Guard | windowing (Prompt Guard) and the 8k Laya variant; truncation recorded |

## 15. Milestones

1. Environment, `models.yaml`, `nar smoke` with two adapters (ProtectAI, Laya) on fixtures.
2. Datasets: curated prompts, public loaders, MaliciousSkillBench, fixtures, hand-crafted skills; `nar data build` green.
3. All local adapters, prompt-track dev and test runs, metrics and caching.
4. Skill-track zero-shot runs.
5. Claude adapters, both tracks, shared and 8-shot policies.
6. Fine-tunes with three seeds, skill-track fine-tuned runs.
7. Report, charts, headline table; final `make bench` from a clean tree.
