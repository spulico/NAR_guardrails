# NAR models as LLM guardrails — research for the lab

Date: 2026-10-14. Target machine: Apple M4, 24 GB unified memory, macOS, `uv` available (Python 3.10–3.12 installable), no Ollama yet.

## 1. TL;DR

- "NAR" here means non-autoregressive decision/classification models: one bidirectional forward pass, typed output (label or probability), no generated tokens. Three families exist today: (a) task-specific encoder classifiers (DeBERTa/ModernBERT fine-tunes), (b) zero-shot "decision models" (Jev, Laya, GLiNER2.5-Decide, GLiGuard, Open-Jev, Clef), (c) hybrids that bolt a classification head onto a decoder backbone (Qwen3Guard-Stream, Open-Jev, Clef).
- "Ayala" in the request is almost certainly **Laya** (`convaiinnovations/laya`), the Apache-2.0 open-source Jev alternative: ModernBERT-large encoder, 421M params, pip-installable, exposes a Jev-compatible `/v1/systemone` server. It runs on CPU and Apple MPS.
- Everything in the recommended core set runs locally on the M4 in fp32 with headroom: Laya (0.8 GB), GLiNER2.5-Decide (0.7 GB), GLiGuard (0.6 GB), ProtectAI DeBERTa (0.7 GB), Prompt Guard 2 (0.3 GB), Qwen3Guard-Stream-0.6B (1.2 GB), Open-Jev-2B (~4 GB bf16). Clef-flash (9B, 19 GB bf16) is a stretch goal only.
- Existing evidence says: on prompt injection, a pre-trained DeBERTa classifier is still the one to beat (fastest and within 0.2 pts of a 35B LLM judge); zero-shot decision models land 85–87% accuracy with policy tuning; on **malicious skills**, zero-shot decision models and even frontier LLMs are weak (Jev 33.8% recall, Claude Haiku 4.5 65.9%, GPT-5.5 72.1% zero-shot) while a fine-tuned 2B decision model or a fine-tuned ModernBERT reach ~96% recall at ~3–4% FPR. So the skill track should include a cheap fine-tune step, or it will mostly demonstrate failure.
- The frontier comparison is cheap to run: Claude Haiku 5.5 costs about $0.06 per 1,000 prompts and $0.30 per 1,000 skills; Sonnet 5.5 about $1.15 and $5.90. Local models cost $0 marginal and answer in 20–300 ms versus 0.5–2 s for an API round trip.

## 2. What counts as NAR, and the model taxonomy

| Family | How it decides | Examples | Zero-shot? |
|---|---|---|---|
| Encoder classifier (task-specific) | Sequence-classification head on an encoder, fixed label set | ProtectAI DeBERTa v2, Llama Prompt Guard 2, Qualifire Sentinel, Granite Guardian HAP 125M, StackOne Defender | No, labels fixed at training |
| Encoder decision model (schema at call time) | Encoder scores each candidate label/option at mask tokens in one pass | Laya, GLiNER2.5-Decide, GLiGuard, GLiNER2.5-multi-Decide | Yes |
| Decoder-backbone decision model (prefill-only) | Frozen decoder + LoRA + scalar head; scores candidates after one prefill, no generation | Open-Jev-2B/9B/27B, Cloudflare Clef-flash/Clef, Qwen3Guard-Stream (token-level heads) | Yes (Qwen3Guard: fixed labels) |
| Closed decision API | Unknown architecture, typed answers | TypeSafe Jev 1.13 | Yes |
| Autoregressive guard LLM (baseline, not NAR) | Generates a verdict string | Llama Guard 3/4, ShieldGemma, Granite Guardian 3.x, Qwen3Guard-Gen, Nemotron Content Safety, Shieldstral | Yes via prompt |
| Frontier LLM judge (baseline) | Generates JSON verdict | Claude Haiku 5.5 / Sonnet 5.5 / Opus 5.5, GPT-5.5 | Yes |

## 3. Candidate models and local viability on the M4 (24 GB)

Memory figures are fp32 unless noted; MPS on Apple runs fp32 (Laya) or fp16/bf16 (transformers, MLX).

### 3.1 Must-test NAR models

| Model | HF id | Backbone / params | Output | Local on M4 | Install | License | Notes |
|---|---|---|---|---|---|---|---|
| Laya (EN) | `convaiinnovations/laya` | ModernBERT-large, 421M, 512 tok | choice / score / noul (P(yes)) | Yes, CPU or MPS; ~0.8 GB | `pip install laya` (py≥3.10); `laya-serve` gives `/v1/systemone` | Apache-2.0 | Jev-compatible request shape. Card admits noul answers can be swayed by option wording; ships over-confident, refit temperatures. CPU 113–464 ms/q, MPS ~37 ms/q. |
| Laya multilingual | `convaiinnovations/laya-multilingual` | mmBERT-base, 322M, up to 8,192 tok | same | Yes; ~0.65 GB | same | Apache-2.0 | 45/51 languages usable, Polish likely OK. 8k context fits whole SKILL.md files. |
| GLiNER2.5-Decide | `fastino/GLiNER2.5-Decide` | DeBERTa-v3-large, 340M | single/multi-label classification with label descriptions, confidence | Yes; `model.to("mps")` or CPU; ~0.7 GB | `pip install "gliner2[local]"` (py 3.10–3.12) | Apache-2.0 | No latency claims on card; vendor blog: 167 ms p50 on 48-vCPU Xeon, ~40 ms GPU. Supports `batch_classify_text`. Multilingual sibling `GLiNER2.5-multi-Decide` (287M). |
| GLiGuard | `fastino/gliguard-LLMGuardrails-300M` | encoder, ~300M | `prompt_safety`, `jailbreak_detection` (12 labels incl. prompt injection, instruction override, system prompt exfiltration), `prompt_toxicity`, response-side tasks | Yes; ~0.6 GB | same `gliner2` package | Apache-2.0 | Purpose-built guardrail; vendor F1 87.7 on prompt harmfulness. Most direct NAR competitor to Prompt Guard. |
| Open-Jev-2B | `ZefanCai/Open-Jev-2B` (+ pinned `Qwen/Qwen3.5-2B`) | Qwen3.5-2B frozen + LoRA r8 + scalar head | choice / score / noul via `/v1/systemone` on :8791 | Probably; ~4 GB bf16; README says CPU and MPS supported but undocumented | `git clone` + `pip install -e '.[train]'` + `hf download` | MIT (code), Apache (Qwen) | Scores candidates after prefill, batched for large choice sets. 64.9% on JevBench public vs Jev. Also the base SkillScreen fine-tuned. Needs a smoke test on MPS before committing. |
| Qwen3Guard-Stream-0.6B | `Qwen/Qwen3Guard-Stream-0.6B` | Qwen3-0.6B + two classification heads (prompt, per-token response) | Safe / Controversial / Unsafe + category | Yes; ~1.2 GB | transformers | Apache-2.0 | Fixed taxonomy, 119 languages. Hybrid: decoder backbone, no generation. 4B variant also fits (~8 GB). |

### 3.2 Pre-trained encoder baselines (the bar NAR must clear)

| Model | HF id | Params | Labels | Local | License | Why include |
|---|---|---|---|---|---|---|
| ProtectAI DeBERTa v2 | `protectai/deberta-v3-base-prompt-injection-v2` | 184M | SAFE / INJECTION | Yes, ~0.7 GB | Apache-2.0 | Red Hat 2026: 89.0% acc, fastest (median 54–80 ms on M1 CPU). Not trained for jailbreaks. |
| Llama Prompt Guard 2 86M / 22M | `meta-llama/Llama-Prompt-Guard-2-86M`, `-22M` | 86M mDeBERTa / 22M DeBERTa-xsmall | benign / malicious | Yes, ~0.3 GB; 512-tok window | Llama 4 Community (gated, form) | AUC 0.998 EN; AgentDojo attack prevention 81.2%. Multilingual (86M). |
| Qualifire Sentinel | `qualifire/prompt-injection-sentinel` | ModernBERT-large, 0.4B | benign / jailbreak | Yes; gated, license "other" | check terms | Vendor F1 93.9 avg vs ProtectAI 70.9 across 4 sets. |
| Granite Guardian HAP 125M | `ibm-granite/granite-guardian-hap-125m` | 125M | hate/abuse/profanity | Yes, tiny | Apache-2.0 | 33 ms median; content-safety baseline only. |
| StackOne Defender | `github.com/StackOneHQ/defender` | MiniLM-L6, 22 MB int8 ONNX | binary | Yes | check | ~45 ms; built for agent payloads; AgentShield 79.8–81.0. Optional. |

### 3.3 Autoregressive local guard models (middle tier, optional)

Llama Guard 3 1B (gated), ShieldGemma 2B (gated), Granite Guardian 3.3 2B (Apache), Qwen3Guard-Gen 0.6B/4B (Apache). All fit the M4 via transformers or MLX (`mlx-lm` 0.31 on PyPI) or Ollama (`brew install ollama`). Qwen3Guard-Gen-4B is the easiest ungated pick and pairs nicely with Qwen3Guard-Stream to show "same backbone, generate vs classify".

### 3.4 Remote comparators

| Model | Access | Price | Latency (vendor) | Role |
|---|---|---|---|---|
| Claude Haiku 5.5 | Anthropic API, `claude-haiku-5-5` | $0.10 / $0.50 per MTok (batch 50% off) | ~0.5–1.5 s | cheap frontier judge |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | $2 / $10 per MTok | ~1–2 s | mid frontier judge |
| Claude Opus 5.5 | `claude-opus-5-5` | $4 / $20 per MTok | ~1–3 s | optional top-end |
| TypeSafe Jev 1.13 | `POST https://api.typesafe.ai/v1/systemone`, also OpenRouter / Cloudflare | $0.042 per M input, output free | 70–500 ms, OpenRouter p50 ~200 ms | closed NAR reference ($5 free credit on signup) |
| Cloudflare Clef-flash / Clef | Workers AI; weights `Cloudflare/clef-flash` (Apache-2.0) | $0.24 per MTok on Workers AI | 38.8 ms / 209 ms (vendor) | optional; local run needs ~41 GB (bf16) or llama.cpp PR #29831 for the GGUF Q4_K_M (6.5 GB) decision head. Stretch goal. |

### 3.5 Local run plan

- Python 3.12 via `uv` (system 3.9 is too old for `gliner2` and `laya`). One venv, `torch` with MPS.
- Keep models fp32 on MPS where the library expects it (Laya), fp16 where supported (transformers classifiers). DeBERTa on MPS has had dtype quirks historically; fall back to CPU per model if outputs look degenerate.
- Memory budget if everything is loaded sequentially (never all at once): peak ~4–5 GB for Open-Jev-2B, <1 GB for the encoders.
- Gated models need `huggingface-cli login` and the Meta / Qualifire forms.

## 4. What existing benchmarks already tell us

- Red Hat (2026-10-02), prompt-injection track, 9 guardrails via NeMo Guardrails: Qwen3.6-35B judge 89.3% acc / 312 ms; ProtectAI DeBERTa 89.0% / 54–80 ms (M1 CPU); DiffusionGemma 87.7%; Jev 86.4% / 348 ms (remote); Laya 85.4% / 119 ms (M1 CPU, recall 91%, precision 84%); Nemotron custom policy 84.8%. On content safety Laya collapsed to 57.9% with the default policy and recovered to 75.2% after policy tuning. Conclusion: decision models give no automatic speed/cost/accuracy win over a good pre-trained classifier; wording of the policy matters a lot.
- Prompt Guard 2 card, AgentDojo attack-prevention rate at 3% utility loss: PG2-86M 81.2%, PG2-22M 78.4%, ProtectAI 22.2%, Deepset 13.5%. So the ProtectAI model that wins on direct injection loses badly on agent-style indirect injection. Our prompt track must separate these.
- "When Benchmarks Lie" (2602.14161): under leave-one-dataset-out, Prompt Guard 2 catches 37% of indirect injections and LlamaGuard 27%; a Llama-3.1-8B judge only 7%. Distribution shift is the main story; our sets must include sources no classifier trained on.
- InjecGuard / NotInject: guard models over-refuse benign prompts containing trigger words ("ignore", "system prompt", "bypass"). We need an explicit over-defense slice.
- SkillScreen (2610.17727), MaliciousSkillBench MSD split, SKILL.md text only, recall / FPR at threshold 0.5: fine-tuned Open-Jev-2B 95.6% / 3.3%; fine-tuned ModernBERT-large 96.0% / 4.1%; TF-IDF SVM 94.6% / 8.9%; GPT-5.5 zero-shot 72.1% / 0.7%; GPT-5.5 8-shot 87.9% / 2.5%; Claude Haiku 4.5 65.9% / 2.9%; TypeSafe Jev 33.8% / 0.2%; SkillSpector static 10.5% / 0.7%; untuned Open-Jev-2B 4.2%. Cost: judges $0.8–8.8 per 1,000 skills; SkillScreen 1,000 skills in 4.5 min on a 10 GB GPU slice. Moving malicious commands into a referenced script dropped recall from 99.8% to 87%.
- MaliciousSkillBench paper: scanners (Cisco, SkillFortify, SkillSpector static) have ~0–25% recall at low FPR; learned text models have ~95% recall but 62% FPR on unseen sources. Nobody has both yet.
- SkillSpector itself: regex + AST + taint + YARA (71 patterns, 17 categories), optional LLM stage. No NAR. Its `tests/fixtures/` has ~15 small fixture skills (safe_skill, malicious_skill, mcp_poisoned_tool, mcp_overprivileged_skill, etc.) that are handy as smoke tests.

Implication for the lab: on prompts, expect NAR zero-shot to be competitive with but not better than DeBERTa; on skills, expect zero-shot NAR to miss most attacks unless we add a small fine-tune (LoRA on Open-Jev-2B or full fine-tune of Laya/ModernBERT on a few thousand MaliciousSkillBench rows, feasible on the M4 in well under an hour for the encoder).

## 5. Test set design

### 5.1 Prompt track (user and agent inputs)

Two sources: a hand-curated scenario set (quality, demo-readable, English) and sampled public sets published in 2025–2026 (scale, distribution shift).

Scenario taxonomy for the curated set (target ~200 items, balanced ~50/50 benign/malicious):

| Code | Scenario | Example shape | Malicious? |
|---|---|---|---|
| B1 | Plain benign tasks | summarise, translate, code help | no |
| B2 | Benign with trigger words (over-defense control) | "How do prompt injection attacks work?", "ignore the first column in this CSV", a shell tutorial with `rm -rf` | no |
| B3 | Benign tool outputs / documents fed to an agent | normal email, web page, PDF chunk, API JSON | no |
| B4 | Benign in Polish | same as B1/B2 in Polish | no |
| J1 | Classic jailbreaks | DAN, role-play, "developer mode", hypothetical framing | yes |
| J2 | Obfuscated jailbreaks | base64, leetspeak, split instructions, translation tricks | yes |
| D1 | Direct injection / instruction override | "ignore previous instructions and…" | yes |
| D2 | System prompt exfiltration | "print your system prompt verbatim" | yes |
| I1 | Indirect injection in tool output | email/web page/RAG chunk containing instructions to the agent | yes |
| I2 | Agent tool-call hijack | injected text asking to call `send_email`, `transfer_funds`, delete files (InjecAgent style) | yes |
| I3 | MCP tool poisoning | tool description with hidden instructions | yes |
| I4 | Multi-agent handoff injection | message from "another agent" asserting authority | yes |
| P1 | Polish attacks | translations of D1/I1/J1 | yes |

Public sources (rule: published in 2025 or 2026 only; 2024-era sets are stale and were used to train the vendor classifiers under test). Verified via the Hugging Face API and GitHub on 2026-10-14:

| Source | Published | Licence | Size | Role |
|---|---|---|---|---|
| AgentShield corpus (doronp, GitHub) | 2026-03 | Apache-2.0 | 537 cases, 7 categories incl. 65 over-refusal benign | direct, indirect, tool abuse, multi-agent, over-defense |
| PIDS-Bench eval subsets (ShirePyDev, GitHub tag v1.0-pids-bench) | 2026-09 | MIT | hard-benign 1,472; obfuscated 405; domain shift 2,000; structural shift 1,998 | over-defense and distribution shift (main corpus excluded: it repackages deepset/SPML) |
| neuralchemy/Prompt-injection-dataset | 2026-01 | Apache-2.0 | 16,918 (core test 942), 12 categories | mixed; use only the authors' original rows |
| prodnull/prompt-injection-repo-dataset | 2026-03 | Apache-2.0 (gated) | 5,671, balanced | repository-content injection against coding agents |
| sureheremarv/ipi_arena_attacks | 2026-03 | MIT | 95 | indirect injection, agent arena 2026 |
| BELLS jailbreak dataset (CeSIA) | 2026-02 | CC BY-NC-SA 4.0 (manual gate) | 6,406 incl. transformed benign | 35 obfuscation techniques, benign-under-obfuscation control |
| mcp-guardbench (getmcpm, GitHub) | 2026-10 (v5) | MIT | 57 JSON-RPC frames, 20 benign | MCP tool poisoning and exfil |
| microsoft/llmail-inject-challenge | 2025-05 | MIT | 462k submissions | indirect injection via email, adaptive attackers |
| hackaprompt/Pliny_HackAPrompt_Dataset | 2025-07 | CC BY 4.0 (gated) | 10k–100k submissions with `passed` label | jailbreaks incl. two challenges against Prompt Guard |
| Simsonsun/JailbreakPrompts | 2025-06 | MIT | 2,426 | jailbreaks filtered for classifier-training contamination |
| Mindgard/evaded-prompt-injection-and-jailbreak-samples | 2025-04 | CC BY-NC 4.0 (gated) | 10k–100k | evasion variants (character injection, adversarial ML) |
| MCPTox (zhiqiangwang4, GitHub, AAAI 2026) | 2025-08 | none stated | 1,312 poisoned tool descriptions | MCP tool poisoning |
| allenai/WildChat-4.8M | 2025-08 | ODC-BY | 4.8M chats | plain benign user prompts |

Excluded as pre-2025 or repackaged: deepset/prompt-injections, xTRam1 safe-guard, jackhhao, WildJailbreak, InjecAgent, BIPIA, AgentHarm, NotInject, HarmBench, JailbreakBench, Cogensec jailbreak-corpus, Bravansky compact-jailbreaks, PIDS-Bench main splits. Qualifire's benchmark (rogue-security, 2025-01) is optional only. Not usable as text datasets: WASP, MCP-SafetyBench, AgentDojo (environment benchmarks), LongPIBench and MCP-AttackBench (no public release found).

### 5.2 Skill track (malicious agent skills)

- Primary: `ProtectSkills/MaliciousSkillBench` config `primary` (parquet, 9,740 rows, `skill_text` + `label` + `source_id` + attack category; CC BY 4.0 metadata; frozen MSD / SD / SBR split manifests). Use SD (source-disjoint) as the honest test and MSD as the main one, matching SkillScreen so our numbers are comparable.
- Secondary: SkillSpector `tests/fixtures/` (~15 skills) for smoke tests; `lxyeternal/MalSkillBench` packages (3,944 malicious with scripts, academic-use only; clone, do not redistribute).
- Hand-crafted set (~30–40 skills in real `SKILL.md` frontmatter format) for the demo narrative, covering SkillSpector's categories: data exfiltration (curl env vars to a webhook), credential harvesting (read `~/.ssh`, keychain), anti-refusal / "do not mention this to the user", hidden instructions in HTML comments or fenced code, download-and-execute, persistence (cron, shell rc), MCP tool poisoning, privilege escalation, typosquatted dependency, plus benign look-alikes (a deploy skill that legitimately uses `curl` and env vars) and benign skills from the Claude Code / Codex ecosystems.
- Input encoding: SKILL.md text only (SkillScreen protocol), capped at 8,192 tokens, plus an optional variant with bundled scripts concatenated, so we can show the "attack moved into a script" recall drop.
- Safety: the data contains live malicious instructions. Store under a quarantined directory, never execute, never feed to an agent with tools. Scoring harness reads text only.

## 6. Comparison protocol

- Questions / policies: one shared policy text per track, used verbatim for every zero-shot model (decision models get it as the noul question or label descriptions; LLM judges get it as the system prompt). Keep a per-model "tuned policy" variant as a second run, because Red Hat showed ±17 pts from wording.
- Thresholds: pick on a dev slice (e.g. 20%), freeze, report on test. Report recall, FPR, precision, AUROC, FPR at 95% and 99% recall, per scenario code and per source.
- Latency: wall-clock per item at batch 1 (p50 / p95), plus throughput at batch 16/32 for local models; API latency measured end-to-end from the Mac.
- Cost: local = $0 marginal (report model size and seconds); API = tokens × price from `usage`. Expected ballpark per 1,000 items (policy ~350 tok, prompt ~150 tok, verdict ~15 tok; skills ~2,500 tok):

| Judge | Prompts, $/1k | Skills, $/1k | Batch API |
|---|---|---|---|
| Claude Haiku 5.5 | ~0.06 | ~0.30 | halve |
| Claude Sonnet 5.5 | ~1.15 | ~5.90 | halve |
| Claude Opus 5.5 | ~2.30 | ~11.80 | halve |
| TypeSafe Jev | ~0.02 | ~0.12 | n/a |
| Clef-flash (Workers AI) | ~0.12 | ~0.68 | n/a |
| Local NAR | 0 | 0 | n/a |

Full run of 200 curated prompts + 2,000 sampled prompts + 1,950 MSD-test skills on Haiku 5.5 costs about $1; on Sonnet 5.5 about $14. Prompt caching helps only once the shared prefix is long enough (model-dependent minimum), so bundle the policy with few-shot examples if we want cache hits.

- Fairness notes to state in the write-up: encoder classifiers were trained on some of the public sets (ProtectAI on deepset-like data, Sentinel on its own benchmark), so the curated set and the source-disjoint splits are the fair comparison; vendor numbers are self-reported.

## 7. Open questions for the lab owner

1. Is the goal a demo-grade comparison for a talk (fast, visual, ~1 afternoon of compute) or a rigorous benchmark (full splits, seeds, CI)? This decides sample sizes.
2. API access available: Anthropic key (yes?), TypeSafe Jev account, OpenRouter, Cloudflare Workers AI?
3. Hugging Face gated access: Meta (Prompt Guard 2, Llama Guard), Qualifire Sentinel, AI2 WildJailbreak?
4. Should Polish be a first-class slice (recommended, it is where the multilingual NAR variants shine or fail)?
5. Include a fine-tuning step on the skill track (recommended, otherwise the NAR skill result is predictably ~5–35% recall)?

## 8. Sources

- GLiNER2.5-Decide card: https://huggingface.co/fastino/GLiNER2.5-Decide ; blog: https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model ; library: https://github.com/fastino-ai/GLiNER2
- GLiGuard: https://github.com/fastino-ai/GLiGuard
- Laya card: https://huggingface.co/convaiinnovations/laya ; repo: https://github.com/NandhaKishorM/laya
- Open-Jev: https://github.com/Zefan-Cai/Open-Jev ; https://huggingface.co/ZefanCai/Open-Jev-2B
- Jev / TypeSafe: https://docs.litellm.ai/docs/pass_through/typesafe ; https://openrouter.ai/typesafe/jev-1.13 ; https://developers.cloudflare.com/ai/models/typesafe/jev/
- Cloudflare Clef: https://www.theregister.com/ai-and-ml/2026/10/01/cloudflare-tries-to-outplay-jev-with-open-weight-clef-models/5300649 ; https://huggingface.co/Cloudflare/clef-flash ; https://huggingface.co/ggml-org/Clef-Flash-GGUF
- Red Hat benchmark: https://developers.redhat.com/articles/2026/10/02/benchmarking-ai-decision-models-against-traditional-guardrails
- Prompt Guard 2: https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
- ProtectAI DeBERTa v2: https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
- Qualifire Sentinel: https://huggingface.co/qualifire/prompt-injection-sentinel
- Qwen3Guard: https://github.com/QwenLM/Qwen3Guard ; https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B
- StackOne Defender / AgentShield: https://www.stackone.com/blog/training-22mb-prompt-injection-classifier/ ; https://doronp.github.io/agentshield-benchmark
- InjecGuard / NotInject: https://arxiv.org/abs/2410.22770 ; When Benchmarks Lie: https://arxiv.org/abs/2602.14161 (code https://github.com/maxf-zn/prompt-mining)
- InjecAgent: https://huggingface.co/papers/2403.02691 ; AgentDojo: https://arxiv.org/abs/2406.13352 ; AgentHarm: https://arxiv.org/abs/2410.09024 ; BIPIA: https://github.com/microsoft/BIPIA
- SkillSpector: https://github.com/NVIDIA/SkillSpector
- SkillScreen: https://arxiv.org/abs/2610.17727 ; MaliciousSkillBench: https://arxiv.org/abs/2608.19901 , https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench ; MalSkillBench: https://github.com/lxyeternal/MalSkillBench
- Claude pricing: Anthropic model table as of 2026-10-06 (Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10, Opus 5.5 $4/$20 per MTok; batch API 50% off)
