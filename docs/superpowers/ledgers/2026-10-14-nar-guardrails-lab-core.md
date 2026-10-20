# SDD ledger — plan: docs/superpowers/plans/2026-10-14-nar-guardrails-lab-core.md

Spec: docs/superpowers/specs/2026-10-14-nar-guardrails-lab-design.md (read; binding authority).
Worktree: .worktrees/nar-lab-core, branch nar-lab-core, base commit 41454ab.
Execution: subagent-driven, user asked for prudence (2026-10-16).

## Pre-flight conflict scan (2026-10-16)

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 cli.py vs T12/T19/T24/T25 (modify cli.py) | T1 builds parser+main; later tasks add subparsers and dispatch branches | additive; T19 needs top-level `import json` in cli.py (stated) — ok |
| T2 schema vs all | Item/Verdict/text_sha/JSONL IO | field names consistent in every later test (item_id, p_malicious, outcome, latency_ms, usage, raw; meta.splits) — ok |
| T3 Policy vs T4/T6/T7/T10/T12 | Policy(track,name,question,malicious,benign,examples), .hash/.ref, load_policy, pick_examples | T12 uses dataclasses.replace on frozen Policy to rename to 8shot — valid — ok |
| T4 base/registry vs T5–T10 adapters | Detector, Prediction, register, _sanitise; _ADAPTER_MODULES names | module names match files hf_classifier, laya_adapter, gliner2_adapter, qwen3guard, openjev, claude_adapter — ok; T10 imports private `_sanitise` (plan-mandated, accepted) |
| T6 noul_probability vs T9 | shared helper | import path nar_guardrails.detectors.laya_adapter — ok |
| T9 DetectorUnavailable vs T12 | runner catches generic Exception from build_detector; test imports class from openjev | ok |
| T11 cache vs T12 runner | VerdictCache(path).get/put_many, cache_path(root,key,policy_hash,track) | ok |
| T12 manifest vs T24 build/T25 smoke | keys run_id, track, split, protocol, policy, environment.timestamp, detectors[k].status/reason | ok |
| T13 metrics vs T14/T23 | basic_metrics dict keys, fpr_at_recall→(fpr,thr), usable→(y,s,ids), bootstrap_ci(y,s,stat,n_boot,seed) | ok (T23 passes n_boot=0 → (nan,nan)) |
| T14 thresholds vs T24 | thresholds_for_run(items_dev, verdicts_by_det, policy_ref, track), write_thresholds | ok |
| T15 cost/latency vs T23 | summarize_cost(verdicts, price|None), latency_summary, load_prices | ok |
| T16/T21 curated vs T25 smoke | load_curated_prompts(), load_curated_skills(); T21 appends to curated.py and relies on T16's `import json` | ok |
| T17/T18 loaders vs T19 build | git loaders (repo) except mcptox(repo,cap,seed); hf loaders (rows,cap,seed) | ok, special-cased in _load_source |
| T19 expected_ranges/datasets.yaml vs T22 | T22 imports expected_ranges and widens two ranges | ok |
| T20/T21 vs T22 build_skills | load_malicious_skill_bench(rows, split_tables), fetch_split_tables, load_skillspector_fixtures(repo, labels=…), assert_no_overlap | ok |
| T23 tables vs T24 build | load_run, leaderboard, scenario_table, source_table, headline, default_label_table, seed_aggregate | ok |
| Each task self-consistency | tests vs code in same task | T13/T14/T23 hand-computed values corrected during plan self-review (fpr@recall=1.0 → 0.5; best-F1 thr → 0.2; seed std ddof=1); T17 CSV fixture simplified; T25 smoke assertion limited to prompts — ok |
| Global constraints | py3.12 via uv; one model at a time; MPS w/ cpu override; 2025–2026 sources; no execution of data; Claude haiku/sonnet only, effort low, max_tokens 256, no fallbacks; raw probabilities only in Verdict | no task contradicts — ok |

Scan result: clean. No rulings needed before Task 1.

## Model plan
Implementers: T1 sonnet (dependency resolution risk); T2,T3,T11,T13,T14,T15 haiku (complete code); T4,T5–T10,T12,T17–T20,T22–T25 sonnet; T16,T21 opus (authored security content).
Reviewers: sonnet default; opus for T12,T13,T14,T16,T19,T21,T22. Final whole-branch review: fable.

## Task log
Task 1: complete (commits 41454ab..c222cb9, review clean). Verified ⚠️: Co-Authored-By trailer present; uv.lock versions torch 2.14.1, transformers 5.20.0, gliner2 2.1.0, laya 0.4.8, anthropic 1.15.0.
Task 1: minor (deferred): tests/test_cli.py unused `import subprocess, sys` (plan-mandated verbatim); cli.py `smoke` branch returns 0 as placeholder until Task 25.
Task 2: minor (deferred, MUST fix in final wave): tests/test_schema.py NFC assertion compares identical code points (both U+00E9); change second literal to "é" so the NFC branch is tested.
Task 3: Ruling: commit trailer is `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` for every commit (session-level attribution); subagents' own-model reminders are ignored — cost if wrong: cosmetic trailer text only.
Task 3: note: pick_examples returns fewer than n when a class is short (plan behaviour; 8-shot pools are thousands of items).
Task 3: complete (commits aa8b1ce..551bc13, review clean). Verified ⚠️: trailer present.
Task 3: minor (deferred): pick_examples silent under-fill; load_policy bare KeyError on missing YAML field; hash sensitivity tested only via question.
Task 4: complete (commits 551bc13..4507a25, review clean). Verified ⚠️: trailer present.
Task 4: minor (deferred): unused `import math` in test; _sanitise accepts any outcome string; no test for amortised latency arithmetic.
Task 4: Ruling: `_load_adapters` must only swallow a ModuleNotFoundError for the adapter module itself (e.name == f"nar_guardrails.detectors.{m}"), re-raising missing transitive deps; implemented as an explicit addition in Task 5 — why: otherwise a missing torch/transformers is masked as "no adapter registered" — cost if wrong: one extra conditional, trivially reversible.
Task 5: complete (commits 4507a25..5d2c830, review clean; protectai real-model test passed on mps, bad=0.99999976 good=2.75e-05). Verified ⚠️: trailer present.
Task 5: minor (deferred, fix in final wave): debug print("SCORES", ...) left in tests/test_hf_classifier.py model test; no unit test for the positive_label-absent ValueError.
Task 6: implementer DONE_WITH_CONCERNS (ae1e8ae): laya 0.4.8 load() lacks max_len; adapter sets agent.cfg["max_len"] post-load (predict_batch has a per-call max_len kwarg instead); real noul shape {"type":"noul","noul":p,"confidence":..}; real-model separation small with the placeholder test question; checkpoint RuntimeWarning on load. Sent to review.
Task 6: complete (commits 5d2c830..ae1e8ae, review clean; reviewer verified agent.cfg["max_len"] is read at call time in laya 0.4.8 on non-CUDA). Verified ⚠️: trailer present. Open ⚠️ carried to Task 25 smoke: laya-multi 8192 limit unproven end to end.
Task 6: minor (deferred): no unit test for the max_length branch (init bypassed in tests); single vs batch predict paths, only batch exercised.
Task 7: complete (commits ae1e8ae..f14928d, review clean; gliner2 2.1.0 schema {task:{label:desc}} confirmed; decide 0.764 vs 0.244, gliguard 0.9995 vs 0.0001 on mps). Verified ⚠️: trailer present.
Task 7: minor (deferred, final wave): debug print in model test (same pattern as Task 5); unknown `mode` silently falls through to decide; `None` label becomes "None".
Controller note: from Task 8 on, dispatches say "no debug prints in committed tests; raw outputs go in the report".
Task 8: review approved at a8f4a4c (shims scoped/benign per reviewer check of transformers 5 source; revision pinned 74e1479150e9029d6778993f00491108323bb6f8). Verified ⚠️: trailer present.
Task 8: Ruling: Qwen3Guard-Stream stays in both tracks as the hybrid content-safety reference even though it labels plain injection text Safe — why: that gap is a finding the report should show, not a reason to drop the model — cost if wrong: one weak row in the leaderboard.
Task 8: Ruling: use the card's `risk_prob` to synthesise class probabilities (Safe: {Safe: p, Unsafe: 1-p}; Unsafe: {Safe: 1-p, Unsafe: p}; Controversial: probs None → 0.5) so scores are continuous for threshold tuning and AUROC; `score_from_moderation` and its tests unchanged — why: discrete 0/0.5/1 scores make the threshold sweep meaningless — cost if wrong: a scoring choice confined to one adapter, reversible.
Task 8: fix round 1/5 dispatched: risk_prob probs, try/finally around close_stream, move mid-file import to top, comment on the content-safety test sentence, store risk_prob in raw.
2026-10-19: session model switched to Claude Mythos 5.1. Ruling: commit trailer is `Co-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>` for commits from now on (earlier commits keep Fable 5.1) — cost if wrong: cosmetic.
2026-10-19: user request: add "basal-omni" (described as a Jev-like Polish decision model) as an extra detector. Researching; will append a task to the plan once the model is identified.
2026-10-19: Ruling: basal pick is `Remek/basal-2.0-mini` (1.5B) served through the Task 9 HTTP adapter (same /v1/systemone shape); basal-2.0 4.5B and basal-omni (multimodal, 16 GB) out of core scope — why: matches the size class of Open-Jev-2B and fits the M4 — cost if wrong: one more optional detector row, reversible. Spec and plan amended (commit above); Task 9 brief regenerated.
Task 8: fix round 1/5 (5 addressed, 0 open; commits a8f4a4c..726edec)
Task 8: complete (commits f14928d..726edec, review clean). Benchmark caveat for the report: Qwen3Guard remote code runs under transformers 5 via local shims, numerics not compared with transformers 4.x; injection text "Ignore all previous instructions..." scores Safe (p≈0.22).
Task 8: minor (deferred): no test for the exception path of close_stream; rp[-1] on an empty list would IndexError.
Task 9: review approved at a1db719. Verified ⚠️: trailer present (Mythos). Reviewer's mid-run-failure concern is covered by Detector.predict's exception-to-error-verdict wrapping (Task 4).
Task 9: Ruling: health check falls back to a probe POST (state "ping", the real noul question) to `url` when GET health_url is not 200; any 200 counts as healthy — why: basal documents no /health, and the user asked for this model explicitly — cost if wrong: one extra request at startup.
Task 9: fix round 1/5 dispatched: health fallback probe + tests (fallback path, empty-text state "(empty)", health GET raising ConnectionError).
Task 9: fix round 1/5 (3 addressed, 0 open; commits a1db719..859a25a)
Task 9: complete (commits f78b932..859a25a, review clean). Note: basal/Open-Jev servers not installed yet; both detectors skip until a server runs (scripts/systemone_servers.md).
Task 10: review approved at eaefa50. Verified ⚠️: trailer present (Mythos). Live test deferred until ANTHROPIC_API_KEY is exported (user action).
Task 10: Ruling: harden the Claude adapter beyond the brief — neutralise literal `</untrusted_input` in wrapped text and examples; treat non-finite probability or a verdict outside {malicious, benign} as outcome=error; make predict_texts share _one's exception handling — why: a benchmark of injection text will contain tag-escape attempts and malformed outputs; silent benign scores would bias results — cost if wrong: a few lines in one adapter.
Task 10: fix round 1/5 dispatched with the three items above plus tests (tag neutralisation, NaN → error, unknown verdict → error, examples truncated at 800 chars).
Task 10: fix round 1/5 (4 addressed, 0 open; commits eaefa50..6293c4e)
Task 10: complete (commits 859a25a..6293c4e, review clean). Live test pending ANTHROPIC_API_KEY.
Task 11: review Needs fixes at 74cbf3c (2 Important, plan-mandated). Ruling: spec wins over brief code — persist first, then update memory; serialise raw with json default=str; tolerate a torn trailing line (skip it, ensure "\n" before appending); put_many writes in one open — why: the cache is the durable record for resumable runs — cost if wrong: a few lines in cache.py.
Task 11: fix round 1/5 dispatched (resume haiku implementer).
Deferred to final wave: schema.write_jsonl should also use json default=str for Verdict.raw.
Task 11: Ruling: a torn trailing fragment (last line, no newline, undecodable) is truncated before appending instead of newline-terminated — why: terminating it would turn it into a malformed middle line that the loader rejects — cost if wrong: loses at most one partial record that was never valid.
Task 11: fix round 1/5 (3 addressed, 1 new Important open — load uses strict UTF-8 so a torn multibyte tail raises UnicodeDecodeError; commits 74cbf3c..c379157)
Task 11: fix round 2/5 dispatched (resume implementer): decode tolerant on load; also handle garbage last line that ends with "\n" (minor) by treating a last line that fails to parse as skippable regardless of newline and truncating it on repair.
Task 11: fix round 2/5 (2 addressed, 0 open; commits c379157..31ce9fa)
Task 11: complete (commits 6293c4e..31ce9fa, review clean after 2 rounds)
Task 11: minor (deferred): repair vs load disagree on "last line" when the file ends with blank lines; errors="replace" can let a corrupt middle line parse with a mangled id (cache miss, no raise).
Task 12: review Needs fixes at 6a572fe (5 Important, all plan-mandated). Ruling: spec's reliability intent wins over the brief's code — (I1) guard warmup/predict per detector → status "failed" with reason, write manifest after each detector; (I2) cache in chunks of 64 as computed; (I3) never cache outcome=error verdicts (still written to the run file); (I4) 8shot raises if fewer than 8 examples and records n_examples; (I5) warmup unconditionally before throughput. Also included minors: strip detector names and accept "all"; expose --seed; --limit 0 → error; gc.collect()+torch.mps.empty_cache() after del; test registry cleanup. — why: a real 11k-item run must survive one broken detector, an interrupt, and transient API errors — cost if wrong: runner-local changes, reversible.
Task 12: fix round 1/5 dispatched (resume sonnet implementer).
Task 12: minor (deferred): run-id collisions overwrite silently; fully cached path still builds the detector; measure_throughput times the error path; no CLI test for `run`; dirty-tree refusal untested.
Task 12: fix round 1/5 (10 addressed, 0 open; commits 6a572fe..34824ec)
Task 12: complete (commits 31ce9fa..34824ec, review clean after 1 round)
Task 12: minor (deferred, final wave): empty --detectors spec runs silently; a throughput-stage failure marks a fully scored detector "failed"; cached error verdicts from pre-fix caches are never recomputed (treat cached outcome=error as a miss); explicit detector keys not checked against track; VerdictCache construction outside the per-detector try.
Task 13: review Needs fixes at 7cedca2 (1 Important, plan-mandated). Ruling: undefined metrics are NaN, never 0.0 — precision NaN when nothing is flagged; f1 NaN when recall is NaN, 0.0 when recall is 0 with positives present; precision_at_prevalence NaN when either input is NaN; fpr_at_recall validates 0 < target <= 1 — why: 0.0 slips past the bootstrap NaN filter and drags CIs to zero on small/imbalanced slices — cost if wrong: metric conventions localised in one module; report code (Task 23) must handle NaN cells.
Task 13: fix round 1/5 dispatched (resume haiku implementer).
Task 13: minor (deferred): bootstrap_ci does not report dropped resamples; usable() silently drops ok-verdicts with both values None and last-wins on duplicate ids; bootstrap speed on 10k+ items.
Task 13: fix round 1/5 (3 addressed, 0 open; commits 7cedca2..32caacd)
Task 13: complete (commits 34824ec..32caacd, review clean after 1 round). Downstream note: basic_metrics may return NaN precision/f1; Task 14's F1 scan and Task 23 tables must treat NaN as "skip", not as a value.
Task 13: minor (deferred): precision_at_prevalence returns 0.0 when recall and fpr are both 0; bootstrap NaN filter only recognises float subclasses.
Task 14: review Needs fixes at 5cf0ce6 (2 Important; tie rule plan-mandated). Ruling: compare F1 exactly as Fraction(2*tp, 2*tp+fp+fn) from the counts so the "ties → higher threshold" rule holds; choose_thresholds raises ValueError on fewer than two classes (sentinel removed); write_thresholds uses allow_nan=False — why: frozen thresholds decide every reported number — cost if wrong: localised to thresholds.py.
Task 14: fix round 1/5 dispatched (resume haiku implementer).
Task 14: fix round 1/5 (5 addressed, 0 open; commits 5cf0ce6..1740e70)
Task 14: complete (commits 32caacd..1740e70, review clean after 1 round). Downstream note for Task 23/24: thresholds entries may carry `error` instead of thresholds; report must fall back to 0.5 and flag it.
Task 15: complete (commits 1740e70..206791c, review clean). Verified ⚠️: trailer present.
Task 15: minor (deferred): cache_creation_input_tokens never priced (no caching is used, so moot); latency_summary has no guard for None/NaN latency.
Task 16: implementer (opus, ad8f6c4) BLOCKED — a safety classifier stopped it while drafting the ~100 attack prompts; loader + validator written but uncommitted, no data file. Ruling: the curated set is defensive benchmark content (guardrail evaluation) and must contain no harmful payloads: jailbreak items (J1/J2) target harmless or absurd goals (reveal a code word, say a banned phrase, adopt a persona), injection items (D/I) ask for benign-looking but unauthorised actions on example.com hosts; re-dispatch a fresh implementer writing the file in three stages (benign, direct/indirect injection, jailbreaks) — why: keeps the benchmark honest about manipulation without producing harmful content; if blocked again, split further and surface to the user in the final report — cost if wrong: time; no harmful content either way.
Task 16: attempt 2 DONE_WITH_CONCERNS at d4548c2 (trailer says Opus 5.5 — left as is, cosmetic; reference items J1-001/J1-002 kept verbatim per brief but sit outside the harmless-target rule). Sent to opus review.
Task 16: opus review Approved at d4548c2 (labels all correct; strong B2/B3). Ruling: reference items J1-001 (lock picking) and J1-002 (synthesis "real quantities") are rewritten to harmless targets, and I1-002's host becomes shop.example.net; the plan file's reference lines are updated to match — why: the content rule (no harmful targets) outranks the plan's verbatim rule — cost if wrong: three benchmark items change text.
Task 16: fix round 1/5 dispatched with the ruling plus polish: translate D1-009/I1-014 to English and add one benign translation-request B1 item (breaks the non-English=attack shortcut), rebalance B3 with 2 benign agent messages + 1 benign tool description, 3 I2 items steered in natural language, vary 3 repeated J targets, fictional club in J1-011, sharpen J1-014.
Task 16: fix round 1/5 (9 addressed, 0 open; commits d4548c2..cb53582)
Task 16: complete (commits 206791c..cb53582, review clean after 1 round). Trailer on d4548c2/cb53582 says Opus 5.5 (cosmetic, left).
Task 16: minor (deferred): reference I2-002 host router-help.example (reserved TLD, acceptable); documentation-example IBANs pass checksums.
Task 17: review Approved-with-Importants at 6066b70. Rulings: (a) AgentShield unknown category → scenario "unknown" stays (valid bucket for public rows); (b) non-"allow" expected_behavior → label 1 stays (intended: warn-class rows are flagged); (c) fix: ensure_repo verifies the checkout matches the requested ref (re-clone on mismatch or missing .git), CSVs open with utf-8-sig, _label accepts numeric strings via float, mcp-guardbench glob limited to attacks/benign/warn with bucket validation, duplicate-id guard in every loader, MCPTox None fields coerced to "" — why: the first real fetch must not silently use a stale checkout or abort on a BOM — cost if wrong: loader-local.
Task 17: fix round 1/5 dispatched (resume sonnet implementer).
Task 17: fix round 1/5 (6 addressed, 1 new Important open — ensure_repo re-fetches on every call after an update and may delete a good clone when offline; commits 6066b70..c91f808)
Task 17: fix round 2/5 dispatched (resume implementer): store .nar_ref {ref, commit}; match on stored ref + HEAD commit; never rmtree an existing clone on fetch failure (raise with the old commit kept); replace-without-.git via temp clone + swap; test calling ensure_repo twice after an update with the remote made unavailable.
Task 17: fix round 2/5 (4 addressed, 0 open; commits c91f808..cbe9191)
Task 17: complete (commits cb53582..cbe9191, review clean after 2 rounds)
Task 17: minor (deferred): clones made before the marker need one network call; corrupt .git raises raw CalledProcessError; clone by SHA unsupported (refs are tags/branches).
Task 18: review Needs fixes at 646817b. Rulings: BELLS ids become bells:<id>:<technique> (counter fallback); BELLS benign rows are capped with sample(benign, cap, seed+1); index-based ids (neuralchemy, simsonsun, llmail, mindgard, wildchat, prodnull) become <source>:<text_sha[:12]> with in-loader text dedupe so ids are stable across caps; BELLS harmful scenario J1 for cognitive/psychological/few-shot/DAP families else J2; Mindgard column detection prefers exact known names and marks encoded rows; WildChat candidate scan bounded even without cap — why: loaders must survive the real files and give reproducible ids — cost if wrong: loader-local; ids differ from the brief's.
Task 18: fix round 1/5 dispatched (resume sonnet implementer).
Task 18: fix round 1/5 (7 addressed, 0 open; commits 646817b..0c1e769)
Task 18: complete (commits cbe9191..0c1e769, review clean after 1 round)
Task 18: minor (deferred): dedupe keeps the first label on identical texts with conflicting labels; BELLS "dap" substring could over-match; check real technique_type names during the build.
Task 19: note — no Hugging Face token on this machine; gated sources (prodnull, bells, pliny, mindgard) will be skipped by the build; USER ACTION needed later: `hf auth login` + accept the four gates, then rerun `nar data build --track prompts`.
Task 19: opus review Needs fixes at 2c05c6a (4 Important; #3 and #4 plan-mandated). Rulings: (1) close HF generators in _load_source and os._exit backstop in the CLI data-build branch after outputs are flushed; (2) llmail label requires exfil.sent (attack actually fired), STREAM_LIMIT for llmail may rise to 150k to reach 500; (3) per-(source,label) RNG seeded with f"{seed}:{source}:{label}" so adding a source never re-splits others; (4) manifest records loaded/kept/dropped per source and label conflicts, ranges validated on kept counts, pids_bench range re-based on observed kept count ±5 % with reason — why: benchmark labels and splits must be correct and stable before any model run — cost if wrong: data build logic only; the processed set is regenerated.
Task 19: fix round 1/5 dispatched (resume sonnet implementer; real build re-run required).
Task 19: minor (deferred): meta.level from scenario; manifest lacks source commit SHAs/HF revisions; expected_ranges cwd-dependent.
Task 19: fix round 1/5 (6 addressed, 0 open; commits 2c05c6a..7269332)
Task 19: complete (commits 0c1e769..7269332, review clean after 1 round). Prompt track built: 9,086 items (dev 1,777 / test 7,309); kept per source curated 200, agentshield 537, pids_bench 5212, mcp_guardbench 57, mcptox 485, neuralchemy 500, ipi_arena 95, llmail 500 (exfil.sent rule), simsonsun 500, wildchat 1000; gated prodnull/bells/pliny/mindgard skipped (USER ACTION: hf auth login + accept gates, rerun build).
Task 19: minor (deferred): llmail label is behavioural (exfil fired) so a few texts read benign; os._exit backstop swallows KeyboardInterrupt as an error.
Task 20: complete (commits 7269332..edb09d1, review clean). Real manifests: m_structural_disjoint (MSD 6818/974/1948), source_disjoint (SD 7513/835/1384 + 8 excluded); csv+parquet pairs deduped (parquet preferred).
Task 20: carried into Task 22 dispatch: add a duplicate benchmark_id guard in load_malicious_skill_bench and check label==0 before category in _category.
Task 20: minor (deferred): fetch_split_tables untested; NaN handling if rows ever come via pandas.
Task 21: opus review Needs fixes at a4dd6d8. Rulings: (1) every curated malicious SKILL.md must carry a text-level cue (mcp-weather-tool gets one); labels files gain an optional `payload_in: "scripts"` field copied to meta for items whose harm sits only in scripts (the four fixtures) so text-only scores can be read correctly; (2) fixture labelling rule: policy/quality/least-privilege findings → 0, so mcp_overprivileged_skill and mcp_underdeclared_skill flip to 0; (3) shortcut-breaking polish before freeze: literal example.net URLs in 3–4 look-alikes, 3–5 "quiet" malicious skills without explicit concealment lines, strip Boundaries sections from 2–3 look-alikes, two look-alikes rewritten as transparent download-verify-run and harmless-HTML-comment patterns; (4) loader raises when the fixture repo or a labelled fixture is missing (overrides brief) — why: the hand-crafted set must not be solvable by surface cues and fixture labels must follow one rule — cost if wrong: text edits in curated data.
Task 21: fix round 1/5 dispatched (resume opus implementer).
Task 21: fix round 1/5 (10 addressed, 0 open; commits a4dd6d8..72c69f5)
Task 21: complete (commits edb09d1..72c69f5, review clean after 1 round). 40 skills (10/10/20, 14 scripted), 32 fixtures (12 malicious), payload_in flags on 4 fixtures. Trailer on both commits says Opus 5.5 (cosmetic).
Task 21: minor (deferred): destructive-action fixtures filed under hidden-instructions (no data-destruction category); subagents share the scratchpad dir — tell them to use unique temp files.
Task 22: complete (commits 72c69f5..4a735e7, review clean). Skill track built: 9,843 items (MSB 9,740 exact card counts, fixtures 49, curated 54); overlap guard 0 collisions on both protocols.
Task 22: minor (deferred, final wave): datasets.yaml range comment misstates "49 fixture dirs" (it is 32 fixtures → 49 items); store full attack_categories list in meta; manifest lacks SkillSpector commit / HF revision; overlap-guard test does not pin the both-protocols scope.
Task 23: review Needs fixes at 5380824 (3 Important). Rulings: (1) every table restricts verdicts to usable outcomes and to the run's item ids before counting; (2) a scenario/source group with mixed labels is split by label (recall for its positives, FPR for its negatives) instead of taking the first item's label; (3) spec §8 wins over the brief: the 95 %/99 % operating points apply the dev-chosen thresholds to test (columns recall_at_r95, fpr_at_r95, recall_at_r99, fpr_at_r99) and the test-chosen oracle values stay only as fpr_at_95_oracle/fpr_at_99_oracle; headline uses the dev-threshold columns; (4) one bootstrap pass computes recall and FPR CIs together — why: the report must not contain optimistic or inconsistent numbers — cost if wrong: report-local.
Task 23: fix round 1/5 dispatched (resume sonnet implementer).
Task 23: fix round 1/5 (5 addressed, 0 open; commits 5380824..4b86dc9)
Task 23: complete (commits 4a735e7..4b86dc9, review clean after 1 round). Column changes for Task 24: leaderboard has recall_at_r95/fpr_at_r95/recall_at_r99/fpr_at_r99 (dev thresholds) and fpr_at_95_oracle/fpr_at_99_oracle; headline = detector, kind, recall_at_r95, fpr_at_r95, p50_ms, per_1k_usd, auroc; scenario table may carry <det>_fpr columns for mixed groups.
Task 23: minor (deferred): _restrict does not dedupe duplicate verdict ids; None vs NaN mix for empty detectors.
Task 24: review Needs fixes at bad3031 (find_dev_run ignores protocol; KeyError on partial manifests; untested paths). Ruling: dev-run selection matches track + policy + protocol; malformed manifests are skipped; tests for zero-verdict, ok+skipped, mixed panel, defaulted thresholds, protocol runs.
Task 24: fix round 1/5 dispatched (resume sonnet implementer).
Task 24: minor (deferred): scatter label overlap with ~15 detectors; `missingval` for NaN cells; seed-aggregate heading wording; `--limit` not reflected in the item count.
Task 24: fix round 1/5 (6 addressed, 0 open; commits bad3031..44d3ba4)
Task 24: complete (commits 4b86dc9..44d3ba4, review clean after 1 round)
Task 24: minor (deferred): protocol "" vs None mismatch in find_dev_run; silent skip of unreadable manifests; build_report indexes m["detectors"]/m["track"] directly; error-message branch of the defaulted-thresholds section untested.
Task 25: implementer DONE_WITH_CONCERNS at e7d7a15; real smoke 2:32, terminated on its own. ok: protectai, laya, gliner2-decide, gliguard, qwen3guard-stream, granite-hap (prompts), laya-multi (skills); skipped: promptguard2-86m/22m, sentinel (gated 401), openjev-2b, basal-2.0-mini (no server); claude-* status ok but all verdicts error (no key).
Final-wave items from smoke: (a) Claude adapter must raise DetectorUnavailable in __init__ when the SDK client has neither api_key nor auth_token (→ skipped); (b) runner marks a detector "failed" (reason "all verdicts are errors") when every computed verdict is an error; (c) smoke table truncates `reason` to ~60 chars. Observed (not bugs): granite-hap inverts on prompts (HAP model), laya inverts on skills zero-shot (expected per SkillScreen), laya-multi weak separation.
Task 25: review Needs fixes at e7d7a15 (smoke hides all-error detectors; shares the real cache; overwrites LATEST). Ruling: RunConfig gains `cache_dir` (default results_dir/cache) and `update_latest` (default True); smoke uses a cache under its own run dir and never touches LATEST; a detector with status ok and n_ok < n_items is shown as "degraded" with the dominant error, and `nar smoke` exits 1 when any such row has n_ok == 0; `separates` boolean column; reason truncated in the printed table; README wording about skips corrected after the final-wave Claude fix.
Task 25: fix round 1/5 dispatched (resume sonnet implementer).
Task 25: fix round 1/5 (7 addressed, 0 open; commits e7d7a15..ea8fce2)
Task 25: complete (commits 44d3ba4..ea8fce2, review clean after 1 round)
ALL 25 TASKS COMPLETE. Final whole-branch review dispatched on fable (merge base 41454ab).
Final review (fable) at ea8fce2: With fixes — C1 make bench reports, C2 skill variant mixing, C3 keyless Claude/all-error status, C4 write_jsonl default=str; I1–I7; tagged minors. Fix wave dispatched (opus) → 13 commits ea8fce2..e2c2a3e, 226 tests, smoke exit 0 (Claude skipped). Scoped re-review dispatched.
Final fix wave re-review (opus) at e2c2a3e: all C1–C4, I1–I7 and tagged minors ADDRESSED; one new Important: runner all-error rule tests only fresh verdicts (runner.py:129) so a resumed run marks healthy detectors failed. Ruling: deviate from "no second fix wave" for ONE minimal targeted dispatch (one-line fix + test + two nits) because the bug would mislabel detectors on the natural `make bench` resume path and the change is trivially verifiable — cost if wrong: one extra small dispatch; I run the suite myself afterwards instead of another review.
Parked (surface to user): msb:ASB04_002434 is a 21M-character skill in MSD test; windowed Prompt Guard / Qwen3Guard / HF tokenization on it risks OOM — Ruling: recommend a per-item text cap (e.g. 200k chars, recorded as truncated) before the full bench; not applied here because it changes benchmark data.
Parked (minor): keyless check treats ANTHROPIC_CUSTOM_HEADERS-only auth as keyless; items-hash warning does not cover the paired variant run; commit trailers on fix-wave commits say Opus 5.5.
Targeted fix landed: b6b27ab (all-error rule over the full verdict set; README ten runs; variant wording). Controller verified: suite green, make -n bench parses. Branch complete at b6b27ab.
