# Campaign 2026-09-16 — evidence analysis (snapshot 19:48 UTC)

Analyst: Claude (worktree `wf_6b35c04d-4b8-3`, on top of integration commit `2a83880`).
Results root: `/workspace/ts-main/results/` on the VM (`campaign/`, `repro/`, `managed-tests*/`, `gates1/`).
Binaries under test: campaign build of commit **8aae2f0** (`libGgmlOps.so` sha256 `48abe2b1…`, `TensorSharp.Server.Host.dll` sha256 `7f04752c…`, recorded in every `profile.json`). None of the integration-branch fixes (2a83880) are in these binaries.

The campaign was still running when this snapshot was taken (lane B on `muse-glimmer/base`, lane C on `modalities/h3-fl2va`, lane D on `gemma4-26b/draft-gpu0` and `gemma4-31b/base-gpu1`). Every number below comes from files that existed at 19:48 UTC; directories marked *running* are partial.

Method: every `profile.json` (status, error, load time, per-suite exit codes), every suite JSON (per-case status, detail, final assistant text, TTFT/decode rates), every `server.log` (errors/warnings/fallbacks/declines) and every AgentTurnBench `rows.json`/`atb.log` were parsed by scripts kept in the analyst scratchpad (`analyze.py`, `atb.py`, `perf.py`, `compare.py`, `mods.py`). "Strict" is the harness verdict. "Lenient" additionally counts a case as passing when the final text is the exact expected JSON wrapped in a single ```` ```json ```` fence (the harness's `check_answer` re-applied after unfencing); no other tolerance is applied. Failure classes: **product** (server/engine defect), **harness** (strict check or runner defect), **environment** (sandbox, VRAM, model file, box-wide kill), **model** (wrong answer, reasoning consumed the budget), **performance**.

---

## 1. Headline findings

| # | Finding | Class | Models | Status vs integration branch |
|---|---|---|---|---|
| 1 | Parallel tool calls streamed with `tool_calls[].index=0` for both calls → any index-merging client sees one call `get_weatherget_weather` (`two_calls` 0/5 on every Gemma 4 size). | product | gemma4 e4b/12b/26b-qat/26b/31b | reported fixed on 2a83880 (not re-verified here) |
| 2 | `KV_CACHE_DTYPE=q8_0/q4_0` silently downgraded to f16 for Gemma 4 and GPT-OSS (one info line, server continues) → the kv-q8/kv-q4 variants measured f16; decode outputs byte-identical to the f16 reference (30/30). Managed test `EngineParallelInferenceTests.Gemma4E4B_Q8_0/Q4_0_LongTurnPrefixReuse` fails for the same reason ("Expected Q8_0, Actual F16"). | product (+doc) | gemma4-e4b, gptoss20b | open (2a83880 only adds DeepSeek-scoped KV-dtype refusals) |
| 3 | Speculative decoding on the **server** path is not greedy-lossless for Gemma 4: ngram and MTP draft-head variants differ from the plain server on 6/6 solo 512-token decode cases (E4B, 12B, 26B-QAT). In-process AgentTurnBench on E4B is identical (192/192), on 12B it diverges at token 10. | product | gemma4 e4b/12b/26b-qat | open |
| 4 | Nemotron-H 8B/47B are served with the wrong chat template (ChatML rendered; the GGUF template is `<SPECIAL_10>System …<SPECIAL_11>User …<SPECIAL_11>Assistant`, eos `<SPECIAL_11>`): outputs carry `</think>`, `<\|im_start\|>` self-dialogue, `<unk>` repetition; quality 15/40, tool-policies 3/29 on 47B. | product | nemotron-h8b, nemotron-h47b | open |
| 5 | Nemotron-H / Nemotron 3.5 / Nemotron Omni long-prompt prefill allocates a single GGML buffer proportional to the prompt (`addmmbatch` 37,312 MiB for 32k on 8B; 12,544 MiB on 47B at 8k; 12,480 MiB on 3.5 at 32k; 18,688 MiB on Omni at 32k) → `cudaMalloc failed: out of memory` → HTTP 500 on every 32k/64k request and on 47B's 8k requests. Prefill that does fit runs at ~100 tok/s (TTFT 72–89 s for 7.9k tokens). | product (perf + memory) | nemotron-h8b, nemotron-h47b, nemotron35, nemotron-omni | open |
| 6 | `NemotronModel.TryMoEPrefillBatchedByExpert` throws `IndexOutOfRangeException` (NemotronModel.cs:2192) when four 51-token prompts prefill in one BatchedPaged step → "Engine executor step failed; failing 4 affected request(s)", HTTP 500 for all 12 `decode@c4` cases. | product | nemotron35 (nemotron_h_moe) | open |
| 7 | `--tp 2` on Nemotron-H 8B/47B aborts at load with an **unhandled** `KeyNotFoundException: TP column-parallel weight 'blk.29.attn_qkv.weight' not found in sharded weights` (47B: `blk.49`) — not the explicit constraint message Gemma 4 gives (`Gemma4 TP validation failed: Global KV heads (1) not divisible by global TP degree (2)`; also an unhandled exception, exit −6). | product (error surface) | nemotron-h8b, nemotron-h47b, gemma4-12b | open |
| 8 | Nemotron 3.5 + DSpark drafter: 0 drafted / 0 accepted in every server request, yet solo greedy output differs from plain on 6/6 decode cases and decode is 0.55× plain; AgentTurnBench `--spec-diagnostic` shows logit error growing 0 → 2.42 across rejected verify windows (SSM state rollback not exact). | product | nemotron35 | open (2a83880's DSpark precision work targets DeepSeek) |
| 9 | Muse-Glimmer HTTP: with `response_format=json_object` the server generates the correct object (`chat.complete … assistantOutput="{"name":"Mars",…}"`) but streams `content: null` (0 tokens delivered, finish=stop); with `json_schema` it returns **422** after generating the correct object; the final tool-answer turn under `json_object` is empty (structured-tools 0/10); `think=false` still leaks reasoning into `multi_turn`. | product | muse-glimmer | open |
| 10 | Mistral-Small-3.1 (`mistral3` catalog entry, bartowski GGUF) cannot load: `NotSupportedException: Unsupported architecture: llama` (registry aliases only `mistral3`). | product/catalog | mistral3 | open |
| 11 | Hunyuan `Hy-MT2-1.8B` (`hunyuan-dense`) answers every `/v1/chat/completions` with 500: `Continuous-batching engine is unavailable for this model (the model supports neither IBatchedPagedModel.ForwardBatch nor IModelArchitecture.SupportsKVStateSnapshot)`; translation 0/90, quality 0/10. | product | hunyuan-dense | open |
| 12 | DiffusionGemma: every final answer starts with raw `<\|channel>thought` (and sometimes `<channel\|>`) markup with thinking off (1/39), and `tool_choice=required` returns 200 prose instead of a 400. | product | diffusiongemma | known, fix in progress |
| 13 | `think=true` + `response_format` → HTTP 400 `response_format cannot be combined with think=true` for every protocol without a `ThinkingGrammarActivationTrigger` (Gemma 4, GPT-OSS, Nemotron-H, Nemotron 3.5): 15/40 quality cases fail in every `--thinking` run. | product (API limit) | gemma4-*, gptoss20b, nemotron-h8b, nemotron35 | known (GPT-OSS), fix in progress |
| 14 | GPT-OSS reasons in the Harmony analysis channel with `think=false` (reasoning_text 2,519 chars on a 512-token budget; content empty) → decode 3/30, long_64k 0/1, json_unicode@c4 0/4 (finish=length). | product / model | gptoss20b | known, fix in progress |
| 15 | Gemma 4 E4B emits its own thought channel after tool results with thinking off: `agentic` final turn = 256 tokens of `<\|channel>thought`, empty content (0/5); `tool_round_trip` final turn duplicated around a stripped `<channel\|>` (`…}\n``````json\n{…`, 0/5 with skills preamble, 2–3/5 without). 12B/26B do not do this. | product (channel stripping) + model | gemma4-e4b | open |
| 16 | ggml_cuda embeddings fail the release batch-vs-single gate (cos ≥ 0.9999): MiniLM min 0.99981, Snowflake min 0.99958 (both ≥ 0.999; rankings identical; cross-backend vs ggml_cpu within the documented 0.999/0.005 gates). | product (precision) or harness (gate) | minilm, snowflake | open |
| 17 | `WeightFusionSplitPathTests.Gemma4AnswersTheSameWhetherOrNotItsGateAndUpWereFused` (ggml_cpu): "only 8 of 24 tokens agree between the fused and split FFN paths; that is structural divergence" on gemma-4-E4B-it-Q8_0. | product (CPU fusion) | gemma4-e4b (CPU) | open |
| 18 | Gemma 4 31B Q4_0 at `MAX_CONTEXT=65536`, 4 running sequences on a 45 GB A40: the process fills the whole GPU (telemetry max 45,497 MiB) and every concurrent request dies with `cudaMalloc failed` (HTTP 500 / ChunkedEncodingError); the server admits work it cannot fit instead of refusing. | product (admission) / environment (card size) | gemma4-31b (running) | open |

Environment/harness items that are **not** product bugs but shaped the numbers: the box-wide SIGKILL at 19:37:01–19:39:25 UTC (killed the nemotron-omni server mid-`decode_8k@c4`, the wan22-t2v-a14b server mid-denoise, the wan-a14b/wan22-i2v/nemotron-omni think+audio runners); no bubblewrap on the host (all 20 agent execution cases per run are "blocked"/refused by design); CPU contention (load average 81–106 on 96 cores during lane A) inflating short-prompt TTFT; the shared runner omitting `--companion-sha256` so `media` exited 2 in `gemma4-e4b/base-gpu0` (re-run as `media-gpu1`, 30/30); the fenced-JSON strict check (see lenient columns).

---

## 2. Per-model results (variant × suite)

Cell format: `strict/total` and, when different, `(lenient)`. "—" = suite not in the variant's plan. `500` = HTTP 500 from the server. Quality/decode/tool suites run at concurrency 1 and 4 (c1/c4); "quality" = short, short_zh, json, json_schema, json_unicode, multi_turn, tool_round_trip, agentic (5 cases each = 1×c1 + 4×c4).

### 2.1 gemma4-e4b (gemma-4-E4B-it-Q8_0 + mmproj, GPU 0/1) — `campaign/gemma4-e4b/`

| variant | quality (40) | structured-tools (10) | tool-policies (29) | long (2) | decode (30) | media (30) | audio (30) | agent (30) | arrivals (21) | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| base-gpu0 (skills-dir on, f16) | 25 (30) | 10 | 24 | 0 (2) | 30 | exit 2 (runner) | 30 | 5 | 15 | agentic 0/5 empty+length (thought channel), tool_round_trip 0/5 duplicated content, two_calls 0/5 index bug, arrivals long_8k 6 fenced |
| think-gpu1 | 10 (25) | — | — | — | — | — | — | — | — | 15 × HTTP 400 (think+response_format), 15 fenced |
| kv-q8_0-gpu1 | 25 (37) | — | — | — | 29 | — | — | — | — | **fell back to f16**; decode outputs byte-identical to ref 30/30; 1 c4 decode never said "collision" |
| kv-q4_0-gpu1 | 25 (35) | — | — | — | 29 | — | — | — | — | **fell back to f16** |
| ref-f16-noskills-gpu0 | 25 (37) | — | — | — | 29 | — | — | — | — | plain reference for the spec/kv comparisons |
| ngram-gpu1 | 25 (34) | — | — | — | 28 | — | — | — | — | engaged; 6/6 c1 decode outputs differ from ref |
| draft-gpu0 (assistant.Q8_0 MTP head) | 25 (37) | — | — | — | 28 | — | — | — | — | engaged (28 % acceptance); 6/6 c1 decode differ from ref; quality 40/40 identical to ref |
| long64k-gpu1 (MAX_CONTEXT 131072) | — | — | — | 0 (1) | — | — | — | — | — | 60,778-token prompt, all 3 codes retrieved, TTFT 14.1 s @ 4,317 tok/s |
| media-gpu1 | — | — | — | — | — | 30 | — | — | — | image_ocr / multi_image / image_follow_up / image_long_context / video_order / video_timestamp, c1+c4 |
| agent-codeexec-gpu0 (TS_CODE_EXEC=1, TS_SKILLS_ALLOW_EXEC=1) | — | — | — | — | — | — | — | 5 | — | server: "--code-exec is set but code execution is unavailable: this host provides no OS sandbox (bwrap (bubblewrap) is not installed on this host) … The shell tool will NOT be offered to the model"; 20 exec cases = blocked, skill_selection 0/5 (sentence around the correct string), skill_run 5/5 |

Quality failure breakdown (every variant): `multi_turn` 0/5 fenced (lenient pass), `long_*` fenced (lenient pass); `tool_round_trip` and `agentic` are the E4B-specific channel problem (#15). `short`, `short_zh`, `json`, `json_schema`, `json_unicode` 25/25 everywhere except the think run.

AgentTurnBench `agentturnbench/e4b-draft-auto.json`: PASS (4,203 fresh tokens in 5 prefill steps); spec plain 81.2 → ngram 298.3 (187 drafted / 160 accepted) → draft head 152.0 tok/s (179/160), **streams identical to plain (192/192)**; newchat B + ngram identical (96/96); json+ngram identical; tool turn 3 reused 3,869/3,889; conc 2 → 66.7, conc 4 → 64.3 tok/s aggregate (PLE declines the batched fused-decode path, warning logged once).

### 2.2 gemma4-12b (gemma-4-12B-it-qat-UD-Q4_K_XL + mmproj-BF16) — `campaign/gemma4-12b/`

| variant | quality | structured-tools | tool-policies | long | decode | media | audio | notes |
|---|---|---|---|---|---|---|---|---|
| base-gpu1 | 25/40 (40) | 10/10 | 24/29 | 0/2 (2) | 30/30 | 30/30 | 30/30 | all 15 failures are fenced JSON; two_calls 0/5 index bug |
| think-gpu0 | 10/40 (25) | — | — | — | — | — | — | 15 × 400; tool calls themselves correct under thinking |
| draft-gpu0 (mtp-gemma-4-12B-it.gguf) | 25/40 (40) | — | — | — | 30/30 | — | — | engaged (draftHeads=16, 32–41 % acceptance), **no speedup** (61.6 vs 60.3; 8k 50.4 vs 63.1 tok/s), 6/6 c1 decode differ from base |
| tp2-gpu01 | server aborted at load (exit −6) | | | | | | | `System.InvalidOperationException: Gemma4 TP validation failed: Global KV heads (1) not divisible by global TP degree (2); Layer 5 KV heads (1) not divisible by global TP degree (2)` (Gemma4Model.TensorParallel.cs:98) — explicit constraint, but surfaced as an unhandled exception |

AgentTurnBench `agentturnbench/12b-draft-auto.json`: PASS batching; spec plain 63.9 / ngram 67.9 / draft head 60.3 tok/s — **both spec streams diverge from plain greedy at token 10 of 192** (`'Collections.Generic;\n   2  '` vs `'\n   1  using System.Collections'`); newchat B + ngram diverges at token 88/96; json+ngram diverges at token 56/69; tool turn 3: "Live-cache continuation declined … would need a 29-token rewind (limit 16)" → only 1,024/3,891 reused, 2,867 re-prefilled (TTFT 1.19 s).

### 2.3 gemma4-26b-qat (gemma-4-26B-A4B-it-qat-UD-Q4_K_XL) — `campaign/gemma4-26b-qat/`

| variant | quality | structured-tools | tool-policies | long | decode | media | notes |
|---|---|---|---|---|---|---|---|
| base-gpu1 | 25/40 (40) | 10/10 | 24/29 | 0/2 (2) | 30/30 | 30/30 | fenced only; two_calls 0/5; "batched paged attention: unavailable (model opt-out …)" |
| draft-gpu0 (mtp-gemma-4-26B-A4B-it.gguf) | 25/40 (40) | — | — | — | 30/30 | — | engaged; **slower**: decode c1 79.6 vs 111.7, 8k 98.3 vs 114.8 tok/s; 6/6 c1 decode differ from base |

### 2.4 gemma4-26b (gemma-4-26B-A4B-it-Q4_0, lane D) — `campaign/gemma4-26b/`

| variant | quality | structured-tools | tool-policies | long | decode | media | notes |
|---|---|---|---|---|---|---|---|
| base-gpu0 | 25/40 (40) | 10/10 | 24/29 | 0/2 (2) | 30/30 | 30/30 | same pattern as QAT; decode c1 98.0, c4 25.5×4 = 102 aggregate |
| draft-gpu0 (assistant.Q8_0) | *running* | | | | | | draft head ready (draftHeads=16, fusedDraft=yes) |

### 2.5 gemma4-31b (gemma-4-31B-it-Q4_0, lane D, *running*) — `campaign/gemma4-31b/base-gpu1/`

quality 15/40 (18 lenient): every c1 case correct (fenced), **c4 cases fail with HTTP 500 / empty content**: `Native GGML gemma4_model_decode failed … failed to allocate persist backend buffer … allocating 258.73 MiB … cudaMalloc failed: out of memory` (×53), plus `RmsNorm` 13 MiB / `Mul` 4 MiB failures; 1,006 `cudaMalloc failed` lines; telemetry shows GPU 1 at 45,497 MiB (the card's full 45,498). structured-tools 4/10, tool-policies 22/29 (2 × 500 + two_calls 0/5), long 0/2 (2) with 8k TTFT 8.0 s @ 960 tok/s and 32k 32.7 s @ 930 tok/s, decode c1 23.8 tok/s (3/3 so far). Load 42 s.

### 2.6 gptoss20b (gpt-oss-20b-MXFP4, GPUs 2,3) — `campaign/gptoss20b/`

| variant | quality | structured-tools | tool-policies | long | decode | agent | arrivals | notes |
|---|---|---|---|---|---|---|---|---|
| base (skills-dir on) | 31/40 (36) | 10/10 | 24/29 | 2/2 | 30/30 | 10/30 | 21/21 | json_unicode@c4 0/4 finish=length (analysis channel 847–912 chars ate the 256 budget), tool_round_trip 0/5 fenced (skills preamble changes the format; plain run passes 5/5); two_calls: 3 × length loops, 2 × only one call parsed; agent: skill_selection/skill_run 5/5, exec cases refused (no sandbox) |
| think | 24/40 (25) | — | — | — | — | — | — | 15 × 400; short@c4 TTFT 13 s (analysis channel inside TTFT) |
| kvq8 | 39/40 (40) | — | — | — | 3/30 | — | — | `[kv-cache] GptOssModel cannot read a q8_0 K/V cache on any of its attention paths; using f16 instead.`; decode fails only because the analysis channel consumes the 512 budget (reasoning_text 1,730–2,519 chars, content empty or truncated) |
| ngram | 39/40 (40) | — | — | — | 4/30 | — | — | `Speculative decoding: requested but unavailable (architecture has no speculative trunk)`; every plan "rejected: SpeculativePerSequence" |
| plain | 40/40 | — | — | — | 3/30 | — | — | comparison baseline (no skills) |
| long64k | — | — | — | 0/1 | — | — | — | 61,430-token prompt fits; analysis channel (865 chars) finds all 3 codes, final answer truncated after 20 chars (256 budget); TTFT 27.6 s @ 2,222 tok/s |

Parity: plain vs kvq8 and plain vs ngram — every c1 case identical (quality + decode); c4 cases differ in 11/30 and 10/30 decode cases → batched greedy is not run-to-run deterministic on gpt-oss (use c1 rows for parity). AgentTurnBench `gptoss20b/atb-ngram`: drafted=0 on every row (`rejected: SpeculativePerSequence: requested (--spec) but this architecture has no speculative trunk`), PASS batching, tool turn 3 reused 3,840/3,881, conc 4 → 309.9 tok/s aggregate.

### 2.7 nemotron-h8b (nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M) — `campaign/nemotron-h8b/`

| variant | quality | structured-tools | tool-policies | long | decode | arrivals | notes |
|---|---|---|---|---|---|---|---|
| base | 15/40 | 4/10 | 25/29 | 0/2 | 30/30 | 6/21 | json/json_schema/json_unicode 15/15 (grammar-forced); every free-text case wrong: `'17 + 25 = 42\n</think>\n\n42\n'`, `'42\n<|im_end|>\n<|im_start|>user\n…<think>…'`, c4 `'101'`, `'1000000…'` (finish=repetition), `'1<unk><unk>…'`; long_8k answer correct but followed by `</think>\n\n<|im_start|>assistant…` (TTFT 71.8 s @ 110 tok/s); long_32k **500** (`addmmbatch … allocating 37312.00 MiB … out of memory`) |
| kvq8 | 13/40 | — | — | — | 30/30 | — | q8_0 KV actually applied (no fallback line; c1 outputs differ from base); same template failures |
| nobatch (TS_NEMOTRON_BATCHED=0) | 15/40 | — | — | — | — | — | c1 identical to base; c4: 3 markup + 1 empty content (finish=stop) + 1 `did not acknowledge` (content null) — the batched path is not the only c4 problem |
| think | 8/40 | — | — | — | — | — | 15 × 400; the rest length/wrong (`'The sum of 12 and 12 is 24'`) |
| tp2 | abort −6 | | | | | | `Unhandled exception. System.Collections.Generic.KeyNotFoundException: TP column-parallel weight 'blk.29.attn_qkv.weight' not found in sharded weights.` |
| long64k | — | — | — | 0/1 | — | — | **500** `addmmbatch … allocating 37312.00 MiB on device 0: cudaMalloc failed: out of memory` after 234.7 s |

### 2.8 nemotron-h47b (nvidia_Nemotron-H-47B-Reasoning-128K-Q4_K_M, GPUs 2,3) — `campaign/nemotron-h47b/`

| variant | quality | tool-policies | long | decode | notes |
|---|---|---|---|---|---|
| base | 15/40 | 3/29 | 0/2 (500) | 15/30 | load 86 s; same template failure as 8B plus `'4<unk><unk>…'` (finish=repetition) and `42\n42\n42…`; tool-policies: every case `finish=length` with the same call repeated 2–6 times and ChatML self-dialogue in content; long_8k/32k and all 15 `decode_8k` → **500** (`addmmbatch … allocating 12544.00 MiB … out of memory`); decode c1 11.7 tok/s, c4 2.9 tok/s per stream |
| tp2 | abort −6 | | | | `KeyNotFoundException: TP column-parallel weight 'blk.49.attn_qkv.weight' not found in sharded weights.` |

### 2.9 nemotron35 (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MXFP4_MOE) — `campaign/nemotron35/`

| variant | quality | structured-tools | tool-policies | long | decode | notes |
|---|---|---|---|---|---|---|
| base | 40/40 | 9/10 | 29/29 | 1/2 | 18/30 | structured c4 one wrong answer `{"temperature": 15}`; long_32k **500** (`addmmbatch … allocating 12480.00 MiB … out of memory`); **decode@c4 0/12: `IndexOutOfRangeException` in `NemotronModel.TryMoEPrefillBatchedByExpert` (NemotronModel.cs:2192) via `ForwardBatch` → "Engine executor step failed; failing 4 affected request(s)"** (3/3 waves, four 51-token prompts prefilling together); decode_8k c1 41.5 tok/s with TTFT 72 s @ 109 tok/s |
| think | 12/40 | — | — | — | — | 15 × 400; multi_turn "OK" turn returns content null; agentic/tool_round_trip finish=length |
| dspark (NVFP4-DSpark drafter) | 39/40 | — | — | — | 30/30 | drafter attached ("dspark(layers=6, block=8, drafts=8 …)", "20 NVFP4 scale2 sidecars", **"DFlash fused drafter disabled: layer 0 has a non-quantized projection"**); `Speculative decoding stats … drafted=0 accepted=0` on every solo request; **decode c1 23.7 tok/s vs 43.2 plain (0.55×)**; c1 outputs differ from base 6/6; decode@c4 passed (BatchedPaged, same prompts) so #6 is timing/batch-composition dependent |
| cpumoe8 (--n-cpu-moe 8) | 40/40 | — | — | — | 26/30 | c1 outputs identical to base (6/6); decode c1 41.8 (vs 43.2), c4 9.1 per stream; 4 c4 cases never reached "collision" in 512 tokens; decode@c4 did **not** hit #6 |

AgentTurnBench `nemotron35/atb-dspark`: prefill 116–173 tok/s, long turn TTFT 35.8 s; spec row drafted 0 / accepted 0 / plain 63; long/tool rows drafted 2–9 accepted 0; **ngram and draft-head streams both diverge from plain at token 51/61** (`' \`Mean\`, and \`Dispose\`'` vs `' and \`Mean\` methods, and proper'`) although nothing was accepted; newchat B + ngram: 36 drafted / 36 accepted, identical. `atb-dspark-diag/rows.json` (`--spec-diagnostic`): logit `max_abs` vs plain 0 at prefill, 0.19 after the first verify+rollback, 1.38 at row 50, 2.42 max; mismatch at row 51 (expected token 2000, actual 1321, margin 0.14).

### 2.10 nemotron-omni (NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_M + mmproj-BF16) — `campaign/nemotron-omni/`

| variant | quality | tool-policies | long | decode | media | notes |
|---|---|---|---|---|---|---|
| base | 38/40 (40) | 29/29 | 1/2 | 17/30 | not run | load 88 s; 2 fenced c4; long_32k **500** (`addmmbatch … allocating 18688.00 MiB … out of memory`); decode_8k c1 TTFT 79.6 s @ **99 tok/s** prefill; decode_8k c1-r2 and all c4 lost to the 19:37:01 SIGKILL (`server_exit_code=-9`, "Server became unavailable after suite 3"); media suite never executed |
| think | killed at start (rc −9, status "starting") | | | | | |
| audio (laneB-custom, short + audio) | killed at start (rc −9) | | | | | |

### 2.11 diffusiongemma (diffusiongemma-26B-A4B-it-Q4_K_M, GPU 4) — `campaign/diffusiongemma/gpu1/`

diffusion suite 1/39 (+ warmup fail): `openai-final-blocking` 0/15, `openai-final-stream` 0/15, `web-replace` 0/7, `web-cancel` 1/1, `unsupported-tool-rejection` 0/1. Every content starts with `<|channel>thought\n` (some `<|channel>thought\n<channel|>`) before a ```` ```json ```` block with the right answer (`json.loads` fails: "Expecting value: line 1 column 1"); `tools`+`tool_choice=required` → 200 with prose ("I do not have the capability…"). Load 22.4 s, 16 GB resident, no CUDA errors; client cancel handled.

### 2.12 muse-glimmer (Muse-Glimmer-30B-UD-Q4_K_XL + mmproj, lane B, *running*) — `campaign/muse-glimmer/base/`

quality 20/40: `short`/`short_zh` 10/10; `json`/`json_unicode` 0/10 — server log `chat.complete tokens=15 … assistantOutput="{"name":"Mars","moons":2,"habitable":false}"` but the SSE stream delivered `content: null` with no token events (harness `t_first_abs: null`, finish=stop); `json_schema` 0/5 — **HTTP 422** after the server logged the correct object (`OpenAIChatAdapter.cs:718/814`: `StructuredOutputValidator.NormalizeOutput` on the parser-filtered content); `multi_turn` 0/5 — the "OK" turn returns null or leaked reasoning (`'.\n\nWe need to parse the instruction…'`) with `think=false`; structured-tools 0/10 (json_object final turn → empty content, one `finishReason=repetition` output beginning `{"sto to=self}…`); tool-policies 8/8 so far. AgentTurnBench `atb-dflash2`: "DFlash fused drafter armed" (block=16, drafts=15), spec plain 32.5 → ngram 48.4 → DFlash2 47.6 tok/s (188 drafted / 57 accepted = 30 %), spec/json streams identical, **newchat B + ngram diverges at token 20/96** (`'   2  using System.Linq;\n'` vs `'...\n```\n\nWe need to repeat this text'`), conc 4 → 35.8 tok/s aggregate.

### 2.13 hunyuan-dense (Hy-MT2-1.8B-Q8_0) — `campaign/hunyuan-dense/`

translation 0/90, quality-short 0/10: every request 500 in ~35–60 ms — `System.InvalidOperationException: Continuous-batching engine is unavailable for this model (the model supports neither IBatchedPagedModel.ForwardBatch nor IModelArchitecture.SupportsKVStateSnapshot)`; startup also logs "Prefix cache warm-up failed" with the same message. Load 22–24 s.

### 2.14 mistral3 (bartowski mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M + f16 mmproj) — `campaign/mistral3/`

base, kvq8, long64k: server exits −6 during load: `Unhandled exception. System.NotSupportedException: Unsupported architecture: llama. Registered: deepseek4, deepseek41, diffusion-gemma, gemma4, glm-dsa, gptoss, hunyuan-dense, minimax-h3, mistral3, muse-glimmer, nemotron_h, qwen3, qwen35, qwen4exp, qwen_image, wan.` (`ModelArchitectureRegistry.Resolve`, ModelBase.cs:2669). The GGUF's `general.architecture` is `llama`; `Mistral3Architecture` registers only the alias `mistral3`.

### 2.15 Non-chat modalities (GPU 4) — `campaign/modalities/`

| profile | status | result | latency (indicative, box loaded) |
|---|---|---|---|
| minilm-ggml_cpu, minilm-cpu, snowflake-ggml_cpu, snowflake-cpu | passed | batch/single min cosine 1.0, 4/4 rankings, 11 invalid requests rejected, 8 concurrent ok, 4 protocols | ggml_cpu: MiniLM 9.4 ms single-short; Snowflake 159 ms single-short / 838 ms medium; pure-C# Snowflake 311 ms / 2.3 s |
| minilm-ggml_cuda, snowflake-ggml_cuda | **failed** | batch-vs-single gate (> 0.9999): MiniLM min 0.999806 (12/13 inputs below), Snowflake min 0.999581 (13/13 below); `*-ggml_cuda-diag`: both pass the 0.999 gate, cross-backend vs ggml_cpu cos ≥ 0.99957 / max component err ≤ 0.0039, rankings identical | ggml_cuda MiniLM 2.4–13.6 ms, Snowflake 5.4–67.7 ms medians (p95 40–50 ms jitter) |
| qwen-image-base (edit-2511 Q4_K_M + VL + VAE) | passed | 19/19 structural (single-image-edit, multi-image-edit c1/c2, stream) | single edit ~85–95 s, multi-image ~252 s; lane C visual check: single edit correct (red→blue, code intact), **multi-image-edit output has both codes garbled and a 3-panel cut layout** (suspected multi-image conditioning defect; no reference impl available here) |
| qwen-image-lightning (4-step LoRA) | passed | 19/19 structural | single ~12 s, multi ~25 s |
| wan-turbo (TI2V-5B-Turbo) | passed | 20/20 structural (t2v/i2v c1/c2, stream, base64) | ~21 s t2v / ~19 s i2v per 33-frame 640×480 clip; frame stats show real motion (mean abs frame diff 26.0) |
| wan-base (TI2V-5B) | passed | 20/20 | ~77 s per clip |
| wan21-13b | passed | 11/11 | ~101 s per clip |
| wan21-14b (Q4_K_M) | passed | 11/11 | ~458 s per clip |
| wan22-t2v-a14b (High+Low noise Q4_K_M) | failed (env) | 2/3 structural then `RemoteDisconnected` — server SIGKILLed at 19:37:06 mid-denoise (`server_exit_code=-9`) | ~711 s per clip |
| wan22-i2v-a14b-base, wan-a14b (lightx2v 4-step) | killed (env) | runners SIGKILLed 19:37:47 / ~19:39:25 before any case; wan-a14b warmup `RemoteDisconnected` | — |
| h3-fl2va (MiniMax-H3 fl2va) | *running* | 7/7 text-to-video structural so far | ~42 s per clip |
| baselines | n/a | `/workspace/tensorsharp-no-patch-20260915/modalities/baseline` does not exist → no PSNR/SNR comparison against the previous build was possible for any modality | |

### 2.16 AgentTurnBench-only models (lane B, GPU 3)

| bench | drafter | spec engaged | speed (plain → spec, tok/s) | greedy parity | other |
|---|---|---|---|---|---|
| qwen36-27b/atb-spec (Qwen3.6-27B-Q4_K_M) | none in file (0 NextN tensors) | `[Warning] Speculative decoding was requested but is not available: the loaded checkpoint carries no draft head`; ngram engages | 28.1 → ngram 71.6 (2.5×, 147/138) | all identical | newchat B: "prompt diverges from the live cache at token 750 of 768, would need an 18-token rewind (limit 16)" → 746/950 reused via checkpoint clone |
| qwen36-moe-mtp/atb-spec (Qwen3.6-35B-A3B-UD-Q4_K_M) | NextN layer 40 (moe=yes) | yes | 98.5 → draft head 147.6 (1.5×, 143/139 = 97 %), ngram 181.2 | spec/newchat identical; **json + ngram diverges at token 50/63** | `Speculative decoding cannot arm … per-token algorithm does not resume after a reused KV prefix (783 tokens adopted)` on tool turn 2 (documented) |
| qwen38-27b-nvfp4/atb-dflash2-default (NVFP4 trunk + DFlash2 Q4_K_M) | DFlash2 (drafts=7, block=8; NextN layer 64 present but DFlash2 takes precedence) | yes | 32.0 → DFlash2 52.5 (1.64×, 152/136 = 89 %), ngram 63.7 | all identical | long prompt decode 14.9 tok/s (34/14) |
| qwen38-27b-nvfp4/atb-dflash2-draft7 (TS_SPEC_DRAFT=7) | same | yes | 32.0 → 47.9 (202/151), ngram 68.6 | all identical | drafts=7 is already the clamp |
| muse-glimmer/atb-dflash2 | DFlash2 (block=16, drafts=15) | yes | 32.5 → 47.6 (1.46×, 188/57 = 30 %) | spec/json identical; newchat B + ngram diverges at token 20 | conc 4 → 35.8 aggregate |
| nemotron35/atb-dspark | DSpark (per-op, fused disabled) | armed, **0 accepted** | 34.7 → 17.4 (0.5×) | diverges at token 51 (both ngram and draft head) | see #8 |

---

## 3. Consolidated bug list with evidence

Severity: S1 = wrong output / crash / feature unusable; S2 = degraded or misleading; S3 = cosmetic/strictness. Paths are under `/workspace/ts-main/results/`.

| id | sev | class | title | evidence |
|---|---|---|---|---|
| B1 | S1 | product | Nemotron-H 8B/47B wrong chat template (ChatML rendered; GGUF wants `<SPECIAL_10>/<SPECIAL_11>`) → `</think>` and `<\|im_start\|>` self-dialogue in content, `<unk>` repetition, quality 15/40, tool-policies 3/29 (47B) | `campaign/nemotron-h8b/base/quality.json` (short-c1: `'17 + 25 = 42\n</think>\n\n42\n'`), `campaign/nemotron-h47b/base/tool-policies.json`, `campaign/nemotron-h47b/base/server.log` (`finishReason=repetition … "4<unk><unk>…"`); source `ChatTemplate.RenderNemotron` via `ChatProtocolRegistry.cs:330`; `docs/models/nemotron.md:625` states ChatML |
| B2 | S1 | product | Nemotron-H / 3.5 / Omni long-prompt prefill allocates one buffer proportional to prompt length → CUDA OOM → HTTP 500 at 32k/64k (and 8k on 47B) | `campaign/nemotron-h8b/long64k/server.log` (`addmmbatch … allocating 37312.00 MiB`), `campaign/nemotron-h47b/base/server.log` (12544 MiB, ×10), `campaign/nemotron35/base/server.log` (12480 MiB), `campaign/nemotron-omni/base/server.log` (18688 MiB) |
| B3 | S1 | product | `NemotronModel.TryMoEPrefillBatchedByExpert` IndexOutOfRangeException on a 4-sequence batched prefill (Nemotron 3.5 MoE) → all 4 requests 500 | `campaign/nemotron35/base/server.log:1831-1843` (stack: NemotronModel.cs:2192 → MoEForward:1839 → NemotronModel.BatchedForward.cs:711 → BatchExecutor.cs:868 → InferenceEngine.cs:361), `campaign/nemotron35/base/decode.json` decode@c4 0/12 |
| B4 | S1 | product | Nemotron-H `--tp 2` load crash: unhandled `KeyNotFoundException: TP column-parallel weight 'blk.29.attn_qkv.weight' not found in sharded weights` (47B: `blk.49`) — a crash, not a constraint message | `campaign/nemotron-h8b/tp2/server.log`, `campaign/nemotron-h47b/tp2/server.log`, `profile.json` `server_exit_code: -6` |
| B5 | S1 | product | Nemotron 3.5 + DSpark: 0 tokens ever accepted, greedy output diverges from plain (6/6 solo decode cases), decode 0.55× plain; diagnostic shows trunk logits drift after each rejected window | `campaign/nemotron35/dspark/server.log` (`drafted=0 accepted=0` ×20, "DFlash fused drafter disabled: layer 0 has a non-quantized projection"), nemotron35 base vs dspark decode c1 6/6 differ (first diffs at chars 299/45/12/123), `campaign/nemotron35/atb-dspark/rows.json`, `campaign/nemotron35/atb-dspark-diag/rows.json` |
| B6 | S1 | product | Muse-Glimmer JSON mode: correct object generated but stream delivers `content: null`; `json_schema` → 422; tool final turn under `json_object` empty; reasoning leaks with `think=false` | `campaign/muse-glimmer/base/server.log:276-300,344-352`, `campaign/muse-glimmer/base/quality.json` (json-c1: `assistant_message.content: null`, `t_first_abs: null`), `structured-tools.json` 0/10; source `TensorSharp.Server/ProtocolAdapters/OpenAIChatAdapter.cs:706-718` (stream-parser output fed to `StructuredOutputValidator`) |
| B7 | S1 | product/catalog | `mistral3` catalog GGUF has `general.architecture=llama`; registry refuses it | `campaign/mistral3/base/server.log` (NotSupportedException), `eng/validation/release-model-catalog.json` (`bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF`), `TensorSharp.Models/Models/Mistral3/Mistral3Architecture.cs:16-18` |
| B8 | S1 | product | Hunyuan `Hy-MT2-1.8B`: no batched forward and no KV snapshot → continuous-batching engine unavailable → every chat request 500 | `campaign/hunyuan-dense/translation/server.log` (×97), `campaign/hunyuan-dense/quality-short/quality.json` |
| B9 | S1 | product | Gemma 4 streams two parallel tool calls both with `index: 0` → merged into `get_weatherget_weather` | `campaign/gemma4-*/base*/tool-policies.json` (`calls=["get_weatherget_weather"]`), raw SSE in `repro/gemma-e4b-repro.txt` (two_calls: two deltas, both `"index": 0`, distinct ids) — reported fixed on 2a83880 |
| B10 | S1 | product | Gemma 4 E4B post-tool thought channel with thinking off: `agentic` final turn 256 tokens of `<\|channel>thought` → empty content; `tool_round_trip` content duplicated around a stripped bare `<channel\|>` | `campaign/gemma4-e4b/base-gpu0/quality.json` (agentic-c1: content "" finish=length; tool_round_trip-c1: `…}\n``````json\n{…`), `repro/gemma-e4b-repro.txt` agentic-turn3 (18 completion tokens, content "") |
| B11 | S1 | product | Speculative decoding on the server path changes greedy output for Gemma 4 (ngram and MTP head): 6/6 solo decode cases differ on E4B, 12B, 26B-QAT while the in-process bench on E4B is identical | e4b ref vs ngram decode c1 first diff at chars 976/228/992/1104, ref vs draft 477/228/25/45; 12b base vs draft 674/259/176; 26b-qat 5/40/683/161 (near-tie synonym flips, e.g. "deterministic algorithm" vs "mathematical function"); `agentturnbench/e4b-draft-auto.json` identical vs `12b-draft-auto.json` diverges at token 10 |
| B12 | S2 | product (+doc) | `KV_CACHE_DTYPE=q8_0/q4_0` downgraded to f16 for Gemma 4 and GPT-OSS with a single info line; docs (`USAGE.md:290`, `docs/models/gemma4.md` §8) promise quantized caches on the native flash path | `campaign/gemma4-e4b/kv-q8_0-gpu1/server.log` (`[kv-cache] Gemma4Model cannot read a q8_0 K/V cache on any of its attention paths; using f16 instead.` then `KV cache: 1044 MB (dtype: f16 …)`), same for kv-q4_0 and `campaign/gptoss20b/kvq8/server.log`; `managed-tests2/models-cpu.log` `Gemma4E4B_Q8_0_LongTurnPrefixReuse` Expected Q8_0 Actual F16 |
| B13 | S2 | product | Gemma 4 E-series / 26B MoE decline the batched fused-decode path → c4 aggregate ≈ c1 (E4B 16.0×4 = 64 vs 61.5 c1; 26B-QAT 27.9×4 = 112 vs 112 c1); 12B batches (35.9×4 = 144 vs 60 c1) | "The model declined the default batched fused-decode path for a N-sequence decode step …" in every gemma4-e4b/26b `server.log`; §5.1 |
| B14 | S2 | product | Gemma 4 MTP draft heads give no speedup on 12B (61.6 vs 60.3) and a slowdown on 26B-QAT (79.6 vs 111.7 tok/s) despite 32–41 % acceptance; E4B gains 1.67× (102.6 vs 61.5) | §5.1; `campaign/gemma4-12b/draft-gpu0/server.log` spec stats |
| B15 | S2 | product | `think=true` + `response_format` → 400 for protocols without `ThinkingGrammarActivationTrigger` (Gemma 4, GPT-OSS, Nemotron-H, Nemotron 3.5) | every `*/think*/quality.json` json* 0/15; message at `OpenAIChatAdapter.cs:270` |
| B16 | S2 | product/model | GPT-OSS reasons at think=false (analysis channel), consuming the completion budget: decode 3/30, long_64k truncated after 20 answer chars, json_unicode@c4 0/4 | `campaign/gptoss20b/plain/decode.json` (decode-c1-r0: content "" reasoning_text 2,519 chars finish=length), `campaign/gptoss20b/long64k/long-context.json` |
| B17 | S2 | product | DiffusionGemma leaks `<\|channel>thought` markup and accepts `tool_choice=required` with 200 prose | `campaign/diffusiongemma/gpu1/diffusion.json` (38/39 fail), `server.log` |
| B18 | S2 | product | Gemma 4 31B at 64k context with 4 sequences fills the 45 GB card and fails requests with `cudaMalloc failed` rather than refusing admission | `campaign/gemma4-31b/base-gpu1/server.log` (1,006 OOM lines), `telemetry.jsonl` GPU 1 max 45,497 MiB |
| B19 | S2 | product (precision) | ggml_cuda embeddings below the 0.9999 batch-vs-single gate (MiniLM 0.999806, Snowflake 0.999581); ≥ 0.999 and rankings identical | `campaign/modalities/minilm-ggml_cuda/embeddings.json`, `…-diag/embeddings-diag.json` |
| B20 | S2 | product (CPU) | Fused vs split Gemma 4 FFN paths on ggml_cpu agree on only 8/24 tokens | `managed-tests2/models-cpu.log` `WeightFusionSplitPathTests.Gemma4AnswersTheSameWhetherOrNotItsGateAndUpWereFused` |
| B21 | S2 | product | Nemotron-H batched path at c4 produces wrong answers (`101`, `1000000…`, `107\n</think>…` loop) that the serial path does not; `nobatch` c4 instead yields one empty and one null content | `campaign/nemotron-h8b/base/quality.json` vs `campaign/nemotron-h8b/nobatch/quality.json`; base vs nobatch: c1 identical, c4 16/20 differ. Managed test `NemotronBatchedCorrectnessTests.Nemotron_Greedy_LegacyAndBatchedAgree` could not run (backend init) |
| B22 | S2 | product | ngram speculation under a JSON grammar diverges from plain on qwen35moe (token 50/63) and gemma4-12b (token 56/69); ngram over a shared-prefix checkpoint clone diverges on muse-glimmer (token 20/96) and gemma4-12b (token 88/96) | `campaign/qwen36-moe-mtp/atb-spec/rows.json`, `agentturnbench/12b-draft-auto.json`, `campaign/muse-glimmer/atb-dflash2/rows.json` |
| B23 | S3 | product | Gemma 4 12B tool follow-up turn re-prefills 2,867 tokens because the re-rendered prompt differs from the raw tokens at the truncated turn-2 boundary (29-token rewind > 16 limit); E4B reuses 3,869/3,889 on the same turn | `agentturnbench/12b-draft-auto.json` tool turn 3 note |
| B24 | S3 | product | qwen-image multi-image edit visually wrong (both codes garbled, 3-panel cut) while single-image edit is correct — structural pass only | `campaign/modalities/qwen-image-base/generation/multi-image-edit-*/` (lane C visual inspection; no reference implementation on the box) |
| B25 | S3 | product (error surface) | Gemma 4 12B `--tp 2` constraint is explicit but still an unhandled exception (SIGABRT, exit −6) | `campaign/gemma4-12b/tp2-gpu01/server.log` |

Native-gate and managed-test failures that belong to the binaries but are already targeted by 2a83880 (per its message; not re-run here): `gates1/tp7-checkpoint-shape` (`TP relative error exceeds the full-weight reference tolerance` at layer 0, tokens=16, rel_l2 4.06e-05 vs 1.2e-07 for 1/5 tokens); `gates1/dspark-cuda1-prefix17`, `dspark-cuda2-prefix17`, `dspark-cuda2-state` (`AssertionError: rewound_confidence_p1_accepted1`, 16/17, 17/18, 13/14 checks; CPU variants pass); `DeepSeek4NativeSpecBoundaryTests` ×5 ("Sequence contains no matching element") and `Qwen4ExpMtpIntegrationTests` ×6 (`Assert.Single() … collection was empty`) under the pinned env of `managed-tests3` (2a83880 changes both test files).

---

## 4. Speculative decoding cross-check

| model / variant | drafter | engaged (drafted/accepted) | plain → spec decode tok/s | greedy parity (solo) | verdict |
|---|---|---|---|---|---|
| gemma4-e4b/ngram-gpu1 (server) | ngram n=2..8 | yes (per request 2–24 % acceptance) | 55.8 → 80.3 (decode), 77.7 → 76.5 (8k) | **differs 6/6** (first diff at char 228–1104) | product (B11) |
| gemma4-e4b/draft-gpu0 (server) | assistant.Q8_0 MTP (draftHeads=4) | yes (e.g. 923/224, 808/225 on 512-token cases) | 55.8 → 102.6 (1.84×), 8k 77.7 → 85.3 | **differs 6/6** (chars 25–477) | product (B11) |
| gemma4-e4b ATB (in-process) | ngram / MTP | 187/160, 179/160 | 81.2 → 298.3 / 152.0 | identical 192/192 | ok |
| gemma4-12b/draft-gpu0 (server) | mtp-gemma-4-12B (draftHeads=16) | yes (32–41 %) | 60.3 → 61.6 (1.0×), 8k 63.1 → 50.4 (0.8×) | **differs 6/6** | product (B11, B14) |
| gemma4-12b ATB | ngram / MTP | 214/123, 248/150 | 63.9 → 67.9 / 60.3 | **diverges at token 10** | product (B11) |
| gemma4-26b-qat/draft-gpu0 (server) | mtp-gemma-4-26B-A4B | yes | 111.7 → 79.6 (0.71×), 8k 114.8 → 98.3 | **differs 6/6** | product (B11, B14) |
| gptoss20b/ngram (server) + ATB | ngram | **0 drafted** ("architecture has no speculative trunk") | 101.8 → 115.9 (noise) | identical (c1) | not engaged, explicit |
| nemotron35/dspark (server) + ATB | DSpark (dflash block 8) | armed, **0/0** | 43.2 → 23.7 (0.55×) | **differs 6/6**; ATB diverges at token 51 | product (B5) |
| qwen36-27b ATB | none in file / ngram | ngram 147/138 | 28.1 → 71.6 | identical | ok (`--spec` warns: no draft head) |
| qwen36-moe-mtp ATB | NextN layer 40 | 143/139 (97 %) | 98.5 → 147.6; ngram 181.2 | identical; json+ngram diverges at 50/63 | ok except B22; "cannot arm" after reused prefix (documented) |
| qwen38-27b-nvfp4 ATB (default / draft7) | DFlash2 | 152/136, 202/151 | 32.0 → 52.5 / 47.9; ngram 63.7 / 68.6 | identical | ok |
| muse-glimmer ATB | DFlash2 (block 16) | 188/57 (30 %) | 32.5 → 47.6; ngram 48.4 | spec identical; newchat+ngram diverges at 20/96 | ok except B22 |

Prefix-reuse notes: `newchat` chat B reuses 746–776 tokens through the shared-prefix checkpoint clone on every model (e4b 776/977, 12b 776/981, qwen 746/950, muse 512/937, gptoss 768/983); tool turn 3 reuses ≥ 3,584/3,8xx on every model except gemma4-12b (1,024, B23). The only "cannot arm" warning is the documented qwen35moe per-token one; nemotron35 logs one plan "rejected: SpeculativePerSequence: sequence K/V lives in paged storage; linear cache would be empty".

---

## 5. Performance

Box caveat: a shared 7×A40 (45 GB) host; during lane A the load average was 81–106 on 96 cores (another lane ran `InferenceWeb.Tests` and an nvcc build), which inflates every fresh-prompt TTFT (e.g. gemma4-12b short@c1 397 ms for 43 tokens = 108 tok/s "prefill"). Decode rates and long-prompt prefill rates (GPU-bound) are more trustworthy than short-prompt TTFT. Medians over cases; decode tok/s measured on 512-token answers.

### 5.1 Load, TTFT, decode (medians from the suite JSONs)

| model/variant | load s | short@c1 ttft ms | decode@c1 tok/s (ttft) | decode@c4 per-seq tok/s (x4 = aggregate) | decode_8k@c1 tok/s (ttft ms, prefill tok/s, prompt) | decode_8k@c4 per-seq | long_8k ttft/prefill | long_32k ttft/prefill | long_64k ttft/prefill |
|---|---|---|---|---|---|---|---|---|---|
| gemma4-12b/base-gpu1 | 16.3 | 397.4 | 60.3 (248.6) | 35.9 (x4=143.6) | 63.1 (3079.6, 2482.1, 7644) | 15.1 | 3324.2 ms / 2307.3 | 12504.2 ms / 2433.9 | - |
| gemma4-12b/draft-gpu0 | 12.3 | 294.3 | 61.6 (345.4) | 34.8 (x4=139.2) | 50.4 (3304.7, 2313.1, 7644) | 15.0 | - | - | - |
| gemma4-26b/base-gpu0 | 26.3 | 339.5 | 98.0 (366.9) | 25.5 (x4=102.0) | 106.4 (2334.9, 3273.8, 7644) | 24.6 | 2141.0 ms / 3582.4 | 8856.1 ms / 3436.5 | - |
| gemma4-26b-qat/base-gpu1 | 20.4 | 280.3 | 111.7 (276.0) | 27.9 (x4=111.6) | 114.8 (2099.5, 3640.9, 7644) | 26.8 | 2056.6 ms / 3729.5 | 9088.9 ms / 3348.5 | - |
| gemma4-26b-qat/draft-gpu0 | 14.3 | 267.9 | 79.6 (414.8) | 26.9 (x4=107.6) | 98.3 (2312.4, 3305.7, 7644) | 26.2 | - | - | - |
| gemma4-31b/base-gpu1 (running, OOM at c4) | 42.2 | 912.6 | 23.8 (2716.7) | - | - | - | 7992.7 ms / 959.6 | 32717.5 ms / 930.2 | - |
| gemma4-e4b/base-gpu0 | 20.4 | 199.5 | 61.5 (195.8) | 16.0 (x4=64.0) | 63.9 (1697.8, 5033.5, 8546) | 16.2 | 1960.4 ms / 4372.6 | 7834.1 ms / 3999.9 | - |
| gemma4-e4b/draft-gpu0 | 12.3 | 242.5 | 102.6 (181.8) | 19.4 (x4=77.6) | 85.3 (1546.3, 4940.7, 7640) | 18.7 | - | - | - |
| gemma4-e4b/kv-q4_0-gpu1 (=f16) | 28.4 | 298.3 | 57.5 (369.4) | 13.7 (x4=54.8) | 59.0 (2106.5, 3626.8, 7640) | 18.6 | - | - | - |
| gemma4-e4b/kv-q8_0-gpu1 (=f16) | 18.4 | 354.3 | 46.1 (402.3) | 11.0 (x4=44.0) | 50.0 (1988.2, 3842.6, 7640) | 13.9 | - | - | - |
| gemma4-e4b/long64k-gpu1 | 14.3 | - | - | - | - | - | - | - | 14079.1 ms / 4316.9 (60778 tok) |
| gemma4-e4b/ngram-gpu1 | 14.3 | 197.0 | 80.3 (186.5) | 19.8 (x4=79.2) | 76.5 (1472.6, 5188.3, 7640) | 18.9 | - | - | - |
| gemma4-e4b/ref-f16-noskills-gpu0 | 24.5 | 398.1 | 55.8 (450.2) | 14.5 (x4=58.0) | 77.7 (1790.5, 4266.9, 7640) | 18.8 | - | - | - |
| gptoss20b/base | 22.3 | 132.8 | 114.9 (151.7) | 66.6 (x4=266.4) | 96.4 (2485.4, 3467.9, 8619) | 60.7 | 2607.0 ms / 3314.9 | 10896.6 ms / 2901.4 | - |
| gptoss20b/kvq8 (=f16) | 30.6 | 389.1 | 82.6 (201.2) | 58.9 (x4=235.6) | 96.9 (2389.2, 3260.1, 7789) | 50.9 | - | - | - |
| gptoss20b/long64k | 24.5 | - | - | - | - | - | - | - | 27648.3 ms / 2221.8 (61430 tok) |
| gptoss20b/ngram (not engaged) | 22.4 | 131.6 | 115.9 (122.6) | 63.7 (x4=254.8) | 90.4 (2406.4, 3236.8, 7789) | 51.7 | - | - | - |
| gptoss20b/plain | 22.4 | 117.2 | 101.8 (165.4) | 76.2 (x4=304.8) | 127.2 (1904.0, 4090.9, 7789) | 63.0 | - | - | - |
| muse-glimmer/base (running) | 36.3 | 465.3 | - | - | - | - | - | - | - |
| nemotron-h47b/base | 86.2 | 1808.8 | 11.7 (1952.7) | 2.9 (x4=11.6) | 500 (OOM) | 500 | 500 | 500 | - |
| nemotron-h8b/base | 22.3 | 869.3 | 42.6 (961.1) | 13.0 (x4=52.0) | 42.8 (71016.1, 111.0, 7884) | 3.2 | 71805.5 ms / 110.2 | 500 | - |
| nemotron-h8b/kvq8 (q8_0 applied) | 20.3 | 352.9 | 37.5 (485.2) | 13.6 (x4=54.4) | 11.5 (43649.5, 180.6, 7884) | 3.3 | - | - | 500 (long64k variant) |
| nemotron-omni/base | 88.4 | 1935.9 | 42.7 (633.5) | 8.8 (x4=35.2) | 39.3 (79574.7, 99.0, 7867) | killed | 88799.9 ms / 88.9 | 500 | - |
| nemotron35/base | 36.4 | 553.3 | 43.2 (613.6) | 500 (B3) | 41.5 (72243.1, 108.9, 7867) | 5.5 | 79530.5 ms / 99.3 | 500 | - |
| nemotron35/cpumoe8 | 40.3 | 736.9 | 41.8 (1718.6) | 9.1 (x4=36.4) | 41.0 (89350.3, 88.0, 7867) | 5.3 | - | - | - |
| nemotron35/dspark | 30.2 | 541.4 | 23.7 (681.0) | 7.3 (x4=29.2) | 23.2 (76425.4, 102.9, 7867) | 4.7 | - | - | - |

GPT-OSS run-to-run spread between three identical f16 configs (plain/kvq8/ngram) is ~25 % (101.8 / 82.6 / 115.9 tok/s at c1): treat ±25 % as noise on this box.

### 5.2 Comparisons the lane plan asked for

- **plain vs spec**: E4B MTP head 1.67–1.84× at c1 (server), ngram 1.44×; 12B MTP 1.0×; 26B-QAT MTP 0.71×; Nemotron 3.5 DSpark 0.55×; in-process: E4B ngram 3.7× / MTP 1.9×, qwen35moe NextN 1.5×, qwen38 DFlash2 1.64×, muse DFlash2 1.46×, qwen36-27b ngram 2.5×. Every server-side Gemma spec variant fails greedy parity (B11).
- **kv f16 / q8_0 / q4_0**: Gemma 4 and GPT-OSS never ran quantized KV (B12) — their "q8"/"q4" rows above are f16 measured under different box load. Nemotron-H 8B did apply q8_0: decode c1 37.5 vs 42.6 tok/s, decode_8k c1 11.5 vs 42.8 (TTFT 43.6 s vs 71.0 s — the 8k prefill was faster with q8_0, 180.6 vs 111 tok/s), outputs differ from f16 as expected for a quantized cache; quality is dominated by B1 either way.
- **tp1 vs tp2**: no tp2 run produced output. Gemma 4 12B: explicit KV-head constraint (B25); Nemotron-H 8B/47B: crash (B4). Lane D's `gemma4-31b tp2`, `qwen35-35b tp2 *` and lane B's `qwen36-moe-mtp tp2` are still queued.
- **cpu-moe**: Nemotron 3.5 `--n-cpu-moe 8`: c1 decode 41.8 vs 43.2 (−3 %), decode_8k TTFT 89.4 s vs 72.2 s (+24 %), c4 per-stream 9.1 vs (500); greedy output identical at c1 (6/6). Queued: gemma4-26b `cpumoe-all`, muse-glimmer `cpumoe8`, qwen36-moe `cpumoe8-spec`, qwen35-35b `cpumoe12`.
- **concurrency 4**: aggregate gains 4.0× on gpt-oss (66.6×4 = 266 vs 115 c1 — note that c4 is a *per-stream* figure, so aggregate = 4× it), 2.4× on gemma4-12b (144 vs 60), ≈1.0× on Gemma 4 E4B and 26B (B13), 1.2× on nemotron-h8b (52 vs 42.6), 0.8× on nemotron-omni (35 vs 43).
- **long context**: 64k works on Gemma 4 E4B (60,778 tok, 14.1 s, 4,317 tok/s, all codes retrieved) and GPT-OSS (61,430 tok, 27.6 s, 2,222 tok/s, codes found in the analysis channel); 32k on Gemma 4 12B/26B/31B (2.4k / 3.4k / 0.93k tok/s prefill); every Nemotron 32k/64k request → 500 (B2). Gemma 4 31B prefill (930–960 tok/s) is 3.7× slower than 26B-A4B on the same card.
- **media**: Gemma 4 E4B image_ocr c1 TTFT 1.9 s, image_long_context 2.5 s (c4 3.7–10.2 s); audio transcribe warmup TTFT 30.4 s (1,111 prompt tokens at 36.5 tok/s prefill — the audio encoder dominates), 30/30 on E4B and 12B.
- **modalities** (GPU 4, 640×480×33 frames): wan-turbo 19–21 s, wan-base 77 s, wan21-1.3B 101 s, wan21-14B 458 s, wan22-A14B 711 s, MiniMax-H3 fl2va 42 s; qwen-image edit 85 s (base) / 12 s (lightning); embeddings ggml_cuda 2–14 ms (MiniLM), 5–68 ms (Snowflake).

---

## 6. Managed tests and native gates (`managed-tests*/`, `gates1/`)

| lane | result | classification of failures |
|---|---|---|
| portable (CPU, 4,243 tests) | 4,238 pass, 2 fail, 3 skip | `SkillHostOptionsTests.Capabilities_AFamilyNothingCanParse_IsNotOfferedTools(qwen4exp)` and `TensorAgentMauiProjectTests.Head_RetainsBonsaiNativeEntryPointsInRelease` (expected `TSGgml_Qwen4ExpTestQsa` in `GgmlExportedSymbols.targets`) — both pass (51/51) in `managed-tests2/portable-refix` after rebuilding the test project → stale test binary; both files are also touched by 2a83880 |
| cuda-required (159) | 157 pass, 2 skip | clean |
| models-cuda (152) | 64 pass, 20 fail, 68 skip | 6 `Qwen4ExpMtpIntegrationTests`: `DirectoryNotFoundException …/fixtures/qwen4exp-mtp/sample.json/manifest.json` (env var pointed at the file, not the dir) — environment; 5 `DeepSeek4NativeSpecBoundaryTests` + 2 `DeepSeek41DsparkIntegrationTests`: `Expected "0"/"1024" Actual null` (pinned env vars absent) — environment; 7 × `Failed to initialize ggml-cpu. A different GGML backend was already initialized in this process` (`WeightFusionSplitPathTests` ×2, `NemotronBatchedCorrectnessTests`, `DeepSeekNativeRetentionFixtureTests` ×4) — harness (mixed backends in one process) |
| models-cpu (152) | 61 pass, 23 fail | as above plus: `WeightFusionSplitPathTests.Gemma4…` real divergence 8/24 tokens (B20); `EngineParallelInferenceTests.Gemma4E4B_{Q8_0,Q4_0}_LongTurnPrefixReuse` Expected Q8_0/Q4_0 Actual F16 (B12); `Gemma4CacheResidencyTests`, `MuseGlimmerKvSnapshotTests` ×2 need ggml-cuda in a cpu process (harness); `DeepSeekNativeRetentionFixtureTests` ×4 `NullReferenceException` on CPU (needs the pinned env of managed-tests3) |
| managed-tests3 (DSV4.1 + Qwen4Exp, pinned env) | cpu 141/156, cuda 137/156 | `DeepSeek4NativeSpecBoundaryTests` ×5 `Sequence contains no matching element`; `Qwen4ExpMtpIntegrationTests` ×6 `Assert.Single() … empty`; `DeepSeek41DsparkIntegrationTests` ×2 and `DeepSeekNativeRetentionFixtureTests` ×4 backend-init (cuda lane) / NRE (cpu lane) — the two test families are modified by 2a83880 ("test fixes"); not re-run on the new tree |
| native ctest | 28/28 pass | the "Failed graph destroyed rank=…" lines are expected negative-test output |
| c-abi-boundary | 32/32 pass | |
| tp7-checkpoint-shape | **fail** | `TP relative error exceeds the full-weight reference tolerance` (layer 0, tokens=16, rel_l2 4.06e-05); 2a83880 "TP oracle stability" rewrites `dsv41_tp_test.cpp` |
| dspark CPU state/prefix17 | pass | |
| dspark CUDA (1 GPU prefix17, 2 GPU state, 2 GPU prefix17) | **fail** | `AssertionError: rewound_confidence_p1_accepted1` (16/17, 13/14, 17/18 checks pass); 2a83880 "DSpark batch-invariant precision dispatch" + `eng/tests/dsv41-dspark-inference.py` |
| qwen-mtp CPU/CUDA (8 configs) + target snapshot | pass | |
| qwen-qsa CPU/CUDA | pass (20 numeric + 21 ABI cases) after the fixture was supplied (first attempt rc=2: missing fixture) | |
| slot-retention cpu/cuda | pass | |
| dspark-converter-tests | not run | `No module named pytest` — environment |

---

## 7. Environment and harness issues

1. **Box-wide SIGKILL 19:37:01–19:39:25 UTC**: `nemotron-omni/base` server (mid `decode_8k@c4`, exit −9), `nemotron-omni/think` and `/audio` runners (rc −9 at "starting"), `modalities/wan22-t2v-a14b` server (mid-denoise, exit −9; 2/3 clips done at 711 s each), `wan22-i2v-a14b-base` and `wan-a14b` runners (rc 137, `RemoteDisconnected` on warmup). No OOM record (container has no kernel log); host RAM 503 GB with 100 GB used; lane D's notes also record the event as external. Every affected run must be re-queued.
2. **No bubblewrap / user namespaces**: `--code-exec` and `--skills-allow-exec` are refused by the server ("this host provides no OS sandbox (bwrap (bubblewrap) is not installed on this host) … The shell tool will NOT be offered to the model") — all 20 execution cases per agent run are refused/blocked by design; the harness `--sandbox-unavailable` mode marks them `blocked`. The one unconfined repro (`repro/agent-unconfined.json`, `--code-exec-unconfined`) is explicitly not accepted by the suite and itself passed only shell_run and code_generation_run (skill script not found in the unconfined workspace; code_edit hit IndentationError loops).
3. **CPU contention** (load 81–106 / 96 cores during lane A; nvcc + `InferenceWeb.Tests` from other lanes) — short-prompt TTFT and small-prompt "prefill tok/s" are unreliable; decode and ≥ 8k prefill rates are usable.
4. **Runner defect**: shared `tools/run_model.py` omits `--companion-sha256` for the media suite → `validate_deepseek41_media.py` exits 2 (`gemma4-e4b/base-gpu0` suite-5); lanes A/D use a patched copy.
5. **Strict JSON check**: `multi_turn`, `long_*`, `tool_round_trip`, `agentic` require bare JSON; Gemma 4 (all sizes) and GPT-OSS (with the skills preamble) answer in ```` ```json ```` fences → 15/40 quality and 2/2 long cases per Gemma run fail strictly but pass leniently (see the lenient columns). The arrivals suite's `long_8k` semantic check has the same strictness (6/21 on E4B are exactly this).
6. **`decode` budget vs reasoning models**: the 512-token decode answer must contain "collision"; GPT-OSS spends the budget in its analysis channel (B16) and Nemotron 3.5 / Gemma 4 E4B occasionally write 2,000+ chars without reaching collisions — model behaviour, not a serving fault.
7. **Skills preamble**: with `--skills-dir` every request carries ~900 prompt tokens (`prompt_tokens` 945 vs 44) and changes model formatting (GPT-OSS tool_round_trip fenced only with the preamble) — the `base` variants with skills are not comparable with no-skills variants.
8. **Missing baselines**: `/workspace/tensorsharp-no-patch-20260915/modalities/baseline` absent → no PSNR/SNR regression check for any generative modality.
9. **ts-verify build** (the user's working tree) failed to configure at 19:34 (`add_subdirectory ../ExternalProjects/ggml is not an existing directory`) and was rebuilt successfully at 19:45 (native sha256 `3e701f6e…`); no campaign result yet uses it.
10. `nemotron-h8b/base` runner hit a one-line `laneB-run.sh` syntax error (`unexpected EOF while looking for matching '"'`) after the run finished — cosmetic, results intact.

---

## 8. Coverage matrix (lane plan vs executed at 19:48 UTC)

| lane | planned | executed | pending / not run |
|---|---|---|---|
| A (gemma4 E4B/12B/26B-QAT) | e4b: base, think, kv-q8, kv-q4, ref, ngram, draft, media, long64k, agent-exec, ATB; 12b: base, think, draft, tp2, ATB; 26b-qat: base, draft | all 17 (tp2 = load abort) | qwen35 moved to lane D |
| B (GPUs 2,3) | gptoss20b ×6 + ATB; nemotron-h8b ×6; nemotron-h47b ×2; nemotron35 ×4 + ATB ×2; nemotron-omni ×3; hunyuan-dense ×2; mistral3 ×3; muse-glimmer ×5 + ATB; qwen36-27b ×6 + ATB; qwen36-moe-mtp ×5 + ATB; qwen38-27b-nvfp4 ×4 + ATB ×2 | gptoss ×6 + ATB; h8b ×6; h47b ×2; nemotron35 ×4 + ATB ×2; omni base (partial), think/audio (killed); hunyuan ×2 (500s); mistral3 ×3 (load abort); muse ATB + base (running); qwen36-27b ATB; qwen36-moe ATB; qwen38 ATB ×2 | muse think/dflash2/cpumoe8/long64k; qwen36-27b base/plain/think/spec/kvq8/long64k; qwen36-moe base/think/spec/cpumoe8-spec/tp2; qwen38 base/dflash2/dflash2-draft7/long64k; omni think/audio/media re-run |
| C (GPU 4) | embeddings ×6 (+2 diag); diffusiongemma; qwen-image ×2; wan ×7; minimax-h3 (fl2va, ref2va) | embeddings 8/8; diffusiongemma; qwen-image 2/2; wan-turbo, wan-base, wan21-13b, wan21-14b; wan22-t2v (2/3, killed) | wan22-i2v-a14b-base, wan-a14b (killed, re-run needed); h3-fl2va running; h3-ref2va not started |
| D (GPUs 0,1; started 19:39) | gemma4-26b base/draft/cpumoe-all + ATB; gemma4-31b base/think/ctx32k/tp2; qwen35-9b base/think/long64k/kvq8/kvq4/ngram + ATB; qwen35-35b cpumoe12/tp2-base/tp2-think/tp2-ngram; then qwen38 on the ts-verify build | gemma4-26b base; gemma4-26b draft (running); gemma4-31b base (running) | everything else |
| never planned on this box | glm52, glm53, glm53-flash, deepseek4/deepseek41 (+DSpark) HTTP runs, qwen38 (catalog `qwen38`) | — | the integration branch's GLM-5.3-Flash speculation, DSpark precision policy and Qwen 3.8 video changes have **no campaign evidence** yet (only the fixture gates in §6) |

Suites that produced no usable evidence anywhere: Nemotron-Omni media/audio, Mistral 3 anything, Hunyuan translation, `--tp 2` throughput on any model, quantized KV on Gemma 4 / GPT-OSS.

---

## 9. Documentation contradictions to fix (behaviour observed vs docs)

- `USAGE.md:290` / `:687` and `docs/models/gemma4.md` §8: `q8_0`/`q4_0` KV caches are described as supported on the native GGML flash path; on this build Gemma 4 and GPT-OSS silently serve f16 (B12). Either the fallback must become a refusal (as 2a83880 did for DeepSeek) or the docs must list the models that honour the setting.
- `docs/models/gemma4.md` §13a ("Gemma 4 runs under `--tp N`"): not for 12B (single global KV head) — document the KV-head divisibility constraint per size.
- `docs/models/nemotron.md:625` ("Chat template uses the ChatML format"): the Nemotron-H Reasoning-128K GGUFs ship a `<SPECIAL_10>/<SPECIAL_11>` template with `{'reasoning': True|False}` in the system prompt (B1).
- `eng/validation/release-model-catalog.json` `mistral3`: the referenced bartowski GGUF is architecture `llama` and cannot load (B7).
- Agent workflows docs / harness: state that on hosts without user namespaces the agent execution cases are expected to be refused (they are, by design) so the 5/30 does not read as a regression.

No repository source or docs were modified for this analysis; this file is the only artifact.
