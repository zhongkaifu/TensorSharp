# System-prefix warmup and session reuse — 2026-09-17

The render-safe common system/tool prefix can be cached at startup and reused by
fresh conversations. Private conversation scopes remain separate.

| Application | Eager warmup | New chat/session in the loaded engine | Later process launch |
| --- | --- | --- | --- |
| Server | Configured startup model, before serving; both thinking modes | Reuses matching public prefix | Restores disk checkpoint when the model supports export/import |
| Interactive CLI | System/tool prefix at startup | `/new` and `/reset` retain the engine and rotate private scope | Starts cold; no CLI disk store |
| TensorAgent | After model load, selected default thinking mode; may defer/skip for active work | Reuses matching public prefix | Restores disk checkpoint when the model supports export/import |

Only the matching rendered tokens are reusable. Tool schemas, tokenizer, chat
template and thinking settings affect that prefix. Cache budgets and eviction
still apply. Server/TensorAgent declare prefixes of at least 64 tokens; CLI
eager warmup has the same minimum, while actual CLI requests can populate
shorter prefixes. Copyable checkpoints preserve exact boundaries; page-only
models reuse complete pages. Models without copyable checkpoints or reusable
pages cannot share a system checkpoint across sessions. Persistence additionally
requires export/import support; Qwen4Exp currently has in-memory checkpoints but
does not advertise persistence.

`--no-prefix-cache` and `TS_SCHED_PREFIX_CACHE=0` disable reuse and eager warmup.
Independent CLI JSONL requests use separate private scopes and can share their
declared system prefix. Extra `--warmup-runs` now include the configured system
prompt and retain eligible public cache state for the actual run.

## Gaps corrected

- CLI startup warmup previously declared no public prefix, and `/new` destroyed
  the engine. It now publishes the shared boundary and preserves public cache
  state when changing conversation scope.
- Runtime clamped a public prefix to one token shorter than the prompt. A
  system-only warmup can now checkpoint its entire prompt; admission still
  leaves one token to forward when an identical prompt needs fresh logits.
- Host shared-prefix render memoization omitted tokenizer/template identity and
  complete tool schemas. Reloads or schema changes now use the correct render.
- TensorAgent and Server use the same warmup request/completion checks. Empty,
  unterminated or aborted streams cannot report successful warmup, and disabled
  runtime caching skips it. A completed request is not itself proof of retained
  state, so the success text no longer claims that guarantee.

## Validation

| Check | Result |
| --- | --- |
| Main non-benchmark suite excluding external-model, CUDA and MLX requirements | 5,159 passed, 4 skipped, 0 failed |
| TensorAgent warmup, memory policy and background generation | 66 passed, 0 skipped, 0 failed |
| Actual Gemma 4 E2B Q8_0 interactive CLI | Warmed 641 tokens once; first chat and chat after `/new` both reused 641/641 |
| Gemma 4 E2B Q8_0 on Metal | System-only one-token warmup; two fresh sessions reused 1,217 tokens; checkpoint restoration matched cold output |
| Qwen 3.5 9B IQ4_XS on Metal | System-only one-token warmup; two fresh sessions reused 1,176 tokens; checkpoint restoration matched cold output |

Both model tests used context 4,096 and compared the second session's 12 output
tokens against a cold engine, then repeated the exact comparison after checkpoint
export/import into a fresh engine. Persisted payload sizes were 13.1 MB for Gemma
and 87.0 MB for Qwen. These tests validate copied model state; host render tests
separately cover system/developer/tool boundaries, tokenizer reloads, template
changes and schema changes. Runtime regressions also cover full 1-token and
non-page-aligned 13-token warmups, session isolation and restored reuse.

The four skipped tests and reasons are unchanged from
[the default-integration validation](radix-default-2026-09-17.md#automated-coverage)
and are not counted as passes. Main-suite TRX:
`InferenceWeb.Tests/TestResults/radix-startup-unit.trx`. Local native results:
`/tmp/radix-startup-native/`; CLI log: `/tmp/radix-cli-startup-smoke.log`;
TensorAgent log: `/tmp/radix-tensoragent-system-warmup-tests.log`.

The CLI smoke run used a long system file and piped `Say apple.`, `/new`,
`Say pear.`, `/quit` into the real CLI with `--interactive --max-tokens 1` and
`--backend ggml_metal`. Only one startup warmup appeared in the log.

No native or upstream ggml source changes were made. The upstream checkout stays
clean at `456172ec733a135778adcd32d00e576a58232e45`; the existing native library
identity is recorded in the default-integration validation. Physical iOS devices,
CUDA/Vulkan, every model/backend combination, and throughput were not tested.
