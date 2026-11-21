# Muse-Glimmer KV grow / sliding-window ring probe

Drives one Muse-Glimmer model directly (no engine) through a prompt and a decode and
records the logits of every step, so two runs that differ only in KV-cache
configuration can be compared position by position. Written for the 2026-09-29 fix
(`docs/models/muse-glimmer.md`, "What the 2026-09-29 fix changed"): the full-attention
cache grows on demand (2048 rows initially on `ggml_metal`) while the 39
sliding-window layers keep a fixed 4352-row ring, and a grow, a KV snapshot or a
rewind must not change a single logit.

```bash
dotnet build eng/validation/MuseGlimmerKvGrowProbe -c Release
P=eng/validation/MuseGlimmerKvGrowProbe/bin/Release/net10.0/MuseGlimmerKvGrowProbe
M=Muse-Glimmer-30B-UD-IQ2_XXS.gguf

# Reference: a cache big enough from the start (no grow before 8192).
MAX_CONTEXT=8192 $P --model $M --prompt-file long.txt --steps 500 --out ref.json --logits-out ref.f16
# Candidate: the default capacity, teacher-forced to the reference's tokens.
$P --model $M --prompt-file long.txt --steps 500 --teacher ref.json --out cand.json --logits-out cand.f16
python3 eng/validation/MuseGlimmerKvGrowProbe/compare.py ref.json cand.json --logits ref.f16 cand.f16 --mark 2048
```

**Do not build the probe into another app's output folder on its own.** Its dependencies are older than the CLI's (it
resolves Microsoft.Extensions.Logging.Abstractions 6.0, the CLI 10.0), so `dotnet build eng/validation/MuseGlimmerKvGrowProbe
-o TensorSharp.Cli/bin` copies the 6.0 assembly over the CLI's, and the shipped CLI then dies at startup with "Could not
load file or assembly 'Microsoft.Extensions.Logging.Abstractions, Version=10.0.0.0'" (exit 134). Stage S6 hit this on the
GPU box. Build the probe into its own folder, as above, or rebuild `TensorSharp.Cli` after it into the same folder.

Options: `--backend` (default `ggml_metal`), `--system-file`, `--prompt-file` (rendered
as a user message through the GGUF chat template), `--steps`, `--teacher <report>`,
`--logits-out` (fp16 logits of every step), `--tp N`, `--layer-split N` (whole layers over N GPUs; the
runtime family, legacy refuses it), the runtime family's choices below, and `--scenario`:

| Scenario | What it does |
|---|---|
| `decode` | Prefill the prompt in one `Forward`, then decode. |
| `split` | Prefill `--split` tokens, then the rest in a second `Forward` (a grow inside the prefill). |
| `snapshot` | Legacy: prefill, extract 256-token KV blocks covering `--split` tokens, scramble and reset the cache, inject them, forward the rest (what a pooled radix hit does). Runtime (no block snapshots where layers ring): checkpoint the cache at `--split`, write the checkpoint file, release the cache and the checkpoint, forward other rows over it, read the file back and clone it into a new request's slot, continue there (what a restarted server's persisted prefix does). |
| `truncate` | Prefill, rewind to `--split`, forward the rest (what a live-cache rewind does). A rewind the runtime's rings refuse (they overwrote rows the next row reads) is recorded, and the history re-prefilled, as the server's turn does. |
| `checkpoint` | Runtime: checkpoint the cache at `--split`, move the source on with other rows, clone the checkpoint into a new request's slot and continue there (a shared-prefix or prompt-end checkpoint). |
| `slot-growth` | Runtime: one request's slot reserved for its prompt and first token, stepped through the unified batch entry as the engine serves it; the steps grow it as the decode outgrows it. |

With `--split` at or past the prompt's end (teacher-forced runs), `truncate`, `checkpoint`
and `snapshot` act in the decode, when the cache holds `--split` positions: a decode-time
`truncate` forwards the next `--rewind` (16) teacher tokens, rewinds to `--split` and lets the
decode forward them again; `checkpoint` and `snapshot` restore mid-generation. The report's
`Events` list each action, refusal and capacity change, and every step records its cache's
`Capacity` (the runtime's active cache or the stepped slot, read through its internal
accessors; legacy's full-attention capacity) and `RingCells`/`SwaRows` the sliding-window
cells (the runtime's rings, legacy's ring rows).

`--logits-ref <f16>` (and `--logits-ref2`) take another run's `--logits-out` for the same
prompt and teacher and record each step's cosine against it (`Steps[].Cos`, `Cos2`), so a
long run needs no logits file of its own; `compare.py` reads them (`--cos-field Cos2`) when
no `--logits` pair is given, and `--margins` prints the top-two gaps at every argmax
difference (the near-tie evidence). `--nll-text` takes any scenario, its `--split` inside
or past the prefix.

`--nll-text <file>` scores a fixed raw text instead (no chat template, the tokenizer's
special prefix only): the first `--nll-prefix` tokens (default 64) are prefilled in one
`Forward`, and every later token is teacher-forced through one-row forwards, so the run
crosses the same cache grows a generation does. The report's `Nll` object and the last
console line give the mean NLL, the perplexity and the top-1 agreement over the scored
tokens (`--steps` caps them; default all). The unified-runtime migration
(`docs/design/muse-glimmer-unified.md`) scores
`benchmarks/engine_comparison/assets/prefill_corpus.txt` this way, on the legacy model and
on the runtime alike:

```bash
$P --model $M --backend ggml_metal --nll-text benchmarks/engine_comparison/assets/prefill_corpus.txt \
   --out artifacts/muse-glimmer-kv-grow/nll-metal.json
```

`--family legacy|runtime` picks the model: `legacy` (the default) builds the legacy
`MuseGlimmerModel` through its oracle seam (`MuseGlimmerArchitecture.CreateLegacyModel`; stage S9
of `docs/design/muse-glimmer-unified.md` deletes it), `runtime` the unified-runtime family, which
the descriptor builds since stage S8 (`MuseGlimmerArchitecture.CreateRuntimeModel`), so both arms
of an A/B run in one binary. `--scenario bench` times the CLI's `--benchmark` procedure (greedy, end to
end) on either: a prompt of `--bench-prefill` tokens (default 512) cycling 17 ids in one
`Forward`, then `--bench-decode` (128) greedy one-row forwards, `--bench-runs` (2) times;
the report has every run's prefill and decode rates and the best of each.

```bash
$P --model $M --backend ggml_metal --family runtime --scenario bench --bench-prefill 4096 --out bench-rt.json
```

Useful A/B knobs (read at model construction): `MAX_CONTEXT`, `TS_KV_INITIAL_TOKENS`,
`TS_MUSE_GLIMMER_SWA_RING=0` (uniform caches).

The runtime family's measured settings (stage S7 of the design) take an arm each, through the model's internal test seams,
so both sides of an A/B run in one binary:

| Option | Arm |
|---|---|
| `--ring-step-rows N` | the rows a prompt step writes into a sliding-window ring: rings of align256(window + N) cells and steps of at most N rows (the runtime's 512 by default; legacy's chunks are 2048 rows over 4352-row rings) |
| `--ring-step-rows linear` | linear caches on the sliding-window layers (no rings); with `--bench-refill` and `TS_PREFILL_CHUNK=512` the prompt still runs in 512-row steps |
| `--gate-up apart\|fused` | every same-type `ffn_gate`/`ffn_up` pair kept apart or fused, in place of the family's per-backend rule |
| `--no-prune` | a prompt step's last layer runs its feed-forward over every row (`PrunesLastLayerRows` off) |
| `--bench-refill` | the bench prompt goes through `ForwardRefill` (chunks of `TS_PREFILL_CHUNK` rows, the last token's logits only) instead of one `Forward` |
| `TS_METAL_GRAPH_OPTIMIZE=0` (environment) | ggml-metal's concurrency reorder off (the runtime reorders its decode graphs; legacy its persistent decode graph) |

The bench report records each run's host embedding time (`PrefillEmbeddingMilliseconds`, `DecodeEmbeddingMilliseconds`: the
runtime looks a step's rows up on the host and uploads them) and the settings it ran with (`Settings`).

`compare.py` prints argmax agreement and (with `--logits`) cosine similarity before and
after each `--mark` position and exits 1 on any argmax difference or a cosine below
`--min-cos`. `webui_chat_e2e.py` drives a running `TensorSharp.Server.Host` through the
Web UI API (`/api/sessions`, `/api/chat`) and fails a reply in which a short unit
repeats back to back (`--min-repeats`, default 12) or whose tail has almost no distinct
8-grams left.

Outputs belong in the ignored `artifacts/` or `docs/validation/` directories.
