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

Options: `--backend` (default `ggml_metal`), `--system-file`, `--prompt-file` (rendered
as a user message through the GGUF chat template), `--steps`, `--teacher <report>`,
`--logits-out` (fp16 logits of every step), `--tp N`, and `--scenario`:

| Scenario | What it does |
|---|---|
| `decode` | Prefill the prompt in one `Forward`, then decode. |
| `split` | Prefill `--split` tokens, then the rest in a second `Forward` (a grow inside the prefill). |
| `snapshot` | Prefill, extract 256-token KV blocks covering `--split` tokens, scramble and reset the cache, inject them, forward the rest (what a pooled radix hit does). |
| `truncate` | Prefill, rewind to `--split`, forward the rest (what a live-cache rewind does). |

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

Useful A/B knobs (read at model construction): `MAX_CONTEXT`, `TS_KV_INITIAL_TOKENS`,
`TS_MUSE_GLIMMER_SWA_RING=0` (uniform caches).

`compare.py` prints argmax agreement and (with `--logits`) cosine similarity before and
after each `--mark` position and exits 1 on any argmax difference or a cosine below
`--min-cos`. `webui_chat_e2e.py` drives a running `TensorSharp.Server.Host` through the
Web UI API (`/api/sessions`, `/api/chat`) and fails a reply in which a short unit
repeats back to back (`--min-repeats`, default 12) or whose tail has almost no distinct
8-grams left.

Outputs belong in the ignored `artifacts/` or `docs/validation/` directories.
