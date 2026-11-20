# LongContextQualityProbe

Measures what an arithmetic DEFAULT CHANGE does to a model's output at long context, between two arms, on one loaded
model. It was written for two changes of issue #376:

- the CUDA executor's 8-bit prompt attention (design t2-attn8 sec. 12);
- q8_0 as the default KV cache on `--backend cuda` (design t3 sec. 3, b2).

Its companion `eng/validation/long-context-quality.py` builds the case file, pairs and grades the runs, checks the
pre-registered gates, and estimates GPU time. Everything it writes goes under ignored directories (`artifacts/` or
`docs/validation/`).

## Arms

- **Arms that differ in an in-process seam run in one process.** `--arms f16,8bit` sets the CUDA executor's
  `PromptAttentionForTests` (the path `--prompt-attention` chooses) per arm, case by case, with `ResetKVCache()` before
  each.
  - `default` keeps the process's own choice (`--prompt-attention`, f16 unless set). `f16` keeps the 16-bit prompt
    kernel. `f32acc` runs the prompt kernels with F32 accumulation. `8bit` asks for the 8-bit prompt attention.
  - An `8bit` arm is refused where the image has no 8-bit kernels. The arm would otherwise silently run the 16-bit
    kernel.
  - Names are free: `--arms ref=f16,cand=8bit`.
  - `q8act` keeps the build's prompt attention and runs every NVFP4 product on the q8 matvec, 16 rows a call
    (`CudaAllocator.AllNvfp4RowsOnMatvecForTests`). Prompt rows then keep the q8 activations of decode rows, where they
    are otherwise quantized to NVFP4 for the FP4 GEMMs: `--arms default,q8act` over `ppl` and `continue` cases measures
    what the prompt path's NVFP4 activations cost. `f16+q8act` and `8bit+q8act` pair a prompt attention with q8
    activations, so `--arms ref=f16+q8act,cand=8bit+q8act` compares the two attentions without the W4A4 rounding of NVFP4
    prompt rows (about 0.16 KL on Qwen3.8-27B prose) in either arm. Records carry `q8act` next to
    `prompt_attention`. A model without NVFP4 weights runs an arm with q8act as the same arm without it. An arm with
    q8act is refused where the image has no FP4 tensor-core products (anything but an sm_120a / sm_121a build on its GPU):
    NVFP4 prompt rows never take the W4A4 GEMMs there, so the arm would silently run without it.
- **Arms that differ in what is fixed at load run as separate processes over the same case file.** For example,
  `--kv-cache-dtype f16` and `--kv-cache-dtype q8_0`, which spell the option as the CLI and server do.
  - The report pairs them by case id and checks that their inputs are identical (the sha256 of the token ids).
  - `--reference RUN.jsonl:ARM` makes a second process compute each scored token's KL divergence against the first
    process's stored top-K distribution.
  - A reference record whose input or scored rows differ from the case's is refused (`kl_refused`), never compared row
    by row. A KL names the distribution it was computed against (`kl_reference_topk_sha256`, the sha256 of that record's
    `topk`), and the report shows it only when that is its own reference record's.
- **No arm is ever chosen through process environment variables.**

## Case kinds

| kind | what runs | what a record holds |
|---|---|---|
| `ppl` | A corpus window's prefix of `tokens - score_tokens` as 2,048-row prompt chunks (`--prefill-chunk`, the served solo chunk). Then the scored rows as marked prompt chunks of `--score-chunk` (512) rows with every row's logits (`StandardDecoderModel.ForwardPromptRowsForTests`). | Per scored token: NLL, argmax, the top-K log-probabilities (`--topk`, 32), and the KL against the reference. Also the PPL. |
| `continue` | The same prefix, then the scored tokens one decode row at a time (`Forward`). This is the served situation: a long prompt, then generation. | As `ppl`. |
| `greedy` | A code prompt cut at a member boundary, then `new_tokens` greedy decode rows. | The generated ids, each step's top-2 logit gap (also the step that chose end of sequence, so `gaps` then has one more entry than `generated`), and the text. |
| `chat` | A chat prompt rendered with the model's own template. Its `{{document}}` holds corpus paragraphs, with the case's inserts at depth fractions, fitted to `fit_tokens` exactly with the model's tokenizer. Then greedy decoding to end of sequence or `max_new_tokens`. Used for retrieval, GSM8K and HumanEval. | The prompt tokens and sha256, the generated ids and gaps, and the text. Grading is in the report. |
| `layers` | An `8bit` arm's prompt, with the executor's check hook: each 8-bit launch is followed by the 16-bit kernel on the same rows. | The relative RMS and max \|d\| of each launch. |

Every record also carries:
- the 8-bit launches, band refusals and marked prompt-chunk attentions it caused. These show that each arm ran what it
  claims.
- the prefill, scoring and decode times. `estimate` extrapolates from these.
- the free device memory after the case.

How failures are recorded:
- An out-of-memory case is recorded as an error.
- The larger cases of its kind and arm are then recorded as "not run".
- `--max-tokens` skips cases above a footprint.
- Skips and errors stay in the output and in the report. They are never counted as passes:
  - a case the reference ran and the candidate failed makes its gate FAIL;
  - any other case of a gate that did not pair (skipped, not run, out of memory in both arms) makes a passing gate
    INCOMPLETE, and a gate with no paired case is NOT RUN;
  - `report` exits 0 only when every gate passed, 1 when one failed, and 3 when one was NOT RUN or INCOMPLETE.
- `--resume` continues a killed run. It also retries the cases an earlier pass skipped, such as a first pass under `--max-tokens`.

## Build and run (an sm_120a / sm_121a build for the 8-bit arm)

Outputs go under ignored directories: `artifacts/prompt-attention/` for the 8-bit prompt attention and `artifacts/kv-q8/`
for q8_0 K/V (the integration plan's session S5).

```
dotnet build eng/LongContextQualityProbe -c Release
python3 eng/validation/long-context-quality.py cases --preset full --out artifacts/prompt-attention/cases
dotnet eng/LongContextQualityProbe/bin/Release/net10.0/LongContextQualityProbe.dll \
    --model Qwen3.8-27B-NVFP4-MTP-N4_0.gguf --cases artifacts/prompt-attention/cases/cases.jsonl \
    --output artifacts/prompt-attention/attn8.jsonl --arms f16,8bit
python3 eng/validation/long-context-quality.py report --ref artifacts/prompt-attention/attn8.jsonl:f16 \
    --cand artifacts/prompt-attention/attn8.jsonl:8bit --gates t2-attn8 --execute-humaneval \
    --out artifacts/prompt-attention/report.md --json artifacts/prompt-attention/report.json
```

For q8_0 as the default KV, measured with the 8-bit prompt attention on, as t3 sec. 8 orders it. It reuses the same case
file and the 8bit arm above as its reference:

```
dotnet .../LongContextQualityProbe.dll --model M --cases artifacts/prompt-attention/cases/cases.jsonl \
    --output artifacts/kv-q8/kvq8.jsonl --kv-cache-dtype q8_0 --arms 8bit \
    --reference artifacts/prompt-attention/attn8.jsonl:8bit
python3 eng/validation/long-context-quality.py report --ref artifacts/prompt-attention/attn8.jsonl:8bit \
    --cand artifacts/kv-q8/kvq8.jsonl:8bit --gates t3-kvq8 --execute-humaneval --out artifacts/kv-q8/report.md
```

- After the q8_0 default lands, `--arms default` with and without `--kv-cache-dtype f16` compares the shipped default with
  F16 K/V.
- A configuration that cannot load is written as a `load_failed` run record, and the probe exits 3. The report then names
  the load error instead of pairing anything.

**NVFP4 models: the ppl gate on q8 activations.** On the FP4 tensor cores NVFP4 prompt rows run W4A4 (decode rows keep q8
activations), and that rounding (about 0.16 KL a prompt row on Qwen3.8-27B prose) swamps an attention change on the prompt
path, which only re-seeds it: the 8-bit attention's served-arm ppl swung -8% and +1% on two windows. So for such a model
the 8-bit prompt attention's `ppl` gate is read from a second process whose arms keep q8 activations, and every other gate
from the served arms. `report --kinds` grades and gates only the kinds it names:

```
dotnet .../LongContextQualityProbe.dll --model M --cases C --output artifacts/prompt-attention/q8act.jsonl --kinds ppl \
    --arms f16q=f16+q8act,a8q=8bit+q8act
python3 eng/validation/long-context-quality.py report --ref artifacts/prompt-attention/q8act.jsonl:f16q \
    --cand artifacts/prompt-attention/q8act.jsonl:a8q --kinds ppl --gates t2-attn8 --out artifacts/prompt-attention/report-ppl.md
python3 eng/validation/long-context-quality.py report --ref artifacts/prompt-attention/attn8.jsonl:f16 \
    --cand artifacts/prompt-attention/attn8.jsonl:8bit --kinds continue,layers,greedy,retrieval,gsm8k,humaneval \
    --gates t2-attn8 --execute-humaneval --out artifacts/prompt-attention/report-served.md
```

**Noise floor.** Run the same cases on another backend, such as `--backend ggml_cuda --arms gg=default --reference
attn8.jsonl:f16`. Then `report --floor RUN.jsonl:gg` tables that numerics difference, which the project already ships,
beside the candidate's, without gates. Read the candidate's ΔPPL and KL against it.

**GPU time.** `python3 eng/validation/long-context-quality.py estimate --run SMOKE.jsonl --cases FULL/cases.jsonl` turns
a small run's measured timings into GPU hours, per arm and summed:
- Prefill is fitted as a L + b L² from the run's prompt-path records, and decode and scored rows as linear in the
  context.
- The `smoke` preset has calibration points at 4K-64K.

## Inputs (pinned)

| input | source | pin |
|---|---|---|
| prose | Moby-Dick (Project Gutenberg #2701, public domain), from its mirrors. The license header and footer are stripped; paragraphs are whitespace-collapsed. | sha256 `f12eb530…` |
| code | `TensorSharp.Runtime/**/*.cs` at 4c819ed2 via `git show`, as issue #376's reproducer builds it. | git revision |
| GSM8K | openai/grade-school-math `test.jsonl` at 3101c7d5 (MIT). This is the same file `InferenceWeb.Tests/Fixtures/Qwen38/gsm8k-mini.json` pins. | sha256 `3730d312…` |
| HumanEval | openai/human-eval `HumanEval.jsonl.gz` at b9453e8f (MIT). | sha256 `b796127e…` |

- Downloads are cached under `artifacts/quality/downloads`.
- The case file's header records each corpus's sha256, and the probe refuses a corpus that differs.
- `report --execute-humaneval` runs generated code with these limits:
  - `python3 -I -S` in a fresh directory and session;
  - CPU, address-space, file-size and file-count limits, with a 10 s wall clock;
  - common destructive calls removed.
- Where the limits cannot be set (macOS address space), the report refuses to grade instead of counting anything.
