# Muse-Glimmer fixtures

## Chat template

`chat_template.jinja` is the unmodified `tokenizer.chat_template` metadata from
`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`, inspected on 2026-10-11. It ignores
`enable_thinking` and ends every generation prompt with a bare assistant header.
The fixture exercises the actual template's reasoning, recipient, and tool
branches without requiring model weights or a native backend.

## llama.cpp goldens (`LlamaCpp/`)

`MuseGlimmerParityTests` checks Muse-Glimmer against these. They were made on
2026-11-20 by `eng/validation/muse-glimmer-llamacpp-goldens.py` with llama.cpp
**d3c3e242c** (a clean Release Metal build; it includes the upstream Muse Glimmer
vision fix #32433, and its text graph is 1a24df163's after a refactor) from
`Muse-Glimmer-30B-UD-IQ2_XXS.gguf` (sha256 `3e9bd0e6...`, unsloth/Muse-Glimmer-30B-GGUF
at `faa5b025`). Each file records the llama.cpp commit, the server arguments and the
model's sha256.

| File | Contents |
|---|---|
| `ref_text.json` | five chat prompts (English prose, the Chinese prompt that looped in September 2026, code, arithmetic, verse): the prompt text, llama.cpp's prompt tokens, its 128-token greedy continuation, and each generated token's top log-probabilities (10 at the first position, 4 after) |
| `ref_text_long.json` | a 4,664-token prompt (more than twice the 2048-token window and past legacy's 4352-row ring) and its continuation, as above |
| `ref_text_dflash.json` | the five prompts through llama.cpp's DFlash path (`dflash-kquant.gguf`, `--spec-draft-n-max 15`): every continuation is its plain one (Metal) |

The prompts are rendered by the template's rules without its date line, so a
regeneration reproduces them. Regenerate (about 90 s on an M5 Pro, peak 16 GB wired):

```bash
python3 eng/validation/muse-glimmer-llamacpp-goldens.py --llama-cpp ~/work/llama.cpp-lfm3 \
    --model Muse-Glimmer-30B-UD-IQ2_XXS.gguf --draft dflash-kquant.gguf
```

### One set per ggml backend (`LlamaCpp/cuda/`, `LlamaCpp/vulkan/`)

The ggml backends round differently. At IQ2_XXS, llama.cpp's own CUDA build is 0.39-1.72
nats from its Metal build at the first generated position (top-10 log-probabilities), and
its greedy continuations leave Metal's at four of the six prompts. That is past the parity
tests' 0.15-nat bound, so each lane is held to the set of its own ggml backend, at the same
bounds (`MuseGlimmerParity.GoldenSet`):
* ggml_metal and MLX use the Metal set above;
* ggml_cuda uses `cuda/`;
* ggml_vulkan uses `vulkan/`;
* direct CUDA uses the Metal set, which it measures closest to (`MuseGlimmerGoldenSetAffinityTests`:
  0.05 nats, against 0.25 for Vulkan's and 0.75 for CUDA's);
* the managed backend and ggml_cpu measure 1.6-1.9 nats from every set. They take Metal's by
  default and do not meet the first-position bound. llama.cpp's own CPU build is 0.72-1.45
  nats from Metal's set too, and its greedy continuations leave Metal's on all six prompts.

Both sets were made on 2026-11-20 with the same llama.cpp commit d3c3e242c, the same
prompts and the same generator, on an RTX PRO 4500 Blackwell (sm_120, driver 595.112.07,
CUDA 13.2). The builds:
* `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`;
* `-DGGML_VULKAN=ON` (NVIDIA's Vulkan driver).

Each file records the backend under `llama_cpp.backend`. A second CUDA run reproduced
`cuda/` exactly.

| Set | vs Metal, first-position top-10 | Continuations equal to Metal's | DFlash equal to plain |
|---|---|---|---|
| `cuda/` | 0.39-1.72 nats | 2 of 6 (the others leave at 44-109) | 2 of 5: refrigerator, train and haiku leave plain, each first where plain's top-2 margin is a near-tie |
| `vulkan/` | 0.07-0.31 nats | 6 of 6 | 5 of 5 |

```bash
python3 eng/validation/muse-glimmer-llamacpp-goldens.py --llama-cpp ~/llama.cpp --build-dir build --backend cuda \
    --model Muse-Glimmer-30B-UD-IQ2_XXS.gguf --draft dflash-kquant.gguf --out InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/cuda
python3 eng/validation/muse-glimmer-llamacpp-goldens.py --llama-cpp ~/llama.cpp --build-dir build-vk --backend vulkan \
    --model Muse-Glimmer-30B-UD-IQ2_XXS.gguf --draft dflash-kquant.gguf --out InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/vulkan
```

## Frozen legacy outputs

Stage S0 of the unified-runtime migration (`docs/design/muse-glimmer-unified.md`)
froze the legacy model's (`MuseGlimmerModel`) outputs before any stage changed shared
code, so the runtime family can be held to them after the legacy code is deleted.

* `LegacyGoldens/<lane>.json` (`MuseGlimmerGoldens.cs`, `MuseGlimmerFrozenLegacyTests`):
  the logits of every forward of seven scripted cases (a long prefill, prompt steps,
  the ring wrapping, the full layers' cache growing, a rewind; an untied and a tied
  head) on two synthetic fixtures, per lane (cpu, ggml_cpu, ggml_metal, mlx on the M5 Pro; cpu,
  ggml_cpu, ggml_cuda, ggml_vulkan and direct cuda on an RTX PRO 4500 Blackwell box, linux-x64).
* `LegacyReal/<lane>.json` (`MuseGlimmerLegacyRealTests.cs`): on the real IQ2_XXS weights
  (ggml_metal and mlx on the M5 Pro; ggml_cuda, ggml_vulkan and direct cuda on the RTX PRO 4500
  Blackwell box), the greedy continuations of the goldens' prompts with
  each position's top four log-probabilities, and the teacher-forced perplexity of
  `benchmarks/engine_comparison/assets/prefill_corpus.txt`.

The writers are Bench tests that run only when `TS_TEST_FREEZE_MUSE_GLIMMER_GOLDENS` or
`TS_TEST_FREEZE_MUSE_GLIMMER_REAL` names the directory to write into.
