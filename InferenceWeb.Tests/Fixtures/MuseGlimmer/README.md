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

### Image goldens (`ref_vision.json`, one set per backend)

`MuseGlimmerVisionParityTests` teacher-forces llama.cpp's greedy continuations of nine image prompts through both
families: the four `eng/minimax-m3-bench-media` questions, descriptions of two of those pictures and of the wide synthetic
picture in `Vision/`, and a left/right and a top/bottom question about the two synthetic pictures there. They were made with the Q8_0 mmproj by
`eng/validation/muse-glimmer-llamacpp-vision-goldens.py`, llama.cpp d3c3e242c, on 2026-11-20. As for text, there is one
set per backend, and each lane takes its own (`MuseGlimmerParity.GoldenSet`):
* `ref_vision.json`: Metal (M5 Pro);
* `cuda/ref_vision.json` and `vulkan/ref_vision.json`: the RTX PRO 4500 Blackwell builds above. A second CUDA run
  reproduced `cuda/` exactly, timings aside.

Only the float-activation lanes (ggml_metal, MLX, direct CUDA) are held to their set's first position
(`MuseGlimmerParityTests.FirstPositionRuleOf`); ggml_cuda, ggml_vulkan and the CPU lanes log it.

| Set | vs Metal, first-position top-10 | Continuations equal to Metal's |
|---|---|---|
| `cuda/` | 0.09-1.01 nats | 6 of 9 (the three descriptions leave at 14, 22 and 32) |
| `vulkan/` | 0.03-0.22 nats | 6 of 9 (they leave at 4, 50 and 50) |

Every short answer is the same in the three sets: 7319, Red, OCEAN, 3, Left, bottom.

```bash
python3 eng/validation/muse-glimmer-llamacpp-vision-goldens.py --llama-cpp ~/llama.cpp --build-dir build --backend cuda \
    --model Muse-Glimmer-30B-UD-IQ2_XXS.gguf --mmproj mmproj-Muse-Glimmer-30B-Q8_0.gguf \
    --out InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/cuda
python3 eng/validation/muse-glimmer-llamacpp-vision-goldens.py --llama-cpp ~/llama.cpp --build-dir build-vk --backend vulkan \
    --model Muse-Glimmer-30B-UD-IQ2_XXS.gguf --mmproj mmproj-Muse-Glimmer-30B-Q8_0.gguf \
    --out InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/vulkan
```

### Legacy's image figures (`legacy_vision.<lane>.json`, next to the lane's set)

Stage S9 deletes the legacy model and its tower. The image gates that hold the runtime family relative to legacy read
legacy's figures, frozen per lane next to the set that lane uses. `MuseGlimmerLegacyVisionWriterTests` wrote them on
2026-11-20 on the RTX PRO 4500 Blackwell box (sm_120, driver 595.112.07, CUDA 13.2, ggml d9fe7bff unchanged), with the
IQ2_XXS model and the Q8_0 mmproj:

| File | Lane | Legacy's argmax agreement | top-k \|dlogprob\| mean / p50 / p90 / p99 / max | Rows: 1 - cosine to the F32 tower |
|---|---|---|---|---|
| `legacy_vision.cuda.linux-x64.NVIDIA.RTX.PRO.4500.Blackwell.json` | direct CUDA (Metal set) | 215/216 (99.54%) | 0.125 / 0.090 / 0.260 / 0.458 / 0.673 | (not frozen: the lane multiplies F32) |
| `cuda/legacy_vision.ggml_cuda.linux-x64.NVIDIA.RTX.PRO.4500.Blackwell.json` | ggml_cuda | 207/212 (97.64%) | 0.320 / 0.273 / 0.575 / 0.956 / 1.066 | 8.3e-4 to 4.7e-3 |
| `vulkan/legacy_vision.ggml_vulkan.linux-x64.NVIDIA.RTX.PRO.4500.Blackwell.json` | ggml_vulkan | 210/217 (96.77%) | 0.183 / 0.130 / 0.384 / 0.743 / 1.358 | 6.2e-5 to 1.7e-3 |
| `legacy_vision.ggml_metal.osx-arm64.Apple.M5.Pro.json` | ggml_metal | 214/216 (99.07%) | 0.090 / 0.043 / 0.242 / 0.477 / 0.577 | 8.5e-6 to 3.4e-4 (summed 9.90e-4) |
| `legacy_vision.mlx.osx-arm64.Apple.M5.Pro.json` | MLX | 215/216 (99.54%) | 0.106 / 0.059 / 0.267 / 0.514 / 0.684 | 6.2e-6 to 2.8e-4 (summed 7.44e-4) |
| `legacy_vision.ggml_cpu.osx-arm64.Apple.M5.Pro.json` | ggml_cpu (Metal set) | 210/216 (97.22%) | 0.392 / 0.330 / 0.654 / 1.085 / 1.458 | 8.4e-4 to 5.6e-3 (summed 2.34e-2) |

The three Mac files were written after the switch, just before S9 deleted the legacy tower: from d492bfe9 (the last commit
with the legacy model and tower; 757e0517 plus the S9 freeze of the synthetic figures), with the writer extended to every
lane but direct CUDA, on the M5 Pro (48 GB, macOS 27.2), ggml d9fe7bff unchanged, 2026-11-20; each file's `provenance` says
so. Each picture also records the graph tower's own distance at that commit and its distance to legacy's rows
(`graphOneMinusCos`, `legacyToGraphOneMinusCos`: at most 3.6e-5 on ggml_metal and 4.4e-5 on MLX, which share their rounding; up to 5.0e-3 on ggml_cpu, whose q8_K activations
the two towers round independently, as on ggml_vulkan).

* A position's top-k |dlogprob| is the largest |our log-probability - llama.cpp's| over llama.cpp's listed top tokens
  (ten at the first generated position, four after it): the parity tests' first-position metric, at every position.
* The F32 tower is the graph tower with F32 products (`GraphProgram.FloatProducts`): direct CUDA's where the process can
  open a CUDA device, else the managed executor's. The two agree to 1 - cosine 1e-9.

How `MuseGlimmerVisionParityTests` uses them:
* ggml_cuda and ggml_vulkan (their products quantize the activations): the runtime family's argmax agreement with the
  lane's set is at least legacy's frozen agreement minus one percentage point, and every parting is logged with both
  engines' margins. The graph tower's 1 - cosine to the F32 tower's rows, summed over the nine pictures, is at most 1.25x
  legacy's frozen sum; each picture's ratio is logged. Per picture the ratio is that of two independent roundings at
  1e-4 on ggml_vulkan (0.74-1.67x measured, 1.00x summed), so it is not held.
* direct CUDA (float activations; the Metal set): argmax agreement at least 97%, and the top-k |dlogprob| distribution's
  mean, p50 and p90 no more than 10% past legacy's frozen ones; the p99 and max, and every parting, are logged. The tail is
  not held because it is noise at this size: over 216 positions p99 is the third-largest value, the largest values are the
  same few positions of the picture past the position table for every family and tower, and legacy's own p99 moves 20%
  (0.458 to 0.548) when it is given the graph tower's rows, 1.3e-5 from its own. One position (wide-describe @32) flips
  with such changes too (`ImageRuleOf`). The graph tower's rows are held, picture by picture, to cosine 0.99999 against
  the managed executor's F32 tower.
* ggml_metal, MLX and the CPU lanes keep the near-tie rule for the prompts (97%, every parting a near-tie). Their tower rows
  take the GPU lanes' rule: 1 - cosine to the F32 tower summed over the nine pictures at most 1.25x legacy's frozen sum, each
  picture's ratio logged (the managed lane has no tower-rows test).

A lane with no file fails and names the file it needs. The writer (`MuseGlimmerLegacyVisionWriterTests`) went with the
legacy code in stage S9, so the frozen files cannot be regenerated; they are the record.

## Frozen legacy outputs

Stage S0 of the unified-runtime migration (`docs/design/muse-glimmer-unified.md`)
froze the legacy model's (`MuseGlimmerModel`) outputs before any stage changed shared
code, so the runtime family can be held to them after the legacy code is deleted.

* `LegacyGoldens/<lane>.json` (`MuseGlimmerGoldens.cs`, `MuseGlimmerFrozenGoldensTests`):
  the logits of every forward of seven scripted cases (a long prefill, prompt steps,
  the ring wrapping, the full layers' cache growing, a rewind; an untied and a tied
  head) on two synthetic fixtures, per lane (cpu, ggml_cpu, ggml_metal, mlx on the M5 Pro; cpu,
  ggml_cpu, ggml_cuda, ggml_vulkan and direct cuda on an RTX PRO 4500 Blackwell box, linux-x64).
* `LegacyReal/<lane>.json` (`MuseGlimmerRuntimeRealTests.cs`): on the real IQ2_XXS weights
  (ggml_metal and mlx on the M5 Pro; ggml_cuda, ggml_vulkan and direct cuda on the RTX PRO 4500
  Blackwell box), the greedy continuations of the goldens' prompts with
  each position's top four log-probabilities, and the teacher-forced perplexity of
  `benchmarks/engine_comparison/assets/prefill_corpus.txt`.

* `LegacySynthetic/managed.json` (`MuseGlimmerLegacySynthetic.cs`): on the managed backend, frozen on 2026-11-20 (M5 Pro,
  757e0517, ggml d9fe7bff) just before stage S9 deleted the legacy code: the legacy per-op vision tower's rows for the
  tower tests' six grids over the F32-product and the Q8_0-product synthetic mmprojs and for Muse Spark's width
  (`MuseGlimmerVisionTowerTests`, `MuseSparkVisionTowerTests`), and the legacy model's DFlash drafter features over the
  speculation tests' prompt (`MuseGlimmerRuntimeSpeculationTests`), each with the SHA-256 of the fixture it ran on.

The writers were Bench tests (`TS_TEST_FREEZE_MUSE_GLIMMER_GOLDENS`, `TS_TEST_FREEZE_MUSE_GLIMMER_REAL`,
`TS_TEST_FREEZE_MUSE_GLIMMER_SYNTHETIC`); they went with the legacy code in stage S9, so these files are the record.
