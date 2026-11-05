# NInfer Qwen3.8 shared configurations

The **same JSON file** works with `TensorSharp.Cli` and `TensorSharp.Server.Host`:

| Profile | Configured context |
|---|---:|
| [qwen3.8-27b-ninfer.json](qwen3.8-27b-ninfer.json) | 40,960 |
| [qwen3.8-27b-ninfer-64k.json](qwen3.8-27b-ninfer-64k.json) | 65,536 |

Both run the [NInfer](https://github.com/Neroued/ninfer) Qwen3.8-27B artifact
`qwen3_8_27b_nvfp4.ninfer` (NInfer v3 container) on TensorSharp's native `cuda` backend,
with the artifact's embedded DFlash2 drafter (K7, pMin 0) and greedy sampling. They do not
use `ggml_cuda` or an external draft model, and set `TS_SPEC_ADAPTIVE=0` to keep K7
drafting stable. The tokenizer, special tokens, generation defaults and chat template come
from the artifact; no separate tokenizer file is needed.

Build with a toolkit that supports the target GPU:

```sh
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release -p:CudaArch=compute_120
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -p:CudaArch=compute_120
```

Run from the repository root:

```sh
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/qwen3.8-27b-ninfer-64k.json --chat
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/qwen3.8-27b-ninfer-64k.json --host 127.0.0.1 --port 5088
```

The model defaults to `<repo>/models/qwen3_8_27b_nvfp4.ninfer` and is downloaded if absent
from [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
at a pinned revision: **23,719,715,844 bytes**, SHA256
`74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`. Set `TENSORSHARP_MODELS`
to use another model directory, or pass `--model PATH` to skip the configured download.
Older NInfer v2 files (including Ignis `nvfp4full-v2` artifacts) are refused: upgrade them
with NInfer's `tools/upgrade_ninfer_v2_to_v3.py` or download the v3 release.

The reserved `environment` map sets process defaults after config expansion, host preflight
and downloads; existing process variables win. `${appBaseDirectory}` names the running
application's output directory, so one draft-module path finds each application's own
`cuda_kernels` folder (`TS_Q35_NINFER_DRAFT_PTX` accepts a fatbin).

## What the artifact contains and how TensorSharp runs it

| Weights | Representation | TensorSharp execution |
|---|---|---|
| Token embedding, LM head | FP8 E4M3, one BF16 scale per row | FP8 gather; small-T BF16 tensor-core head (the drafter shares it) |
| Attention Q\|K\|gate\|V, GDN Q\|K\|V\|Z, attention/GDN output | FP8 E4M3 row-scaled | decode/verify: FP8 widened exactly to BF16 MMA operands (activations never quantized); prefill: W8A8 TMA GEMM with per-row E4M3 activations on the full-rate block-scaled FP8 MMA |
| MLP layers 0-55 | NVFP4 (E2M1 + E4M3 block scales + FP32 divisor) | W4A4 with the artifact's FP32 activation divisors |
| MLP layers 56-63 | FP8 E4M3 row-scaled | as the other FP8 projections, SwiGLU fused in the epilogue |
| GDN a/b, norms, conv, A_log, dt_bias | BF16 / FP32 | unchanged |
| DFlash2 drafter | q8_g32 (INT8, F16 group-32 scales), BF16 conv/selector weights and codebooks | small-T BF16 tensor-core kernels, gate\|up SwiGLU fused |

Without speculation (`--no-spec`, one activation row per step) the FP8 and NVFP4 projections
run as CUDA-core GEMVs (16-byte weight loads that bypass L1, activations L1-resident) instead of
the 8-row tensor-core tile, and the drafter is not attached. On the RTX 5090 this decodes at
73 tok/s, up from 53. `TS_Q35_T1_GEMV=off` restores the previous kernels for comparisons.

Row orders are applied once at load (attention Q/gate heads interleaved, gate\|up
interleaved by 8 rows); every code and scale word is copied, never requantized.

KV cache: BF16 by default. `TS_Q35_NINFER_KV=fp8` stores 272-byte rows (256 E4M3 codes and
two F32 group-128 scales), half the BF16 size; decode and verify widen FP8 tiles in shared
memory and prefill reads a BF16 copy of the visible history per layer. On the RTX 5090 it
measured no speed change at 32K-64K and the same answers on the long-context checks; use it
for capacity (more lanes or longer contexts in the same memory). Serve long documents with it:
several chats on one cached document each need a slot of their own (recurrent states and drafter
ring; the document's attention blocks are shared), and with a 64K BF16 cache a 32 GB card has no
room for a second slot, so every new chat re-prefilled the document.

Configured contexts are not file-format limits (the artifact declares 262,144); larger
capacities need more KV memory (a 128K context on a 32 GB card needs `TS_Q35_NINFER_KV=fp8`).

Memory: the FP8 embedding table (1.2 GiB) stays in pinned, device-mapped host memory by
default (a token gather reads one 5 KB row over PCIe); `TS_Q35_NINFER_EMBEDDING=device`
uploads it instead. With `--mmproj`, `TS_Q35_RECLAIM_VISION_POOL=1` returns the vision
tower's cached buffers when KV blocks run short (needed for 128K on a 32 GB card).

Images: pass `--mmproj mmproj-BF16.gguf` (the Qwen3.8 vision tower, the same official BF16
weights the artifact carries); its rows replace the image placeholders' embeddings. The
artifact's own vision and MTP weights are not read by this route.

## ggml backends and MLX

`--model qwen3_8_27b_nvfp4.ninfer` also works with `--backend ggml_metal`, `mlx`,
`ggml_cuda`, `ggml_vulkan` and the CPU backends. TensorSharp converts the artifact once into a
llama.cpp-layout `qwen35` GGUF and loads that. The converted file is written beside the artifact
as `<artifact>.q8_0.gguf`, or under `TS_NINFER_GGUF_DIR`. It records the artifact id, so a
different artifact at the same path is converted again. The conversion takes about 30 s on an
M5 Pro and the file is about 27.7 GB.

- Matrices become Q8_0 (ggml-metal and MLX have no FP8 type, and ggml-metal no NVFP4 matmul).
  The 8-bit codes hold the artifact's E4M3 and E2M1 x E4M3 values almost exactly.
- The embedded MTP block is carried over, so the ggml route can speculate with it. External
  DFlash2 GGUF drafters work as with any GGUF.
- The tokenizer, special tokens and chat template come from the artifact.
- Checked against the same model's Q8_0 GGUF: every tokenizer array and hyperparameter is
  identical (the padding id follows the artifact's `tokenizer_config.json`). All 866 tensors
  have the same names and shapes and correlate 0.9996-1.0 with the reference tensors.

Measured on an M5 Pro (48 GB), plain decoding, about 4.4K-token prompts with 2K tokens of
shared prefix reused:

| Backend | TTFT | Decode |
|---|---:|---:|
| ggml_metal | 6.7 s | 9.7 tok/s |
| mlx | 5.6-5.9 s | 9.1 tok/s |
| ggml_metal + `--draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-draft 7` | 7.2 s | 21.3 tok/s |

MLX decodes the converted model at the same rate as the Unsloth Q8_0 GGUF (9.1 tok/s each,
back to back).

The DFlash2 drafter inside the artifact is not converted; on these backends pass the DFlash2
GGUF with `--draft-model`.

See the [native adapter guide](../TensorSharp.Backends.Cuda/native/ninfer/README.md) for
the kernels and their numerical contracts.
