# KRAFTON Raon-1 nano (`raon1nano`)

[← back to model index](README.md) | [中文](raon_zh-cn.md)

TensorSharp runs [KRAFTON/Raon-1-nano](https://huggingface.co/KRAFTON/Raon-1-nano) from the GGUFs KRAFTON publishes in
[KRAFTON/Raon-1-nano-gguf](https://huggingface.co/KRAFTON/Raon-1-nano-gguf), which were converted with the `raon-1-nano`
branch of KRAFTON's llama.cpp. The architecture is not Qwen 3: the model's own code (`model_type: raon1_nano`,
`trust_remote_code`) is a 24-layer hybrid decoder that combines these parts:

- **mHC:** four hyper-connection streams per token, with a learned read and write mix around every attention and FFN block.
- **PLE:** a 256-wide per-layer embedding per token and layer (a 259,840 × 6,144 table), added to every stream after the FFN.
- **RDA:** 18 recurrent linear-attention layers. Each has a causal conv, gated delta-rule state with a low-rank,
  lower-bounded decay, and separate write and erase gates.
- **MLA:** 6 multi-head latent attention layers (0-based 3, 7, …, 23). They are absorbed, use no positional encoding, and
  have a sigmoid output gate. The cache holds a 576-wide key and a 512-wide value per token.

The family runs on the shared decoder runtime as whole-forward graphs only (`RequiresGraphs`). It reuses existing IR ops:
`GatedDeltaNet` and `SsmConv` for RDA, and `FlashAttnExt` with DK 576 / DV 512 for MLA. It needs no new ggml op and no ggml
patch.

| Property | Value |
|---|---|
| GGUF architecture | `raon1nano` |
| Source class | [`RaonRuntimeModel`](../../TensorSharp.Models/Models/Raon/RaonRuntimeModel.cs) ([graph](../../TensorSharp.Models/Models/Raon/RaonRuntimeModel.Graph.cs)) |
| Architecture plug-in | [`RaonArchitecture`](../../TensorSharp.Models/Models/Raon/RaonArchitecture.cs) |
| Modalities | Text only |
| Thinking | Yes. `--think` opens `<think>\n`; otherwise the template's closed `<think></think>\n\n` is kept. |
| Tool calling | The template's Qwen 3.5-style `<tool_call><function=…>` blocks, parsed by the Qwen 3.5 output parser. Not yet exercised end to end. |
| Backends | `cpu` (managed), `ggml_cpu`, `ggml_metal`, `ggml_cuda`, `ggml_vulkan`. `cuda` and `mlx` are refused at load. |
| Batching | Continuous batching, checkpoints and speculative verify, via the shared runtime |
| Multi-GPU | None. Tensor parallelism, layer split and hybrid placement are declared gaps. |
| MTP | `*_MTP` files load, but their NextN block is skipped (not loaded or run) |

## Run it

```bash
echo "What is the capital of France? Answer in one word." > prompt.txt
dotnet run --project TensorSharp.Cli -- \
    --model models/raon-1-nano-Q8_0.gguf --backend ggml_cuda \
    --input prompt.txt --max-tokens 256 --no-skills
```

KRAFTON recommends temperature 0.1 with presence penalty 1.5 for general use, and temperature 1.0 for agentic use.

## Implementation notes

The GGUF stores a folded layout, and the graph does not apply those constants a second time:

- The ×2 hyper-connection post factor is folded into `attn_output` and `ffn_down`.
- `exp(A_log)` is folded into `ssm_f_b` and `ssm_dt.bias`.
- `inp_gate` is divided by 4, so the PLE gate reads the sum of the streams rather than their mean.
- Each site's `hc_*_chart` packs the stream-mix map K (17 → 16), its row scale r and offset c0. They are split into F32
  constants at load.

Other details:

- **MLA scale:** the query is pre-scaled by √(576/192), because the runtime applies 1/√576 and the model uses 1/√192.
- **PLE table:** the token table stays on the host. Each step's token rows are dequantized there and uploaded as a graph
  input.
- **PLE GELU:** composed from exact ops (tanh, mul, scale), because ggml-cpu's `GELU` reads an fp16 table.

## Validation

[`RaonParityTests`](../../InferenceWeb.Tests/RaonParityTests.cs) compares the runtime with KRAFTON's own modeling code run
in float32 on the CPU (Transformers 5.16.1, torch 2.15.0). The fixtures are
[`Fixtures/Raon/hf_goldens.json`](../../InferenceWeb.Tests/Fixtures/Raon/hf_goldens.json), made by
[`eng/validation/raon-hf-goldens.py`](../../eng/validation/raon-hf-goldens.py): five chat prompts (English, arithmetic,
Korean, code, story) with 32-token greedy continuations, and a ten-string tokenizer corpus. The test requires
`TS_TEST_MODEL_DIR` to hold `raon-1-nano-Q8_0.gguf`, and skips without it.

Results on `raon-1-nano-Q8_0.gguf` (RTX 5090, Windows, 2026-11-22):

| Lane | Teacher-forced argmax agreement | First-position top-10 max \|Δ log p\| |
|---|---|---|
| `ggml_cuda` | 105/105 | 0.09–0.28 nats |
| `ggml_cpu` | 105/105 | 0.14–0.39 nats |
| `cpu` (managed) | 105/105 | 0.13–0.46 nats |

For scale, Transformers with Q8_0-rounded weights and activations moves 0.09–0.24 nats from its own float32 run. Switching
the KV cache between f16 and f32 moves the worst prompt by about 0.26 nats.

The other KRAFTON quants on `ggml_cuda` (`TS_TEST_MODEL_DIR` set to the file; the bounds are Q8_0's):

| File | Teacher-forced argmax agreement | First-position top-10 max \|Δ log p\| |
|---|---|---|
| `raon-1-nano-F16.gguf` | 104/105 (the one miss is a 0.005-nat near-tie) | 0.03–0.07 nats |
| `raon-1-nano-Q4_K_M.gguf` | 98/105 | 0.50–5.2 nats; outside the Q8_0 bounds, not compared with llama.cpp's Q4_K_M |
| `raon-1-nano-Q4_K_M_MTP.gguf` | identical to Q4_K_M (the NextN block is skipped) | identical to Q4_K_M |

F16 shows that the graph reproduces the reference forward. Q4_K_M's larger deltas are expected from 4-bit weights, but they
are measured here only against float32. Its greedy answers stay coherent (for example `17 × 23 = 391`, with the steps).

Tokenization of every prompt and corpus string matches Hugging Face `tokenizers` exactly, including Korean, CJK, emoji,
NFC composition and special tokens. The GGUF chat template renders exactly what `apply_chat_template` produced.

[`RaonRuntimeTests`](../../InferenceWeb.Tests/RaonRuntimeTests.cs) covers the serving mechanisms on a synthetic F32 fixture
([`RaonSyntheticModelBuilder`](../../InferenceWeb.Tests/RaonSyntheticModelBuilder.cs)):

- batched slots against each sequence run alone;
- concurrent engine requests;
- checkpoints, cloned and exported, including device copies on GGML;
- greedy speculative decoding against plain greedy decoding;
- chunked prefill against a single chunk;
- the pinned GGML backend against the managed one.

These tests pass with both the `ggml_cpu` and the `ggml_cuda` pins. They have not been run on Metal or Vulkan.
