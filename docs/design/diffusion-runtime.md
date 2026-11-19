# Diffusion runtime: MiniMax-H3-Image on the unified runtime, with editing and LoRA

Status: stages S1-S4 implemented (MiniMax-H3-Image text to image runs on the runtime; the legacy model is a test
seam until S10); S5-S11 are design. Written 2026-11-19 on `feature/h3-image-unified` (cut from
`origin/main` ec702639). ggml is upstream `cd6d2b75`, unchanged (AGENTS.md). No MiniMax-H3
weights were downloaded or run for this document: every statement about TensorSharp is from
the code, and every statement about upstream is from the sources listed in
[Appendix A](#appendix-a-upstream-sources-read), at the revisions given there.

This document covers four pieces of work:

1. Move MiniMax-H3-Image (text to image) onto the unified execution architecture
   ([unified-execution.md](unified-execution.md)), with no loss of speed or quality, and delete
   the legacy path afterwards.
2. Add a **generic diffusion runtime** that every diffusion family can run on. H3-Image is the
   first family on it. MiniMax-H3 video, Qwen-Image-2.1, Wan and LTX 2.5 come later. Their
   model classes were read so that the interfaces fit them, but only H3-Image's migration is
   planned here.
3. **Editing** for H3-Image: up to 16 reference images, natural-language edits, masks and
   multi-turn, on the CLI, the server and TensorAgent.
4. **LoRA** for H3-Image: selecting, loading and running adapters on the new runtime.

Sections 1-7 are the analysis and the design. [Section 8](#8-staged-implementation-plan) is
the staged plan, and [section 9](#9-decisions-for-the-user) lists the decisions only the user
can make, each with a recommendation.

---

## 1. Today's MiniMax-H3-Image path

### 1.1 One request, end to end

```
TensorSharp.Cli / Server (/api/image-generate[/stream]) / ComfyUI node
  -> IImageGenerationModel.GenerateImage(prompt, QwenImageParams)      (QwenImage/IImageGenerationModel.cs)
     MiniMaxH3ImageModel.GenerateImage                                  (MiniMaxH3ImageModel.cs)
       refuse cfg > 1, a negative prompt or a mask; ResolveSize (2048 default, sides % 32)
       lock (_generationLock) -> Generate()
         text   : MiniMaxH3TextEncoder.Tokenize + Encode (raw prompt, no chat template)
                  -> TSGgml_MiniMaxH3TextEncode (ggml_ops_minimax_h3.cpp:1507), layer-50 hidden
         noise  : TorchCpuRandn.Normal(24*lh*lw, seed)   (MT19937 + Box-Muller, ComfyUI's draw)
                  HWC transpose, MiniMaxH3DiT.PatchifyVideo (2x2 patches, 96 values a token)
         sigmas : MiniMaxH3ImageSampling.Schedule(grid, steps)  (ComfyUI "simple" over the PDD grid)
         loop   : MiniMaxH3ImageLayout.Build(textLen, lh, lw, sigma)     ([text | target], 3-axis RoPE)
                  MiniMaxH3ImageSampling.Interval -> PDD head interval [start, stop)
                  MiniMaxH3DiT.ForwardImage -> PddHead (CombineHead, F32, pinned and bound)
                    -> TSGgml_MiniMaxH3DitForward (ggml_ops_minimax_h3.cpp:1839, audio_count = 0)
                  x += v * (next - sigma)  (host Euler; v is already negated)
         decode : UnpatchifyVideo -> MiniMaxH3ImageVae.Decode (256-px tiles, overlap >= 64,
                  one native call a tile row: TSGgml_MiniMaxH3VitDecodeBatch, host blending)
       LastTimings = (text, denoise, decode, steps)
```

All three components stay resident between images: `_textEncoder ??=`, `_dit ??=` and
`_vae ??=` in `Generate`. Nothing releases the text encoder before the denoise. That is fine on
the 94 GB H100 where it was measured (17 + 40 + 1.2 GB), but it is why the model needs about
60 GB of VRAM today.

### 1.2 Files and what they share with MiniMax-H3 video

| File | Role in H3-Image | Shared with H3 video? |
|---|---|---|
| `MiniMaxH3ImageModel.cs` | the model (`ModelBase`, `IImageGenerationModel`): companion resolution, the loop, timings; `SupportsImageEdit => false`, `EditImage` throws | image only |
| `MiniMaxH3ImageArchitecture.cs` | descriptor: id `minimax-h3-image`, the `hyvid` relabel rule for unsloth GGUFs, `ServingContract` (Bespoke, ggml backends, TP/LayerSplit gaps) | image only |
| `MiniMaxH3ImageLayout.cs` | `[text | references | target]` packed layout; it already takes `textTokenModalities` and `references`, but nothing calls it with them | image only (uses `MiniMaxH3Layout.SpatialAxes`, which is shared) |
| `MiniMaxH3ImageSampling.cs` | PDD grid (`GridFrom`, `ReleaseSigmas`), ComfyUI schedule, `Interval`, `CombineHead`, `TorchCpuRandn` | image only |
| `MiniMaxH3ImageVae.cs` | 2-D ViT decoder (24 x 1024, 16 heads, RoPE on half a head, 4 register tokens), ComfyUI tiling and blending, `TS_H3_IMAGE_VAE_BATCH` | image only; it calls the video VAE's native decode (`TSGgml_MiniMaxH3VideoVaeDecode`, and `TSGgml_MiniMaxH3VitDecodeBatch` for rows) |
| `MiniMaxH3DiT.cs` | the DiT: weight binding by host pointer, `TimeEmbedding` (curve table), `BuildRope`/`GetRopeForForward`, `PatchifyVideo`, **`ForwardImage` + `PddHead` + `_pddWeight/_pddBias/_pddHeads`** (image only), `Forward` (video), residency, LoRA binding, the CUDA graph arena | shared class; the image members are image only |
| `MiniMaxH3Config.cs` | `Detect`, **`IsMiniMaxH3Image`, `PddSteps`**, `HasAudio` | shared; the image members are image only |
| `MiniMaxH3WeightFile.cs` | GGUF **or safetensors** mapping (safetensors only for the image release) | shared |
| `MiniMaxH3TextEncoder.cs` (+ `MiniMaxH3VisionEncoder.cs`) | Qwen3-VL-32B truncated to 50 layers, no final norm; vision tower with DeepStack taps | shared; H3-Image uses the text path only |
| `MiniMaxH3DirectDiT.cs`, `CudaMiniMaxH3Ops`, `MlxMiniMaxH3Ops`, `MiniMaxH3Direct*` | direct (cpu/mlx/cuda) H3 video | **video only**: `MiniMaxH3DiT`'s constructor refuses H3-Image on direct backends (`if (_weights.Gguf == null \|\| !Config.HasAudio) throw`) |
| `MiniMaxH3Pipeline.cs`, `MiniMaxH3Model.cs`, `MiniMaxH3Scheduler.cs`, `MiniMaxH3Layout.cs`, `MiniMaxH3Geometry.cs` | video pipeline. `MiniMaxH3Model.ResolveCompanion` finds H3-Image's companions; `MiniMaxH3Scheduler.VisualConditionTimestep` (0.999) is read by the image layout | video, with those two members used by image |
| `MiniMaxH3LoraSet/Keys/Config.cs` | H3 video LoRA (LightX2V Turbo): diffusers keys only, unmerged, FL2VA only | video only today (H3-Image takes no LoRA) |
| `TensorSharp.GGML.Native/ggml_ops_minimax_h3.cpp` (4,288 lines) | `TSGgml_MiniMaxH3DitForward` (one graph per step), `h3_vit_decode`, `TSGgml_MiniMaxH3TextEncode`, the offload (`h3_compute`, `h3_offload_wants/run`), the BF16 cuBLAS route (`h3_bf16_matmul_wants`), the BF16 fc1->fc2 SwiGLU fusion | shared; the `audio_count == 0` branch, the BF16 routes and `TSGgml_MiniMaxH3VitDecodeBatch` serve only H3-Image today |
| `tsg_cuda_h3_fused.cu` | fused modulation, gated and scaled residuals, attention prep, LoRA GEMM, F16/BF16 GEMM with SwiGLU, int8 deferral | shared |

### 1.3 Backends today

The descriptor and the constructor accept `ggml_cuda`, `ggml_metal` and `ggml_cpu` and
nothing else. Only `ggml_cuda` has been run with real weights (docs/models/minimax-h3-image.md:
"`ggml_metal` and `ggml_cpu` take the same native graphs but have not been run with it yet").
The fused stages, cuDNN attention and the cuBLAS/int8 products are CUDA-only. Metal and CPU run
the unfused graph that the same builder emits.

### 1.4 Hosts and wiring

* **Model creation.** `ModelBase.Create` (ModelBase.cs:3696) checks
  `MiniMaxH3ImageModel.IsImageDenoiserSafetensors` **before** any GGUF probe, refuses `--tp` and
  `--layer-split`, and constructs the model directly. `ModelLifecycleService.ValidateModelFiles`
  (TensorSharp.Chat, line 260) skips the GGUF header check for the same file. A GGUF denoiser
  goes through the registry (`DetectFromTensors`, `RecognizeRelabelledFile` for `hyvid`).
* **CLI.** `Program.cs:1162`. `IImageGenerationModel` goes to `RunImageEdit`. With `--image` and
  `SupportsImageEdit == false` the CLI prints "does not edit images here; drop --image." Flags:
  `--prompt`, `--width`, `--height`, `--diffusion-steps`, `--diffusion-seed`, `--cfg`,
  `--negative-prompt`, `--output`, plus the companions `--video-text-encoder` and `--video-vae`
  (`TS_VIDEO_TEXT_ENCODER`, `TS_VIDEO_VAE`).
* **Server.** `UploadEndpoints.cs` maps `/api/image-generate[/stream]` and
  `/api/image-edit[/stream]` to `WebUiAdapter` and then `WebUiChatService.ImageRequestCoreAsync`.
  Parameters come from `ParseImageParameters`; requests are serialized by
  `lock (_imageEditLock)`; progress and cancellation arrive through `QwenImageParams.OnStep`.
  H3-Image refuses `/api/image-edit` (`RequireImageEditModel` and `SupportsImageEdit`). There is
  **no** OpenAI image route: `OpenAIEndpoints.cs` maps `/v1/videos/generations` only, with a
  body that "follows the images/generations envelope".
* **TensorAgent.** It has no H3-Image entry: `ModelCatalog.cs` lists Qwen-Image 2.1 (and Turbo)
  and MiniMax-H3 video, `LoraCatalog` lists Qwen-Image and H3 Turbo plug-ins, and
  `ImageTurns.FramesFor` would route an H3-Image turn through `LoadedModelMakesImages`
  (`_svc.Model is IImageGenerationModel`).
* **ComfyUI node pack.** `examples/comfyui/profiles/minimax-h3-image.json` and
  `minimax-h3-image-int8.json`. The second sets `env: {"TS_H3_INT8_MATMUL": "1"}`.
* **Config and docs.** `config/minimax-h3-image.json`: the bf16 DiT from
  Comfy-Org@0bb149f5 (40.2 GB), the Q4_K_M text encoder from unsloth/MiniMax-H3-GGUF@d629413c
  with three tokenizer files from MiniMaxAI/MiniMax-H3@42ed227e, and the fp16 VAE.
  `docs/models/minimax-h3-image.md` and `_zh-cn.md`. `config/regression/catalog.json` has entries
  at lines 323 and 2483. README.md, README_zh-cn.md and docs/models/README*.md mention the model.

### 1.5 Settings, options, env vars and switches that touch H3-Image

| Name | Where read | Image-only? |
|---|---|---|
| `TS_H3_IMAGE_VAE_BATCH` (tiles a decode call takes, default 64) | `MiniMaxH3ImageVae` | yes |
| `TS_VIDEO_TEXT_ENCODER`, `TS_VIDEO_VAE` / `--video-text-encoder`, `--video-vae` | `MiniMaxH3Model.ResolveCompanion` | no (H3 video) |
| `TS_H3_INT8_MATMUL` (int8 tensor-core products; BF16 for the image DiT, Q4_K for video) | native offload | no |
| `TS_H3_CUBLAS_MATMUL`, `TS_H3_CUBLAS_MIN_ROWS` | native offload | no |
| `TS_H3_CUDNN_ATTENTION`, `TS_H3_CUDNN_ATTENTION_VERBOSE`, `TS_H3_NO_FLASH`, `TS_H3_FLASH_ATTENTION`, `TS_H3_CUDA_BATCHED_ATTENTION` | native attention | no |
| `TS_H3_FUSED`, `TS_H3_INT8_DEFER`, `TS_H3_INT8_KEEP_WEIGHTS`, `TS_H3_INT8_ONE_PASS`, `TS_H3_FP8_MATMUL` | native fused stages and products | no |
| `TS_H3_CUDA_F32_PRODUCTS`, `TS_H3_CUDA_F32_BANDS`, `TS_H3_CUDA_BF16_PEDANTIC` | precision oracles | no |
| `TS_H3_DIT_LAYERS`, `TS_H3_DIT_TRACE_DIR`, `TS_H3_DUMP_*`, `TS_H3_TRACE`, `TS_H3_STAGE_PROFILE`, `TS_H3_UPLOAD_PROFILE`, `TS_H3_PHASE` | measurement and oracle switches (keep list) | no |
| `TS_H3_UPLOAD_PREFAULT`, `TS_H3_UPLOAD_WORKERS`, `TS_H3_PREFAULT*`, `TS_H3_DIRECT_READ*`, `TS_H3_KEEP_RESIDENT` | weight upload and residency | no |

H3-Image itself has no CLI flags. It reuses the generic image flags and the H3 companion flags.
`QwenImageParams` is the request type, even though the name belongs to another family.

### 1.6 What becomes legacy, and what stays for MiniMax-H3 video

**Deletable once H3-Image runs on the new runtime** (stage S10):

* `MiniMaxH3ImageModel.cs`, `MiniMaxH3ImageArchitecture.cs` and `MiniMaxH3ImageVae.cs`. Their
  logic moves to the new family folder, and the descriptor keeps id `minimax-h3-image`.
* `MiniMaxH3ImageSampling.cs` and `MiniMaxH3ImageLayout.cs` **move**: the grid, schedule, head
  combination and layout become the family's recipe and layout on the new runtime. They are
  not deleted; `TorchCpuRandn` becomes the runtime's noise source.
* In `MiniMaxH3DiT.cs`: `ForwardImage`, `PddHead`, `_pddWeight/_pddBias/_pddHeads`, the
  `PddSteps > 1` branch in the constructor, and the safetensors note on the constructor.
* In `MiniMaxH3Config.cs`: `IsMiniMaxH3Image`. The family's own detector replaces it. Keep
  `PddSteps` only if video ever needs it (it does not today).
* In `MiniMaxH3WeightFile.cs`: the safetensors branch, if no video file needs it. The new
  runtime gets a family-neutral weight source (`SafetensorsModel`/`GgufFile`, as Qwen-Image
  uses).
* `ModelBase.Create`'s and `ModelLifecycleService.ValidateModelFiles`'s H3-Image special
  cases. They are replaced by a descriptor-level safetensors detector in the registry, a
  generic hook that every safetensors-shipped family can use.
* `TS_H3_IMAGE_VAE_BATCH`. The VAE program sizes its own batch.
* The native `TSGgml_MiniMaxH3VitDecodeBatch` export and its `GgmlNative.cs` binding, which
  only `MiniMaxH3ImageVae` calls. `h3_vit_decode` itself stays because the video decode calls it
  with `batch = 1`.
* The `audio_count == 0` branch of `TSGgml_MiniMaxH3DitForward`, once nothing calls the
  native DiT without audio.
* The `minimax-h3-image-int8` ComfyUI profile's env var. It is replaced by the new option
  (decision D4).
* Tests: `MiniMaxH3ImageSamplingTests` and `MiniMaxH3ImageGgufLabelTests` move to the new
  family, and their assertions are kept.

**Stays because H3 video uses it:** `MiniMaxH3DiT.Forward`, `MiniMaxH3DirectDiT` and the
direct ops, `MiniMaxH3TextEncoder` and the vision tower with their native entry points,
`MiniMaxH3VideoVae` (including the single-frame encoder that editing reuses as its weight
layout reference, §4.2), `MiniMaxH3Pipeline`, `MiniMaxH3Layout`, `MiniMaxH3Scheduler`,
`MiniMaxH3LoraSet/Keys/Config`, the whole of `ggml_ops_minimax_h3.cpp` except the image-only
pieces above, `tsg_cuda_h3_fused.cu`, and every `TS_H3_*` switch listed in §1.5 except
`TS_H3_IMAGE_VAE_BATCH`. The BF16 cuBLAS route and the BF16 SwiGLU-in-GEMM fusion are not
deleted. They **move** into the shared DiT offload module (§2.4) that the new runtime and the
legacy video graph both call.

---

## 2. The unified architecture, and what a DiT needs from it

### 2.1 What exists

* **IR.** `TensorSharp.Core/Graph/GraphProgram.cs`. `GraphOp` has 57 ops with ggml semantics,
  and `GraphProgramBuilder` provides `Leaf` (Weight, State, Input), `Op`, `Effect`, `Alias`,
  `Boundary`, `BeginFused/EndFused` and `Output`. `GraphFusion` has 17 region kinds, all from
  the AR families (DeltaNet, HC, SwigluFfn, RoutedExperts, AttentionPrep, ...). A region's ops
  are the reference implementation; an executor with a kernel for the kind may serve the
  region instead.
* **Executors.** `IGraphExecutor` has `GraphCapabilities`, is registered per allocator type
  (`GraphExecutors.Register/For`), and the runtime keeps one per allocator (`ExecutorFor`).
  The implementations are `GgmlGraphExecutor` (one ggml graph per program, leaves bound by host
  pointer, replayable plans cached by signature, compiled through `ggml_ops_graph_builder.cpp`)
  and `TensorGraphExecutor`, the interpreter that `ManagedGraphExecutor`, `MlxGraphExecutor`
  and `CudaGraphExecutor` specialize with their kernels.
* **Emission.** `StandardDecoderModel.Graph.cs` builds programs through the nested `GraphStep`
  (weights, inputs, ops, regions such as `SwigluFfn`, `RoutedExperts` and `SigmoidGate`).
  Tensor parallelism is per-rank programs with `GraphOp.AllReduce` (`EmitRankedStep`,
  `IRankedGraphExecutor`, `TSGgml_GraphRunRanked`).
* **How regions are served on GGML today.** `TSGgml_GraphFusedBegin` / `try_serve_region`. A
  served region becomes a *device operation between graph segments*, with operands in an IO
  buffer outside the graph allocator. This is implemented only for the Metal kinds (`FUSED_DELTA_NET_PROMPT`,
  `FUSED_HC_*`) and the selected attention. `TSGgml_GraphRun` runs the graph with
  `ggml_backend_graph_compute_async` or `graph_compute_profiled`. It has **no node offload**.

By contrast, the H3 graph (`h3_compute`, ggml_ops_minimax_h3.cpp ~800) runs through
`graph_compute_offload` with an `H3Offload` `NodeOffload`. Its offloaded nodes are: cuDNN SDPA
for `FLASH_ATTN_EXT` nodes it can take; wide BF16 and Q4_K `MUL_MAT` on cuBLAS or int8
(`h3_bf16_matmul_wants`, `h3_q4k_gemm_takes`); and `GGML_OP_CUSTOM` fused nodes (`h3_fused_node`
-> `tsg_h3_*`), where int8 results are left unscaled for the fused residual that reads them
(`h3_deferrable`). That mechanism is where today's speed comes from (docs/models/minimax-h3-image.md:
cuDNN attention is about half of a 16 MP step, and the SwiGLU-in-GEMM saves 1.5 ms a block),
and the graph builder does not have it yet.

### 2.2 Gap analysis: what the DiT and the image VAE need

| Need | In the IR today | Proposal |
|---|---|---|
| H3 3-axis RoPE: float positions (t, h, w); the spatial axes are fractional (`_spatial_position_grid`: aspect-normalized and scaled by 32); rotate-half over `RopeRotaryDim` split by axis | `RopeExt` (one integer axis), `RopeMulti` (ggml sections, **int32** positions): neither takes fractional positions | Keep the host tables (`MiniMaxH3DiT.BuildRope`, constant for a request) as **Input** leaves `cos`, `sinSigned` `[rot, seq]`. New region **`GraphFusion.TableRopeQkNorm`**: q/k per-head RMSNorm, then `x*cos + rotate_half(x)*sinSigned` on the first `rot` dims, the rest passed through. Reference ops: `RmsNorm`, `Mul`, `View`, `Concat`, `Add`. Served on CUDA by the attention prep that `H3_FUSED_ATTENTION` does today |
| adaLN modulation per segment (a segment is a modality and timestep run of rows; `mods` is `[hidden, 18*nTimesteps]`) | expressible with views, `RmsNorm`, `Mul`, `Add` and `Concat`, but that costs about 8 nodes per segment and twice a block (the node-budget comment at ggml_ops_minimax_h3.cpp ~2574) | New region **`AdaLnModulate`**. Inputs: `x`, norm weight, `mods`, and a segment table input `I32 [3, nseg]` (start, end, column). Ip: shift and scale index. Fp: eps. Served by `tsg_h3_modulate` |
| Gated residual `base + (delta + mm + bias) * gate` per segment | same as above | New region **`GatedResidual`**, served by `tsg_h3_gated_residual` (and by the int8 deferral: an unscaled int32 `mm` is consumed here) |
| Non-causal attention over 4K-100K tokens, no mask | `FlashAttnExt` (no mask) runs on ggml's kernel | No IR change. The GGML lowering offloads it to cuDNN SDPA (§2.4). Metal pads K to 64 as `h3_attend_padded` does |
| fc1 `[gate | value]` -> SwiGLU -> fc2 without the `[ffn, seq]` F32 intermediate | `SwigluFfn` takes gate rows then up rows, which is H3's order | Reuse `SwigluFfn`. Add Ip[1] = 1 for biases (the VAE's `w1`/`w2` have biases). On CUDA with a BF16 or F16 `w2`, serve it as `H3_FUSED_F16_GEMM` with f0 = 1 (SwiGLU applied while fc2 narrows its input) |
| Unmerged LoRA on a projection | none | New region **`LoraLinear`**. Inputs: `x`, `w`, then (down, up) for each part. Ip: row slots (q, k and v thirds; fc1 halves). The reference is `MulMat(w,x) + sum MulMat(up, MulMat(down, x))`, with scales folded into `up`. Served on CUDA by `H3_FUSED_LORA_GEMM` (Q4_K bases) |
| Patchify and unpatchify (2x2, 24 channels -> 96) | (Reshape, Permute, Cont) | Stay on the host, as today (1.5M floats at 16 MP) |
| PDD head per step (head 0 + a dt-weighted mean of offset heads) | none needed | Combine on the host in F32 (`CombineHead`, unchanged) and bind as an **Input** leaf (96 x 5376 F32, 2 MB a step) rather than a weight. The six steps then share **one** replayable plan instead of six plans, each with its own compute buffer |
| Curve-table time embedding | none needed | Host interpolation (`TimeEmbedding`), passed as an Input leaf; the adaLN projections run in the graph |
| Velocity negation, Euler step, mask blending | none needed | Host, vectorized (as `QwenImageEditMask.Reinject` is) |
| VAE ViT decode of a row of 256-px tiles | `FlashAttnExt` with the batch in `ne[3]` | One program per batch of tiles: register tokens via `Concat` with a weight leaf, a 2-axis RoPE table on half of each head (Input), LayerNorm with `scale1/scale2`. The batched-SDPA offload serves the 4-D attention node (`tsg_cudnn_sdpa_prepare_batched`). Blending stays on the host (the existing `BlendRows`/`BlendColumns`) |
| VAE **encoder** (editing): causal 3-D CNN; with one frame, only the last temporal tap sees the frame (zero causal padding, `MiniMaxH3ImageCausalConv3d`), so it is exactly a 2-D CNN; reflect spatial padding | `GroupNorm`, `Silu`, `Add` and `Pad` (zero padding) exist. **No convolution, no reflect pad** | New ops **`GraphOp.Conv2d`** (srcs: kernel, x; ip: stride, padding, dilation; lowered to `ggml_conv_2d`, or to an un-lowered `CONV_2D` that the existing fast-conv offload, cuDNN or MPSGraph, executes: `graph_compute_fast_conv`) and **`GraphOp.PadReflect1d`** (`ggml_pad_reflect_1d`; 2-D is two passes over a transpose). The weights take the last temporal slice, as `MiniMaxH3VideoVae.LastSlice` already does |
| int8 and BF16 wide products | none | Not IR ops: an **executor product policy** set per program (§2.4). The int8 option is decision D4 |
| Tensor parallelism | `GraphOp.AllReduce`, per-rank programs | Reuse. Heads (56 = 2x28, 4x14, 8x7) and FFN columns (14336/T) are sharded; adaLN, refiner and head are replicated (§3.6) |
| Large tensors (fc1 output at 16 MP is `[28672, 16540]` F32 = 1.9 GB) | `long` extents and strides; `GraphLargeInputViewTests` | No change; covered by a test at 16 MP shapes |

### 2.3 What the GGML plugin must add to keep today's speed

The legacy DiT graph is *one ggml graph a step, with offloaded nodes*. The program lowering
must produce the same node sequence and run it the same way. The work is all TensorSharp-owned
native code; ggml is not touched.

1. **A shared DiT offload module.** Move `H3Offload`, `h3_offload_wants/run`,
   `h3_bf16_matmul_wants/run`, `h3_q4k_gemm_takes`, `h3_fused_node`, `h3_fused_params`,
   `h3_deferrable` and the cuDNN attention predicate out of `ggml_ops_minimax_h3.cpp` into a new
   `ggml_ops_dit_offload.{h,cpp}`. The legacy H3 forward includes it unchanged, which is a
   pure refactor guarded by the existing H3 video tests and probes (`eng/validation/MiniMaxH3DitProbe`).
2. **In-graph served regions.** In `ggml_ops_graph_builder.cpp`, `try_serve_region` learns a
   second serving mode for the kinds `AdaLnModulate`, `GatedResidual`, `TableRopeQkNorm`,
   `LoraLinear` and (BF16/F16) `SwigluFfn` on `BACKEND_TYPE_CUDA`. The region becomes **one
   `GGML_OP_CUSTOM` node** built by the shared `h3_fused_node`, with the region inputs as its
   sources, inside the graph allocator. The region's ops are numbered but not built, as served
   regions are now. On Metal and CPU these kinds are not served, so their reference ops run,
   which is exactly what the legacy graph does there (`TS_H3_FUSED` is CUDA-only).
3. **Node offload in `TSGgml_GraphRun`.** A graph that holds custom nodes, or that carries the
   new program attribute `WideTokenProducts`, runs through `graph_compute_offload` with the
   shared offload. `WideTokenProducts` is set by the diffusion runtime for denoiser steps, and it
   replaces the legacy `tag contains "dit-forward"` test that enables the cuBLAS and int8
   products. The int8 "left unscaled and never read" check (`tsg_cuda_int8_forget_pending`)
   moves with it.
4. **Replay.** The denoiser step is `replayable`: every per-step value (x, the time embedding,
   the PDD head and the per-row timestep indices) is an Input. The segment table and the
   positions are fixed for a request, so the program's signature is fixed for a request
   geometry, and a second request of the same geometry reuses the plan. This matches the
   legacy `NativeGraphArena`, which exists so that ggml-cuda replays successive steps.
5. **Weight upload.** The legacy first-request time (9.1 s at 4 MP, int8) comes from
   `h3_upload_weights` (parallel uploads, direct reads from the file mapping:
   `MiniMaxH3DirectRead`, `TS_H3_UPLOAD_*`). The GGML executor's binding by host pointer must
   use the same uploader for large weight sets, through a `Preload(weights)` call the runtime
   issues once per component load. Without it the first-request gate in §7 fails.

### 2.4 Tensor executors (managed CPU, MLX, direct CUDA)

The same programs run on `TensorGraphExecutor` through the reference ops of every region. Each
plugin may add kernels: MLX has fused SDPA and could add `AdaLnModulate`; direct CUDA could
reuse `tsg_h3_*` through `CudaGraphKernels`. None of these backends run H3-Image today, so
enabling them is new support, not a regression risk. The descriptor lists a backend only after
a real-weights run on it, and the synthetic fixtures (§7.4) prove correctness everywhere.

---

## 3. A generic diffusion runtime

### 3.1 Where it lives, and how it relates to the AR runtime

* **`TensorSharp.Models/SharedDiffusion/`** (namespace `TensorSharp.Models.SharedDiffusion`) holds
  the runtime: pipeline, components, residency, sampling and the mask/reference utilities.
  It is a sibling of `SharedDecoder/`.
* **`TensorSharp.Models/Models/MiniMaxH3Image/`** (namespace `TensorSharp.Models.MiniMaxH3Image`)
  holds the H3-Image family on the runtime: descriptor, layout, recipe, denoiser emission, VAE
  emission, and the LoRA key grammar it shares with video (§5.4).
* **`TensorSharp.Runtime/Scheduling/Diffusion/`** holds the host-side job queue (§3.5), next to
  `InferenceEngine`. It replaces the ad-hoc `_imageEditLock` in `WebUiChatService`.

**Shared with the AR runtime:** the IR and executors; the `GraphFusion` regions (the new kinds
are generic: any adaLN DiT uses them); weight leaves (`IGraphWeight`) and host-pointer binding;
`ModelArchitectureDescriptor` with `ServingContract` and the conformance tests; `DeviceMesh`,
`ITensorParallelGroup` and `AllReduce`; LoRA factor math (`LoraFactorMath`, `LoraFactorBuffers`,
`LoraSpec`/`LoraCliFlags`, `ILoraPluginHost`); telemetry. Emission helpers move out of
`StandardDecoderModel.GraphStep` into a shared `GraphEmitter` (weights, inputs, ops,
regions), which both runtimes use.

**Separate:** no KV cache, slots, paged batches or speculation; a request is one image (or
clip), and a step is a whole-sequence forward. It has its own request model, sampling and
scheduler.

`ServingContract` gains `ServingImplementation.SharedDiffusionRuntime`. For a diffusion family
on it, `ContinuousBatching` is `NotApplicable("step-interleaved jobs; see Scheduling/Diffusion")`,
`TensorParallel` is provided by the runtime's mesh where the family declares shardable blocks,
and `LayerSplit` and `ExpertOffload` stay gaps or not-applicable. The registry rule that new
families must use a shared runtime extends to diffusion families.

### 3.2 Components and interfaces

```csharp
namespace TensorSharp.Models.SharedDiffusion;

/// A family on the runtime: geometry, components and the default recipe.
public interface IDiffusionFamily
{
    DiffusionGeometry Geometry { get; }          // latent channels, VAE factor, patch, size multiple,
                                                 // min/max/default area, max references, aspect range
    DiffusionGuidance Guidance { get; }          // None (distilled) | Cfg | (LTX) multi-guidance
    IConditioner CreateConditioner(DiffusionLoad load);
    IDenoiser CreateDenoiser(DiffusionLoad load);           // emits one step's GraphProgram(s)
    IVaeCodec CreateVae(DiffusionLoad load);                 // encode/decode programs + tiling plan
    SamplingRecipe DefaultRecipe { get; }                    // schedule, sampler, head policy, steps
    IReferencePolicy References { get; }                     // resize rules per consumer, presentation
    INoiseSource Noise(long seed);                           // torch-CPU MT19937 stream | Philox | ...
    DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> refs, LatentShape target,
                         StepState step);                    // segments, positions, row timesteps
}

public interface IConditioner : IDisposable      // text (+ vision) -> hidden rows with modality tags
{
    ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken ct);
}

public interface IDenoiser : IDisposable
{
    DenoiserCaps Caps { get; }                   // shardable (TP), prefix-cacheable rows, outputs
    StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken ct);
}

public interface IVaeCodec : IDisposable
{
    LatentPosterior Encode(RgbImage image, CancellationToken ct);  // moments (mean, logvar), tiled
    RgbImage Decode(LatentTensor latent, CancellationToken ct);    // tiled, blended
    RgbImage Preview(LatentTensor latent, int maxSide);            // cheap preview
}
```

The runtime owns the rest:

* **`DiffusionPipeline`** runs the phases (§3.4), applies `SamplingRecipe` (schedule from the
  recipe: an explicit sigma list, ComfyUI "simple" over a grid, linspace plus shift per stream,
  dynamic shift; plus the sampler: Euler flow, UniPC multistep with state, ...), drives
  guidance, and applies `EditMask` blending and the final composite.
* **`ResidencyPlanner`** decides, per device, which components stay resident between phases
  and between requests, from `GpuMemoryBudget` and the components' declared bytes (§3.4).
* **`AdapterSet`** holds the LoRA plug-ins bound to the denoiser (and, where a family needs it,
  the conditioner), plus at most one **sampling contract** (§5.4).
* **`DiffusionRequest`**: prompt; ordered references (`RgbImage` plus a role); an optional mask
  with its settings; size (width/height, target area, keepSourceSize); seed; steps, cfg and
  negative prompt (validated against the recipe and the contract); adapters; preview count;
  progress sink; `CancellationToken`.
* **`DiffusionResult`**: the image (or frames and audio), the effective recipe (steps, sigmas,
  seed, sampling canvas, adapters), and per-phase timings (text, vision, VAE encode, denoise
  per step, decode, composite), so every host prints and records the same numbers that
  `LastTimings` gives today.

### 3.3 The request and turn model

The runtime is stateless across requests. A **turn** (multi-turn chat) is a host concept: the
host resolves which pictures a turn uses and in what order, and passes an ordered reference
list. That is what `ImageTurns.Planning.cs` already does for Qwen-Image (`ImagePlan.Edit`,
`Change`, `Again`; `ImageRecipe` holds what a picture was made from). The runtime guarantees:
the reference order defines `<Picture n>`; a seed and the same inputs reproduce the same
picture; and `keepSourceSize` returns `<Picture 1>`'s exact size.

### 3.4 Phases and memory residency

```
Prepare      validate the request against the family and contract; resolve canvas, references, mask
Condition    IConditioner.Encode (text, plus the vision tower for references)   -> ConditioningSet
EncodeRefs   IVaeCodec.Encode for each reference (and for the mask source)       -> LatentRef[]
[release]    the conditioner and the VAE encoder, if the planner says the denoiser needs the room
Denoise      for each step: Pack -> IDenoiser.Step -> sampler update -> mask reinjection -> preview?
[release]    the denoiser's activations; its weights too if the decoder needs the room
Decode       IVaeCodec.Decode -> resize to the requested size -> mask composite -> result
```

`ResidencyPlanner` keeps everything resident when the sum fits (the legacy behaviour on an
H100: about 58 GB, so no regression). Otherwise it releases a phase's components after the
phase through `IGraphExecutor.Invalidate` on their weight leaves. For example, on an A40 the
bf16 DiT (40.2 GB) cannot sit beside the Q4_K_M text encoder (18.2 GB). The weights stay
mapped on the host, so a re-upload is a device copy, not a disk read. `MiniMaxH3DiT.ReleaseDeviceResidency`
and `QwenImage21Pipeline` (which disposes its conditioner before the DiT) are the two
precedents that this generalizes. The planner logs its decision once per load and per geometry
class, and the decision is visible in telemetry.

### 3.5 Scheduling and batching

* **Queue.** One `DiffusionJobQueue` per loaded diffusion model, FIFO. A job runs phase by
  phase on the model's compute thread. Between steps it yields to waiting compute turns
  (`DiffusionComputeTurns` already gives a Jev read or an image encode its turn in
  `DiffusionBatchScheduler`), so a short job's text phase is not stuck behind a 16 MP denoise.
* **Step interleaving (optional, later).** With two jobs of different sizes, the queue may
  alternate steps for fairness and preview latency. It is off by default: a 16 MP step is
  compute-bound at about 2.6 s, interleaving only delays the first job, and two jobs'
  activations must both fit.
* **Batching.** Not in this series. At 4 MP and above every product already has 4K or more
  rows and is compute-bound. Same-shape small jobs (1 MP pDMD previews) could pack into one
  sequence with block-diagonal attention: the per-row modulation segments already allow it,
  and attention would be per-sequence `FlashAttnExt` views. That is recorded as future work
  (decision D15).
* The server's `lock (_imageEditLock)` and the per-route `Task.Run` become `Submit` on the
  queue. Progress and previews (`OnStep`) and cancellation keep their observable behaviour.

### 3.6 Multi-GPU: tensor parallelism

The precedent is Qwen-Image-2.1's DiT TP (`QwenImage21DiT(..., ITensorParallelGroup tpGroup)`:
whole heads and MLP columns per rank, two all-reduces a block, the native
`QwenImage21ForwardTp`). On the runtime, a denoiser that declares `Shardable` is emitted per
rank, as `EmitRankedStep` does for decoders. Each rank takes 56/T heads of the qkv projection
(qkv is stacked `[q; k; v]` in the Comfy-Org file, so a rank takes rows of each third) and
14336/T FFN columns. The adaLN projection, the token refiner, the final layer and the PDD head
are replicated. The out-projection and fc2 partials go through `AllReduce` before
`GatedResidual`. On GGML this is `IRankedGraphExecutor` with `TSGgml_GraphRunRanked`. Exact F32
sums are required for parity (`ExactTensorParallelSums`): a 6-step distilled sampler amplifies
rounding (§7.3). The value of TP here is fitting the bf16 DiT on two 46 GB A40s; on a single
H100 it is slower than one GPU. This is stage S9, optional.

### 3.7 LoRA application

LoRA is unmerged, always (§5.4). The runtime's `Linear(x, w)` emits `LoraLinear` when the
projection has bound parts. Several adapters on one projection are concatenated along the rank
(Qwen's packing), so there is one shrink per input. q, k and v share one stacked down factor
(H3's native packing). Strength x alpha / rank is folded into `up` in F32, then
component-balanced and stored as F16 unless it would overflow (the `MiniMaxH3LoraSet` rules,
F32 on `ggml_cpu`, ranks padded to 64 on Metal and 16 elsewhere). An adapter change invalidates
only the factor leaves (`InvalidateDeviceCopies`), not the base weights.

### 3.8 Cancellation, progress and previews

The `CancellationToken` is checked between phases, between steps and between VAE tile rows. A
cancelled job releases its activations and leaves weights resident. Progress is
`(phase, step, total, preview?)`. Previews decode the current x0 estimate (`x - sigma * v` for
flow models) at a reduced size through `IVaeCodec.Preview`, with masked previews composited
(Qwen's `PreviewCount` semantics). The server's SSE frames (`imageGenerate`/`imageEdit` with
`step`, `total`, `image`) and TensorAgent's `image_step` frames are unchanged.

### 3.9 Does it fit the other families? (read, not planned)

| Family | Needs from the runtime | Fits by |
|---|---|---|
| MiniMax-H3 video (`MiniMaxH3Pipeline`) | two streams (video, audio) with different shifts and a velocity scale (`ShiftSlope`); keyframes and Ref2VA references; audio VAE; video preview | `StepOutput` with several named outputs; a recipe with per-stream sigmas; `IVaeCodec` per modality; the same layout and segment machinery (H3-Image's layout is a subset) |
| Qwen-Image-2.1 (`QwenImage21Pipeline`) | CFG (two branches); a prefix K/V cache (text and reference rows are modulated at t = 0 and do not attend to the target, so their K/V are fixed across steps); Philox noise; DiT TP; PDD LoRA heads (Fun-Acc: per-step heads replacing `proj_out`) | `DiffusionGuidance.Cfg`, batched as two sequences; `DenoiserCaps.PrefixCacheable` (State leaves for the cached K/V); `INoiseSource`; §3.6; the "heads by step" policy (§5.4), the same idea as H3's PDD |
| Wan 2.x (`WanVideoModel`) | T5 conditioner; UniPC multistep (sampler state); CFG; causal 3-D VAE in temporal chunks; A14B's two experts switched by sigma | stateful `ISampler`; a recipe with `DenoiserStage[]` selected by sigma range |
| LTX 2.5 (`Ltx25Pipeline`) | Gemma prompt encoder plus connector; audio and video; two stages (base, then latent upsampler, then refine with a distilled LoRA); IC-LoRA references | `SamplingRecipe.Stages` with a latent transform between stages; per-stage adapter sets |

---

## 4. Editing: upstream semantics

### 4.1 What the official sources say

MiniMax's README (MiniMax-AI/MiniMax-H3-Image@40850901): editing takes **up to 16 reference
images** ("each at most 64 MiB, with sides of 128 to 8192 pixels and an aspect ratio from 1:4 to
4:1"), and their upload order defines `<Picture 1>`, `<Picture 2>`, ... in the prompt.
`num_inference_steps` must be omitted or 6, and `guidance_scale` omitted or 1. A negative prompt
is rejected. The prompt is used as given and truncated to 4,096 tokens. "**Not yet supported:
mask-based inpainting**". SGLang serves `/v1/images/edits` as multipart with `image` fields
(examples/edit.py). The architecture section: "The text prompt and the reference images are
encoded by the H3 text encoder; the reference images are also encoded by the image VAE. All
tokens form one packed sequence."

The reference implementations agree on that: **both** conditioning paths are used. Below is
what diffusers (`modular_pipelines/minimax_h3_image/*`, main@d5df05ed) does, with ComfyUI
(`comfy_extras/nodes_minimax_h3.py`, `comfy/ldm/minimax/{model,vae}.py`, master@995456d0)
noted where it differs.

### 4.2 How references are conditioned

1. **Upright.** `ImageOps.exif_transpose(image).convert("RGB")`. A side below 32 px is refused.
2. **Two resized copies** (`MiniMaxH3ImageEditResizeStep`, `bound_short_edge`): a short edge
   inside `[min, max]` keeps the original pixels; otherwise the image is scaled to the nearer
   bound and both sides are rounded to multiples of 32.
   * The **VAE copy** is bounded to 1024-2048 (`reference_min/max_short_edge`).
   * The **vision copy** is bounded to exactly 1024 (`vision_min = vision_max = 1024`). After
     that, the Qwen3-VL processor's own smart resize applies (`MiniMaxH3VisionEncoder` has the
     same rule: multiples of 32, pixel bounds).
   * References are **never** resized to the output size. ComfyUI does the same
     (`IMAGE_REF_SHORT_EDGE = 1024`, `IMAGE_REF_VAE_MAX_SHORT_EDGE = 2048`) and resizes with
     Lanczos through 8 bits. diffusers uses `VaeImageProcessor.resize`. TensorSharp's
     `ImageIO.Resize` must use the same filter, checked by a test.
3. **Vision path (text encoder).** The presentation is, for each reference i, the label
   `"Picture {i}: "` (**no angle brackets**, unlike H3 video's `"<Picture i>: "`; ComfyUI's
   `picture_label="Picture %d: "` agrees), then `<|vision_start|>`, one `<|image_pad|>` per
   merged patch, `<|vision_end|>`. After all references comes the prompt, verbatim: no chat
   template and no special tokens. The prompt and each label are truncated to 4,096 tokens. The
   vision tower's output replaces the pads, and the DeepStack taps are injected after layers 0-2.
   That is exactly what `MiniMaxH3Pipeline.BuildReferencePrompt` and `MiniMaxH3TextEncoder.Encode`
   do for Ref2VA, except for the label. The conditioning is the layer-50 hidden state of every
   row. **Token tags:** label and prompt rows are text (1); every row of a vision block,
   including its start and end markers, is image (0). That matches `MiniMaxH3ImageLayout`'s
   `textTokenModalities` comment.
4. **VAE path.** Crop the pixels to a multiple of 16, then normalize with ImageNet statistics
   (`(x/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]`). Encode with **tiling** (256-px tiles,
   overlap at least 64, the same `_split_tiles` as the decoder; the **moments** are blended with
   `_stitch_tiles`, a linear ramp against the already-blended neighbour, then cropped). Sample
   the **posterior**: `mean + exp(0.5 * clamp(logvar, -30, 20)) * eps`. Round to float16 and
   back. Crop the latent to whole 2x2 patches. Normalize with `latents_mean/std` (the same
   constants as `MiniMaxH3ImageVae.ReleaseLatentsMean/Std`). Then **augment**: `0.999 * z + 0.001 * eps2`
   (`scheduler.scale_noise` at `reference_noise_aug = 0.999`). The augmented reference rows are
   fixed for the whole loop; only the target rows are updated.
   * The VAE file TensorSharp already downloads (`minimax_h3_image_vae_fp16.safetensors`) holds
     the encoder (read from its header: `encoder.conv_in`, `encoder.down.N.block.N.*`,
     `encoder.down.N.downsample.conv`, `encoder.norm_out`, `encoder.conv_out`, `quant_conv`
     `[48,48,1,1,1]`). These are the names `MiniMaxH3VideoVae` binds. No new download is needed.
5. **Positions and modulation** (`build_image_packed_sequence`). Text row i is at `(i, 0, 0)`.
   Reference k occupies time slot `textLen + k` over its **own** latent grid (the same
   aspect-normalized axes). The target is at time `textLen + nRefs`. Text and target rows are
   modulated at `t = 1 - sigma` and reference rows at `0.999`. `MiniMaxH3ImageLayout.Build`
   already produces exactly this: `rt = max(t, 0.999)`, which is 0.999 on every step because
   `t` stays at or below 0.768 on the six-step grid, and the cursor advances one slot per
   reference.
6. **Output.** The final layer reads only the target rows (`layout.segments[-1]` in ComfyUI;
   `latents[num_reference_rows:]` in diffusers).

### 4.3 Seeds and noise streams

* **diffusers**, one CPU generator per request, in this order: each reference's posterior sample
  (over the **uncropped** latent shape), then the target noise `(1, 24, 1, lh, lw)`, then each
  reference's augmentation noise (over the cropped shape). With references, the target noise at
  a seed therefore differs from text-to-image's at the same seed, and an edit never restarts from
  its source's noise. (TensorAgent's planning notes rely on that property for Qwen.)
* **ComfyUI**: the target noise comes from `prepare_noise(seed)`. Every reference's augmentation
  restarts `Generator().manual_seed(payload seed)` ("every condition intentionally restarts the
  same RNG stream"). The VAE posterior uses a fixed `IMAGE_ENCODE_SEED = 42`.
* **Design:** follow diffusers, MiniMax's documented pipeline whose request fields SGLang
  mirrors (decision D1). `TorchCpuRandn` becomes a **stateful** stream: the same MT19937 and
  Box-Muller blocks of 16, with the state carried across calls, which is how successive
  `torch.randn(..., generator=g)` calls behave. Text-to-image is unchanged: one draw from a
  fresh generator, as today.
* **Schedule.** diffusers steps on the exact grid (`sample_sigmas`, rounded through float32
  `1 - sigma`) and uses head i at step i. ComfyUI's "simple" schedule lands within 0.001 of the
  grid and picks the nearest grid interval. TensorSharp today follows ComfyUI. The design keeps
  that for text-to-image so that the no-regression gate can be bit-exact, and uses it for edits
  too, so that one model has one schedule (decision D2).

### 4.4 Output size for edits

* diffusers: `height`/`width` default to 2048 for both workflows. An area below
  `min_generation_area` (default 4,194,304 = 2048^2) is generated on a larger canvas of the same
  aspect, then resized back with Lanczos (`get_image_canvas_size`, `decoders.py`). Canvas sides
  are rounded to 32. SGLang requires `size`. ComfyUI: width or height 0 means "the VAE size of
  image_1" (short edge 1024-2048), or 2048 without images.
* TensorSharp rules:
  1. explicit width and height win (multiples of 32, as today);
  2. otherwise an edit uses `<Picture 1>`'s aspect at the request's target area. The default is
     the VAE copy's own size, which is ComfyUI's rule and keeps a small photo small;
  3. `keepSourceSize` returns `<Picture 1>`'s exact pixel size: sample at its aspect within the
     area bounds, then resize the decode back. These are Qwen-Image's semantics, which the Web
     UI and TensorAgent already send;
  4. a **generation floor** of 4 MP: smaller requests sample at the floor and are resized down
     with Lanczos (upstream behaviour; decision D3). A speed adapter's contract may lower the
     floor (§5.4). Above 20 MP the request is refused, because the model's range is 4-20 MP.

### 4.5 Masked edits (TensorSharp's own method; upstream has none)

ComfyUI's generic inpainting is the reference (`comfy/samplers.py` `KSamplerX0Inpaint`): before
every model call, `x = x*m + noise_scaling(sigma, noise, latent_image)*(1-m)`, where for flow
models `noise_scaling = sigma*noise + (1-sigma)*latent_image` and `noise` is the sampler's fixed
noise; after the call, the x0 prediction is replaced by `latent_image` where `m = 0`
(0 = keep, 1 = regenerate). The mask reaches the latent grid by bilinear `reshape_mask`. For an
Euler flow step this is algebraically identical to re-injecting the source, re-noised with the
same fixed noise, at the *next* sigma after each update. That is what TensorSharp's Qwen path
does (`QwenImageEditMask.Reinject` at `sigmas[step + 1]`). Writing x0 = src in the kept region:
`x_{i+1} = (1-sigma_{i+1}) * src + sigma_{i+1} * eps`. Note that ComfyUI's H3-Image model
ignores `denoise_mask` (its `forward` takes no such argument), so ComfyUI does no per-row
timestep trick for images. Its H3 *video* model does one (`rows_t = 1 - m*sigma`).

The method:

1. **Mask input and interpretation:** `QwenImageEditMask` semantics, moved to
   `SharedDiffusion/EditMask`. Grayscale white edits and black keeps, or alpha transparent
   edits; `maskInvert`; inward feather (0-1024 px); optional `maskCrop` with padding. The mask
   must match `<Picture 1>`'s exact size, and `<Picture 1>` is the picture being edited.
2. **Canvas:** the source's aspect (or the crop's) at the request's area, with the floor of
   §4.4. The output is always the source canvas.
3. **Source latent:** the posterior **mean** of `<Picture 1>` at the sampling canvas. It is the
   VAE copy already encoded for the reference when that copy has the same size; otherwise it is
   a separate tiled encode. The mean keeps the blend independent of the RNG stream. ComfyUI's
   `latent_image` is a posterior sample at seed 42; the difference is the posterior's standard
   deviation, and the parity test feeds ComfyUI's latent where needed.
4. **Latent mask:** max-pool the pixel weights over each 16x16 cell, as Qwen's `LatentWeights`
   does, so that a thin brush stroke stays editable. Broadcast over the 24 channels, then
   patchify alongside x, which gives 96 values a row.
5. **Loop:** `x_0 = blend(noise, src at sigma_0 = 1)`, which is pure noise. After each Euler
   update, `x = m*x + (1-m)*((1-sigma_{i+1})*src + sigma_{i+1}*eps)`, with `eps` the request's
   initial target noise. The last step (sigma = 0) leaves `src` exactly in the kept cells.
6. **Composite:** decode, resize to the source (crop) size, then `EditMask.Composite`. Pixels
   with weight 0 are the source's bytes exactly, alpha included. An empty mask returns the
   source without inference, as Qwen does. A full mask is an unmasked edit.
7. The reference rows still carry the **whole** `<Picture 1>` (its context), so the model sees
   what it must blend with. This is the Qwen design too.

Optional, behind a recipe field rather than a user switch: mask-aware modulation (kept rows
modulated at `t = 1 - m*sigma`, as ComfyUI's H3 video does). It is off by default for parity;
it is an experiment for stage S6 to measure.

### 4.6 Multi-turn

This uses the same convention as Qwen-Image's turns (`ImageTurns.Planning.cs`): the planner
picks a picture (`Change`, `Edit`, `Again`). For H3-Image the **picture being changed is
`<Picture 1>`**, and photos attached to the same message follow as `<Picture 2..n>`. Today
`Edit(photos)` sends only the attached photos. The planner gains one generic rule: when a
message attaches photos **and** refers to an earlier picture (planner choice `edit` of a picture
in the chat), the sources are `[that picture, ...attached]`. That is also correct for Qwen,
which takes several references. The seed policy is unchanged (`Again` = seed + 1). The mask
belongs to `<Picture 1>`. `keepSourceSize` is set on every edit, as now. The prompt is passed
verbatim (decision D10). TensorAgent's "intent scorer" (`QwenImageIntentScorer`) needs an LM
head that H3's 50-layer encoder does not have, so H3-Image turns take the existing
`ImageIntentUnavailableException` route (shape-only planning).

### 4.7 Limits and cost

Upstream allows 16 references, each up to a 2048 short edge in the VAE copy (4,096 DiT rows for
a square 2048 reference) plus about 1,024 vision rows in the text part. Sixteen of them beside a
16 MP target come to about 98K DiT rows: attention costs about 35 times a text-to-image step,
and activations grow by about 5x. The runtime computes a **row budget** from free memory
(`GpuMemoryBudget`) and refuses a request above it, with a message that names the rows and
suggests fewer or smaller references (decision D11). The cap of 16 references is checked in
`Prepare`. Note that `MiniMaxH3Geometry.MaxReferences = 9` is Ref2VA video's cap and not this
one.

---

## 5. LoRA research

### 5.1 Adapters of MiniMaxAI/MiniMax-H3-Image (HF API, 2026-11-19)

Fifteen repositories list `base_model:adapter:MiniMaxAI/MiniMax-H3-Image`. Headers were read with
HTTP range requests (no weights downloaded).

| Repo @ revision | What it does | File, format, keys | Rank / alpha | Licence | Catalog? |
|---|---|---|---|---|---|
| **linoyts/MiniMax-H3-Image-pDMD-2steps** @c07d96e5 | 2-step pDMD distillation (teacher: the base 6-step sampler), T2I **and editing**, trained about 1 MP | `pytorch_lora_weights.safetensors`, 1,383,680,256 B, sha `c0a88d9c...`; **diffusers PEFT** `transformer.transformer_blocks.N.attn.to_{q,k,v}`, `to_out.0`, `ff.net.0.proj` (`[value; gate]`), `ff.net.2`, plus `token_refiner.refiner_blocks.N.*`; 624 tensors | 128 / 128 (metadata `alpha`, `rank`) | MiniMax-H3-Image Community License | **yes, Speed**. Recipe: 2 steps, `sample_sigmas = [1, 12/13, 0]`, **heads in order** (head 0, then head 1), `min_generation_area = 1024` (generate at the asked size, about 1 MP) |
| **akhaliq/minimax-h3-image-pdmd-2step** @ef1e3b9a | the pDMD 2-NFE LoRA from pdmd2026 ("byte-identical to the upstream pDMD release"); its grids were rendered in **P-H3-Image's stack** (Pruna's 4B text encoder), and the ComfyUI recipe on Comfy-Org's checkpoint is shift 12, "simple", 2 steps (the same sigmas `[1, 12/13, 0]`), with ComfyUI's grid-interval heads | `lora_model_0.safetensors` (diffusers PEFT, 1,383,680,648 B, sha `3179e630...`; a different file from linoyts'), `minimax_h3_pdmd_2nfe_comfyui.safetensors` (**ComfyUI native**, fused `qkv_proj` block-diagonal B, `alpha x 3`, fc1 halves swapped back to `[gate; value]`, 1,956,194,168 B, sha `5feaeef1...`), a 40.2 GB pre-merged checkpoint | 128 / 128 (comfy: qkv 384 / 384) | card says apache-2.0; it is a derivative of a Community-License model | config only (second speed recipe, decision D7) |
| Kevin67456/minimax-h3-image-pdmd-2step @3f0ef8f3 | mirror of akhaliq | same | same | same | no (duplicate) |
| hugmynugs/RealemV2-H3 @6a24162e | style ("Realem" trigger; card sample "Neon street"), ai-toolkit 0.13.25, step 1500 | `Realem-MinimaxH3_000001500.safetensors`, 155,110,352 B, sha `1f7e11fc...`; **ComfyUI native** `diffusion_model.blocks.N.attn.qkv_proj.lora_A/B` (one shared rank over fused qkv), `out_proj`, `mlp.fc1`, `mlp.fc2`, refiner blocks; no alpha tensors | 16 / none (scale 1, as ComfyUI) | none stated | config only (decision D7); a format fixture |
| waryusya/waryusya_MiniMaxH3 @4ab9f672 | character or style ("waryusya" trigger), ai-toolkit via Civitai, 59 images at **512x512**, 10 epoch files | ComfyUI native (as Realem), 310 MB each | 32 / 32 (`ss_network_alpha`) | apache-2.0 (card) | no: unknown subject, 512 px training far below the model's 4-16 MP range |
| SimpleTuner/minimax-h3-image-domokun-2048 @57214626 | test LoRA (the Domo-kun character), 500 steps, attention only | `pytorch_lora_weights.safetensors`, 310,155,360 B, sha `3fd1899c...`; **ComfyUI native with `.alpha` tensors**, fused qkv with rank 96 (3 x 32) | 32 (qkv 96) / per-module tensors | Community License | no (experiment, trademark character); **test fixture candidate** |
| SimpleTuner/minimax-h3-image-domokun-4096, -2048-4096, -LoRA-experiments | training logs, no weights | — | — | Community License | no |
| SimpleTuner/minimax-h3-image-training-assistant @bd2142e9 | tagged **not-for-all-audiences** | diffusers | — | Community License | **excluded (NSFW)** |
| ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter @f1a45475 | a **training** adapter: "keep it on while you train, leave it off when you use the LoRA" | **kohya** `lora_unet_blocks_N_attn_qkv_proj.lora_down/up/.alpha`, 600 tensors, 149 MB | 16 / alpha tensors | Community License | no (not for inference); a kohya format fixture |
| t8star/Minimax-H3-Image-Exp-Comfy @8f1d3909 (and the Resize8004 mirror) | experiments: single-head diagnostics (`mean_head.safetensors`: "head-only diagnostic, NOT recovered teacher"), MLP corrections ("sampling validation pending"), a 66 GB full checkpoint "qkvfix", third-party training adapters | ComfyUI native, `lora_down/lora_up` naming, F32 | 16 | Community License | no (experimental) |
| Tadness/myLoRAs, Tadness/FullNude | NSFW | — | — | — | **excluded (NSFW)** |

Takeaways:

* Only **two** inference adapters are clearly meant for the official stack. Both pDMD 2-step
  adapters have the same sigma schedule but different **head policies**: diffusers uses heads in
  order, ComfyUI the grid interval. A speed contract must therefore declare its head policy.
* **Three key grammars** appear: diffusers PEFT (optionally prefixed `transformer.`), ComfyUI
  native (`diffusion_model.blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}`, `lora_A/B`
  or `lora_down/up`, optional `.alpha` tensors), and kohya (`lora_unet_blocks_N_*`).
  `MiniMaxH3LoraKeys` today accepts only the first, unprefixed, and refuses native names
  ("a checkpoint-format (ComfyUI) name").
* **QKV layout.** diffusers splits `to_q/to_k/to_v`. The Comfy-Org single file stores qkv
  **stacked** `[q_all; k_all; v_all]`, which is what TensorSharp's mapping assumes and what
  matches ComfyUI. MiniMax's **raw** shards store it per-head interleaved; diffusers'
  `convert_minimax_h3_to_diffusers.py` reorders with `reorder_interleaved_qkv`. TensorSharp
  reads only the Comfy-Org file and the unsloth GGUFs (converted from it). A loader for the raw
  `transformer/` shards would have to reorder, or refuse them.
* **fc1 halves.** diffusers is `[value; gate]` (swap into native order). ComfyUI native and
  kohya files are already in native `[gate; value]` (no swap), as `minimax_h3_pdmd_2nfe_comfyui`'s
  metadata states ("Diffusers [value;gate] -> ComfyUI [gate;value]").

### 5.2 Comparison with today's loaders

* `MiniMaxH3LoraSet` (H3 video): unmerged, `strength * alpha / rank` folded into up in F32, fc1
  `lora_B` halves swapped, every tensor accounted for, F16 component-balanced packing, direct
  backends with exact F32 factors. Alpha comes only from metadata or the config ("a per-module
  .alpha tensor" is refused). The plug-in config `minimax-h3-lora` locks the sampling contract
  (`MiniMaxH3SamplingRecipe`: steps, linspace schedule, video and audio shift, cfg; 0 or unset
  means use the recipe, the recipe's own value is accepted, anything else is refused).
* `QwenImage21LoraSet`: unmerged ("a step-distillation LoRA moves the weights by about 0.1-0.5%,
  the size of Q8_0's own rounding step"), several adapters concatenated along the rank,
  `pdd_config.json` replacement heads and norms, at most one sampling recipe.

### 5.3 Design: selection, loading, running

* **Plug-in type** `"minimax-h3-image-lora"` in `LoraCliFlags.PluginTypes`, with config keys
  `type, name, variables, weights{path,urls,sha256}, key_format, alpha, scale, sampling{steps,
  sigmas, heads, cfg, min_generation_area}`. `key_format` is `auto` (detect from the names, and
  refuse a mix), `diffusers`, `comfy` or `kohya`. `heads` is `grid-interval` (the base default,
  ComfyUI's) or `by-step` (diffusers': head i at step i). The schema is written so it can become
  the generic diffusion-runtime LoRA schema when another family migrates.
* **Key grammar** (shared with H3 video; `MiniMaxH3LoraKeys` is extended, not forked):

  | Grammar | Scope | Target | fc1 | Alpha |
  |---|---|---|---|---|
  | diffusers | `[transformer.]transformer_blocks.N`, `[transformer.]token_refiner.refiner_blocks.N` | `to_q/k/v` -> qkv thirds; `to_out.0`; `ff.net.0.proj`; `ff.net.2` | swap halves | metadata `alpha`/`lora_alpha`, else the config, else rank |
  | comfy | `diffusion_model.blocks.N`, `diffusion_model.token_refiner.blocks.N` | `attn.qkv_proj` (whole rows), `attn.out_proj`, `mlp.fc1`, `mlp.fc2` | as stored | `.alpha` tensor, else the config, else rank (ComfyUI: no alpha means scale 1) |
  | kohya | `lora_unet_blocks_N_*`, `lora_unet_token_refiner_blocks_N_*` | as comfy | as stored | `.alpha` tensor |

  Every tensor is used or the load fails naming it, as now. A block-diagonal fused qkv (rank
  3r) applies as one factor pair, because the product equals the block-diagonal form.
* **Running.** Always unmerged, through `LoraLinear` (§3.7). Merging is not offered. On a Q4_K
  or Q8_0 base it loses most of a distillation delta (Qwen's measurement). On the bf16
  safetensors it would need either a second 40 GB copy or writes into a read-only mapping.
  Unmerged BF16 costs `2 * rank * (in + out)` FLOPs a row, which is about 3% of fc1 at rank 128.
  The S7 gate measures it.
* **Several adapters.** Any number of style adapters, each with its own strength, plus **at
  most one speed adapter** (one schedule). The engine refuses two speed adapters, the same rule
  as `LoraKind.Speed` and `TwoPluginsRefusal`.
* **Speed contract.** The adapter's `sampling` fixes steps, sigmas, head policy, cfg (1) and the
  generation floor. Request values may be unset or repeat them; anything else is refused, never
  overridden (`MiniMaxH3SamplingRecipe.Validate/CheckHost` semantics). With pDMD the 2-step
  sigmas `[1, 12/13, 0]` and heads `by-step` select PDD head 0 for step 1 and head 0 + offset 1
  for step 2. `CombineHead` over the single interval `[i, i+1)` gives head 0 + head i, which
  should equal diffusers' fused `pdd_proj_out[i]` (`pdd_fused_heads.safetensors`). A parity test
  checks that against the official file's header and rows before the contract is trusted.
* **Hosts.** `--lora`, `--lora-scale` and `--lora-config` (CLI and server), `TS_LORAS`, the
  request's LoRA list (`ImageEditAsync(..., IReadOnlyList<LoraSpec> loras)`) and
  `ILoraPluginHost.SetLoras`. All of these exist; the new family implements `ILoraPluginHost`.

---

## 6. User surfaces

Spellings are reused everywhere (flag-surface rule: one spelling per option). Every new
user-visible field gets en and zh-cn docs.

### 6.1 TensorSharp.Cli

```
TensorSharp.Cli --config config/minimax-h3-image.json --prompt "..." --output out.png \
  [--width W --height H] [--diffusion-seed S] [--diffusion-steps 6]
  [--image a.png --image b.png ...]          # edit: <Picture 1>, <Picture 2>, ... in order
  [--mask m.png --mask-mode grayscale|alpha --mask-invert --mask-feather N --mask-crop --mask-crop-padding N]
  [--keep-source-size]
  [--lora config/lora/minimax-h3-image-pdmd-2step.json [--lora-scale 1]]
```

All of these flags exist (`Program.cs`, `ImageMaskCliOptions.cs`). H3-Image starts accepting
`--image` and the mask flags. `--cfg` above 1 and `--negative-prompt` stay refused. The image
branch's comment and usage text (Program.cs ~1155, which describe Qwen's `<image1>` tags)
change to say that the prompt names the images the way the model does (`<Picture n>` for
MiniMax-H3-Image). The CLI has no multi-turn image mode, and none
is added: multi-turn is a chat surface.

### 6.2 TensorSharp.Server.Host

* `/api/image-generate[/stream]` and `/api/image-edit[/stream]`: the existing JSON fields
  (`prompt`, `imagePaths`, `width`, `height`, `targetArea`, `seed`, `steps`, `keepSourceSize`,
  `maskPath`, `maskMode`, `maskInvert`, `maskFeather`, `maskCrop`, `maskCropPadding`, `loras`).
  `cfg` above 1 and `negativePrompt` are refused with the model's message, as today. No new
  field is needed (the generation floor is model-declared, decision D3).
* OpenAI `/v1/images/generations` (JSON) and `/v1/images/edits` (multipart, `image` repeated in
  `<Picture n>` order, optional `mask`, `size` as `WxH`, `seed`, `response_format` `b64_json` or
  `url`) are **new routes**, in the shapes SGLang serves (MiniMax's `examples/edit.py` works
  against them unchanged). They sit beside `/v1/videos/generations` in `OpenAIEndpoints.cs`,
  their parsers map to `DiffusionRequest` and share `ParseImageParameters`' validation, and
  `n > 1` runs n jobs in sequence. They are model-generic, so Qwen-Image gets them too.
  `num_inference_steps` and `guidance_scale` map to `steps` and `cfg`; `negative_prompt` maps
  to `negativePrompt`.
* Web UI: no change. It already attaches several images (`imagePaths`), sends masks
  (`mask-editor.js`) and offers "Edit again"; `/api/chat` image turns go through ImageTurns
  where that host is used.

### 6.3 TensorAgent

* **ModelCatalog**: a `MiniMaxH3Image(...)` entry (decision D8):
  * id `minimax-h3-image-q4km`, `Family = CatalogFamily.MiniMaxH3Image` (new), `Kind =
    Diffusion`, `Modalities = Image | ImageOutput`;
  * files: `minimax_h3_image_pruned-Q4_K_M.gguf` (unsloth/MiniMax-H3-Image-GGUF@0d25c734,
    12,721,119,968 B, sha256 `ef784687ff25a974191d00f941388060ef971286fbd2e98b2769fa78c8d599a8`);
    the shared text encoder `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (18,218,065,024 B,
    `11e6efe7...`, the same `CatalogFile` as the H3 video entry, so the store deduplicates it);
    `minimax_h3_image_vae_fp16.safetensors` (Comfy-Org@0bb149f5, `d80e9c2e...`); and the three
    tokenizer files;
  * `MinDeviceMemoryGB = 32` with phase residency (the DiT 12.7 GB, the encoder 18.2 GB released
    before the denoise, about 3 GB of activations at 4 MP). An optional Q8_0 entry (22.0 GB, sha
    `ff5ed88d...`) at 48 GB;
  * licence key `catalog.license.minimaxH3Image`, with the territory exclusion (US, EU, UK, KR)
    and the revenue and hosted-service terms in the notes, `Experimental = true` until the
    quality of Q4_K_M versus bf16 has been measured (§7.3).
* **LoraCatalog**: `minimax-h3-image-pdmd-2step` (linoyts@c07d96e5), `Kind = Speed`,
  `Steps = 2`, `RecipeConfig = "minimax-h3-image-pdmd-2step.json"`, licence = the Community
  License. Nothing NSFW, no training adapter, no experiment (§5.1).
* **ImageTurns**: the multi-reference rule of §4.6 (generic). For H3-Image the turn area is the
  model's floor (2048^2) instead of `DefaultTargetArea` (1024^2). To fix that, the page reads
  the model's declared default area instead of a constant, and the pDMD contract lowers it to
  1 MP.

### 6.4 Config files

* `config/minimax-h3-image.json`: unchanged downloads. The comment gains editing, masks and LoRA,
  plus the new `diffusion-products` key if D4 is accepted.
* `config/minimax-h3-image-q8.json` (new): the unsloth Q8_0 denoiser for 46-48 GB GPUs. A
  `-q4km` config is optional.
* `config/lora/minimax-h3-image-pdmd-2step.json` (linoyts; sampling `{steps 2, sigmas [1,
  0.923076923, 0], heads "by-step", cfg 1, min_generation_area 1024}`).
* `config/lora/minimax-h3-image-pdmd-2step-comfy.json` (akhaliq's ComfyUI file; `heads
  "grid-interval"`).
* `config/lora/minimax-h3-image-realem.json` (style; `licence: none stated` in its comment).
* `config/regression/catalog.json`: modalities `image_generation` plus `image_edit`, and the
  presets.

### 6.5 Docs

`docs/models/minimax-h3-image.md` and `_zh-cn.md`: editing, masks, LoRA, backends and the new
performance table. `docs/env_var_feature_matrix*.md`: drop `TS_H3_IMAGE_VAE_BATCH`; the int8
option per D4. `USAGE*.md` image section. `examples/comfyui` profiles. The README model table
row.

---

## 7. Validation plan and no-regression gates

All logs, reports and images go to the ignored `docs/validation/` or `artifacts/`. Reusable
tools go in `eng/` (e.g. `eng/validation/H3ImageAB/`), and the required fixtures in
`InferenceWeb.Tests/Fixtures/MiniMaxH3Image/`. Every report records the ggml revision
(`cd6d2b75`), the GPU, the driver and the exact files. A skipped or unavailable scenario is
reported as **not run**, never as passed.

### 7.1 Legacy reference (docs/models/minimax-h3-image.md, H100 NVL 94 GB, warm, 156 tokens, 6 steps)

| | Text | Denoise | VAE decode | Job (ComfyUI node) |
|---|---:|---:|---:|---:|
| 2048^2 bf16 | 0.10 s | 3.09 s (0.51 s/step) | 0.28 s (121 tiles) | 4.38 s |
| 2048^2 int8 | 0.09 s | 2.55 s (0.43 s/step) | 0.27 s | 3.88 s |
| 4096^2 bf16 | 0.09 s | 15.8 s (2.63 s/step) | 0.95 s (441 tiles) | 19.6 s |
| 4096^2 int8 | 0.09 s | 13.1 s (2.18 s/step) | 0.9 s | 16.9 s |
| first job, 4 MP | | | | 9.1 s int8, 9.5 s bf16 |

Before any change, stage S3 **re-measures** the legacy numbers on the validation GPU in the same
session (memory: an A/B is only valid within one box and session; evict the page cache before
each load arm).

### 7.2 Performance gates (ggml_cuda)

Run **legacy then new, interleaved, three rounds**, the same binary build where possible, at
2048^2 and 4096^2, bf16 and int8 (if D4 keeps int8):

* the text phase, denoise per step (the median of steps 2-6), VAE decode, the whole request, and
  the first request after load: **new <= legacy x 1.02** each (2% is the measured run-to-run
  noise; set it from the legacy rounds' spread if wider);
* peak device memory: **new <= legacy + 2%** with everything resident. With phase residency on
  smaller GPUs it is reported, not gated;
* editing has no legacy. Its numbers (VAE encode per reference, vision encode, denoise per step
  with 1/4/16 references) are recorded as the new baseline. The A40 and PRO 4500 runs are
  recorded too, without a legacy comparison except at 2048^2 (Q8_0, where legacy fits the GPU).

Metal (`ggml_metal`), MLX and direct CUDA have no legacy real-weights numbers. They get
correctness gates on the fixtures, and real-weights numbers only if a Mac with the disk space
becomes available.

### 7.3 Quality gates

* **Determinism floor.** Legacy twice with the same seed (expected bit-exact on CUDA). Then
  legacy with one legitimate kernel variant (`TS_H3_CUDNN_ATTENTION=0`) to measure how far
  rounding alone moves a 6-step distilled picture. The doc already shows that int8 versus bf16
  is 17.5-19.8 dB, so PSNR is a coarse instrument here.
* **Per-step parity (the primary gate).** Feed the same x, conditioning and layout to the legacy
  `ForwardImage` and the new denoiser step: the velocity must be **bit-exact** when the lowering
  emits the same node sequence (the goal), otherwise relative L2 <= 1e-4 and no worse than the
  floor's per-step error. The same applies to the VAE decode (pixels) and to the conditioner
  (hidden states).
* **End to end.** Same seed, 2048^2 and 4096^2, 8 prompts (the doc's prompt, rendered text, a
  portrait, a landscape and 4 from the upstream showcase): PSNR(new, legacy) **>= 45 dB, or
  bit-exact**, and never below the floor measured above.
* **Editing versus diffusers (independent reference).** diffusers at d5df05ed with the official
  checkpoint, CPU generator: dump `prompt_embeds`, `text_token_tags`, `position_ids`, the
  reference latents (before and after augmentation), the target noise and step-1 `noise_pred`.
  The TensorSharp bf16 DiT must reproduce the layout and positions exactly (float32), the tags
  and the reference resizes exactly, and the RNG stream bit-exact (the stateful MT19937 against
  `torch.Generator`). The VAE encoder moments must be within F16 rounding of FP32 diffusers, and
  the step-1 velocity at cosine >= 0.9999 given diffusers' own inputs. The text encoder
  (Q4_K_M versus bf16) is compared by cosine, recorded but not gated. End-to-end pixels across
  engines are not gated: the doc measured 9-12 dB between engines for text-to-image. Review
  composition and rendered text side by side instead.
* **Masked edits.** Weight-0 pixels are byte-identical to the source (gated). The inside of the
  mask is compared with ComfyUI's SetLatentNoiseMask workflow given the same latent mask and
  latent image, as a visual review.
* **LoRA.** Factor mapping per grammar against an independent merge: (W + scale * B A) x in
  NumPy over rows of the real files, as `turbo-comfy-reference-rows.json` does for H3 video. The
  diffusers and ComfyUI conversions of pDMD must give the same delta for the stacked qkv. The
  pDMD 2-step picture is compared with diffusers' at the same seed (composition review), and the
  unmerged overhead is measured.
* **Quantized catalog files.** Q8_0 and Q4_K_M versus bf16 on the 8 prompts (PSNR, plus a review)
  before the catalog entry loses `Experimental`.

### 7.4 Synthetic fixtures (the Mac, no weights)

A generator in the test project writes a tiny H3-Image-shaped checkpoint at test time from a
seed (2 blocks, hidden 256, 2 heads of 128, 1 refiner block, 6 PDD heads, a 2-block VAE ViT, a
2-level VAE encoder). The tests:

* the new denoiser step versus legacy `ForwardImage` on `ggml_cpu` and `ggml_metal`
  (`TSGgml_MiniMaxH3DitForward`), and versus the managed executor (`cpu`), with references and
  several segments; error bounds set from the measured floor (memory: synthetic fixtures carry
  about 40x the rounding noise);
* the VAE decode program versus legacy `MiniMaxH3ImageVae.Decode`, and the VAE encode program
  versus a NumPy oracle file (`eng/`) for one tile and the moment stitching;
* layout, positions and tags versus diffusers' functions ported into the test as golden arrays
  (no Python at test time); the stateful randn against stored `torch` outputs; the reference
  resizes; canvas rules; the mask maths (Reinject, Composite, empty and full masks);
* LoRA grammar resolution for all three formats (tiny generated files), refusal messages,
  contract validation, and two speed plug-ins refused;
* before S10 deletes legacy, the legacy outputs on the fixtures are frozen into
  `Fixtures/MiniMaxH3Image/` (a few hundred KB), so coverage survives the deletion.

Tests never set process environment variables (memory rule); they use model seams.

### 7.5 Hardware and disk

| Set | Files | Disk |
|---|---|---:|
| bf16 (legacy reference) | DiT 40.23 GB + TE Q4_K_M 18.22 GB + VAE 1.17 GB + tokenizer | 59.6 GB |
| Q8_0 | 22.03 + 18.22 + 1.17 | 41.4 GB |
| Q4_K_M | 12.72 + 18.22 + 1.17 | 32.1 GB |
| diffusers reference | `transformer/` 66.3 + MiniMax-H3 `text_encoder/` 66.7 + `vae/` 2.3 + PDD heads | about 136 GB |
| ComfyUI reference | Comfy-Org bf16 DiT (shared) + TE int8 27.1 + VAE (shared) | +27 GB |
| LoRAs | linoyts 1.38, akhaliq diffusers 1.38 and comfy 1.96, Realem 0.16, domokun 0.31, Fizgig 0.15 | 5.3 GB |

* **H100 80 or 94 GB (sm_90), required.** It holds the legacy reference numbers. Everything
  resident in bf16 (about 60 GB plus 16 MP activations) fits only here. diffusers needs 80 GB
  plus about 140 GB of host RAM with offload (its card: "a 4736x3552 image peaks at about
  65 GB"). Disk: **at least 400 GB** for all sets, references and the Python environment.
* **A40 46 GB (sm_86).** Ampere coverage of cuDNN SDPA, BF16 cuBLAS and int8. Q8_0 fits at
  16 MP. bf16 needs phase residency (DiT 40.2 GB plus about 3 GB of activations at 4 MP, tight;
  16 MP does not fit). **2x A40** for the TP stage (bf16 at 20 GB a rank) if peer-to-peer works
  (memory: one 2x A40 VM had broken PXB P2P; the host-staged fallback exists). Disk about 150 GB.
* **RTX PRO 4500 32 GB (sm_120).** The TensorAgent tier: Q4_K_M with phase residency. About
  60 GB of disk.
* **This Mac:** synthetic fixtures only (`cpu`, `ggml_cpu`, `ggml_metal`). No real weights: the
  disk is 96% full.

---

## 8. Staged implementation plan

Each stage is reviewable and pushable on its own. It leaves main-quality code (tests green,
docs updated for what it changes), records its validation in `docs/validation/` (ignored), and
names its gates. Estimates are agent-days including GPU runs.

| Stage | Deliverable | Main files | Tests and gates | Effort |
|---|---|---|---|---|
| **S1** IR and GGML lowering for DiT regions | `GraphFusion.AdaLnModulate`, `GatedResidual`, `TableRopeQkNorm`, `LoraLinear`; `SwigluFfn` biases; `GraphOp.Conv2d`, `PadReflect1d`; the shared `ggml_ops_dit_offload.{h,cpp}` (moved from the H3 file, a pure refactor); in-graph custom-node serving and the `NodeOffload` run in `ggml_ops_graph_builder.cpp`; the `WideTokenProducts` program attribute; reference ops in `TensorGraphExecutor` | `TensorSharp.Core/Graph/*`, `TensorSharp.GGML.Native/ggml_ops_graph_builder.cpp`, new `ggml_ops_dit_offload.*`, `ggml_ops_minimax_h3.cpp` (includes it), `GgmlGraph.cs` | region reference-op tests on managed, ggml_cpu and ggml_metal; served-versus-reference on CUDA; H3 video regression: `MiniMaxH3DitProbe` bit-exact before and after the move | 5 |
| **S2** Diffusion runtime core | `SharedDiffusion/` interfaces, `DiffusionPipeline` phases, `ResidencyPlanner`, `SamplingRecipe` (explicit, ComfyUI-simple-over-grid, linspace+shift), Euler flow sampler, stateful `TorchCpuRandn` stream, `EditMask` (moved from `QwenImageEditMask` as a shared type; Qwen keeps using it), the shared `GraphEmitter` (extracted from `GraphStep`), `ServingImplementation.SharedDiffusionRuntime`, the registry safetensors-detector hook, `DiffusionJobQueue` | `TensorSharp.Models/SharedDiffusion/*`, `TensorSharp.Models/Architecture/*`, `TensorSharp.Runtime/Scheduling/Diffusion/*`, `SharedDecoder/StandardDecoderModel.Graph.cs` | unit tests: schedules, randn stream versus stored torch outputs, mask maths (the existing Qwen mask tests still pass), queue cancellation; AR runtime regression (decoder tests unchanged) | 4 |
| **S3** H3-Image denoiser on the runtime | `MiniMaxH3Image/` family: layout (from `MiniMaxH3ImageLayout`), recipe (from `MiniMaxH3ImageSampling`), denoiser emission (refiner, 50 blocks, final layer; PDD head as an Input leaf), weight binding plus `Preload` with the H3 uploader; a test-only seam to the legacy `ForwardImage` | `TensorSharp.Models/Models/MiniMaxH3Image/*`, `GgmlGraphExecutor.cs` | synthetic parity versus legacy (ggml_cpu, ggml_metal, managed); **H100**: per-step velocity bit-exact (or within the floor) at 2048^2 and 4096^2, bf16 and int8; per-step time <= legacy x 1.02 | 6 |
| **S4** Full text-to-image on the runtime; switch | VAE decode program (batched tile rows, cuDNN batched SDPA offload), the conditioner adapter wrapping the native `MiniMaxH3TextEncoder` (shared with video), the descriptor switched to the runtime (same id; `hyvid` rule kept), `ModelBase.Create` and `ModelLifecycleService` special cases replaced by the registry hook; legacy image classes reachable only through the test seam | family folder, `ModelBase.cs`, `ModelLifecycleService.cs`, `MiniMaxH3ImageArchitecture.cs` -> new descriptor | **H100**: all §7.2 gates (text, step, decode, job, first job; bf16 and int8; 2048^2 and 4096^2); §7.3 end-to-end PSNR >= 45 dB or bit-exact on 8 prompts; **A40** Q8_0 run at 4096^2 recorded; `ServingContractConformanceTests` updated | 5 |
| **S5** Editing (unmasked) | VAE encoder program (Conv2d, reflect pad, last-tap slices, tiled moments with `_stitch_tiles`, posterior sample, F16 round trip), reference policy (EXIF, bound_short_edge, Lanczos), vision presentation (`"Picture i: "`, tags), the diffusers RNG order, output size rules and generation floor, 16-reference cap and row budget; `SupportsImageEdit = true`; CLI `--image`, `/api/image-edit`; the new OpenAI `/v1/images/generations` and `/v1/images/edits` routes (§6.2) | family folder, `TensorSharp.Cli/Program.cs` (help text), `WebUiChatService.cs` (requests through the queue), `OpenAIEndpoints.cs`, `WebUiAdapter.cs`, `API_EXAMPLES*.md` | component parity versus diffusers dumps (§7.3) on **H100**; synthetic encoder versus the NumPy oracle; refusals (17 references, row budget, cfg, negative prompt); route tests (multipart order defines `<Picture n>`; MiniMax's `examples/edit.py` against a fake model); 1/4/16-reference timings recorded | 9 |
| **S6** Masked edits and multi-turn | masks through `EditMask` (latent max-pool, reinjection, composite, crop), `keepSourceSize`; ImageTurns' `[picture, ...attached]` rule; the TensorAgent turn area from the model; optional mask-aware modulation measured (off by default) | family folder, `TensorAgent.Core/Hosting/ImageTurns*.cs`, `wwwroot/index.html` (none expected) | weight-0 pixels byte-exact; ComfyUI SetLatentNoiseMask comparison (review); `ImageTurnsTests`, `MaskedImageTurnsTests` extended for H3; a multi-turn E2E on **A40** (Q8_0) | 4 |
| **S7** LoRA | the three grammars (extended `MiniMaxH3LoraKeys`, which H3 video also gets), unmerged `LoraLinear` binding, several style adapters plus one speed contract (sigmas, heads, floor), the `minimax-h3-image-lora` type, `config/lora/minimax-h3-image-*.json`, `ILoraPluginHost` | `MiniMaxH3LoraKeys.cs`, family folder, `TensorSharp.Runtime/LoraSpec.cs`, `config/lora/*` | grammar and refusal tests (generated tiny files); row parity against an independent merge for linoyts, akhaliq-comfy, Realem and domokun (header plus rows only); pDMD versus diffusers at the same seed on **H100** (review); unmerged overhead <= 5% per step at rank 128 | 6 |
| **S8** Catalogs, configs, docs | TensorAgent `ModelCatalog` entry (Q4_K_M, optional Q8_0), `CatalogFamily.MiniMaxH3Image`, licence strings (en, zh-Hans, ja), `LoraCatalog` pDMD entry; `config/minimax-h3-image-q8.json`; `docs/models/minimax-h3-image*.md`, `USAGE*`, the env-var matrix, the regression catalog, ComfyUI profiles | TensorAgent `Catalog/*`, `Localization/*`, `config/*`, `docs/*` | catalog tests (sizes, sha256, licences), `ModelCatalogTests`; a TensorAgent download-and-run on the **PRO 4500** (Q4_K_M) | 2 |
| **S9** (optional) DiT tensor parallelism | per-rank emission of the denoiser, exact F32 all-reduce, the ServingContract `TensorParallel` gap closed for ggml_cuda | family folder | **2x A40** bf16 at 4096^2: velocity versus one GPU within the floor; timings recorded | 5 |
| **S10** Legacy deletion | delete the §1.6 deletable list; retire `TS_H3_IMAGE_VAE_BATCH` (and `TS_H3_INT8_MATMUL` per D4) through the removed-names registry (a hard error naming the replacement); golden fixtures frozen before deletion; docs | `MiniMaxH3/*Image*`, `MiniMaxH3DiT.cs`, `MiniMaxH3Config.cs`, `MiniMaxH3WeightFile.cs`, `ggml_ops_minimax_h3.cpp`, `GgmlNative.cs`, `examples/comfyui/profiles/*` | the whole suite; H3 video regression (DitProbe bit-exact, Turbo LoRA tests); a final **H100** smoke run of T2I and edit | 2 |
| **S11** (later, with H3 video) | the conditioner and vision tower on the shared decoder runtime (prefill to the layer-50 hidden state, DeepStack injection) | `SharedDecoder/*` | text-phase <= legacy x 1.02; hidden-state parity | 5+ |

Order: S1 -> S2 -> S3 -> S4 (text-to-image parity, production switched) -> S5 -> S6 (editing)
-> S7 (LoRA) -> S8 (surfaces) -> S9 (optional) -> S10 (deletion). S8 can land piecemeal with
S4, S5 and S7. The total is about 43 agent-days, plus 5 for S9.

---

## 9. Decisions for the user

| # | Decision | Recommendation |
|---|---|---|
| D1 | Which upstream defines an edit's noise and seed streams: diffusers (MiniMax's pipeline; SGLang mirrors its API) or ComfyUI (fixed seed 42 for the posterior, references restart the seed) | **diffusers**. Same seed plus same inputs gives MiniMax's own stream; ComfyUI is reviewed visually |
| D2 | Text-to-image schedule: keep ComfyUI's "simple over the grid" (today's) or switch to the exact grid (diffusers) | **keep today's** (the bit-exact no-regression gate); revisit separately |
| D3 | Adopt upstream's 4 MP generation floor (smaller requests sample larger and are resized down) | **adopt**, model-declared, no new flag; the pDMD contract lowers it to 1 MP; TensorAgent turns use the model's area |
| D4 | int8 products (`TS_H3_INT8_MATMUL`, 17% faster at 16 MP, 17.5-19.8 dB from bf16) | **keep as an option**, one spelling for H3 image and video: config key `diffusion-products` (`bf16` default, `int8`), the CLI `--diffusion-products`; retire `TS_H3_INT8_MATMUL` with a hard error |
| D5 | When legacy goes: in the switching change (S4, the unified-execution policy) or after editing and LoRA (S10) | **S10**, with legacy reachable only through a test seam after S4 (no user switch), because legacy is the text-to-image oracle while S5-S7 change shared code |
| D6 | Text encoder and vision tower: wrap the native ones (shared with H3 video) now, or migrate them in this series | **wrap now**; migrate with H3 video (S11) |
| D7 | Catalog LoRAs | TensorAgent: **linoyts pDMD 2-step only**. Configs: linoyts, akhaliq-comfy, Realem. Exclude NSFW (Tadness x2, training-assistant), training adapters (Fizgig), experiments (t8star/Resize8004, SimpleTuner), mirrors and waryusya |
| D8 | TensorAgent tier | **Q4_K_M at 32 GB** with phase residency (Experimental until measured); optional Q8_0 at 48 GB; no bf16 entry |
| D9 | Listing a model whose licence excludes the US, EU, UK and KR in TensorAgent | **Decided by the user (2026-11-19): list it.** TensorAgent shows the terms and requires acceptance at download. Today a catalog entry carries only a licence name (`CatalogModel.License`) and nothing asks for acceptance, so S8 adds the flow: the entry declares its terms (a summary in every locale, the territory and revenue clauses, and a link to the licence), the download does not start until the user accepts, and the acceptance is stored with the licence's revision so a changed licence asks again |
| D10 | Multi-turn prompts: verbatim, or auto-prefix "Edit `<Picture 1>`." | **verbatim** (upstream: "used verbatim"; ComfyUI prefixes only for p-h3-image) |
| D11 | References beyond the memory row budget | **refuse** with the row count and a suggestion; no silent downscale |
| D12 | Companion flags `--video-text-encoder` and `--video-vae` for an image model | **keep** for now (no flag churn); generic `--text-encoder` and `--vae` when a second family migrates |
| D13 | Validation hardware | an **H100 80/94 GB with >= 400 GB disk** (required); A40 46 GB; 2x A40 for S9; RTX PRO 4500 for the catalog tier |
| D14 | Tensor parallelism in this series | **optional S9**, after LoRA |
| D15 | Batching concurrent requests | **not now** (compute-bound at 4 MP and above); FIFO queue with yielding |

## 10. Open questions and risks

* **Bit-exactness of the lowering.** The legacy graph's node order (e.g. `h3_token_rows`
  copies, the V pre-scale in `h3_attend`) must be reproduced by the emission. Where it cannot be,
  the per-step gate falls back to the floor, and a 6-step distilled sampler magnifies small
  differences (§7.3).
* **Per-geometry plan memory.** A replayable 16 MP plan holds GBs of compute buffer. The executor
  must release cached plans under memory pressure, or the A40 runs fail after a size change
  (compare the 27B ggml_cuda VRAM-leak lesson in memory).
* **Raw MiniMax shards** (interleaved qkv) are not supported. Only Comfy-Org's file and the
  unsloth GGUFs are.
* **pDMD on the official stack.** Only linoyts' card documents it. akhaliq's grids were rendered
  with Pruna's 4B text encoder, so its quality with the 32B encoder is unknown until S7 measures
  it.
* **Q4_K_M quality** of the H3-Image DiT has not been measured anywhere: a catalog blocker
  (§7.3).
* **The VAE resize filter**: Lanczos parity with PIL or torch must be checked. A one-pixel
  difference in a reference changes the conditioning.

---

## Appendix A: upstream sources read

| Source | Revision | Used for |
|---|---|---|
| github.com/MiniMax-AI/MiniMax-H3-Image | 40850901 (2026-11-11) | README (architecture, request fields, 16 references, sizes, no masks), `examples/edit.py`, `context_ir.py`, `prompt.edit.txt` |
| huggingface.co/MiniMaxAI/MiniMax-H3-Image | 8ced6006 | adapter listing, licence |
| huggingface/diffusers `src/diffusers/modular_pipelines/minimax_h3_image/{before_encoder,encoders,before_denoise,denoise,decoders,modular_blocks_minimax_h3_image,modular_pipeline}.py`, `models/transformers/transformer_minimax_h3_image.py`, `models/autoencoders/autoencoder_kl_minimax_h3_image.py`, `scripts/convert_minimax_h3_to_diffusers.py`, `docs/.../pipelines/minimax_h3_image.md` | main d5df05ed (2026-11-19) | reference resizing, presentation, tags, layout, RNG order, posterior, augmentation, tiling, heads, canvas rules, qkv interleave |
| comfyanonymous/ComfyUI `comfy_extras/nodes_minimax_h3.py`, `comfy/ldm/minimax/{model,vae}.py`, `comfy/text_encoders/minimax.py`, `comfy/samplers.py`, `comfy/model_sampling.py` | master 995456d0 (2026-11-19) | edit node, picture label, ref sizes, encode seed 42, ref noise, denoise-mask handling, inpaint blending |
| LoRA repos in §5.1 (cards plus safetensors headers by range request) | the revisions in §5.1 | formats, ranks, alpha, licences, recipes |
| unsloth/MiniMax-H3-Image-GGUF | 0d25c734 | quant sizes and sha256 |
| Comfy-Org/MiniMax-H3-Image | 0bb149f5 | file sizes; the VAE header (encoder present) |

SGLang and vLLM-Omni were not read beyond MiniMax's README description of their request fields.
Their edit semantics are assumed to follow MiniMax's pipeline, which is what diffusers ports.
