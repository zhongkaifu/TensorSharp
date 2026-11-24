# Qwen3-VL (`qwen3vl`, `qwen3vlmoe`)

[← back to model index](README.md) | [中文](qwen3vl_zh-cn.md)

TensorSharp runs Qwen's Qwen3-VL vision-language models: the dense checkpoints llama.cpp's converter labels
`qwen3vl` and the mixture-of-experts ones it labels `qwen3vlmoe`, such as
[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF). Text, images and video
(`video_url`) are supported.

The family is built on the shared decoder runtime
([`StandardDecoderModel`](../../TensorSharp.Models/SharedDecoder/StandardDecoderModel.cs); see
[unified execution](../design/unified-execution.md)), so it gets continuous batching, prefix reuse, tensor
parallelism (`--tp T`), whole-layer placement (`--layer-split P`), their hybrid and routed experts on the host
(`--n-cpu-moe`, `--cpu-moe`) without code of its own. Its own parts are the Q/K treatment, the interleaved M-RoPE
sections and DeepStack.

| Property | Value |
|---|---|
| GGUF architecture | `qwen3vl` (dense), `qwen3vlmoe` (mixture of experts) |
| Source class | [`Qwen3VLModel`](../../TensorSharp.Models/Models/Qwen3VL/Qwen3VLModel.cs) |
| Architecture plug-in | [`Qwen3VLArchitecture`](../../TensorSharp.Models/Models/Qwen3VL/Qwen3VLArchitecture.cs) |
| Vision tower | [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs) (the Qwen3-VL tower Qwen 3.5 / 3.6 also use), with its DeepStack mergers |
| Projector | the model's `mmproj` (`clip.projector_type = qwen3vl_merger`); found beside the model as `mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf`, `mmproj-F16.gguf` and similar, or passed with `--mmproj` |
| Modalities | Text, image, video (`video_url`, sampled into timed frames) |
| Thinking | Thinking checkpoints only (their template opens `<think>`); Instruct checkpoints have no reasoning block |
| Tool calling | Hermes `<tool_call>` JSON, as the template declares it |
| Batched / paged forward | From the shared runtime |
| Speculative decoding | Weight-free n-gram only (`--spec --spec-type ngram`); no published drafter |
| Backends | Every backend for text; media prompts run as whole-forward graphs (DeepStack), which every backend has |

## Run it

```bash
dotnet run --project TensorSharp.Server.Host -- \
    --model models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf \
    --mmproj models/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf \
    --backend ggml_cuda --port 5000
```

A clip goes in as an OpenAI `video_url` part (a data URL or a URL the server may fetch):

```json
{"role": "user", "content": [
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,...", "fps": 2, "max_frames": 64}},
  {"type": "text", "text": "What happens in this clip?"}]}
```

`VIDEO_SAMPLE_FPS=2` (and `VIDEO_MAX_FRAMES=64`) make the server sample clips the way the Qwen3-VL processor does
when a request does not say: 2 frames per second, merged in pairs (see [Video](#video)).

## Architecture

The text graph matches llama.cpp's `qwen3vl` / `qwen3vl-moe` graphs and Hugging Face's `Qwen3VLMoeTextModel`:

```text
tokens -> embedding (media rows: the tower's main merger)
  -> N x [ RMSNorm -> Q, K, V
            -> per-head Q/K RMSNorm -> NeoX RoPE over the whole head
               (interleaved M-RoPE: rope.dimension_sections (T, H, W) for media rows)
            -> causal GQA -> output projection -> residual
          RMSNorm -> SwiGLU (dense) or top-K routed experts (renormalized, no shared expert) -> residual
          after layer i < n_deepstack_layers: + DeepStack tap i at the media rows ]
  -> RMSNorm -> LM head
```

Qwen3-VL-30B-A3B: 48 layers, hidden 2048, 32 query and 4 KV heads of 128, 128 routed experts (top 8, 768 wide),
sections `[24, 20, 20, 0]`, three DeepStack layers.

### DeepStack

The tower's blocks 8, 16 and 24 (`clip.vision.is_deepstack_layers`) feed three more mergers (`v.deepstack.N.*`):
each takes the residual stream after its block, merges every 2 x 2 patches into one 4,608-wide row, normalizes that
row (the final merger normalizes before merging) and projects it through fc1, exact GELU and fc2 to the language
model's width. The language model adds tap i to the residual stream at the media rows after decoder layer i.

- The tower returns `[main | tap 0 | tap 1 | tap 2]` per media row (`ConcatenateDeepStack`), computed in the same
  merged graph as the blocks on the GGML backends (`TSGgml_Qwen35VisionEncoderMergedBatchF32`).
- `Qwen3VLModel.SetVisionEmbeddings` hands the main part to the runtime's media rows and keeps the taps, anchored to
  the KV index of the forward that injects their media (`StandardDecoderModel.OnMediaInjected`), so a prefix-cache
  trim or a split prompt keeps them aligned.
- Each graph step adds the taps of the rows it holds as one per-run input per tap layer (zeros on text rows), in
  `EmitBlockOutput`. A step that cannot run as a graph while taps wait fails instead of answering without them.
- `TS_QWEN3VL_DEEPSTACK=0` leaves the taps out, as a diagnostic A/B.

### Vision tower

Qwen3-VL's tower is Qwen 3.5's: 27 blocks of 1152, 16 heads of 72, patch 16 with two temporal slices, 2 x 2 spatial
merge. Its blocks use the tanh GELU and its mergers (final and DeepStack) the exact GELU. Images are resized to
multiples of 32 within the processor's pixel budget, as for Qwen 3.5.

### Video

A `video_url` part is sampled into timed frames (`fps` / `max_frames` in the part, `VIDEO_SAMPLE_FPS` /
`VIDEO_MAX_FRAMES` otherwise) and rendered as the Qwen3-VL processor's layout
([`QwenVideoFrames`](../../TensorSharp.Runtime/QwenVideoFrames.cs)): frames at most 0.575 s apart share a temporal
patch, every patch is a `<t seconds><|vision_start|><|video_pad|>...<|vision_end|>` block, and the clip is resized
as a whole against the processor's 25,165,824-pixel budget. TensorSharp also writes the sampled frame times as a line
of text before the clip, which the Hugging Face processor does not.

A clip's pairs are encoded together (`Qwen35VisionEncoder.EncodeVideoPairs`): their frames are decoded in parallel
and, on the GGML backends, the merged graph stacks up to `TS_QWEN_VISION_BATCH_PATCHES` (8,192) patches of pairs per
call, with attention kept within each pair. Loading the projector encodes one small blank frame pair on a GGML GPU
backend (`WarmUp`, about 0.4 s), so the first clip does not pay the tower's one-time device setup (about 10 s on CUDA
beside the language model); `TS_QWEN_VISION_WARMUP=0` skips it.

Where OpenCvSharp's native library does not load (it is not shipped for every platform), frames are read through the
`ffmpeg` and `ffprobe` executables instead
([`FfmpegVideoDecoder`](../../TensorSharp.Models/Media/Desktop/FfmpegVideoDecoder.cs); `TS_FFMPEG` / `TS_FFPROBE`,
or `PATH`).

## Chat template

The `qwen3vl` [chat protocol](../../TensorSharp.Runtime/ChatProtocolRegistry.cs) renders the GGUF's template as
shipped. It differs from Qwen 3.5's in the reasoning block: an Instruct template has none, so the generation prompt
ends `<|im_start|>assistant\n` exactly as the template renders it (`KeepsGenerationPromptWhitespace`), and only a
Thinking template's `<think>\n` is restored in front of a cached assistant turn. Tool calls are Hermes
`<tool_call>` JSON.

## Performance

One CUDA GPU, Qwen3VL-30B-A3B-Instruct Q4_K_M with the F16 projector on `ggml_cuda`, against vLLM 0.32 serving
QuantTrio's AWQ checkpoint (4-bit language model, BF16 tower, FP8 KV cache). Three security-camera clips of 20-37 s,
sampled at 2 frames per second (64 frames, 11.9K-12.5K prompt tokens), one request at a time, with the checkpoint's
sampling (temperature 0.7, top_p 0.8, top_k 20):

| | TensorSharp | vLLM |
|---|---|---|
| First token, a clip not seen before | 7.9-8.0 s (10.1 s for the first clip after start-up) | 4.2-4.7 s |
| First token, another question about the same clip (TensorSharp reuses the sampled frames and the tower output, vLLM the decoded frames) | 4.3-4.7 s | 3.0-3.4 s |
| Decode after a 12K-token video prompt | 64-66 tok/s | 78 tok/s |
| Decode after a short text prompt | 98 tok/s | — |

A new clip's 12.5K-token prompt spends about 1.2 s in frame extraction, 0.3 s decoding the frames, 2.2 s in the tower
(34 pairs of 1,400 patches; 3.3 s with F32 linears) and 4-5 s in prefill. The prefill is ggml's: llama.cpp b12030 takes
2.9 s for an 11.3K-token text prompt on the same GGUF where TensorSharp takes 3.5 s, and vLLM's AWQ kernels are faster
still. On 15 short questions about the clips, checked against answers written from the frames beforehand, TensorSharp
answered 11 right, 2 partly and 2 wrong, vLLM 10, 3 and 2.

`top_k` 0 (no limit) samples over the whole 151,936-token vocabulary on the host and drops decode to about 42 tok/s;
vLLM's default is the checkpoint's `top_k` 20.
