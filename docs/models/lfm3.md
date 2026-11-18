# Liquid LFM3 and LFM3-Omni (`lfm3`)

[← back to model index](README.md) | [中文](lfm3_zh-cn.md)

Liquid AI's LFM3 checkpoints declare `general.architecture = lfm3`: **LFM3-1.7B**, a text
model with reasoning and tool calls, and **LFM3-Omni-2B**, the same decoder with one
multimodal projector that takes images, video and audio in and speaks (its speech
generator and codec: [Speech output](#speech-output)). Both share one chat protocol; the
GGUF's own template tells them apart.

| Property | Value |
|---|---|
| GGUF architecture | `lfm3` (both checkpoints) |
| Model class | `Lfm3RuntimeModel` on the shared decoder runtime (`TensorSharp.Models/Models/Lfm3/`) |
| Chat protocol | `lfm3` in [`ChatProtocolRegistry`](../../TensorSharp.Runtime/ChatProtocolRegistry.cs), fallback renderer [`ChatTemplate.Lfm3.cs`](../../TensorSharp.Runtime/ChatTemplate.Lfm3.cs) |
| Reference template | the GGUF `tokenizer.chat_template` (byte-identical to `LiquidAI/LFM3-1.7B` / `LFM3-Omni-2B` `chat_template.jinja`) |
| Modalities | 1.7B: text. Omni: image, video and audio in (with the projector); text out |
| Thinking | 1.7B: yes (`<think>`; see [the default](#reasoning)). Omni: no |
| Tool calling | Yes: Qwen3-Coder XML (`<tool_call><function=…><parameter=…>`) |
| Ready-made configs | [`config/lfm3-1.7b.json`](../../config/lfm3-1.7b.json), [`config/lfm3-1.7b-nvfp4.json`](../../config/lfm3-1.7b-nvfp4.json), [`config/lfm3-omni-2b.json`](../../config/lfm3-omni-2b.json) |
| TensorAgent | `lfm3-1.7b-q4km`, `lfm3-omni-2b-q4km` (experimental, 12 GB tier) |
| License | LFM Open License v1.0 |

## Validation status

As of 2026-11-16 (llama.cpp reference `d3c3e242c`):

- **Validated:** the chat protocol. 32 OpenAI request bodies (plain chat, system
  prompts, multi-turn with reasoning, tools, single and parallel tool rounds, typed
  arguments, reasoning inside a tool round, Unicode, and Omni image / audio /
  image+audio / history turns) go through TensorSharp's OpenAI message and tool
  parsers and render **byte for byte** what llama-server's `/apply-template` renders,
  through the GGUF template and through the fallback renderer alike
  (`InferenceWeb.Tests/Lfm3ChatProtocolTests.cs`). Reply parsing is checked on real
  greedy replies of both Q4_K_M GGUFs from llama.cpp, streamed in chunks of 1 to 4096
  characters. The pinned files' SHA-256 equal the Hugging Face LFS object ids.
- **Validated (2026-11-17), Omni media input end to end** against llama-server
  (`--mmproj`, same GGUF and mmproj, greedy, identical OpenAI request bodies; see
  [Media input](#media-input)): 7 image prompts (photos, a newspaper page, a flowchart, a
  receipt, a 2048-row landscape, two images at once), 8 ASR clips (6 to 47 s), 2 audio
  questions and 2 image multi-turn conversations give the same prompt token count (one
  MP3 clip excepted: its decoders differ by one row's worth of samples) and
  **token-identical replies** in 18 of these 19 cases on ggml_metal (a multi-turn reply
  diverges after 97 identical tokens) and 18 of 19 on MLX (a flowchart description
  diverges after 93), and in all 4 image and audio cases run on ggml_cpu and the managed
  CPU. ASR word error rate against the spoken text is 0 on 5 of the 7 clips with known
  text; the other two misread one word each, identically in both engines. Videos are read
  the model's way (2 fps with the soundtrack) and agree in layout to one audio row (see
  below); their replies differ after 6 to 14 tokens because the decoders' frames differ.
- **End to end (text, 2026-11-17, `eng/lfm3/e2e`):** the server's OpenAI chat
  completions, Responses and Web UI surfaces and the CLI, checked by 20 scripted
  scenarios (streaming, stop strings, seeds, thinking and its budget, tool calls and
  `tool_choice`, structured output, prefix reuse, cancel, robustness) on ggml_metal for
  Q4_K_M, Q8_0, QAD-Q4_0, BF16, NVFP4 and LFM3-Omni-2B as a text model, and on MLX,
  ggml_cpu and the managed CPU for a subset. Against llama-server on the same GGUF,
  greedy, the same rendered prompts: GSM8K (200 questions thinking off, 50 on) scores
  the same on Q4_K_M (81.5% / 98%) and Q8_0 (86% / 98%), and 92-93% of the thinking-off
  replies are the same text; needle retrieval at 8K-128K finds every needle on both engines
  with identical replies; the 33-prompt instruction-following and 35-case
  function-calling sets score equal or higher, the differences being the two
  behaviors below (stray reasoning with thinking off, `tool_choice`). 4 and 16
  concurrent conversations, n-gram speculation (identical replies) and client
  cancellation hold. Prompt processing past 16K tokens is slower than llama.cpp's
  (about 3,000 vs 5,400 tokens/s at 32K on an M5 Pro); decoding after a long prompt is
  on par.
- **CPU lanes, quality (2026-11-18, M5 Pro, after the CPU optimizations):** ggml_cpu and the
  managed CPU executor against llama.cpp's CPU backend (`d3c3e242c`, CPU-only builds) on the
  same GGUFs. CPU products round their activations to 8 bits, so two correct CPU
  implementations that differ only in summation order land 2-4% apart in the logits:
  llama.cpp's own CPU kernel variants do (median relative L2 1.7-3.8e-2, teacher-forced over
  320 rows). Teacher-forced through llama.cpp's histories (5 prompts of 14 to 9,294 tokens,
  64 steps each), ggml_cpu lands at 1.03-1.16x that floor and the managed CPU at 1.17-1.37x
  (median relative L2 against llama.cpp's no-llamafile, no-repack kernel variant): 1.8 and
  2.0-2.1e-2 on Q8_0 and QAD-Q4_0 (variant 1.7e-2), 4.4e-2 and 4.7e-2 on Q4_K_M (variant
  3.8e-2), 1.9e-2 and 2.4e-2 on NVFP4 (variant 1.8e-2, which flips 4 of ggml_cpu's 5 rows
  itself). Each lane has 1-5 (ggml_cpu) and 0-9 (managed) top-1 differences in 320 rows per
  file, each at a near-tie (llama.cpp's own top two logits within 0.5 of each other) except
  two Q4_K_M rows that llama.cpp's own kernel variant also flips or lands 18% away on. Both
  lanes are as far from llama.cpp's Metal run as its own CPU run is (within 7% on the
  quantized files). ForcedLogitProbe's pass gate (relative L2 1e-3, the GPU lanes' bar)
  fails on every CPU run, llama.cpp's variant included. BF16 no longer rounds activations on
  either lane: 4e-3 from llama.cpp's CPU run and 4e-4 from its Metal run, ten times closer
  to Metal than llama.cpp's own CPU run. Perplexity on Q4_K_M (32 x 1,024 tokens) is 26.79
  on ggml_cpu and 26.75 on the managed CPU against llama.cpp CPU's 26.79. The 20 server
  scenarios pass on both lanes, as do the 8 CPU speech-synthesis tests (engine speech equal
  bit for bit to standalone synthesis). GSM8K (first 200 questions, thinking off, Q4_K_M)
  scores 79.0% on ggml_cpu and 80.0% on the managed CPU against llama.cpp CPU's 81.0%
  (paired McNemar p = 0.50 and 0.82; on Metal both engines score 81.5%). The run first
  stopped at 50 questions, where ggml_cpu trailed 37 to 43 (p = 0.07; 0.03 against an
  earlier llama.cpp CPU run). Scored on the content alone (what a client reads with thinking
  off: TensorSharp moves text before a stray `</think>` into `reasoning_content`,
  llama-server leaves it in the content), both lanes score 76.5% against 81.0% (p = 0.09).
  Greedy CPU replies part from llama.cpp's CPU replies at near-ties within a few hundred
  characters, as llama.cpp's own CPU and Metal replies part from each other. Four concurrent
  conversations equal their one-at-a-time runs token for token on the managed CPU; on
  ggml_cpu they part at a near-tie, as on llama.cpp, because a batched decode step reads the
  repacked weights through ggml-cpu's GEMM and a single one through its GEMV
  (`TS_GGML_CPU_REPACK=0` restores the equality at half the prompt speed). Checked before
  the optimizations and not repeated: needles at 8K and 32K, the CLI checks, LFM3-Omni image
  and audio input (prompt token counts equal to llama-server's) and speech output that
  transcribes back at word error rate 0.
- **CPU lanes, speed (2026-11-18, M5 Pro with 6 performance and 12 efficiency cores,
  `eng/lfm3/bench/run-matrix.sh cpu cpu-serve cpu-omni`):** TensorSharp at its defaults
  against llama.cpp at its best (the faster of its Accelerate BLAS and ggml-cpu-only builds,
  prompts on 18 threads, decode on 16: 12 is 4-11% slower at depth 0 and 4K and 21% on BF16
  and at 16K, 18 is 6% slower), flash attention on, 3 interleaved rounds on a quiet machine
  (serving: 2). Decode is the median of 5 runs on both sides after an untimed run
  (re-measured after review: the first matrix set the faster of TensorSharp's two runs
  against the mean of llama.cpp's two, which hid the managed lane's JIT warm-up run). The
  table is under [Benchmarking](#benchmarking). *ggml_cpu* decodes faster than llama.cpp on
  every file (1.03-1.16x at depth 0 and 4K, 1.60x at 16K, 3.9x on BF16, whose ggml-cpu dot
  is scalar on ARM), runs prompts of 512 tokens and more at 0.93-1.00x (NVFP4 1.11-1.20x)
  and 128-token prompts at 0.91-0.97x; the same cell moves by up to 5% between sessions
  (Q4_K_M pp512: 0.93-0.98x over six sessions). It decodes one served stream as fast as
  llama-server (6.4-6.5 vs 6.5-6.6 ms a token) with a 20-75 ms longer time to first token
  (0.60-0.65 vs 0.53-0.60 s for 512 tokens), serves four streams at 0.97-1.08x its
  aggregate, encodes a 512 x 512 image in about 0.6x llama.cpp's time (0.58x warm, 0.66x for
  a process's first encode) and audio in 0.48-0.50x (0.50-0.63x with the mel front end),
  both engines on 18 threads, and speaks at 6.8-7.0x real time against 4.2x. The *managed
  CPU* runs NVFP4 prompts at 1.4-2.0x llama.cpp's speed and BF16 decode at 3.5x, Q4_K_M
  prompts of 512 tokens and more at 0.92-0.95x, Q8_0 and Q4_0 ones at 0.77-0.84x, these
  three files' 128-token prompts at 0.54-0.70x (0.70-0.88x once the JIT has tiered its
  kernels up), quantized decode at 0.70-0.86x (0.94x at 16K) and BF16 prompts at 0.44x of
  llama.cpp's Accelerate build. Against vLLM 0.34's CPU backend (fp16 with its macOS
  short-conv workaround; it has no quantized CPU path here), 512-token prompts: ggml_cpu on
  Q4_K_M answers in 0.64 s against 1.9 s and decodes at 6.2 against 37.9 ms a token (that
  harness's TPOT; llama-server 6.6 ms in it); on BF16 1.1 s and 17.2 ms. The knobs are in
  the [environment variable
  matrix](../env_var_feature_matrix.md#out-of-matrix-ggml-cpu-backend-knobs).
- **Not yet validated:** the TensorAgent app on a device, and every CUDA and Vulkan route
  (`config/lfm3-1.7b-nvfp4.json` in particular has never run). vLLM was compared on the
  CPU only (above); SGLang not at all (its CPU backend is x86 AMX only).

## Run it

```bash
echo "What is 17*23?" > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/lfm3-1.7b.json --think --input prompt.txt
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/lfm3-omni-2b.json
```

The configs download the pinned files on first use into
`${TENSORSHARP_MODELS:-../models}/lfm3/`.

| File | Bytes | SHA-256 | Repository @ revision |
|---|---|---|---|
| `LFM3-1.7B-Q4_K_M.gguf` | 1,099,627,808 | `3793e1e6…b05c93c4` | `LiquidAI/LFM3-1.7B-GGUF@cd0c74c5` |
| `LFM3-1.7B-NVFP4.gguf` | 1,280,019,776 | `462c2379…59a67bec` | `LiquidAI/LFM3-1.7B-GGUF@cd0c74c5` |
| `LFM3-Omni-2B-Q4_K_M.gguf` | 1,099,630,080 | `9f8d30e9…cc9893e2` | `LiquidAI/LFM3-Omni-2B-GGUF@d508b88f` |
| `mmproj-LFM3-Omni-2B-Q8_0.gguf` | 506,842,528 | `8a25f9d1…9c6b27b1` | `LiquidAI/LFM3-Omni-2B-GGUF@d508b88f` |

Full hashes are in the configs and equal the files' LFS object ids at those revisions. The repositories also publish Q5_K_M, Q6_K, Q8_0,
BF16, F16 and a quantization-aware `QAD-Q4_0` (1.7B), Q4_0/Q8_0/BF16/F16 (Omni) and
mmproj BF16/F16/Q4_0/Q4_K_M; they carry the same metadata and load the same way.

## Architecture

24 layers with a 2,048-wide residual stream and a dense SwiGLU FFN (7,680) in every
layer. `lfm3.attention.head_count_kv` is a per-layer array: the 13 layers with 0 KV
heads are **short-convolution** layers (`lfm3.shortconv.l_cache = 7`: a gated causal
depthwise convolution with a 6-column F32 state per sequence), and the 11 others are
GQA attention (16 query heads, 4 KV heads of 128, per-head Q/K RMSNorm, NeoX RoPE, a
sigmoid output gate). Of those, `sliding_window_pattern` marks layers 4, 8, 12, 16
and 20 as sliding-window (window 256, RoPE base 10,000); layers 2, 6, 10, 14, 18 and 22
attend to the whole context (RoPE base 1,000,000). The GGUF declares a 262,144-token
window (1.7B) and 128,000 (Omni); `MAX_CONTEXT` bounds it.

K/V per token at F16: 6 full layers x 4 heads x 128 x 2 x 2 = 12,288 bytes, as
llama.cpp's iSWA cache. The five sliding-window layers keep rings of 768 cells each
(the 256-key window plus a 512-row prompt step, aligned to 256: 7.5 MB a sequence),
so prompts run in steps of at most 512 rows a sequence once a ring would wrap; with
linear caches on all eleven layers it would be 22,528 bytes a token. The short-conv
state is fixed per sequence and cannot be rewound, so a prefix is reused exactly or
not at all (the recurrent-state rule the Qwen 3.5 family follows).

## Chat template

The GGUF template renders as shipped; TensorSharp's Jinja engine treats
`{% generation %}` as transparent and renders `bos_token` empty, because the tokenizer
adds the one `<|startoftext|>` the template asks for (`add_bos_token` is false in the
GGUF). `RenderLfm3` reproduces the same text for a GGUF without a template.

```text
<|startoftext|><|im_start|>system
{system}

<tools>
{"name": "get_weather", "description": "...", "parameters": {...}}
</tools><|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
<think>
```

### Reasoning

- The generation prompt ends `<think>` (thinking on) or `<think></think>` (off), with no
  newline. The model writes its reasoning, `</think>`, then the answer.
- **Default:** the 1.7B template defaults `enable_thinking` to true and the Omni template
  to false, but TensorSharp's request switch defaults to off on every host, as for
  Qwen 3.5. Send `"think": true` or `chat_template_kwargs.enable_thinking: true` (CLI
  `--think`) for the 1.7B card's default, and pin it in any comparison with llama.cpp,
  vLLM or SGLang, whose LFM3 default is on.
- Earlier turns are re-rendered as `<think></think>` + answer, reasoning dropped (the
  template's `preserve_thinking` defaults off; llama-server turns it on by default with
  `--reasoning-preserve`, so its multi-turn prompts differ from HF's and TensorSharp's
  when the client sends `reasoning_content`). TensorSharp's own conversations splice a
  past turn's generated tokens back instead, so the next turn extends the KV cache
  exactly. Reasoning a client sends for a tool round of the turn in progress is kept, as
  the template does.
- `</think>` (124902) is one token: the thinking budget closes the block there and the
  answer follows, and a `response_format` grammar arms after it.
- With thinking off the 1.7B often reasons anyway: it writes reasoning (sometimes after
  a `<think>` of its own) and closes the block itself before it answers (52 of 200
  greedy GSM8K replies and 5 of 33 instruction prompts on the Q4_K_M; llama.cpp's
  replies are the same tokens). The model card warns that thinking off "can be
  inconsistent". How TensorSharp returns such a reply depends on whether the client
  can take text back:
  - A non-streaming reply (OpenAI, Ollama, Responses) and a stream that can take text
    back (the Web UI, TensorAgent): a `</think>` within 4,096 characters makes the text
    before it reasoning, returned as `reasoning_content` (OpenAI) or `thinking` (Ollama,
    Web UI); the Web UI shows the text at once and takes the reasoning back when the
    close arrives.
  - A stream that cannot take text back (the OpenAI, Ollama and Responses streams, the
    CLI's interactive mode): the reply streams as content as it is generated, a
    `</think>` and the text around it included, as llama.cpp, vLLM and SGLang return it.
    Nothing is held, so a short answer arrives token by token; such a stream and the
    non-streaming reply to the same request can therefore differ in where the reasoning
    is. A reply that opens with a `<think>` of its own streams that block as reasoning.
  A block the model opens itself is capped like Gemma 4's unrequested channel (closed at
  a line break, after at most 64 tokens).
- LFM3-Omni does not reason: with thinking on it closes the block at once.

### Tool calls

The model writes Qwen3-Coder XML; parallel calls are separated by a blank line:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

The `Qwen35OutputParser` reads it, as llama.cpp's `qwen3-coder` parser does for this
template: reasoning ends at `</think>` or at a `<tool_call>` written while still
reasoning; declared string parameters keep their text, other declared types are
parsed as JSON. Results render as `<|im_start|>tool\n<tool_response>\n…\n</tool_response>`,
from the tool message alone, so a tool round's generated tokens are spliced back and
the next round continues the live cache. `tool_choice: required` and a named function
are enforced by a grammar (`QwenXmlToolGrammar`): the reply is well-formed calls to the
declared functions (only the named one), required parameters present, values of the
declared types and enums; it arms after `</think>` when the request reasons. `auto`
and `none` stay unconstrained (`none` leaves the tools out of the prompt). llama-server
does not enforce a named choice for this template.

Tool declarations go through TensorSharp's shared tool builder, which keeps each
parameter's `type`, `description` and `enum` but not nested schemas (`items`, nested
`properties`, defaults): a family-wide limit, where llama.cpp and vLLM print the
client's schema as sent.

### Media placeholders

The placeholder contract for LFM3-Omni, before the multimodal expander runs (one
message's media always precede its text):

| Input | Prompt text | The expander replaces |
|---|---|---|
| each image | `<\|vision_start\|><\|image\|><\|vision_end\|>` | `<\|image\|>` (125030) with the encoder's N rows |
| one video (its sampled frames) | `<\|vision_start\|><\|video\|><\|vision_end\|>` | `<\|video\|>` (125029) with, per pair of frames, `"<t.t seconds>"` + `<\|vision_start\|>` + rows + `<\|vision_end\|>` (+ that pair's soundtrack rows) |
| each audio clip | `<\|vision_start\|><\|audio\|><\|vision_end\|>` | `<\|audio\|>` (125031) with the encoder's rows |

These are the templates' own spans for `image`, `video` and `audio` content parts, and
the final layout is llama.cpp's mtmd one (which wraps Omni audio in the same vision
markers) and HF's (`<|image|>` x N inside the pair). One rule decides a message's
items, for the renderer and the expander alike
([`Lfm3MediaLayout`](../../TensorSharp.Runtime/Lfm3MediaLayout.cs)): stills and videos in
attachment order, then audio clips. A video named by its file
(`ChatMessage.VideoFilePaths`) is read from the file; frames a host sampled are a
video only when no file is named.

### Stop tokens

`<|im_end|>` (the GGUF's EOS) and `<|endoftext|>`, as llama.cpp's vocabulary. Omni's
`generation_config.json` also lists `<|text_end|>` (125014); llama.cpp uses it only to
end speech and interleaved generation, and in twelve greedy text chats with
LFM3-Omni-2B-Q4_K_M it never appeared, so text chat does not stop on it.

## Media input

LFM3-Omni takes images, video (with its soundtrack) and audio once its projector is
loaded (`--mmproj mmproj-LFM3-Omni-2B-Q8_0.gguf`; the configs name it). The one mmproj
holds the vision tower (`lfm3vl`) and the audio tower (`lfm3a`), both loaded from it.

| Surface | Image | Audio | Video |
|---|---|---|---|
| CLI | `--image` (repeatable) | `--audio` | `--video clip.mp4` |
| `/v1/chat/completions` | `image_url` (data URI) | `input_audio` / `audio_url` | `video_url` (data URI) |
| `/v1/responses` | `input_image` | `input_audio` | `input_video` `{url}` |
| `/api/chat` (Web UI, TensorAgent) | `imagePaths` | `audioPaths` | `videoFilePaths` (+ `isVideo`, `imagePaths` frames) |

- **Video is read from the file.** The model samples the clip itself as llama.cpp and
  HF do: 2 frames a second at `round(linspace)` indices, clamped to [4, 384], one size
  for all frames, consecutive frames merged in pairs labelled `<t.t seconds>`, and the
  soundtrack (desktop: ffmpeg on `PATH` or `TS_FFMPEG`; Apple: AVFoundation) encoded once
  and split after each pair. `video_url` / `input_video` therefore sample no frames on
  the host for this family (`ChatProtocol.ReadsVideoFiles`), and their `fps` /
  `max_frames` do not apply. A clip without an audio stream is its frames alone; a
  soundtrack that cannot be decoded is warned about and the frames are used without it.
- **Frames sampled elsewhere** (a client that sends only frames) are paired by their
  times: frames at most 0.575 s apart share a patch labelled with their mean time,
  sparser ones fill a patch each; untimed frames are taken at `1 / VIDEO_SAMPLE_FPS`.
  That is not the model's own sampling and has no soundtrack.
- **Differences from llama.cpp on video.** TensorSharp follows HF: exact source frames
  (stepping, not seeking), torchvision's bicubic resize and the decoder's average frame
  rate. llama.cpp resizes with swscale, reads ffmpeg's constant-rate output and clocks
  with the nominal rate, so on clips whose nominal and average rates differ a label can
  read one tenth of a second apart and the soundtrack can take one row more or fewer
  (`test-3.mp4`: 3,221 vs 3,220 prompt tokens).
- **Prefix reuse follows media identity.** Each image, clip, frame pair and soundtrack
  piece is a span named by its content (bytes, not file name): a follow-up question
  about the same image reuses the cached prompt through it, and a different image stops
  reuse at its span (`Lfm3OmniModelTests`).
- **Concurrency.** Decoding files, resizing frames and computing audio features run
  with the model's GPU lock released; a video yields it between frame pairs, audio
  longer than 30 s between chunks, and an image of more than 4,096 patches (above 1,024
  rows) runs as two programs with a yield between them.
- **Refusals.** LFM3-1.7B, or Omni without its projector, refuses image, video and
  audio input with a message naming the projector, before any upload is written where
  the request surface allows; the CLI exits 1 with one line. The Omni projector is
  refused on LFM3-1.7B (a model-load refusal, exit 2), whether named with `--mmproj` or
  found beside the model: LiquidAI publishes the 1.7B as a text model.

## Sampling

| Model | Card / GGUF `general.sampling` | Configs |
|---|---|---|
| LFM3-1.7B | temperature 0.1, top_k 50, repetition_penalty 1.1 (math/STEM: temperature 1.0, top_p 0.95, top_k 20) | the card's general preset, top-p 1, min-p 0, `repeat-last-n -1` |
| LFM3-Omni-2B | greedy, repetition_penalty 1.0 | temperature 0, top-k 1, repeat-penalty 1 |

The server ignores `general.sampling`, so the configs pin these values (the CLI reads
it for chat). TensorSharp's repetition penalty looks back over the last
`repeat_last_n` **generated** tokens; llama-server also counts the prompt's last
`repeat_last_n` tokens, vLLM and HF the whole prompt and output, and SGLang the whole
output. `repeat-last-n -1` matches SGLang; compare with other engines at repeat penalty
1.0 and greedy decoding.

## Speech output

LFM3-Omni's mmproj carries the speech generator (`a.gen.code.*`: a 3-layer bidirectional
depthformer that turns the LM's final hidden row into a frame of 12 codes in 3 MaskGIT
passes) and the LiquidCodec decoder (`a.gen.wav.*`: frames to 24 kHz PCM through a host
ISTFT). Loading the mmproj (`--mmproj`) turns speech on; `TensorSharp.Models.Lfm3.Lfm3SpeechOutput`
is the component, bit-exact with llama.cpp's on ggml-metal.

**How a spoken reply runs.** One engine request whose slots a driver chooses
(`Lfm3SpeechDriver`, llama.cpp's `step_gen` schedule): a text slot samples a token from the
logits; a speech slot samples a frame from the previous position's final hidden row, queues
its 80 ms of PCM, and decodes the frame's feedback row (the sum of its 12 codebook
embeddings) as the next input row. The runtime's slot steps take an embedding row in place
of a token and return the last row's final hidden row, and a step no request wants logits
from skips the 128000 x 2048 LM head, so spoken and text requests batch in the same steps
on every executor. Modes: `audio` (`<|audio_start|>`), `interleaved` (`<|mixed_start|>`: 6 text
tokens, then 12 frames, ..., until `<|text_end|>`; the text is the speech's transcript), and
TTS (the GGUF's `tokenizer.chat_template.tts`, ending at the end-of-speech frame). A
conversation's reply goes on in text after its speech, as the model card's `generate` and
llama.cpp's realtime server do; a TTS reply stops at the end of its speech, as llama-tts does.

Where to use it: CLI `--tts` / `--speech-out` (with `--speech-mode`, `--speech-temperature`,
`--voice`, `--language`, `--seed`), `POST /v1/audio/speech`, and `/v1/chat/completions` /
`/v1/responses` with `modalities` `["text","audio"]` or `["audio"]` — see
[USAGE.md → Speech output](../../USAGE.md#speech-output-lfm3-omni). Voices are `female`
and `male` (named in the system prompt; there is no speaker embedding); an unknown voice,
OpenAI's default `alloy` included, is a 400 that lists them. A spoken chat reply's system
prompt gets llama.cpp's realtime line ("Speak in English using a female voice.") and, for
interleaved replies, the card's "Respond with interleaved text and audio.".

Text replies of LFM3-Omni never draw the media placeholders (`<|audio|>`, `<|audio_out|>`,
`<|image|>`, `<|video|>`), and a text-only reply never opens speech (`<|audio_start|>`,
`<|mixed_start|>` banned), as HF's `generate` does; llama.cpp suppresses neither in plain
chat. (A sampler with suppressed tokens is not a plain argmax, so direct CUDA's on-device
greedy pick is not used for LFM3-Omni's text replies.)

**Validation (2026-11-17, M5 Pro, ggml-metal, Q8_0 LM + Q8_0 mmproj, llama.cpp `d3c3e242c`):**

- The reply driver on llama.cpp's own hidden rows of a llama-tts run reproduces llama-tts's
  WAV to 1 LSB (20 of 130,560 samples differ, end of speech at the same frame 68).
- End to end, TensorSharp's TTS with llama-tts's seed draws the same codes as llama-tts for
  the first 2-7 frames, then parts: the LM's hidden rows sit 2-7e-4 (relative L2) from
  llama.cpp's and a frame's sample flips. At speech temperature 0 (argmax), 2 of 5 texts
  match llama-tts within 1 LSB for the whole utterance (72 and 86 frames).
- Intelligibility (LFM3-Omni's own ASR through llama.cpp, `eng/lfm3/speech-asr.py`): over 5
  texts (English, British English, French, Chinese) x 5 seeds, TensorSharp 0.00% WER/CER (25
  of 25 exact) against llama-tts 1.38% (22 of 25).
- Interleaved replies: the ASR of each English reply's speech equals its transcript (6 of 6,
  0% WER), as for llama.cpp's realtime path driven the same way (`eng/lfm3/speech-chat-ref.cpp`).
  Chinese interleaved speech is not intelligible in either engine (a model limit; Chinese TTS
  is).
- Speech-only replies (`modalities ["audio"]`, which opens `<|audio_start|>` as llama.cpp's
  realtime AUDIO mode does): LFM3-Omni reads the user's message aloud instead of answering it,
  in TensorSharp and llama.cpp alike (8 of 8 questions, ASR = the question), and without a
  voice line its speech is unintelligible. For spoken answers ask for `["text","audio"]`.
- A text request served while a TTS reply speaks keeps its greedy reply; the engine's TTS
  equals the standalone synthesis bit for bit.
- Speed (streaming `/v1/audio/speech`, one client against TensorSharp and llama-server, requests
  interleaved, on a shared and loaded Mac): at parity on Metal - time to first audio 88-114 ms
  against 119-141 ms, real-time factor 1.83x against 1.38-1.92x at load 36-95 (a quiet machine is
  3-5x faster for both). Audio-only steps skip the LM head (-15% of the step). MLX runs at about
  half Metal's speed (its fixed cost per graph run).

History: a spoken assistant turn sent back in a later request (OpenAI's
`{"role":"assistant","audio":{"id":...}}`, kept for an hour) is spliced back as generation decoded
it: the token that opened the mode, the text tokens and one `<|audio_out|>` per decoded frame, each
given that frame's codebook-sum row (the end-of-speech frame included), as HF's processor gives a
past speech turn's placeholders. Its prompt tokens are therefore the reply's own sequence (checked
token for token against generation on LFM3-Omni Q8_0). Each run of frames is one media span keyed
by its codes. An id whose generation is not known renders as its transcript.

Limits: no `/v1/realtime`; speech requests are not retained in the prefix cache, and a later turn
does not yet reuse the cache across a spliced spoken turn (open issue); speech and CUDA are untested
on NVIDIA GPUs.

## Benchmarking

`ignore_eos: true` on `/v1/chat/completions` (and `/v1/responses`) generates exactly
`max_tokens` through end-of-sequence tokens, as vLLM's and SGLang's benchmark clients
expect (`vllm bench serve --ignore-eos`, SGLang's default). Pin `think` and the
sampling on every engine.

CPU lanes on an M5 Pro (6 performance + 12 efficiency cores, 2026-11-18, LFM3-1.7B; tokens
per second, median of 3 interleaved rounds; `eng/lfm3/bench/run-matrix.sh cpu`). TensorSharp
runs at its defaults (prompts on 18 threads; decode on 16 for ggml_cpu, 18 for the managed
CPU). llama.cpp (`d3c3e242c`) is at its best: the faster of its Accelerate BLAS and
ggml-cpu-only builds, prompts on 18 threads, decode on 16 (its fastest count). ppN is an
N-token prompt, tg a 128-token decode after 0 or 4,096 tokens: the median of 5 runs on both
sides after an untimed run, re-measured after review in 3 more rounds (the machine 0-3% busy
before every run). A prompt cell moves by up to 5% between sessions: Q4_K_M pp512 measured
0.93-0.98x llama.cpp's over six sessions.

| file | engine | pp128 | pp512 | pp2048 | pp8192 | tg | tg at 4K |
|---|---|---|---|---|---|---|---|
| Q4_K_M | TS ggml_cpu | 953 | 960 | 885 | 759 | **171** | **148** |
| | TS cpu (managed) | 575 | 898 | 849 | 712 | 128 | 107 |
| | llama.cpp | 1001 | 978 | 919 | 759 | 163 | 139 |
| Q8_0 | TS ggml_cpu | 1178 | 1170 | 1160 | 917 | **118** | **107** |
| | TS cpu (managed) | 880 | 1014 | 934 | 776 | 92 | 81 |
| | llama.cpp | 1255 | 1249 | 1161 | 923 | 115 | 103 |
| QAD-Q4_0 | TS ggml_cpu | 1223 | 1222 | 1158 | 933 | **190** | **164** |
| | TS cpu (managed) | 699 | 994 | 939 | 783 | 131 | 107 |
| | llama.cpp | 1292 | 1288 | 1190 | 938 | 182 | 153 |
| NVFP4 | TS ggml_cpu | 380 | 628 | 645 | 561 | **152** | **135** |
| | TS cpu (managed) | **777** | **820** | **791** | **665** | 114 | 94 |
| | llama.cpp | 399 | 566 | 537 | 488 | 133 | 116 |
| BF16 | TS ggml_cpu | 267 | 487 | 491 | - | **60** | - |
| | TS cpu (managed) | 224 | 228 | 223 | - | 55 | - |
| | llama.cpp | 274 | 520 | 508 | - | 16 | - |

At 16K tokens of context Q4_K_M decodes at 115 (ggml_cpu), 67 (managed) and 71 (llama.cpp)
tokens/s (one run after an untimed one, 3 rounds). The managed CPU's short prompts keep
speeding up over their first few runs while the .NET JIT tiers its kernels up; at steady
state (10 runs) its pp128 is 838 / 935 / 892 on Q4_K_M / Q8_0 / QAD-Q4_0 against llama.cpp's
952 / 1238 / 1270. Before the CPU work (`feature/lfm3` 052db7ed, measured in the same
rounds) ggml_cpu ran pp512 at 498 / 605 / 508 / 361 on Q4_K_M / Q8_0 / QAD-Q4_0 / NVFP4
(BF16: 22, the CPU baseline's figure) and the managed CPU at 20 / 135 / 25 / 28 / 143 on the
five files.

## Current limits

- Speech output has no realtime endpoint (`/v1/realtime`), and a later turn does not yet reuse
  the prefix cache across a past spoken turn (see [Speech output](#speech-output)).
- Media input and speech output on CUDA (direct and ggml-cuda) and Vulkan have not run (no device).
- Speech input as a live stream (a realtime endpoint) is not served; clips are encoded
  whole (30 s chunks).
- A message's still images come before its videos in the prompt whatever the client's
  order; a video sent as `video_url` has no position among them.
- Nested tool schemas are flattened in the prompt (the tool-call grammar reads the
  client's schema).
- The QAD-Q4_0 file is not in TensorAgent's catalog pending a quality and speed
  measurement against Q4_K_M.
- CPU lanes against llama.cpp's CPU builds on an M5 Pro (see [Benchmarking](#benchmarking)):
  ggml_cpu's prompts are 0-9% behind on the repacked quants, most at 128-512 tokens (the
  products are at parity; each 512-row step carries ~25-30 ms of prompt K/V copies llama.cpp
  writes in place), and its NVFP4 prompts under 256 tokens still run ggml's own NVFP4
  product, half the speed of the managed CPU's NEON GEMM. The managed CPU decodes quantized
  files at 0.70-0.86x (its pool waits on the slowest of 6 + 12 unequal cores for small
  products, and the graph interpreter costs a few percent a step), runs Q8_0 / Q4_0 prompts
  of 512 tokens and more at 0.77-0.84x (llama.cpp's repacked GEMMs use i8mm, which .NET 10
  does not expose) and 128-token ones at 0.54-0.70x (0.70-0.88x once the JIT has tiered its
  kernels up), BF16 prompts at 0.44x of llama.cpp's Accelerate build, and its audio encoder
  takes 1.1-2x llama.cpp's time. Served through TensorSharp.Server.Host, the managed CPU
  decodes at ~78 tokens/s against its 128 in the engine benchmark (ggml_cpu: 153 against
  171): its worker pool takes every core, so the server's own threads stall it at every
  join. `TS_CPU_THREADS=16` serves 94 tokens/s; a decode width of its own, as ggml_cpu has,
  is the open fix.
- ggml_cpu keeps its own load-time host copies (the fused gate|up and permuted attention
  weights) resident beside their repacked copies: peak footprint ~2.0 GB for Q4_K_M against
  llama-server's 1.2 GB. `TS_GGML_CPU_REPACK=0` restores the smaller footprint at half the
  prompt speed. With repacking, a batched decode step reads the weights through ggml-cpu's
  GEMM and a single one through its GEMV, so concurrent greedy replies can part from
  one-at-a-time ones at a near-tie (as on llama.cpp); the managed CPU keeps them equal.
  Repacking and the decode pool were measured only on an M5 Pro; the BF16/NVFP4 Accelerate
  paths are Apple-only.
