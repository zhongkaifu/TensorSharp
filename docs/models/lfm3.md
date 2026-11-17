# Liquid LFM3 and LFM3-Omni (`lfm3`)

[← back to model index](README.md) | [中文](lfm3_zh-cn.md)

Liquid AI's LFM3 checkpoints declare `general.architecture = lfm3`: **LFM3-1.7B**, a text
model with reasoning and tool calls, and **LFM3-Omni-2B**, the same decoder with one
multimodal projector that takes images, video and audio in (and carries a speech
generator, which TensorSharp does not serve yet). Both share one chat protocol; the
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
- **Not yet validated:** generation itself on any backend, the TensorAgent app on a
  device, and every CUDA route (no NVIDIA GPU was
  available; `config/lfm3-1.7b-nvfp4.json` in particular has never run). Speed
  against llama.cpp, vLLM and SGLang has not been measured.

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

K/V per token at F16: 11 layers x 4 heads x 128 x 2 x 2 = 22,528 bytes while every
attention layer keeps a linear cache, 12,288 bytes once the five sliding-window
layers keep a fixed 256-row ring (llama.cpp's iSWA cache). The short-conv state is
fixed per sequence and cannot be rewound, so a prefix is reused exactly or not at all
(the recurrent-state rule the Qwen 3.5 family follows).

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
the next round continues the live cache. `tool_choice: required` is accepted but not
grammar-enforced, as for Qwen 3.5.

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

## Benchmarking

`ignore_eos: true` on `/v1/chat/completions` (and `/v1/responses`) generates exactly
`max_tokens` through end-of-sequence tokens, as vLLM's and SGLang's benchmark clients
expect (`vllm bench serve --ignore-eos`, SGLang's default). Pin `think` and the
sampling on every engine.

## Current limits

- Speech output (the projector's `lfm3a_gen` generator and codec) is not served: no
  chat audio-out, `/v1/audio/speech` or realtime endpoint for this family yet.
- Media input on CUDA (direct and ggml-cuda) and Vulkan has not run (no device).
- Speech input as a live stream (a realtime endpoint) is not served; clips are encoded
  whole (30 s chunks).
- A message's still images come before its videos in the prompt whatever the client's
  order; a video sent as `video_url` has no position among them.
- `tool_choice: required` is not grammar-enforced; nested tool schemas are flattened.
- TensorAgent's catalog counts 22,528 K/V bytes per token until the sliding-window
  ring lands; the QAD-Q4_0 file is not in the catalog pending a quality and speed
  measurement against Q4_K_M.
