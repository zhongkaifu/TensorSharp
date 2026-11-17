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
- **Not yet validated:** generation itself on any backend, the Omni vision and audio
  towers, the TensorAgent app on a device, and every CUDA route (no NVIDIA GPU was
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
markers) and HF's (`<|image|>` x N inside the pair). OpenAI `video_url` parts are
sampled into timed frames and capped (`CapsVideoFrames`); llama.cpp samples 2 fps
clamped to [4, 384] frames for Omni.

### Stop tokens

`<|im_end|>` (the GGUF's EOS) and `<|endoftext|>`, as llama.cpp's vocabulary. Omni's
`generation_config.json` also lists `<|text_end|>` (125014); llama.cpp uses it only to
end speech and interleaved generation, and in twelve greedy text chats with
LFM3-Omni-2B-Q4_K_M it never appeared, so text chat does not stop on it.

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

Limits: no `/v1/realtime`; a spoken assistant turn in later history renders as its
transcript (OpenAI's `audio.id` maps to it for an hour), not as its audio frames; speech
requests are not retained in the prefix cache; speech and CUDA are untested on NVIDIA GPUs.

## Benchmarking

`ignore_eos: true` on `/v1/chat/completions` (and `/v1/responses`) generates exactly
`max_tokens` through end-of-sequence tokens, as vLLM's and SGLang's benchmark clients
expect (`vllm bench serve --ignore-eos`, SGLang's default). Pin `think` and the
sampling on every engine.

## Current limits

- Speech output has no realtime endpoint (`/v1/realtime`), and past spoken turns re-render
  as their transcripts (see [Speech output](#speech-output)).
- A video's soundtrack is not extracted (the CLI and Web UI pass frames only).
- Audio sent to LFM3-1.7B, or to Omni without its projector, is not refused yet; the
  multimodal integration has to refuse it rather than drop it.
- `tool_choice: required` is not grammar-enforced; nested tool schemas are flattened.
- TensorAgent's catalog counts 22,528 K/V bytes per token until the sliding-window
  ring lands; the QAD-Q4_0 file is not in the catalog pending a quality and speed
  measurement against Q4_K_M.
