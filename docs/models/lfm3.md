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
- **Not yet validated:** the Omni vision and audio towers, the TensorAgent app on a
  device, and every CUDA route (no NVIDIA GPU was available;
  `config/lfm3-1.7b-nvfp4.json` in particular has never run). vLLM and SGLang were not
  compared.

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
- With thinking off the 1.7B often reasons anyway: it writes reasoning (sometimes after
  a `<think>` of its own) and closes the block itself before it answers (52 of 200
  greedy GSM8K replies and 5 of 33 instruction prompts on the Q4_K_M; llama.cpp's
  replies are the same tokens). The model card warns that thinking off "can be
  inconsistent". TensorSharp parses such a reply as it does Nemotron-H
  Reasoning-128K's: a `</think>` within 4,096 characters makes the text before it
  reasoning, returned as `reasoning_content` (OpenAI) or `thinking` (Ollama, Web UI),
  streamed or not, and a block the model opens itself is capped like Gemma 4's
  unrequested channel (closed at a line break, after at most 64 tokens). llama.cpp,
  vLLM and SGLang return such a reply whole, `</think>` included, as the answer. The
  price: a stream that cannot take text back (OpenAI, Ollama, Responses, CLI) shows a
  thinking-off answer once 4,096 characters are generated or the reply ends, whichever
  comes first; the Web UI shows it at once and takes the reasoning back.
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

## Benchmarking

`ignore_eos: true` on `/v1/chat/completions` (and `/v1/responses`) generates exactly
`max_tokens` through end-of-sequence tokens, as vLLM's and SGLang's benchmark clients
expect (`vllm bench serve --ignore-eos`, SGLang's default). Pin `think` and the
sampling on every engine.

## Current limits

- Speech output (the projector's `lfm3a_gen` generator and codec) is not served: no
  chat audio-out, `/v1/audio/speech` or realtime endpoint for this family yet.
- A video's soundtrack is not extracted (the CLI and Web UI pass frames only).
- Audio sent to LFM3-1.7B, or to Omni without its projector, is not refused yet; the
  multimodal integration has to refuse it rather than drop it.
- Nested tool schemas are flattened in the prompt (the tool-call grammar reads the
  client's schema).
- TensorAgent's catalog counts 22,528 K/V bytes per token until the sliding-window
  ring lands; the QAD-Q4_0 file is not in the catalog pending a quality and speed
  measurement against Q4_K_M.
