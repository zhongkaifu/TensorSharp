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
| Modalities | 1.7B: text. Omni: image, video and audio in (with the projector); text and speech out ([Speech output](#speech-output)) |
| Thinking | 1.7B: yes (`<think>`; see [the default](#reasoning)). Omni: no |
| Tool calling | Yes: Qwen3-Coder XML (`<tool_call><function=…><parameter=…>`) |
| Ready-made configs | [`config/lfm3-1.7b.json`](../../config/lfm3-1.7b.json), [`config/lfm3-1.7b-nvfp4.json`](../../config/lfm3-1.7b-nvfp4.json), [`config/lfm3-omni-2b.json`](../../config/lfm3-omni-2b.json) |
| TensorAgent | `lfm3-1.7b-q4km`, `lfm3-omni-2b-q4km` (experimental, 12 GB tier) |
| License | LFM Open License v1.0 |

## Validation status

Where LFM3-1.7B stands on 2026-11-18 (`feature/lfm3` after the think, worker-pool and second ggml_cpu
rounds were merged; llama.cpp `d3c3e242c` on the same GGUF and device; each figure is detailed in the
items below and under [Benchmarking](#benchmarking)):

| backend (device) | quality against llama.cpp | one sequence against llama.cpp | serving |
|---|---|---|---|
| ggml_metal (M5 Pro) | teacher-forced median relative L2 3-4e-4, no top-1 difference in 320 rows (Q4_K_M, Q8_0, BF16); GSM8K equal (81.5%) | prompts 0.97-1.03x, decode 1.01-1.02x | first token within 5% of llama-server's up to 2K tokens, 3-5% later at 16K-32K |
| MLX (M5 Pro) | Omni image and audio replies token-identical in 18 of 19 cases | prompts 1.12x at 512 tokens, decode 0.87x | - |
| ggml_cpu (M5 Pro) | at the CPU lanes' 8-bit-activation floor (1.03-1.16x llama.cpp's own kernel variants' distance); GSM8K 79.0% against 81.0% (p = 0.50) | decode 0.97-1.14x (BF16 3.9x), prompts 0.95-1.00x (NVFP4 1.9-2.6x) | one stream: first token 0.95x the time, decode 0.96x; four streams 1.17x llama-server's aggregate; against vLLM's CPU backend (fp16): first token 0.30x the time, decode 5.6x, four streams 3.5x (medians; 2 of the 5 one-stream rounds ran 8-10x slower, see [Current limits](#current-limits)) |
| cpu, managed (M5 Pro) | 1.17-1.37x that floor; GSM8K 80.0% (p = 0.82) | decode 0.78-0.95x (BF16 3.7x), prompts 0.86-0.92x (NVFP4 1.6-2.2x, BF16 0.44x) | one stream: decode 0.83x, first token at parity; four streams 1.06x |
| ggml_cuda (A40; RTX PRO 4500) | perplexity within 0.4% (A40) and 0.13% (sm_120; NVFP4's W4A4 prompts +0.4% against llama.cpp's W4A4) | A40 0.99-1.17x (NVFP4 prompts 0.89-0.91x); sm_120 0.92-1.01x (NVFP4 prompts 0.86-0.90x) | A40, BF16: faster than vLLM and SGLang at one request (6.5 against 7.3 / 7.2 ms a token), at parity at 4 (7.6 against 7.6), behind at 16 / 64 (11.0 / 26.6 against 8.1 / 11.0 ms); Q4_K_M ahead of llama-server at 1-16 requests |
| cuda, direct (A40; RTX PRO 4500) | as ggml_cuda (NVFP4's W4A4 prompts +0.7% on sm_120) | prompts 0.58-0.95x, decode 0.78-0.95x; QAD-Q4_0 decode 0.56-0.61x (A40) | A40, BF16: one request 7.0 ms a token (vLLM 7.3), behind from 4 requests (21.5 / 40.3 against 8.1 / 11.0 ms at 16 / 64) |
| ggml_vulkan (A40; RTX PRO 4500) | A40: 0-1 top-1 differences in 320 rows; sm_120: NVFP4 perplexity 26.51, as llama.cpp CUDA's q8 route | A40: 0.80-0.98x on the quantized files, BF16 0.92-1.01x, NVFP4 0.89-1.08x | not measured |

Details (llama.cpp reference `d3c3e242c`):

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
  cancellation hold. Long prompts prefill at llama.cpp's speed (next item; an earlier
  "3,000 vs 5,400 tokens/s at 32K" timed llama-server on a prompt whose first half the
  previous request had left in its cache); decoding after a long prompt is on par.
- **Speed (2026-11-17, M5 Pro, ggml_metal, `eng/lfm3/bench`, each run interleaved with
  llama.cpp `d3c3e242c` on the same GGUF, quiet machine):** LFM3-1.7B Q4_K_M prefills
  4,320 tokens/s at 128 prompt tokens, 6,043 at 512 and 5,971 at 2,048 against
  llama-bench's 4,471 / 5,962 / 5,883 (Q8_0: 4,446 / 6,298 / 6,210 against 4,544 / 6,195 /
  6,084), 4,952 at 16K and 4,129 at 32K against 4,932 / 4,126, and decodes 195 tokens/s
  at depth 0 against 193 (185 against 181 at 4K). Through the servers the first token
  (`max_tokens` 1) comes within 5% of llama-server's from 40 to 2K prompt tokens and
  3-5% later at 16K-32K. On a machine busy with other jobs TensorSharp's multi-step
  prompts lose 5-10% more than llama.cpp's. MLX prefills 512+ token prompts faster than
  ggml_metal (6,589 tokens/s at 512 on Q4_K_M) but decodes about 15% slower (165 tokens/s,
  a per-step host cost). The CPU lanes are the two CPU items below.
- **CUDA and Vulkan (2026-11-17, one NVIDIA A40, sm_86; llama.cpp `d3c3e242c`'s CUDA and Vulkan
  builds on the same GPU):** the text decoder, the Omni media input and the speech output run on
  ggml_cuda, the direct `cuda` backend and ggml_vulkan. Teacher-forced against llama.cpp (5 prompts
  x 64 steps, one of them 9,294 tokens): 0-1 of 320 top-1 mismatches on ggml_vulkan and up to 7 on
  the CUDA lanes, every one at a near-tie - as far from llama.cpp CUDA as llama.cpp's own Vulkan run
  is, because ggml-cuda and direct CUDA requantize each quantized product's activations to q8_1
  (BF16: 0-1 everywhere); perplexity within 0.4% of llama.cpp's for all five quantizations. The
  server's e2e scenarios, the CLI checks and 4 and 16 concurrent conversations pass on all three
  lanes; speech output round-trips through ASR with no word errors (it is not bit-exact to
  llama-tts on CUDA, whose sampler draws other codes from the same hidden rows).
- **Speed on the A40 (`eng/lfm3/bench/gpu-bench.sh`, each arm interleaved with llama-bench, which
  gets the better of flash attention on and off):** ggml_cuda matches or beats llama.cpp CUDA (-1.5% to
  +17%) at every length tested on Q4_K_M, Q8_0, QAD-Q4_0 and BF16. Q4_K_M: 17,004 / 18,283 / 17,132 /
  15,156 prompt tokens/s at 512 / 4K / 16K / 32K against 17,003 / 17,946 / 16,961 / 15,390, and
  347 / 337 / 309 / 280 decoded tokens/s at depth 0 / 4K / 16K / 32K against 334 / 327 / 299 / 268
  (Q8_0, QAD-Q4_0 and BF16 likewise ahead; NVFP4 prompts 9-11% behind, its decode 3-5% ahead).
  The direct `cuda` backend decodes 16-19% below llama.cpp on Q4_K_M (277 / 264 / 246 / 225) and
  prefills about 60% of its speed: its quantized prompt products dequantize every weight to F16
  each step where ggml-cuda runs them on the quantized blocks. ggml_vulkan against llama.cpp's Vulkan
  build: Q4_K_M prefills 10,119 / 10,430 / 9,258 / 7,749 tokens/s against 11,021 / 10,833 / 10,600 /
  9,633 and decodes 272 / 251 / 242 / 217 against 295 / 283 / 254 / 226 (4-20% behind, mostly host
  time between steps); BF16 and NVFP4 at parity.
- **Serving on the A40 (`eng/lfm3/bench/serve-vs.sh`: one client for every engine, random prompts, prefix
  caching off; BF16 against vLLM 0.34 and SGLang, whose short convolution takes its torch fallback on
  this GPU):** one request at a time, ggml_cuda decodes faster (6.7 ms a token against 7.3 and 7.2)
  and a 128-token prompt's first token comes sooner (26 ms against 56 and 43), but a 4K-token prompt's
  comes later (275 ms against 204 and 192). Under concurrency the server falls behind from 16 requests:
  7.6 / 11.0 / 26.6 ms a token at 4 / 16 / 64 requests against vLLM's 7.6 / 8.1 / 11.0, 42% of its
  output at 64 (2,314 against 5,495 tokens/s; 2026-11-19, two rounds, after the device's greedy ids, one
  upload of a step's inputs, the short convolutions' products over the step's rows and balanced joins: 9.2
  / 16.8 / 49.6 ms before), because a batched step still runs each sequence's attention and short convolution
  as ops of its own. Direct cuda: 9.1 / 21.5 / 40.3 ms (28.9 / 67.3 at 16 / 64 before its batched steps
  recorded past 16 sequences and its BF16 products of 5-16 rows reduced each output in one
  reduce-scatter). The server holds 4-6 GB of GPU memory (ggml_cuda) where
  vLLM and SGLang reserve about 40 GB. Against llama-server on Q4_K_M (CUDA build, as many slots as
  requests) both TensorSharp lanes decode faster at 1, 4 and 16 concurrent requests (ggml_cuda 3.0 /
  5.8 / 14.1 ms a token against 3.8 / 7.0 / 20.1).
- **CUDA and Vulkan on an RTX PRO 4500 (Blackwell sm_120, 32 GB, 2026-11-18; llama.cpp `d3c3e242c`'s CUDA
  build on the same GPU):** teacher-forced against llama.cpp CUDA (5 prompts x 64 steps), Q4_K_M, Q8_0 and BF16 land
  as on the A40: 0-3 top-1 differences in 320 rows, median relative L2 2.2e-3 to 2.4e-2; perplexity (32 x 1,024
  tokens) is within 0.13% of llama.cpp's on both CUDA lanes. NVFP4 takes its FP4 tensor-core route there: prompt
  products are W4A4 (the activations quantized to NVFP4 too: `ts_fp4_gemm_w4a4_tma128` on direct cuda, ggml's MMQ on
  ggml_cuda), as in llama.cpp's own default on this GPU, at perplexity 27.47 (direct cuda) and 27.38 (ggml_cuda)
  against llama.cpp's 27.27. With q8 activations (ggml's `GGML_CUDA_MMQ_PREC=q8`) ggml_cuda reaches 26.50 against
  llama.cpp's 26.51, while W4A4 runs prompts 1.51-1.55x faster. ggml_cuda runs Q4_K_M at 0.99 / 1.00 / 0.98x
  llama-bench at pp512 / pp4096 / tg128 (26,160 / 29,027 / 476 tokens/s), BF16 at 0.92 / 0.98 / 0.99x and NVFP4 at
  0.86 / 0.90 / 1.01x; direct cuda runs prompts at 0.58-0.85x and decode at 0.78-0.91x. The 1-2% decode gap is
  TensorSharp's process-wide `GGML_CUDA_PDL=0` on CC 12 devices (set for Qwen 3.5's batched graphs): with PDL on in
  both engines TensorSharp decodes 1.0% faster, and LFM3's one-sequence logits are the same bytes. The server's 20
  e2e scenarios pass on ggml_cuda and direct cuda. The LFM3 test classes pass on ggml_cuda, direct cuda and
  ggml_vulkan except four tolerance tests calibrated on the A40 (two NVFP4 feed-forward tests, which expect q8
  activations, and two synthetic speech-network bounds exceeded by 0.1% and 6%) and one Qwen DFlash test that fails
  at this ggml pin on every CUDA GPU. The tree after the 2026-11-18 lane merges gives the same logit bytes as the tree
  before them on all three GPU lanes (12 file x lane runs, 3,648 rows, and 9 perplexity digests); the speed figures
  were measured on the tree before them, whose GPU code paths are the same. Not measured on this GPU: serving under
  concurrency, ggml_vulkan's speed, and Q8_0 / QAD-Q4_0 speed.
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
  to Metal than llama.cpp's own CPU run. A prompt's last layer runs its feed-forward on the
  output row only (as llama.cpp's graph does), so on the CPU lanes that row's products take
  the one-row kernels: its logits move by at most 8.3e-7 relative L2 against a full-width
  product, with every later row unchanged and the same top-1 in all 14 of 42 measured cases
  where they move (Q4_K_M on ggml_cpu and BF16 on the managed CPU when the prompt's last
  512-row piece is a multiple of 4 rows; BF16 on ggml_cpu from 64 rows; never Q8_0 on either
  lane or Q4_K_M on the managed CPU). Perplexity on Q4_K_M (32 x 1,024 tokens) is 26.79
  on ggml_cpu and 26.75 on the managed CPU against llama.cpp CPU's 26.79. The 20 server
  scenarios pass on both lanes, as do the 8 CPU speech-synthesis tests (engine speech equal
  bit for bit to standalone synthesis). GSM8K (first 200 questions, thinking off, Q4_K_M)
  scores 79.0% on ggml_cpu and 80.0% on the managed CPU against llama.cpp CPU's 81.0%
  (paired McNemar p = 0.50 and 0.82; on Metal both engines score 81.5%). The run first
  stopped at 50 questions, where ggml_cpu trailed 37 to 43 (p = 0.07; 0.03 against an
  earlier llama.cpp CPU run). Scored on the content alone (what a client reads with thinking
  off), both lanes scored 76.5% against 81.0% (p = 0.09) while TensorSharp moved the text
  before a stray `</think>` into `reasoning_content` in replies max_tokens then cut (5 on
  ggml_cpu and 7 on the managed CPU had their answer only there). Such a reply now reads as
  its stream does (Reasoning, below): re-run on ggml_cpu, the 200 replies are the earlier
  run's text and the content alone scores 79.0%, its whole-reply score (p = 0.50 against llama.cpp
  CPU's 81.0%); the managed CPU's recorded replies, re-scored the same way, give 80.0%
  (p = 0.82; not re-run).
  Greedy CPU replies part from llama.cpp's CPU replies at near-ties within a few hundred
  characters, as llama.cpp's own CPU and Metal replies part from each other. Four concurrent
  conversations (Q4_K_M) equal their one-at-a-time runs token for token on the managed CPU. On
  ggml_cpu they do on Q8_0 and QAD-Q4_0 (12 of 12 turns each) and part at a near-tie on Q4_K_M,
  as on llama.cpp: ggml-cpu's repacked K-quant product runs groups of four rows through its GEMM
  and the rest through its GEMV, which sum a row in different orders, and a server step mixes
  prompt rows, other sequences' rows and decode rows (`TS_GGML_CPU_REPACK=0` restores the
  equality at half the prompt speed; sending every K-quant row through the GEMM restored it at
  half the decode speed). Checked before
  the optimizations and not repeated: needles at 8K and 32K, the CLI checks, LFM3-Omni image
  and audio input (prompt token counts equal to llama-server's) and speech output that
  transcribes back at word error rate 0.
- **CPU lanes, speed (2026-11-18, M5 Pro with 6 performance and 12 efficiency cores):** TensorSharp at
  its defaults against llama.cpp at its best (the faster of its Accelerate BLAS and ggml-cpu-only builds,
  prompts on 18 threads, decode on 16), flash attention on, interleaved rounds. The final tree, measured
  that evening (3 rounds; the table is under [Benchmarking](#benchmarking)): *ggml_cpu* decodes Q4_K_M at
  0.97x llama.cpp (155 against 160 tokens/s; 0.97-1.04x by round), Q8_0 at 1.07x, QAD-Q4_0 at 1.03x and
  NVFP4 at 1.14x (BF16 3.9x, whose ggml-cpu dot is a scalar loop on ARM; Q4_K_M 1.56-1.60x at 16K tokens
  of context in the day's earlier rounds), and runs 128- and 512-token prompts at 0.95-0.97x on Q4_K_M,
  1.00x on Q8_0 and 1.9-2.6x on NVFP4. Served (Q4_K_M, 512-token prompts, 128 out) its first token comes
  sooner than llama-server's (570 against 600 ms), one stream decodes at 0.96x (144 against 150
  tokens/s), and four streams reach 1.17x llama-server's aggregate (127 against 108 tokens/s). The
  *managed CPU* decodes the quantized files at 0.78-0.95x (Q4_K_M 0.81x, Q8_0 0.95x, QAD-Q4_0 0.78x,
  NVFP4 0.92x; BF16 3.7x in the worker-pool round), runs Q4_K_M and Q8_0 prompts of 128-512 tokens at
  0.86-0.92x and NVFP4 ones at 1.6-2.2x, and serves one Q4_K_M stream at 0.83x llama-server's decode
  with the first token at parity (589 against 600 ms) and four streams at 1.06x its aggregate. Measured
  earlier the same day and not repeated: ggml_cpu encodes a 512 x 512 image in about 0.6x llama.cpp's
  time and audio in 0.48-0.50x (both engines on 18 threads) and speaks at 6.8-7.0x real time against
  4.2x; the managed CPU speaks at 5.1x real time against llama-server's 4.1x (first audio 70 against
  112 ms) and its audio encoder takes 1.0-1.25x llama.cpp's time; BF16 prompts run at 1.03-1.26x
  (ggml_cpu) and 0.44x (managed) of llama.cpp's Accelerate build. Against vLLM 0.34's CPU backend (fp16 with its macOS short-conv
  workaround, 18 threads; it has no quantized CPU path here) with one client for every engine and no
  repetition penalty anywhere (ISL 512 / OSL 128, `eng/lfm3/bench/vllm-cpu-mac.sh`): one request at a time
  ggml_cpu on Q4_K_M answers in 0.59 s against 1.98 s and decodes at 6.8 against 38.1 ms a token (median of
  5 rounds, 2 of which ran 8-10x slower: see [Current limits](#current-limits); llama-server 0.61 s and 6.6 ms, the managed CPU 0.61 s and 7.5 ms), and four at a time it
  delivers 126 tokens/s against vLLM's 36 (llama-server 112, the managed CPU 114). On BF16 ggml_cpu decodes
  at 18.3 ms a token against vLLM's 38.1, but its first token took 2.1-6.0 s this session on every build
  tested (see [Current limits](#current-limits)). The knobs
  are in the [environment variable
  matrix](../env_var_feature_matrix.md#out-of-matrix-ggml-cpu-backend-knobs).
- **Not yet validated:** the TensorAgent app on a device; tensor parallelism and multi-GPU
  placements on CUDA (one GPU was available); serving under concurrency on a Blackwell or a datacenter GPU
  (vLLM and SGLang were compared on the A40, vLLM also on the Mac's CPU; SGLang's CPU backend is x86 AMX
  only); the CPU lanes' newest code (the managed worker pool's guided splits and decode width, ggml_cpu's
  NVFP4 GEMM and prompt pool) on Linux, Windows and iOS devices, where it was not built or measured.

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
  inconsistent". TensorSharp renders the thinking-off prompt exactly as llama.cpp and
  Hugging Face do (the same token ids for all 200 GSM8K prompts and 50 thinking-on ones;
  `eng/lfm3/prompt-parity.py`), so the stray close is the model's own. How TensorSharp
  returns such a reply depends on whether the client can take text back and on whether
  the model finished it:
  - A stream that can take text back (the Web UI, TensorAgent), and a non-streaming reply
    (OpenAI, Ollama, Responses) the model finished: a `</think>` within 4,096 characters
    makes the text before it reasoning, returned as `reasoning_content` (OpenAI) or
    `thinking` (Ollama, Web UI); the Web UI shows the text at once and takes the
    reasoning back when the close arrives.
  - A stream that cannot take text back (the OpenAI, Ollama and Responses streams, the
    CLI's interactive mode): the reply streams as content as it is generated, a
    `</think>` and the text around it included, as llama.cpp, vLLM and SGLang return it.
    Nothing is held, so a short answer arrives token by token. A reply that opens with a
    `<think>` of its own streams that block as reasoning.
  - A non-streaming reply cut short (`finish_reason: length`: max_tokens, a thinking
    budget, the repetition guard) reads as its stream did: the reasoning and the
    `</think>` stay in the content, and none of it counts as `reasoning_tokens`. The
    answer after the close is incomplete there, and the reasoning before it often holds
    the result: on GSM8K (first 200, thinking off, max_tokens 1,024) 5 replies on
    ggml_cpu and 7 on the managed CPU had their answer only before the close, all cut in
    the answer that followed, and a client reading the content lost them. This holds
    with and without the server's sub-agent loop, whose rounds wait for their end for
    such a client.
  A finished non-streaming reply and its stream therefore still differ in where the
  reasoning is; a client's next request may send back either form, and both continue
  the conversation's KV cache. A block the model opens itself is capped like Gemma 4's
  unrequested channel (closed at a line break, after at most 64 tokens).
- LFM3-Omni does not reason (its card: `enable_thinking=True` is not supported): with
  thinking on it closes the block at once, in 58 of 58 replies of the loop study, so the
  Web UI's thinking switch changes nothing for it (loops 8 against 9 of 58, greedy).

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
| LFM3-Omni-2B | greedy, repetition_penalty 1.0 | the LFM3 card's general preset, as the 1.7B: temperature 0.1, top-k 50, top-p 1, min-p 0, repeat-penalty 1.1, `repeat-last-n -1` |

The server ignores `general.sampling`, so the configs pin these values (the CLI reads
it for chat). TensorAgent's catalog entries send the same values, with the engine's
64-token penalty window.

**Why LFM3-Omni is not decoded greedily.** Decoded as its card says (greedy, no penalty;
llama.cpp takes that from `general.sampling`, vLLM from `generation_config.json`), the 2B
model repeats a sentence, a table row or a stanza until max_tokens ends the reply. On the
50 English and Chinese chats of `eng/lfm3/loop` (58 replies: answers, lists, code,
stories, summaries, multi-turn, image and audio input; 2026-11-20) the share of replies
that loop is 3% on vLLM BF16, 7% on llama.cpp Q4_K_M (Metal) and 7-10% (CUDA), and 7%
(ggml_metal) and 9% (ggml_cuda) on TensorSharp Q4_K_M with the agent features off; on
Metal 45 of TensorSharp's 58 greedy replies are llama.cpp's text exactly, so this is the
model, not the engine. With the server's skills on, as the Web UI, the OpenAI endpoint and
TensorAgent run by default, 9-16% looped: greedy decoding also repeated the same tool
call round after round (see [Agent Skills](../agent_skills.md) for the repeat guard). With
the LFM3 preset the same chats loop in 0-3% of plain replies, and with the repeat guard
too in 0-3% on the Web UI and OpenAI endpoints (ggml_metal and ggml_cuda) and 2-6% in
TensorAgent. Quality holds: GSM8K
(first 200, thinking off) 75.5% greedy against 76.0% (llama.cpp greedy 76.0%; vLLM BF16
78.0% greedy, 79.0% with the preset), the 33-prompt instruction set 31 of 33 either way,
and ASR word error rate 1.5% either way on 8 clips, one of which repeats words on purpose
("no, no, no"). Decoding speed is unchanged (ggml_metal 182-194 tokens/s before and
after, ggml_cuda 411-444 before and 422-444 after).
Spoken replies keep the projector's own speech sampling (temperature 0.4, penalty 1.2
over 64 frames); their text is the reply the preset wrote. TensorSharp's repetition penalty looks back over the last
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
`--voice`, `--language`, `--seed`), `POST /v1/audio/speech`, `/v1/chat/completions` /
`/v1/responses` with `modalities` `["text","audio"]` or `["audio"]`, and the web chat
(`/api/chat` with `speech`: the bundled Web UI's and TensorAgent's **Speak replies** switch, which
play the reply as it is generated and keep a player and a download; while a reply is spoken both pages ask WebKit
for `navigator.audioSession.type = "playback"`, so an iPhone in Silent mode plays the live speech, which WebKit
otherwise plays as ambient sound that Silent mode mutes, give the choice back (`"auto"`) once the reply's audio has
played so the user's music or podcast can resume, and resume the audio after a call, Siri or the app going to the
background interrupts it) — see
[USAGE.md → Speech output](../../USAGE.md#speech-output-lfm3-omni). Voices are `female`
and `male` (named in the system prompt; there is no speaker embedding); an unknown voice,
OpenAI's default `alloy` included, is a 400 that lists them. A spoken chat reply's system
prompt gets llama.cpp's realtime line ("Speak in English using a female voice.") and, for
interleaved replies, the card's "Respond with interleaved text and audio.".

**Languages.** LFM3-Omni's interleaved and audio modes speak English only (as LFM3-Audio's
card says of its voice chat): a Chinese reply in either is read with English sounds whatever
the system prompt says, and the card's interleaved prompt alone makes the model answer a
Chinese question in English (measurements below). Its TTS prompt speaks all 19 speech
languages. So a spoken reply's language is the request's (`speech.language` on `/api/chat`,
`audio.language` on chat completions and Responses, CLI `--language`), else the one the
user's latest message is written in, by its script (Chinese characters `zh`, kana `ja`, Hangul
`ko`, Arabic, Hebrew, Greek; Latin letters leave it to the default, English). An English reply
speaks in the reply itself as above. A reply in any other language is generated as an
ordinary text reply (no speech line in its system prompt: the reply a text request gets), and
each sentence is spoken by the TTS prompt in that sentence's language as soon as it is
complete, on the engine beside the text still being generated
(`TensorSharp.Models.Audio.SpeechReplyPath.Synthesized`). The text is the speech's transcript;
markdown markers, emoji and fenced code are not read aloud, and the first piece ends at a
comma once it has 8 characters so the speech starts early. TTS without a language
(`/v1/audio/speech`, `--tts`) speaks the language its text is written in (American English,
the template's default, for Latin letters), and Chinese text reaches the TTS prompt with its
numerals read out (`2024年10月1日` as `二零二四年十月一日`, `199元` as `一百九十九元`), which
the model otherwise says as noise; English text (and Latin letters with no language) reaches it with its numerals and
the signs between them as words (`366` as `three hundred sixty-six`, `3 + 8 = 11` as `three plus eight equals eleven`,
`29th`, `3.14`, `50%`, `18 °C`), which its TTS otherwise garbles too ("366 days" heard as "state-sex days").

**Spoken turns are text turns.** On `/api/chat` (the web UI's and TensorAgent's **Speak replies**)
every spoken turn, in every language, is the request's text turn - reasoning, named skills and the
catalog, the client's tools, the host's code tools, knowledge, sub-agents, every round of the tool
loop, with the same frames - and its answer is spoken as it is written (`WebUiSpokenTurn`, over
`ReplySpeaker`): each round's answer text goes through `SpeechTextSegmenter`, and each piece is a TTS
request on the engine beside the generation, spoken in order. While the text is still being written,
the next piece waits until the speech already made is less than 3 s ahead of what the listener has heard
(`ReplySpeaker.DefaultMaxLeadSeconds`): speech and text share the engine, and speech made far ahead of the
listener only slowed the text (a long English or Chinese answer's text went from 48-53 to 79-86 updates a second
in the Mac app, and an English one's from about 17.6 to 23 on an iPhone 17 Pro Max, with no silence in the speech
either way); once the text is done, the rest is spoken at once. The text before a tool call is spoken
when the call begins, so the speech goes on while the tool runs; reasoning, tool calls and tool
results are never spoken, nor are fenced code, table rows, emoji, markdown markers, a host note in the
answer (`_(...)_`) or an inline code span of 40 characters or more; a URL is said as its host, a path
of three or more parts as its last name. The turn keeps one WAV and audio id for everything it
spoke. Its prompt is the text turn's (no speech line), so a conversation that switches speech on and
off reuses its cached prefix on every turn; an interleaved reply put "Speak using a female voice.
Respond with interleaved text and audio." in the system prompt, which forked the two modes from the
first tokens. `/v1/chat/completions` does the same for a spoken request with tools, reasoning, skills
or knowledge (or any language but English); a plain English one is still one interleaved generation.

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
  Chinese interleaved speech is English-sounding in either engine (a model limit; Chinese TTS
  is fine), hence the Languages rule above.
- Chinese replies (2026-11-19, Q4_K_M + Q8_0 mmproj, ggml-metal; `eng/lfm3/speech-lang-eval.py`
  and `speech-whisper.py`, Whisper medium as the independent judge): llama.cpp's realtime path
  under six system prompts (none, the card's, "Speak in Chinese ...", "Perform TTS in Chinese ...",
  a Chinese instruction, the voice line) gave 1 of 23 Chinese-text replies that Whisper hears as
  Chinese, and with the card's prompt alone it answered Chinese questions in English (5 of 5).
  Through `/api/chat` as the web chat sends it, 12 Chinese prompts x 2 voices: before, interleaved,
  1 of 24 heard as Chinese (CER of the speech against its transcript 102%); spoken from the text
  reply, 24 of 24 (CER 10.7%, median 3.8%; LFM3's own ASR 18.3%); chat completions 12 of 12 (8.3%),
  with `audio.language: "zh"` 12 of 12 (8.8%, 0 of 12 before). The spoken reply's text is the
  text reply (24 of 24), which equals llama-server's greedy reply for 20 of 24; prompt tokens equal
  llama-server's and HF's for 39 of 39 requests. English replies are unchanged (12 of 12, WER 0-0.4%).
  First audio of a Chinese reply 136 ms median (196 ms p90) against 66 ms for an English one;
  real-time factor 9.2x. TTS of 12 Chinese texts x 2 voices x 3 seeds without a language: CER
  18.6% before ("American English", 70 of 72 heard as Chinese), 5.0% now (72 of 72; texts with
  numerals 39.0% -> 2.0%).
- Spoken turns that run tools (2026-11-19, Q4_K_M + Q8_0 mmproj, ggml-metal, `/api/chat` as the web chat
  sends it, greedy, `eng/lfm3/speech-agent-e2e.py`): 16 turns in English and Chinese (reasoning, a named skill,
  code the host runs, a file created and edited, a three-turn session) with speech on and off: the text, the tool
  steps and the token counts match in 14 of 16; the other two are greedy near-ties that a rerun did not reproduce
  and that unrelated TTS requests beside a text turn flip too (engine batch composition). The first answer token is
  as fast as without speech; the first audio comes 0.12-0.49 s after the request. Conversations that switch speech
  on and off reuse every previous prompt and reply. English replies on a host without tools, interleaved before and
  spoken from the text now: first audio 65-70 -> 139-154 ms, the male voice 182 -> 115 Hz (the interleaved mode
  speaks one voice whatever it is asked), the text is the typed turn's (6 of 6, 0 of 6 before).
- Speech-only replies (`modalities ["audio"]`, which opens `<|audio_start|>` as llama.cpp's
  realtime AUDIO mode does): LFM3-Omni reads the user's message aloud instead of answering it,
  in TensorSharp and llama.cpp alike (8 of 8 questions, ASR = the question), and without a
  voice line its speech is unintelligible. For spoken answers in English ask for
  `["text","audio"]`; a reply in another language is spoken from its text (see Languages), in
  either modality.
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
by its codes. An id whose generation is not known renders as its transcript. The web chat
(`/api/chat`) sends a spoken reply back the same way, by the `audioId` on its assistant message.

Prefix reuse: a spoken reply's end state is retained like a text reply's. Its driver names each run of
the frames it decoded by the same codes-keyed id the expander gives the spliced turn
(`IOutputMediaDriver`, `Lfm3SpeechOutput.FramesContentId`), so the next turn resumes from it through the
whole reply. Checked on LFM3-Omni Q8_0 + Q8_0 mmproj (ggml-metal server, web chat, three spoken turns):
turn 2 reused 84 of its 111 prompt tokens (all of turn 1: prompt, `<|mixed_start|>` and the 40 slots it
computed), turn 3 156 of 179, and turn 3 answered from the spoken turns ("I named Paris first."). The
spoken system prompt names the voice and asks for interleaved replies, so a text turn after a spoken one
(or the reverse) differs from the system message on and re-prefills. A reply spoken from its text (a language
other than English) adds nothing to the system prompt: it is kept and reused as the text turn it is, by either
kind of next turn.

Limits: no `/v1/realtime`. On CUDA the speech output is checked by an ASR round trip (no word errors),
not bit for bit against llama-tts.

## Benchmarking

`ignore_eos: true` on `/v1/chat/completions` (and `/v1/responses`) generates exactly
`max_tokens` through end-of-sequence tokens, as vLLM's and SGLang's benchmark clients
expect (`vllm bench serve --ignore-eos`, SGLang's default). Pin `think` and the
sampling on every engine.

The full CPU matrix, measured at the CPU integration (`wip/lfm3-cpu`, before the worker-pool and
second ggml_cpu rounds below; the final tree's cells follow them): CPU lanes on an M5 Pro (6
performance + 12 efficiency cores, 2026-11-18, LFM3-1.7B; tokens per second, median of 3
interleaved rounds; `eng/lfm3/bench/run-matrix.sh cpu`). TensorSharp
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

**Managed CPU after the worker-pool round** (later the same day, `wip/lfm3-cpu-pool`; the
same machine shared with other jobs, 3 interleaved rounds, rounds that started with the
machine more than 5% busy left out; where a cell is a range, it spans that round's A/B and
a review's re-run of the final branch hours later). Decode-sized steps leave two cores to the process's
other threads (`TS_CPU_DECODE_THREADS`, default every core but two on a Mac with
performance and efficiency cores), the submitter polls for a job's last blocks before it
yields, products split their columns into shrinking blocks so the faster cores take the
tail, and the hosts let hot framework code tier up (the .NET vector `exp` under every
SiLU and softmax ran precompiled under `TieredCompilation=false`). Every logit is the
same bits as before (perplexity digests on all five files at 18, 9 and 4 threads;
teacher-forced decode rows on all five files, at the default width, at 9 threads and,
on Q4_K_M, at a 2-thread decode width).

| | before | after | llama.cpp |
|---|---|---|---|
| tg128, Q4_K_M / Q8_0 / QAD-Q4_0 / NVFP4 / BF16 | 122 / 87 / 125 / 107 / 53 | 142-143 / 95-101 / 131-140 / 123-125 / 58 | 160 / 111-114 / 178-180 / 131 / 16 |
| Q4_K_M pp128 / pp512 | 880 / 924 | 897-937 / 920-939 | 896-920 / 922-946 |
| served, one stream: decode tok/s, TTFT (Q4_K_M, 512 + 128) | 76, 702 ms | 125-126, 575-615 ms | 149-151, 589-605 ms |
| served, four streams: tok/s in all, TTFT | 87, 2,806 ms | 111-113, 1.8-2.4 s | 105-115, 2.4-2.5 s |
| TTS (Omni-2B Q8_0, `speech-bench.py`): real-time factor, first audio | 3.5x, 123 ms | 5.1x, 70 ms | 4.1x, 112 ms |
| audio encoder, 10 s / 30 s clip (15 calls a process, 4 rounds) | 54 / 110 ms | 44 / 98 ms | 35 / 97 ms |

llama.cpp: decode `build-cpu-noblas -t 16`, prompts the same build at `-t 18`; served
`llama-server` from the same build at `-t 16 -tb 18`; TTS `llama-server` from the Accelerate build at `-t 12` (its best of the
two); audio its encoder at 18 threads. The managed audio encoder keeps speeding up over its
first hundred calls while the JIT tiers it up (30 ms on the 10 s clip after 300 calls). At
four streams the time to first token is bimodal on both servers (single rounds 1.8-2.5 s,
depending on whether the four prompts share one prompt step), so it ranks neither.

ggml_cpu after a second round the same day (`wip/lfm3-cpu2`: prompt chunks read their products
and fresh K/V rows as views, LFM3's gate and up weights stay apart where ggml-cpu repacks them,
NVFP4 products from two rows run TensorSharp's GEMM with ggml's own arithmetic, the prompt pool
takes every physical core but one) against the merged tree before it and llama.cpp at its best
(noblas build for the quants, BLAS for BF16 and the better of the two per cell for NVFP4;
prompts on 18 threads, decode on 16), interleaved, 3 rounds, median of 3 runs a round (16K: 1
run):

| file | engine | pp128 | pp512 | pp2048 | tg | tg at 4K |
|---|---|---|---|---|---|---|
| Q4_K_M | ggml_cpu, second round | 881 | 869 | 858 | 158.7 | 141.2 |
| | ggml_cpu before it | 789 | 871 | 823 | 164.5 | 145.6 |
| | llama.cpp | 941 | 935 | 856 | 157.4 | 133.5 |
| Q8_0 | ggml_cpu, second round | 1145 | 1169 | 1095 | 114.5 | 105.3 |
| | ggml_cpu before it | 1127 | 1140 | 1036 | 116.8 | 106.3 |
| | llama.cpp | 1168 | 1218 | 1099 | 113.0 | 101.3 |
| QAD-Q4_0 | ggml_cpu, second round | 1128 | 1193 | 1131 | 181.2 | 157.2 |
| | ggml_cpu before it | 1074 | 1199 | 1053 | 183.0 | 156.0 |
| | llama.cpp | 1141 | 1190 | 1119 | 179.9 | 150.6 |
| NVFP4 | ggml_cpu, second round | **994** | **1021** | **970** | 149.9 | 131.0 |
| | ggml_cpu before it | 384 | 631 | 594 | 149.4 | 133.1 |
| | llama.cpp | 396 | 550 | 532 | 131.0 | 113.4 |
| BF16 | ggml_cpu, second round | 332 | 533 | - | 60.1 | - |
| | ggml_cpu before it | 308 | 477 | - | 59.8 | - |
| | llama.cpp | 264 | 517 | - | 15.6 | - |

Q4_K_M at 16K: 111.4 (second round), 112.7 (before) and 71.2 (llama.cpp) tokens/s. Served
(Q4_K_M, 512-token prompts, 128 out, 3 rounds): one stream's time to first token 562 ms against
603 before and llama-server's 585, decode 151 tokens/s on all three; four streams 131.0 tokens/s
in aggregate against 126.8 and 118.3. Decode pays up to 4% on Q4_K_M and 2% on Q8_0 for the
separate gate and up products, which saved 405-765 MB (below).

A review re-run that evening (same machine and builds, interleaved, 3 rounds; Q4_K_M prompts 4)
put the second round, the tree before it and llama.cpp at: Q4_K_M pp128 / pp512 923 / 922, 910 /
917 and 949 / 942; Q8_0 1122 / 1128, 1087 / 1107 and 1149 / 1137; NVFP4 955 / 971, 357 / 613 and
366 (ggml-cpu build) / 528 (BLAS); Q4_K_M decode at depth 0 / 4K 160.8 / 141.9, 164.5 / 144.4 and
160.7 / 136.3. Served Q4_K_M, the time to first token of one stream was 568 ms against 617 before,
628 for the second round with its prompt pool on 18 threads, and llama-server's 612; four streams
reached 127.9 tokens/s in aggregate against 121.0 and 106.8. Prompt cells move by up to 10%
between sessions: on Q4_K_M and Q8_0 the round's engine-bench prompt gain is 1-3% in the re-run
(the table's 12% at Q4_K_M pp128 came from slow runs of the tree before it), and decode is at parity with
llama.cpp at depth 0 rather than ahead; the served time-to-first-token gain (8-10%) holds and comes
from the prompt pool leaving one core to the server's own threads.

**The final tree** (`feature/lfm3` after the think, worker-pool and second ggml_cpu rounds were merged; the
evening of 2026-11-18). Tokens/s, median of 3 interleaved rounds [min-max of the round medians]; prompts 5
runs a round, decode 5 runs of 128 tokens after an untimed one; llama.cpp's prompts are the faster of its
two builds per cell at 18 threads (Q4_K_M and Q8_0: ggml-cpu alone; NVFP4 pp128 ggml-cpu alone, pp512
BLAS), its decode the ggml-cpu-alone build at 16. Each lane's branch head ran in the same rounds and
measured within 4.5% of these medians, in both directions, with overlapping spreads (its logits are the
same bits).

| file | engine | pp128 | pp512 | tg128 |
|---|---|---|---|---|
| Q4_K_M | TS ggml_cpu | 930 [833-936] | 893 [841-902] | 155 [155-165] |
| | TS cpu (managed) | 846 [795-862] | 862 [856-862] | 130 [126-138] |
| | llama.cpp | 956 [939-960] | 938 [841-940] | 160 [158-160] |
| Q8_0 | TS ggml_cpu | 1167 [1145-1227] | 1169 [1163-1221] | 114 [111-116] |
| | TS cpu (managed) | 1002 [949-1040] | 1021 [988-1053] | 101 [100-102] |
| | llama.cpp | 1161 [1160-1190] | 1165 [1159-1184] | 107 [100-113] |
| QAD-Q4_0 | TS ggml_cpu | - | - | **186** [185-186] |
| | TS cpu (managed) | - | - | 140 [130-141] |
| | llama.cpp | - | - | 180 [179-181] |
| NVFP4 | TS ggml_cpu | **993** [949-999] | **1024** [1023-1026] | **150** [146-150] |
| | TS cpu (managed) | 855 [853-873] | 853 [850-870] | 121 [120-124] |
| | llama.cpp | 384 [382-387] | 548 [546-554] | 132 [132-132] |

Served (Q4_K_M, 512-token prompts, 128 out, `serve-bench.py`, 3 rounds; llama-server from the ggml-cpu-alone
build at `-t 16 -tb 18`): one stream decodes at 144 (ggml_cpu), 125 (managed) and 150 (llama-server) tokens/s
with the first token after 570, 589 and 600 ms; four streams reach 127, 115 and 108 tokens/s in all (first
tokens after 2.1, 1.7 and 2.4 s, the TensorSharp lanes' moving between 1.5 and 2.3 s by round). Peak footprint 1.58, 1.59-1.67 and 1.22 GB at one
stream, 2.05-2.13, 2.11-2.29 and 1.30 GB at four. TensorSharp's decode cells sit below the lanes' own sessions
(ggml_cpu Q4_K_M 155 against 161, managed 130 against 142) while llama.cpp's did not move, and the lane heads
measured the same in these rounds (161 and 129): a session effect, not the merge; its cause was not
isolated. Spotlight indexing ran at the start of some rounds (each cell's process snapshot is kept with the
results). In the same session ggml_metal ran Q4_K_M pp512 / tg128 at 6,041 / 191 against llama.cpp Metal's
5,871 / 188, and MLX at 6,598 / 164.

## Current limits

- Speech output has no realtime endpoint (`/v1/realtime`). On `/v1/chat/completions` and
  `/v1/responses`, a text turn after an interleaved spoken one re-prefills from the system prompt,
  which names the voice only on interleaved turns; the web chat speaks every turn from its text and
  does not (see [Speech output](#speech-output)).
- Many concurrent requests decode slower than vLLM and SGLang (A40: 11.0 / 26.6 ms a token at 16 / 64
  requests on ggml_cuda, 21.5 / 40.3 on direct cuda, against vLLM's 8.1 / 11.0): a batched step runs each
  sequence's attention, K/V writes and short-convolution window as ops of its own, and is traced again
  whenever a request joins or leaves. The grouped-steps design's first, bit-identical stages are in (the
  device's greedy ids on ggml-cuda and ggml-vulkan, a step's inputs uploaded in one copy, the short
  convolutions' products over the step's rows, balanced joins, direct-CUDA recordings of up to 256
  sequences, row-invariant BF16 products of 5-16 rows 1.1-1.75x as fast); the short convolutions'
  windows in one pool per layer and row-list attention (upstream `ggml_flash_attn_ext_rows`) on the ggml
  lanes, a per-step sequence table on direct CUDA, are not built. On direct CUDA, 17-32 rows still take
  two 16-row products per weight (a 32-sequence step is slower than a 64-sequence one, which takes
  cuBLAS). One request at a time is unchanged.
- NVFP4 on a Blackwell GPU (sm_100 / sm_120) runs its prompt products as W4A4, the activations quantized to
  NVFP4 too, as llama.cpp's CUDA build does by default: perplexity +3.3% (ggml_cuda) and +3.6% (direct cuda)
  against q8 activations, for prompts 1.5x faster. Only ggml_cuda can take q8 activations (ggml's own
  `GGML_CUDA_MMQ_PREC=q8`); the direct-CUDA route of `config/lfm3-1.7b-nvfp4.json` has no such option.
- On CC 12 GPUs (sm_120) TensorSharp turns ggml-cuda's programmatic dependent launch off for the whole
  process (`GGML_CUDA_PDL=0`, set for Qwen 3.5's batched graphs), which costs LFM3 about 3% of its decode
  speed; llama.cpp keeps it on.
- The direct `cuda` backend runs parts of the Omni vision and audio towers on its element-by-element
  host fallback (abs, exp, neg, a fill and some of the towers' products have no kernel for their
  operands there): an image or audio prompt's first token comes later than on ggml_cuda.
- Speech input as a live stream (a realtime endpoint) is not served; clips are encoded
  whole (30 s chunks).
- A message's still images come before its videos in the prompt whatever the client's
  order; a video sent as `video_url` has no position among them.
- Nested tool schemas are flattened in the prompt (the tool-call grammar reads the
  client's schema).
- The QAD-Q4_0 file is not in TensorAgent's catalog pending a quality and speed
  measurement against Q4_K_M.
- CPU lanes against llama.cpp's CPU builds on an M5 Pro (the final tree's cells are under
  [Benchmarking](#benchmarking)): ggml_cpu's Q4_K_M prompts of 128 and 512 tokens are 2-7% behind
  depending on the session (Q8_0 and QAD-Q4_0 at parity; the cell moves by up to 10% between
  sessions), and its Q4_K_M decode is at 0.97-1.04x. The managed CPU decodes quantized files at
  0.78-0.95x (what remains is per-step work outside the products: the graph interpreter and the
  single-threaded elementwise ops are ~20% of a decode step's main-thread time), runs Q4_K_M and Q8_0
  prompts at 0.86-0.92x (llama.cpp's repacked GEMMs use i8mm, which .NET 10 does not expose), BF16
  prompts at 0.44x of llama.cpp's Accelerate build, and serves one Q4_K_M stream at 0.83x llama-server's
  decode speed (first token at parity).
- Served ggml_cpu can run 8-10x slower for a whole server's life: on 2026-11-18, 2 of 40 one-stream Q4_K_M
  runs of the vLLM harness's cell (65-69 against 6.7-6.8 ms a token; both in its first two interleaved rounds,
  none in 30 back-to-back runs later that evening) and 3 of 23 BF16 runs, on the final tree and on the tree
  before that day's merges alike, and never llama-server or the managed CPU in the same rounds. During such a
  run the server keeps 15 threads busy instead of the 16 a decode step runs on, as if one of the step's threads
  (a pool worker or the calling thread) were not running while the others wait for it at every node; the cause
  is not isolated. Served BF16 prompts on ggml_cpu also took 2.1-6.0 s for 523 tokens that day on every build
  tested, the CPU integration's included, against the engine's ~1.0 s.
- ggml_cpu's repacked weight copies are anonymous memory: served peak footprint 1.56 GB for
  Q4_K_M (2.31 GB Q8_0, 1.52 GB QAD-Q4_0) against llama-server's 1.20 GB (1.91, 1.15), down from
  1.96 GB (3.08, 1.92) since LFM3 keeps the gate and up weights apart there; NVFP4 (1.57 GB
  against 0.84) and BF16 (2.33 GB against 0.29) still keep their fused gate|up weights as
  anonymous copies, the faster layout for them. `TS_GGML_CPU_REPACK=0` restores the smaller
  footprint at half the prompt speed. With repacking, ggml-cpu's K-quant products (Q4_K_M) give
  a row other bits through its GEMM than through its GEMV, so concurrent greedy replies can part
  from one-at-a-time ones at a near-tie (as on llama.cpp; Q8_0 and QAD-Q4_0 stay equal); a
  TensorSharp one-row kernel with the GEMM's arithmetic would close it. The managed CPU keeps
  them equal on Q4_K_M (measured), but its BF16 products take activation rows four at a time
  and a lone row another way, so BF16 replies can part there too.
  Repacking and the thread pools were measured only on an M5 Pro; the BF16 Accelerate path is
  Apple-only.
