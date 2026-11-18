# LFM3 validation tools

Reusable tools for comparing TensorSharp's LFM3 (`Lfm3RuntimeModel`) with llama.cpp on the same GGUF. Outputs belong
in the ignored `artifacts/lfm3/`.

- `tokenizer-fixture.py` regenerates `InferenceWeb.Tests/Fixtures/Lfm3Tokenizer/llamacpp-reference.json` from
  `llama-tokenize`.
- `llamacpp-forced.cpp` decodes each case (a UTF-8 prompt file) greedily with libllama and writes
  `eng/ForcedLogitProbe`'s reference layout (`cases.json` and one `.f32` file of logits per case). Build it against a
  llama.cpp build tree (see the file's header), then teacher-force TensorSharp through the same histories:

  ```
  llamacpp-forced MODEL REF_DIR 64 99 cases/*.txt          # 99: Metal; -1: ggml-cpu only
  dotnet eng/ForcedLogitProbe/bin/Release/net10.0/ForcedLogitProbe.dll MODEL OUT_DIR ggml_metal --reference REF_DIR
  ```

  ForcedLogitProbe's `metrics.json` holds every row's top-1 agreement, relative L2, KL and largest logit difference.
- `Lfm3Probe` (`dotnet build eng/lfm3/Lfm3Probe -c Release`):
  - `ppl MODEL BACKEND FILE CTX CHUNKS` computes perplexity as `llama-perplexity -f FILE -c CTX --chunks CHUNKS` does
    (no BOS, special tokens not parsed, the second half of each chunk scored), and prints a digest of every scored
    logit's bits: two builds whose kernels claim the same arithmetic must print the same digest;
  - `tokens MODEL FILE OUT` writes the file's ids, to compare with `llama-tokenize --ids --no-bos --no-parse-special`;
  - `bench MODEL BACKEND [PP,...] [TG] [DEPTH,...]` measures prefill and decode tokens/s as llama-bench's pp / tg /
    tg@d do, with the decode steps' graph counters (runs, advances in place, builds).

- `omni-in/`: LFM3-Omni image, video and audio input end to end against llama-server (media set, request
  comparison, cold and concurrent TTFT, the three chat surfaces, a layout / encode probe); see its README.

MLX runs need `TENSORSHARP_MLX_LIBRARY_DIR` pointing at a directory holding `libmlxc` (for example the test project's
output).

## Speech output

- `speech-ref.sh` / `speech-ref.cpp`: llama.cpp's TTS pipeline on a fixed text and seed, with the LM hidden row of every
  frame, the depthformer logits, codes, PCM and timings; the fixture of `Lfm3SpeechRealWeightsTests` and the
  `TS_LFM3_SPEECH_REF` directory of `Lfm3SpeechIntegrationRealModelTests` (hidden.f32, llama-tts.wav).
- `speech-chat-ref.cpp`: llama.cpp's speech pipeline continuing a conversation (its realtime server's path: the turn as
  text, `n_past > 0`, out_mode MIXED or AUDIO, greedy text slots with the pipeline's logit bias) for each question of a
  TSV; each reply's WAV and transcript. Build it like `speech-ref.cpp` (see its header).
- `speech-asr.py LLAMA_BIN LM MMPROJ MANIFEST.tsv OUT.json`: round-trip intelligibility. LFM3-Omni's own ASR
  (`llama-mtmd-cli`, "Perform ASR in <Language>.", greedy) transcribes each WAV, scored against the text spoken (WER, or
  CER for Chinese/Japanese/Korean; case, punctuation and English numbers normalized).
- `speech-chat.py URL OUT_DIR [interleaved|audio]`: spoken chat replies from an OpenAI-compatible server
  (`modalities` with "audio"), each reply's WAV beside its transcript, and a manifest for `speech-asr.py` that checks
  the speech says what the transcript says.
- `speech-bench.py OUT.json REPEATS NAME=URL ...`: streaming `/v1/audio/speech` from several servers in turn (TensorSharp,
  llama-server) with one client: time to the first audio byte, total time, real-time factor.

## End-to-end suite (`e2e/`)

Black-box checks of the real servers and CLI, with llama-server on the same GGUF as the reference. Standard
library Python only; outputs go to the ignored `artifacts/lfm3/e2e/`.

- `ts-server.sh PORT LOG [args]` / `llama-server.sh PORT LOG MODEL [args]` start the two servers
  (`config/lfm3-1.7b.json`, agent features off; llama-server with `--jinja --reasoning-format deepseek
  --no-reasoning-preserve`). Start TensorSharp with `--sampling-precedence request` so requests can pin
  greedy decoding.
- `e2e.py --url TS [--ref LLAMA]`: 20 scenarios with automated checks (chat completions / Responses / Web UI,
  streaming, system prompts, stop strings, max_tokens, seeds, card sampling, ignore_eos, thinking on/off,
  past reasoning, thinking budget, streamed tool-call deltas, tool_choice, an agent loop, structured output,
  prefix-cache reuse and branching, robustness inputs, cancel and client disconnect).
- `gsm8k.py`, `ifeval.py` (33 verifiable prompts), `toolcall_eval.py` (35 cases in `toolcall_cases.py`) and
  `compare.py` (paired accuracy, Wilson intervals, exact McNemar); `quality.sh` runs all of them on one server.
- `needle.py` (needle in a haystack at 8K-128K, sized with llama-server's `/tokenize`) and `longctx_bench.py`
  (interleaved prefill / decode speed after a long prompt).
- `spec.py` (speculative server == plain server, acceptance from the log), `cli.py` (TensorSharp.Cli checks,
  `--dump-prompt` against llama-server), `matrix.sh` (one model x backend cell: server, scenarios, quality
  sets, optional llama-server reference).
- Concurrency: `eng/validation/parallel-multiturn-webui.py --conversations 16 --greedy --plain`, with a
  `--sequential` run as the `--compare-with` control.

Performance: `bench/` (README there) compares TensorSharp with llama.cpp - the llama-bench matrix on Metal, MLX and the
CPU lanes, and serving against llama-server - with every arm interleaved.
