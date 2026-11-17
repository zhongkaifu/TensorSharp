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
    (no BOS, special tokens not parsed, the second half of each chunk scored);
  - `tokens MODEL FILE OUT` writes the file's ids, to compare with `llama-tokenize --ids --no-bos --no-parse-special`;
  - `bench MODEL BACKEND [PP,...] [TG] [DEPTH,...]` measures prefill and decode tokens/s as llama-bench's pp / tg /
    tg@d do, with the decode steps' graph counters (runs, advances in place, builds).

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
