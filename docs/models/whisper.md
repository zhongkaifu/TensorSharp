# Whisper (speech recognition)

[English](whisper.md) | [中文](whisper_zh-cn.md)

OpenAI's [Whisper](https://github.com/openai/whisper) (MIT) transcribes speech in about 100 languages and
times it. TensorSharp runs the Hugging Face checkpoints (large-v3, large-v3-turbo, and the smaller multilingual
and English-only models) from a self-contained GGUF, on `--backend cuda` (resident F16 weights) or `--backend cpu`.

Decoding follows openai/whisper's `transcribe()`: 30-second windows of the clip's log-mel features, each decoded
with timestamps (greedy, falling back to sampling at temperatures 0.2 ... 1.0 when the text repeats itself or is
improbable), split into segments at the timestamp pairs, the next window starting where the last complete segment
ended, each window conditioned on the text before it, and silent windows skipped. Word timestamps come from the
checkpoint's alignment heads: the window is decoded again teacher-forced, the cross-attention is median-filtered and
aligned to frames with dynamic time warping, as `whisper.timing` does.

## Make the GGUF

```bash
huggingface-cli download openai/whisper-large-v3-turbo --local-dir whisper-large-v3-turbo
python eng/audio-models/whisper_to_gguf.py --input whisper-large-v3-turbo --out whisper-large-v3-turbo-f16.gguf
```

The converter needs only numpy. It keeps the Hugging Face tensor names and F16 precision, and stores
`config.json`, `generation_config.json` (special tokens, language ids, suppressed tokens, alignment heads), the
Slaney mel filterbank and the byte-level BPE vocabulary in the file, so nothing else sits beside it.

## Run it

```bash
TensorSharp.Cli --model whisper-large-v3-turbo-f16.gguf --backend cuda --audio talk.mp3
TensorSharp.Cli --model whisper-large-v3-turbo-f16.gguf --backend cuda --audio talk.wav --language zh \
    --word-timestamps --output talk.json
```

`--output` picks the format from its extension: `.srt`, `.vtt`, `.json` (OpenAI's `verbose_json`, with `words` when
`--word-timestamps` is given) or plain text. `--language` skips detection; `--translate` translates to English
(large-v3-turbo was fine-tuned without translation data and transcribes instead, as it does in transformers);
`--prompt` conditions the first window. Audio is read from WAV, MP3 and Ogg directly and from other formats through
the platform decoder, at any rate and channel count.

The server takes OpenAI's request shape:

```bash
curl http://localhost:5000/v1/audio/transcriptions -F file=@talk.mp3 -F response_format=verbose_json \
    -F "timestamp_granularities[]=word"
curl http://localhost:5000/v1/audio/translations -F file=@talk.mp3
```

`response_format` is `json`, `text`, `srt`, `vtt` or `verbose_json`; `language`, `prompt` and `temperature` are
honoured.

## What was checked

Against Hugging Face transformers 5.19 (FP32, the same long-form settings: condition on previous text, temperature
fallback, compression-ratio 2.4, log-probability -1, no-speech 0.6) on two concatenated narrations:

| Clip | Segments | Identical text | Largest timestamp difference |
|---|---|---|---|
| English, 98.7 s | 19 | 19 | 0.02 s (one timestamp step) |
| Mandarin, 90.4 s | 14 | 14 | 0.01 s |

The log-mel features match `WhisperFeatureExtractor` to 3e-5 and the encoder output to cosine 0.9999 at every
position. Whisper's GELU is the exact (erf) form; the tanh approximation alone moves single encoder positions in
silent stretches to cosine 0.4 and changes the punctuation of a Mandarin segment, so the port uses
`DirectOps.GeluErf`. With `--backend cuda` the 98.7 s clip transcribes in about 4.8 s, a little over 20x real time on
one GPU. `--backend cpu` produces the same text but is slow (tens of seconds per window).

Not implemented: beam search and `best_of` (decoding is greedy with temperature fallback, the transformers and
openai/whisper default for `transcribe`), and the `hallucination_silence_threshold` heuristic.
