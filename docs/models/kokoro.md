# Kokoro (speech synthesis)

[English](kokoro.md) | [中文](kokoro_zh-cn.md)

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) is an 82M-parameter text-to-speech model:
StyleTTS 2 with a PL-BERT text encoder, a style-conditioned duration and prosody predictor and an iSTFTNet vocoder,
speaking 24 kHz mono in a voice chosen from a pack of style vectors. TensorSharp runs v1.0 on `--backend cpu` in
managed C#, faster than real time.

Text reaches the model as phonemes. This build reads **Mandarin** text the way Kokoro's own pipeline does for
v1.0 (`misaki.zh.ZHG2P`): `cn2an` reads numerals out (dates digit by digit, percentages, fractions, ranges,
temperatures), Chinese punctuation becomes the marks the model knows, jieba splits words (its dictionary DAG and
HMM), pypinyin reads each word (phrase dictionary first, then each character's usual reading), and misaki's
pinyin-to-IPA rules turn syllables into Kokoro's phoneme alphabet. Every table comes from those libraries through
the converter, so the C# side holds only the algorithms. Other languages can be spoken from phonemes
(`--phonemes`, `SpeechSynthesisOptions.Phonemes`); an English text front end is not included yet.

## Make the GGUF

```bash
huggingface-cli download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth voices/zm_yunjian.pt voices/zf_xiaoxiao.pt \
    --local-dir kokoro-82m
python eng/audio-models/kokoro_to_gguf.py --input kokoro-82m --out kokoro-82m-v1.0.gguf
```

The converter needs torch (to read the `.pth` checkpoint) and, for the Mandarin tables, `misaki[zh]`, `pypinyin`,
`jieba` and `cn2an`; `--no-zh` leaves those out. Weight norm is folded and every tensor stored F32 (322 MiB with five
voices and the Mandarin tables). Each `voices/*.pt` found is included; `--voices` picks some.

## Run it

```bash
TensorSharp.Cli --model kokoro-82m-v1.0.gguf --backend cpu --prompt "今天天气很好，我们去公园散步吧。" \
    --voice zm_yunjian --output speech.wav
TensorSharp.Cli --model kokoro-82m-v1.0.gguf --backend cpu --phonemes "ni↓xau↓" --voice zf_xiaoxiao
```

`--speed` scales the speaking rate. Long text is split at newlines and packed into chunks of whole sentences, as
`KPipeline` does, and the chunks are joined. The server takes OpenAI's speech request:

```bash
curl http://localhost:5000/v1/audio/speech -H "content-type: application/json" \
    -d '{"input": "今天天气很好。", "voice": "zm_yunjian", "response_format": "wav"}' -o speech.wav
```

`response_format` is `wav` or `pcm` (16-bit little-endian mono at 24 kHz).

## What was checked

Against `kokoro` 0.9.4 / `misaki` 0.9.4 on a sentence with a date, numbers and a temperature:

- The phonemes are identical to `KPipeline`'s, character for character.
- With the harmonic source's random phase and noise zeroed in both (`TS_KOKORO_DETERMINISTIC=1`), the durations and
  the sample count are identical, the F0 and energy curves match to 1e-3, and the waveform correlates at 0.992
  (SNR 17.7 dB). The rest is rounding in the phases of near-empty STFT bins of the harmonic source, which have no
  defined angle; Kokoro's own alternative STFT (`disable_complex=True`) differs from its default by more (correlation
  0.936, SNR 4.9 dB).
- Whisper large-v3-turbo transcribes the speech back to the input text.

About 2x real time on the CPU.
