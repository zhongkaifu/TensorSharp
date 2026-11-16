# Whisper（语音识别）

[English](whisper.md) | [中文](whisper_zh-cn.md)

OpenAI 的 [Whisper](https://github.com/openai/whisper)（MIT 许可）可识别约 100 种语言的语音并给出时间戳。TensorSharp
从自包含的 GGUF 运行 Hugging Face 的检查点（large-v3、large-v3-turbo 以及较小的多语言和纯英文模型），支持
`--backend cuda`（F16 权重常驻显存）和 `--backend cpu`。

解码流程与 openai/whisper 的 `transcribe()` 一致：按 30 秒窗口处理整段音频的 log-mel 特征，每个窗口带时间戳解码
（贪心解码；当文本出现重复或概率过低时，依次以 0.2 … 1.0 的温度重新采样），按时间戳对切分片段，下一个窗口从上一个完整
片段的结尾开始，每个窗口以之前的文本为条件，静音窗口会被跳过。词级时间戳来自检查点的对齐注意力头：对窗口进行一次
teacher forcing 解码，对交叉注意力做中值滤波，再用动态时间规整（DTW）对齐到帧，与 `whisper.timing` 相同。

## 生成 GGUF

```bash
huggingface-cli download openai/whisper-large-v3-turbo --local-dir whisper-large-v3-turbo
python eng/audio-models/whisper_to_gguf.py --input whisper-large-v3-turbo --out whisper-large-v3-turbo-f16.gguf
```

转换脚本只依赖 numpy。它保留 Hugging Face 的张量名与 F16 精度，并把 `config.json`、`generation_config.json`
（特殊 token、语言 id、需抑制的 token、对齐注意力头）、Slaney mel 滤波器组和字节级 BPE 词表一并写入文件，旁边无需
其他文件。

## 运行

```bash
TensorSharp.Cli --model whisper-large-v3-turbo-f16.gguf --backend cuda --audio talk.mp3
TensorSharp.Cli --model whisper-large-v3-turbo-f16.gguf --backend cuda --audio talk.wav --language zh \
    --word-timestamps --output talk.json
```

`--output` 按扩展名选择格式：`.srt`、`.vtt`、`.json`（OpenAI 的 `verbose_json`，带 `--word-timestamps` 时包含
`words`）或纯文本。`--language` 跳过语言检测；`--translate` 译为英文（large-v3-turbo 微调时没有翻译数据，会照常转写，
与 transformers 中的表现相同）；`--prompt` 作为第一个窗口的上文。WAV、MP3、Ogg 直接读取，其他格式经平台解码器读取，
采样率和声道数不限。

服务器接受 OpenAI 的请求格式：

```bash
curl http://localhost:5000/v1/audio/transcriptions -F file=@talk.mp3 -F response_format=verbose_json \
    -F "timestamp_granularities[]=word"
curl http://localhost:5000/v1/audio/translations -F file=@talk.mp3
```

`response_format` 可为 `json`、`text`、`srt`、`vtt` 或 `verbose_json`；支持 `language`、`prompt` 和 `temperature`。

## 验证结果

与 Hugging Face transformers 5.19（FP32，相同的长音频设置：以前文为条件、温度回退、压缩比阈值 2.4、对数概率阈值
-1、无语音阈值 0.6）在两段拼接旁白上的对比：

| 音频 | 片段数 | 文本完全相同 | 最大时间戳差 |
|---|---|---|---|
| 英语，98.7 秒 | 19 | 19 | 0.02 秒（一个时间戳步长） |
| 普通话，90.4 秒 | 14 | 14 | 0.01 秒 |

log-mel 特征与 `WhisperFeatureExtractor` 相差 3e-5，编码器输出在每个位置的余弦相似度为 0.9999。Whisper 使用精确
（erf）形式的 GELU；仅用 tanh 近似就会让静音段的个别编码器位置的余弦相似度降到 0.4，并改变一段普通话的标点，因此本
移植使用 `DirectOps.GeluErf`。使用 `--backend cuda` 时，98.7 秒的音频约 4.8 秒完成，单块 GPU 上略高于 20 倍实时。
`--backend cpu` 结果相同，但较慢（每个窗口数十秒）。

未实现：束搜索与 `best_of`（采用贪心解码加温度回退，这是 transformers 与 openai/whisper 中 `transcribe` 的默认方式），
以及 `hallucination_silence_threshold` 启发式规则。
