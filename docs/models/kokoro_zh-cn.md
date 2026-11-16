# Kokoro（语音合成）

[English](kokoro.md) | [中文](kokoro_zh-cn.md)

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)（Apache-2.0 许可）是一个 8200 万参数的文本转语音模型：
StyleTTS 2 架构，包含 PL-BERT 文本编码器、受风格向量调制的时长与韵律预测器以及 iSTFTNet 声码器，用从音色包中选取的
音色输出 24 kHz 单声道语音。TensorSharp 在 `--backend cpu` 上以托管 C# 运行 v1.0，速度快于实时。

文本以音素形式输入模型。本版本按 Kokoro 自身管线在 v1.0 中的方式（`misaki.zh.ZHG2P`）读取**普通话**文本：`cn2an`
把阿拉伯数字读出来（日期逐位读，百分数、分数、范围、温度），中文标点转换为模型认识的符号，jieba 分词（词典 DAG 加
HMM），pypinyin 为每个词注音（先查词组词典，再取每个字的常用读音），misaki 的拼音转 IPA 规则把音节转为 Kokoro 的音素
字母表。所有词表都由转换脚本从这些库导出，C# 部分只包含算法。其他语言可以直接用音素朗读（`--phonemes`、
`SpeechSynthesisOptions.Phonemes`）；英文文本前端暂未包含。

## 生成 GGUF

```bash
huggingface-cli download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth voices/zm_yunjian.pt voices/zf_xiaoxiao.pt \
    --local-dir kokoro-82m
python eng/audio-models/kokoro_to_gguf.py --input kokoro-82m --out kokoro-82m-v1.0.gguf
```

转换脚本需要 torch（读取 `.pth` 检查点）；普通话词表还需要 `misaki[zh]`、`pypinyin`、`jieba` 和 `cn2an`，用 `--no-zh`
可省略。权重归一化会被折叠，所有张量以 F32 存储（含五个音色和普通话词表共 322 MiB）。找到的每个 `voices/*.pt` 都会写入，
`--voices` 可指定其中一部分。

## 运行

```bash
TensorSharp.Cli --model kokoro-82m-v1.0.gguf --backend cpu --prompt "今天天气很好，我们去公园散步吧。" \
    --voice zm_yunjian --output speech.wav
TensorSharp.Cli --model kokoro-82m-v1.0.gguf --backend cpu --phonemes "ni↓xau↓" --voice zf_xiaoxiao
```

`--speed` 调整语速。长文本按换行拆分，并像 `KPipeline` 一样按整句打包成若干段，最后拼接。服务器接受 OpenAI 的语音请求：

```bash
curl http://localhost:5000/v1/audio/speech -H "content-type: application/json" \
    -d '{"input": "今天天气很好。", "voice": "zm_yunjian", "response_format": "wav"}' -o speech.wav
```

`response_format` 为 `wav` 或 `pcm`（24 kHz 单声道 16 位小端）。

## 验证结果

与 `kokoro` 0.9.4 / `misaki` 0.9.4 在一句含日期、数字和温度的句子上对比：

- 音素与 `KPipeline` 的输出逐字相同。
- 双方都把谐波源的随机相位和噪声置零（`TS_KOKORO_DETERMINISTIC=1`）后，时长与采样点数完全相同，F0 与能量曲线相差在
  1e-3 以内，波形相关系数 0.992（信噪比 17.7 dB）。剩余差异来自谐波源近乎为空的 STFT 频点的相位舍入，这些频点本身没有
  确定的相位角；Kokoro 自带的另一种 STFT（`disable_complex=True`）与其默认实现之间的差异更大（相关系数 0.936，信噪比
  4.9 dB）。
- Whisper large-v3-turbo 能把合成的语音转写回原文。

CPU 上约 2 倍实时。
