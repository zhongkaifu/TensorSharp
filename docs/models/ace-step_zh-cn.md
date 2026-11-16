# ACE-Step 1.5（音乐生成）

[English](ace-step.md) | [中文](ace-step_zh-cn.md)

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)（Apache-2.0 许可）可根据风格描述、歌词以及可选的速度、调式
和拍号生成 48 kHz 立体声歌曲，演唱或纯音乐均可。TensorSharp 按 ComfyUI 模板的方式运行 turbo 检查点，使用
`--backend cuda`（BF16 权重常驻显存；`--backend cpu` 也可运行，但较慢）：

1. Qwen3 规划语言模型根据描述、歌词和元数据（其思维链以 YAML 写出元数据）每秒生成 5 个音频编码，以空思维链作为负向
   条件做 classifier-free guidance（2），温度 0.85，top-p 0.9；
2. Qwen3-0.6B 编码风格描述和元数据；歌词以其 token 嵌入输入，音色以参考"静音"潜变量输入，音频编码经 FSQ 量化器和
   反量化器解码为提示；
3. 2.4B 参数的 DiT（24 层，AdaLN，隔层滑动窗口自注意力，对条件做交叉注意力）在 shift 为 3 的流匹配调度上用 8 步
   Euler 对 25 Hz、64 通道的潜变量去噪；
4. Oobleck VAE 解码（在 CPU 上分块重叠解码），当结果的标准差高于 0.2 时按 `VAEDecodeAudio` 的方式缩小音量。

## 生成 GGUF

使用 ComfyUI 的一体化检查点（`ace_step_1.5_turbo_aio.safetensors`，Comfy-Org/ace_step_1.5_ComfyUI_files）和一份
ComfyUI 源码（提供 Qwen 分词器文件和静音潜变量）：

```bash
python eng/audio-models/acestep_to_gguf.py --input ace_step_1.5_turbo_aio.safetensors --comfyui path/to/ComfyUI \
    --out ace-step-1.5-turbo.gguf
```

各部分保持发布时的精度（BF16，文本编码器为 F16）；VAE 解码器折叠权重归一化后存为 F32。仅用于参考音频的 VAE 编码器和
音频分词器的池化层不写入（共 9.4 GiB）。

## 运行

```bash
TensorSharp.Cli --model ace-step-1.5-turbo.gguf --backend cuda --prompt "upbeat lofi hip hop, mellow piano" \
    --lyrics "[Instrumental]" --duration 30 --bpm 90 --key "C major" --time-signature 4 --diffusion-seed 7 \
    --output song.wav
```

`--lyrics` 接受文本或文件，可用 `[verse]` / `[chorus]` 段落标记；`--language` 指定歌词语言（`en`、`zh` 等）。省略
`--bpm`、`--key` 或 `--time-signature` 时由规划模型决定。`--diffusion-steps` 可改变 8 步的默认值。服务器路由：

```bash
curl http://localhost:5000/v1/audio/music -H "content-type: application/json" \
    -d '{"prompt": "lofi hip hop, piano", "lyrics": "[Instrumental]", "duration": 30, "bpm": 90, "seed": 7}' -o song.wav
```

## 验证结果

与以 FP32 运行同一检查点的 ComfyUI 对比，把 ComfyUI 规划模型生成的编码交给 TensorSharp（`TS_ACESTEP_CODES`；规划模型
用 torch 的随机数生成器采样，无法复现），并使用相同种子：

| 阶段 | `--backend cpu` | `--backend cuda` |
|---|---|---|
| 风格描述编码状态 | 相对误差 3e-6 | 7e-4 |
| 歌词嵌入 | 完全相同 | 完全相同 |
| 8 步后的最终潜变量 | 相对误差 9.5e-4，相关系数 1.000000 | 1.6%，相关系数 0.99988 |
| 解码音频 | 相关系数 0.99998 | 0.9977 |

CUDA 上的差异来自与 FP32 参考对比时的 BF16 激活。规划模型第一步的条件与非条件 logits 与 ComfyUI 的相关系数为 0.9999，
前 10 个编码相同。单块 GPU 上生成 10 秒歌曲约需 15 秒（规划 5 秒、编码器 1 秒、扩散每步 0.18 秒、VAE 在 CPU 上 7 秒）；
下一步应把 VAE 移到 GPU 上。

未实现：参考音频（翻唱、局部重绘）、负向风格描述，以及 4B 规划模型版本。
