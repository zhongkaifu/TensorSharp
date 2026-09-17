# Muse-Glimmer

[English](muse-glimmer.md) | [中文](muse-glimmer_zh-cn.md)

[← 返回模型索引](README_zh-cn.md)

| 属性 | 值 |
|---|---|
| GGUF 架构标识 | `muse-glimmer` |
| 源码类 | [`MuseGlimmerModel`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.cs)（传统单序列） |
| 投机草稿模型 | [`MuseGlimmerModel.DFlash.cs`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.DFlash.cs) + [`DFlashConfig`](../../TensorSharp.Models/Speculative/DFlashConfig.cs) |
| 视觉编码器 | [`MuseGlimmerVisionEncoder`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerVisionEncoder.cs) |
| 图像预处理 | [`MuseGlimmerImageProcessor`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerImageProcessor.cs) |
| 示例模型 | Muse-Glimmer-30B |
| 模态 | 文本、图像 |
| 思维链 | 支持（聊天模板会输出 `assistant to=self` 推理通道） |
| 工具调用 | 支持（聊天模板中的 ATEM XML 标记） |
| 批处理 / 分页前向 | 不支持（传统单序列） |
| 张量并行 | 支持 —— GGML CUDA / Vulkan，最高 `--tp 2`（30B 只有 2 个 KV 头） |

## 快速开始

```bash
# 文本
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --input prompt.txt --backend ggml_cuda --max-tokens 256

# 图像理解（需要 mmproj）
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --mmproj models/mmproj-Muse-Glimmer-30B-Q8_0.gguf \
  --image photo.png --input question.txt --backend ggml_cuda --max-tokens 300

# DFlash 投机解码（无损；输出与普通贪心 decode 一致）
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --draft-model models/dflash-kquant.gguf \
  --spec-draft 15 --input prompt.txt --backend ggml_cuda
```

`--draft-model` 也可以用环境变量 `TS_MUSE_GLIMMER_DFLASH` 指定。

### 结构化输出

生成 prompt 停在 `<|start|>assistant`，因此正常回复以模型自己写的路由头开始（推理为 ` to=self<|message|>`，答案为 ` to=user<|message|>` 或 `<|message|>`）。使用 `response_format` 时 JSON 语法从第一个 token 起生效，模型会直接写出对象而没有任何头部。`MuseGlimmerOutputParser` 把不可能是头部开头的回复（JSON 的 `{`、`[`、`"` 或数字）视为答案内容，并在流结束时返回未加框架的文本而不是丢弃。此前解析器一直等待被语法排除的 `<|message|>`：流式响应返回 `content: null`，`json_schema` 在生成正确对象后返回 HTTP 422（2026-09-16 验证活动，B6）。非流式路径现在也对解析后的 content 而不是原始输出做校验。`response_format` 也可以与 `"think": true` 同时使用。模型先在 ` to=self<|message|>` 消息中推理，再以 `<|start|>assistant to=user<|message|>` 开始答案，因此协议把 `to=user<|message|>` 声明为 `ThinkingGrammarActivationTrigger`，语法在那里启用。此前该组合返回 HTTP 400（首次复测中每个 `--thinking` json 用例都是如此）。用 `validate_inference.py --thinking` 实测（json / json_schema / json_unicode，c1 与 c4，重复三次，Q4_K_XL，单张 RTX PRO 6000）：38/45，模型写出的 41 个答案头全部是 `to=user<|message|>`。7 个失败都停在 `max_tokens` 256：推理会先复述 prompt 再作答，而 Muse-Glimmer 没有声明可以提前关闭推理的思考预算结束 token。

## 1. 文本架构

52 层稠密层，`n_embd` 6656，`n_ff` 19968，32 个查询头 / 2 个 KV 头，
`head_dim` 128，词表 202048。

* **交错滑动窗口注意力。** `muse-glimmer.attention.sliding_window_pattern`
  是一个标量周期 P（30B 为 4）。当 `l % P < P - 1` 时第 `l` 层是滑动窗口层，
  因此每第 P 层（l = 3, 7, 11, … 51）是完整因果注意力 —— 30B 共 39 个 SWA 层
  + 13 个全注意力层。这与 `llama_hparams::set_swa_pattern(P)`（`dense_first = false`）
  一致。窗口判定采用 llama.cpp 的标准 SWA 规则：位置 `p0` 的 key 对位置 `p1`
  的 query 可见当且仅当 `p1 - p0 < n_swa`（2048）。
* **只有滑动窗口层使用 RoPE**，全注意力层是 NoPE。RoPE 风格是 ggml 的 NORM
  （相邻元素成对，`mode 0`），不是 NeoX —— 转换时已经把 transformers 的
  rotate_half 排布还原回来。
* **逐头 QK RMSNorm。** Q 的 norm 权重在转换时被合成，用来承载模型的
  `qk_scale_factor`；K 的 norm 权重全为 1。
* **注意力输出门控。** `attn = attn * sigmoid(W_gate @ attn_norm(x))`，
  在 `o_proj` 之前施加。门控与 Q/K/V 一样从同一个 post-norm 张量投影而来。
* **每层 4 个 RMSNorm。** `attn_norm` 与 `ffn_norm` 使用模型的
  `f_norm_rms_eps`（1e-5）；`post_attention_norm` 与 `post_ffw_norm` 使用
  **硬编码的 1e-8**（见 `llama.cpp/src/models/muse-glimmer.cpp`）。这个 epsilon
  写错会造成静默的数值偏差。
* **输入 embedding 上的无权重 RMSNorm** —— 它取代了其他 Gemma 类模型使用的
  `sqrt(hidden_size)` 缩放。
* **稠密 SwiGLU FFN**（没有 MoE）。
* **输出路径：** `logits = lm_head(h)`，然后 `logits *= logit_scale`（0.19612），
  再做 `final_logit_softcapping`（20.0）的 tanh 软上限。缩放在软上限**之前**。

## 2. 视觉塔

50 层 ViT，`n_embd` 1536，`n_ff` 8960，16 个头（`head_dim` 96），patch 14，
带 bias 的 LayerNorm，普通的 2 层线性 MLP，使用精确的 **erf** GELU（不是 tanh
近似），以及可学习的 32x32 位置 embedding。

* **预处理是纯拉伸** —— 不填充、不切块。合并 token 的网格由 llama.cpp 的
  `muse_glimmer_grid_size` 选定（在四个 floor/ceil 候选中取长宽比最接近者，
  平局时偏向更多 token，上限 4096 个合并 token），然后用与 Pillow 兼容的
  **Lanczos-3** 滤波把图像缩放到 `grid * 28` 像素，并用 mmproj 的
  `image_mean` / `image_std`（0.5 / 0.5）归一化。
* **稀疏窗口注意力。** patch 被重排进 32x32 的窗口（窗口边长为
  `sqrt(position_embd_rows)`），边缘窗口裁剪。满足 `(il + 1) % 4 == 0` 或
  `il == n_layer - 1` 的层做全局注意力，其余层只在窗口内注意。由于该重排让掩码
  在连续行区间上呈块对角，编码器按窗口逐个执行注意力，而不需要物化
  `n_tok x n_tok` 的掩码。
* **2D RoPE**（`theta` 10000）：`head_dim` 的前一半按 1-indexed 的 patch 列做
  旋转，后一半按 1-indexed 的行，两者都用成对交错（NORM）风格。
* **2x2 像素混洗采用通道在外的打包**：`out[o][c * 4 + s]`，不是 `s` 在外。
  这是经典陷阱，写反了会静默出错。
* **适配器**：6144 -> 4096 -> 4096 -> 6656，中间是 erf GELU，无 bias。

在 GGML 后端上视觉塔的 2D matmul 权重保持 GGUF 量化状态直接送入 `AddmmQuant`。
把这个塔反量化成 F32 需要约 7.4 GB，会把语言模型常驻的权重挤出显存；
`TS_MUSE_GLIMMER_VENC_F32=1` 可以恢复 F32 路径用于 A/B 对比。

### 提示词管线

聊天模板把一个图像内容段渲染成单个 `<|patch|>`。
`ChatTemplate.InjectMultimodalTokens` 负责输出它，宿主（CLI 或
`ModelMultimodalInjector`）再把每个 `<|patch|>` 展开成
`<|image_start|>` + N 个占位行 + `<|image_end|>`，其中 N 是编码器的合并 token 数。
占位行会在输入 RMSNorm **之前**被投影后的 embedding 覆盖 —— llama.cpp 也是把图像
embedding 直接喂进 `build_inp_embd`，然后对合并后的序列做 norm。

## 3. DFlash 投机解码

DFlash 是一个**块级**草稿模型：独立的 5 层 GGUF
（`general.architecture = dflash`），一次前向即提出整个投机窗口。它复用主干的
`token_embd` 与 `output`（lm_head），并维护自己的 SWA KV 环。三个阶段：

1. **编码** —— 把主干在 `dflash.target_layers`（`[2, 14, 26, 38, 50]`）处的逐层
   *输入*残差拼成每个位置一行 33280 宽的向量，经 `fc.weight` 投影后由
   `enc.output_norm` 做 RMSNorm。
2. **注入** —— 该行喂给每个草稿层的 `attn_k` / `attn_v`；key 会经过逐头 RMSNorm
   和 **NeoX** RoPE（草稿模型的 rope 风格与主干不同），位置取目标位置，然后写入
   草稿模型的环形缓存。
3. **起草** —— `[anchor, MASK x (block_size - 1)]` 在
   `[环形窗口 | 该块自身的 key]` 上非因果地过完 5 个块，再用主干的 lm_head 得到
   `block_size` 行 logits。第 0 行是 anchor 自己的预测，会被丢弃。

草稿模型的 logits **既不**乘主干的 `logit_scale`，**也不**做软上限
（llama.cpp 的 dflash 图就止于 lm_head 矩阵乘）。argmax 对两者都不变，但接受
置信度会变，因此交给执行器的逐位置置信度是原始草稿 logits 的 softmax。

验证以贪心方式对齐主干，所以输出的 token 流就是普通贪心 decode 的流
（浮点意义下的"无损"含义见[输出一致性](#输出一致性)）。

草稿模型与主干验证都作为融合的、可被 CUDA 图捕获的原生图运行；正是这一点让投机
从亏损变成了收益。与 llama.cpp 自带 DFlash 实现的最新对比 —— 我们在哪里领先、
哪里落后、以及原因 —— 见[第 5 节 性能](#5-性能)。

## 4. 与 llama.cpp 的对齐

`InferenceWeb.Tests/MuseGlimmerParityTests.cs` 用运行同一批 GGUF 的
`llama-server` 采集的黄金输出来校验实现（`.parity/gen_ref.py`、
`.parity/gen_ref_vision.py`）：

| 检查项 | 结果 |
|---|---|
| 分词器（5 个提示，`add_special`） | token 完全一致 |
| 贪心续写（5 个提示，每个 26-32 token） | token 完全一致 |
| 长上下文（4651 token 提示，为滑动窗口的 2.3 倍） | token 完全一致 —— 覆盖 SWA 掩码、填充窗口的翻转与 KV 缓存扩容 |
| DFlash 贪心续写 | 与非投机路径以及 llama.cpp 均 token 一致 |
| 图像几何（`ComputeTargetSize` / `ComputeTokenCount`） | 与 `muse_glimmer_grid_size` 一致；1024x1024 -> 1036x1036 -> 1369 token，336x336 -> 144 token（已与 llama-server 的 `prompt_tokens` 核对） |
| 图像描述 | 近乎逐字一致，包括 OCR 出来的叠加文字 |

重新生成黄金数据：

```bash
llama-server -m Muse-Glimmer-30B-UD-IQ2_XXS.gguf --mmproj mmproj-Muse-Glimmer-30B-Q8_0.gguf -ngl 99 --port 8899
python .parity/gen_ref.py http://127.0.0.1:8899 .parity/ref_text.json
python .parity/gen_ref_vision.py http://127.0.0.1:8899 .parity/ref_vision.json
```

然后 `TS_TEST_MODEL_DIR=<模型目录> dotnet test --filter MuseGlimmerParityTests`。

## 5. 性能

2026-08-13 重新测量。本节替换了此前在 16 GB 笔记本 GPU 上用 2-bit 量化得到的表格，
那批数字一个都没有保留。后面的工程小节仍保留其原始 A/B 数据，因为它们记录的是
某项改动**为什么**要做 —— 每处都注明了测量机器。

### 测试环境

| | |
|---|---|
| GPU | 1x **NVIDIA RTX PRO 6000 Blackwell Server Edition**（97,887 MiB），驱动 580.126.20，PCIe 5.0 x16。机器上有两张；除[双 GPU](#双-gpu)一节外所有行都固定 `CUDA_VISIBLE_DEVICES=0` |
| CPU / 内存 | 2x Intel Xeon 6952P（384 线程），1.5 TiB |
| 模型 | `Muse-Glimmer-30B-Q8_0.gguf`（27.6 GiB） |
| 草稿模型 | `dflash-kquant.gguf`（1.5 GiB） |
| TensorSharp | commit `5098e3f`，内置 ggml `8846b79`（2026-08-12），`--backend ggml_cuda`，原生库以 `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real` 构建 |
| llama.cpp | master `8e7f22b`（2026-08-13，libggml 0.19.0 —— 与内置 ggml 相差一天以内），相同 CUDA 架构，`-DGGML_CUDA=ON -DLLAMA_CURL=OFF` |
| 采样 | 两侧都是贪心（llama.cpp 用 `--temp 0`；TensorSharp **不传**任何采样参数） |
| 生成长度 | 128 token |
| 批大小 | llama.cpp `-b 2048 -ub 2048`，与 TensorSharp 默认的 `TS_MUSE_GLIMMER_PREFILL_CHUNK` = 2048 对齐 |
| 重复次数 | 每个点 2 次，**同一上下文内两个引擎交替执行** |

**两个引擎 prefill 的是同一串 token。** TensorSharp 会套用聊天模板而 llama.cpp 的
`-no-cnv -f` 不会，因此若都喂原始问题，就变成 60 token 的提示对 21 token 的提示。
做法是先用 `TensorSharp.Cli --dump-prompt` 导出渲染后的提示词，把**那段文本**交给
`llama-cli`；再用 `llama-tokenize` 确认 token 数与 TensorSharp 报告的一致
（60 / 501 / 2050 / 16126 / 32274 / 64575 / 123931）。

### 纯文本生成

两次重复的均值，tok/s。比值列是 TensorSharp / llama.cpp，大于 1.00x 表示
TensorSharp 领先。

| 提示 token 数 | llama.cpp prefill | TS prefill | 比值 | llama.cpp decode | TS decode | 比值 |
|---|---:|---:|---:|---:|---:|---:|
| 60 | 362 | **459** | 1.27x | 34.7 | **35.0** | 1.01x |
| 501 | 927 | **1135** | 1.23x | **36.2** | 34.3 | 0.95x |
| 2050 | 1132 | **1317** | 1.16x | **35.0** | 33.5 | 0.96x |
| 16126 | **1325** | 1249 | 0.94x | **32.2** | 30.9 | 0.96x |
| 32274 | **1303** | 1211 | 0.93x | **32.1** | 29.9 | 0.93x |
| 64575 | **1256** | 1150 | 0.92x | **32.4** | 29.1 | 0.90x |
| 123931 | **1166** | 1073 | 0.92x | **30.7** | 26.6 | 0.86x |

整条阶梯的形状是一致的：TensorSharp 的融合整模型图在短提示上领先 1.16-1.27x，
两个引擎在 2K 到 16K 之间交叉，再往上 llama.cpp 保持 6-8% 的 prefill 优势。
decode 在 60 token 上下文时打平，到 128K 降到 0.86x —— 差距随 KV 长度增大，
指向注意力路径而不是 FFN。

两个引擎都能在单卡上跑满 128K 上下文。

> **关于四行长上下文的说明。** 60 / 501 / 2050 三个提示对两个引擎逐字节一致。
> 16126 / 32274 / 64575 / 123931 四行是在[基准方法说明](#基准方法说明)中提到的
> CRLF 归一化之前测的，因此这四个点上 TensorSharp prefill 的是同一文档的 CRLF
> 形式 —— 同样的文本，多出 1.2% 的 token（16322 / 32666 / 65359 / 125412）。
> 吞吐是速率，对 tok/s 影响很小，但这四行上两侧生成的续写并不严格可比。
> 修正后的重跑已排队，但基准机在跑完之前下线了。

### DFlash 投机解码

同一批运行加上 `--draft-model dflash-kquant.gguf --spec-draft 15`，
对手是 llama.cpp 的 `-md … --spec-type draft-dflash --spec-draft-n-max 15 -ngld 99`。
单位是 decode tok/s；括号内是两次重复的范围（仅在差距大时给出）。

| 提示 token 数 | llama.cpp | TensorSharp | TS，`--spec-pmin 0` |
|---|---:|---:|---:|
| 60 | 45.5 | **50.9** | 43.5 |
| 501 | 117.5 | 164.6（150-179） | **180.3** |
| 2050 | 24.9 | **43.5**（30-57） | 34.7 |
| 16126 | **80.2** | 55.8（37-75） | 33.2 |
| 32274 | **60.7**（43-79） | 33.8（31-36） | 29.9 |
| 64575 | **66.1** | 48.7（34-64） | 49.1 |
| 123931 | **69.0** | 42.3（30-55） | 59.8 |

投机在两个引擎上都要付出 *prefill* 代价，因为草稿模型的编码器也要过一遍提示词：

| 提示 token 数 | llama.cpp 普通 → DFlash | TensorSharp 普通 → DFlash |
|---|---:|---:|
| 60 | 362 → 203（0.56x） | 459 → 341（0.74x） |
| 501 | 927 → 495（0.53x） | 1135 → 700（0.62x） |
| 2050 | 1132 → 259（0.23x） | 1317 → 703（0.53x） |
| 16126 | 1325 → 988（0.75x） | 1249 → 826（0.66x） |
| 64575 | 1256 → 985（0.78x） | 1150 → 780（0.68x） |
| 123931 | 1166 → 920（0.79x） | 1073 → 742（0.69x） |

#### 为什么 TensorSharp 那一列是区间而 llama.cpp 不是

TensorSharp 在草稿模型前面放了一个**自适应成本调控器**
（[`SpeculativeExecution`](../../TensorSharp.Runtime/Speculative/SpeculativeExecution.cs)
的 `AdaptiveSpeculation`，默认开启）。投机只是速度优化，因此执行器会分别测量
带起草与不带起草的 ms/token，一旦起草更慢就把草稿模型**暂停**
`ParkedProbeInterval = 64` 步再重新探测。llama.cpp 没有这套机制 —— 它每步都起草。

在只生成 128 token 的运行里，一次错误的探测就会毁掉一半测量。下面两次 16K
重复用的是同一个提示、同一个二进制、同一份权重，唯一的差别是调控器的判定：

| 16K 第几次 | 起草 / 接受 | 验证步数 | 暂停步数 | decode |
|---|---|---:|---:|---:|
| 1 | 64 / 48（75%） | 13 | **67** | 36.7 tok/s |
| 2 | 132 / 103（78%） | 22 | 3 | **74.8 tok/s** |

第 1 次在*暂停之后*的重新探测里测到投机是 **14.0 ms/token，而普通解码是 37.8** ——
快 2.7 倍 —— 说明当初把它暂停的判定是错的。prefill 之后最初几步投机要为验证批的
形状付一次性的建图开销，而探测采样到的正是这几步。32K、64K、128K 上被暂停的那些
重复都有同样的指纹（`drafted` 卡在 64-84：一个探测窗口，然后就没有了）。

若以未被暂停的那次作为稳态，TensorSharp 的融合 DFlash 达到
**16K 时 llama.cpp 的 94%（74.8 对 79.8）、64K 时 96%（63.6 对 66.5）** ——
而不是本文档旧版本在另一套硬件上报告的落后 1.6-2.4 倍。修复方式很明确：
在探测之前先预热验证形状，或者把第一步投机从采样中剔除；目前尚未实现。

#### 置信度下限

TensorSharp 默认 `confMin = 0.35`；llama.cpp 这条路径上的 `p_min` 默认是 0，
也就是永远起草满窗口。`--spec-pmin 0` 能让两侧策略可比，但它**并非**
一律更好：在 501 与 128K 上它赢（180 对 165、60 对 42），在 16K 与 32K 上输得很惨
（33 对 56、30 对 34）—— 那里接受率从约 75% 掉到 24-42%，而每个被拒绝的行仍然
占用一个验证槽位。这个下限应该是自适应的，而不是常数 —— 与更早的笔记本测试得到
的结论一致。

#### 2K 处的异常属于 llama.cpp

在 2050 token 的提示上，llama.cpp 的 DFlash decode（24.9 tok/s，两次都是）
*低于*它自己的普通 decode（35.0），而它的 DFlash prefill 从 1132 塌到 259 tok/s ——
4.4 倍的代价，远差于它在 16K 以上付出的 0.75-0.79x。TensorSharp 在同一点付出
0.53x，decode 为 43.5。这个提示没什么特别，只是它停在文档中间，因此续写比其他
尺寸上"问题 + 回答"式的提示更难预测。

### 显存峰值

单卡整进程峰值，每 2 秒采样一次（MiB）：

| 提示 token 数 | llama.cpp 普通 | TS 普通 | llama.cpp DFlash | TS DFlash |
|---|---:|---:|---:|---:|
| 501 | 28329 | 29655 | 31881 | 29887 |
| 16126 | 28585 | 30567 | 32191 | 34089 |
| 64575 | 29401 | 32003 | 33007 | 35809 |
| 123931 | 30471 | 33787 | 34641 | 37769 |

普通路径上 TensorSharp 比 llama.cpp 多占 1.3-3.3 GB，加载草稿模型后大约多 3 GB。
两者在 96 GB 卡上跑 128K 都绰绰有余；在 40 GB 卡上，128K + DFlash 是第一个装不下
的组合。

### 输出一致性

贪心验证在*精确算术*下让 DFlash 无损 —— 验证批会用主干重新给每个起草行打分，
只保留主干自己也会产生的前缀。但在浮点下，验证 GEMM 的形状与单行 decode 的 GEMM
不同，logits 的末位不同，接近平局的位置就可能翻转。这批运行的实测：

* TensorSharp 是**确定性的**：每个配置与自己的重复运行逐字节一致。
* TensorSharp 普通 vs TensorSharp DFlash：在 60 / 501 / 2050 / 16126 上完全一致，
  在 32274 上分叉。
* llama.cpp 普通 vs llama.cpp DFlash：除 2050 外处处一致。

两个引擎在同一语料上表现出同样的行为，因此这是平局翻转，而不是验证 bug。
跨引擎对比时，两条续写在前 127-636 个字符内一致，之后分开 —— 同样的权重上不同的
内核与不同的归约顺序，这是预期结果。

### 基准方法说明

有五件事对数字的影响大于被测效应本身，因此记录在此，免得重新踩：

* **交替执行两个引擎。** 先跑完一个引擎的整条阶梯再跑另一个，会让后者偏低。
  这里每个上下文都是 llama.cpp → TensorSharp → llama.cpp DFlash → TensorSharp DFlash。
* **给 llama.cpp 套好模板的提示词**（见上）。在 60 token 那一点，差别是 3 倍的
  提示长度。
* **先统一换行符。** 长提示文件原本是 CRLF。把 LF 版本交给 llama.cpp 后，
  TensorSharp prefill 了 16322 个 token 而 llama.cpp 只有 16126 —— 同样的文本、
  多 1.2% 的 token、不同的续写。在相信任何长上下文数字之前，先比对两侧报告的
  提示 token 数。
* **做投机对比时，两侧都用贪心。** CLI 从 `SamplingConfig.Greedy` 起步，这正
  对应 llama.cpp 的 `--temp 0`，所以一个采样参数都不要传。投机本身如今可与采样
  组合（验证会用本次运行自己的采样器抽取每一行），但块级草稿器的提案不带惩罚项，
  接受率会随之变化——那样得到的就不是一次干净的对照。检查日志里有
  `cli.inference speculative:` 一行，确认投机确实启用了。
* **128 个生成 token 对投机对比来说太短。** 它比两个暂停区间还短，一次调控器误判
  就能让数字差 2 倍（见上面的 16K 两次重复）。

整个测试期间 GPU 一直报告 `HW Power Brake Slowdown: Active`，频率 2280-2347 MHz，
功耗 180-270 W（上限 450 W），温度 28-42 C —— 这是主机层面的功率制动，不是温度
降频，而且对两个引擎一视同仁。大约每二十次运行会有一次在 prefill 和 decode 上
同时慢约 40%，且没有任何频率或温度上的痕迹（llama.cpp 32K DFlash 第 2 次最明显：
603 / 42.8 对 1017 / 78.6，而两次运行逐字节一致）。在这台机器上，对单次重复的
差值要保持怀疑。

### 语料比引擎更影响数字

长提示来自 `.parity/gen_long_prompts.py`，它生成的是高度重复的合成文档
（"Chapter *n*. The *n*th study …"），提出的问题的答案几乎逐字引用某一章。
在这种文本上起草几乎全中 —— 501 token 那一点在**两个引擎上都是 100% 接受率**，
所以它的 DFlash 数字（117-180 tok/s）是普通 decode 的 3-5 倍，不应被当作通用的
投机加速比。自然文本上的接受率更接近 16K-128K 各行显示的 55-78%。用这些行来
比较引擎，但不要把 DFlash 的绝对值当成聊天负载会看到的数字。

### 逐算子路径为什么慢

瓶颈不是算术，是调度。逐算子前向在 52 层上提交约 600 个 GGML 算子，每个都带一次
主机可见的往返。在 RTX 3080 Laptop 上实测：1 行前向约 262 ms，74 行前向约
1332 ms，也就是每次前向约 262 ms 的*固定*开销加每行约 14 ms。本仓库里所有达到
llama.cpp 级 decode 吞吐的模型都是靠同一个办法 —— 一个整模型内核。

### 融合内核

[`ggml_ops_muse_glimmer.cpp`](../../TensorSharp.GGML.Native/ggml_ops_muse_glimmer.cpp)
导出 `TSGgml_MuseGlimmerModelForward`，把整个模型 —— 全部 52 层、最终 norm、
LM head、logit 缩放与 tanh 软上限 —— 构建成**一张** ggml 图。它在数值上与
`TransformerBlock` 是同一条链：相同的算子顺序、相同的 epsilon、同样只在滑动窗口层
上使用 NORM 风格 RoPE、同样的注意力输出门控。`TS_MUSE_GLIMMER_FUSED=0` 可强制
走逐算子路径做 A/B。

真正起作用的设计点：

* **持久、可捕获的图。** 图只构建一次且张量地址稳定（用裸 `ggml_init` +
  `ggml_backend_alloc_ctx_tensors`，而不是会按生命周期重排地址的 gallocr），
  ggml-cuda 因此可以捕获它，一个 token 就是一次重放。图的拓扑在步与步之间保持
  逐字节一致：KV 用 `ggml_set_rows` 写入（写入行号是一个 I64 *输入*），读取的是
  按 256 行对齐填充的窗口，配一个 F16 掩码输入 —— 于是每 256 个 token 才重建一次，
  而不是每个 token 都重建。
* **图池的键是 `(模型, KV 持有者, n_tokens)`。** 投机运行会交替 1 行 decode 与
  k 行验证批，k 还会变；若键里不含 `n_tokens`，这些形状会互相把对方挤出同一个
  槽位，每一步都要完整重建。
* **prefill 用共享 gallocr，decode 不用。** prefill 分块的中间结果按行增长 ——
  仅 `[2*n_ff, n_tokens]` 的 gate/up 在 1024 行时每层就是 163 MB —— 给每个节点
  单独分配需要约 47 GB。`alloc_graph_reuse_gallocr` 按生命周期打包，这才让多行
  融合图装得下。
* **KV 缓存在分配时清零。** `ModelBase.InitializeCacheTensor` 在 GgmlCuda 上跳过
  清零，因为逐算子注意力只读自己写过的行。融合内核读的是*填充后*的窗口，多出来的
  行用 `-inf` 掩掉 —— 而 `-inf + NaN` 仍是 NaN，所以那些行必须是有限值。
* **仅 CUDA / Vulkan。** GgmlCpu 保留逐算子路径：1024 行的融合图在那里会崩，
  提供优雅回退好过发布一个崩溃。
* **图内 embedding 是可选项。** 内核可以自己做 embedding gather 与无权重输入 norm，
  但绑定 202K x 6656 的表会额外钉住约 1.1 GB 张量；在 16 GB 卡上这会挤掉层权重，
  代价超过省下的两次调度（18.5 -> 16.1 tok/s）。因此只有当 LM head 与该表绑定
  （本来就常驻）或显式设置 `TS_MUSE_GLIMMER_INGRAPH_EMBED=1` 时才启用。

### 融合的 DFlash 草稿模型

草稿模型曾经是投机亏损的根源：每个投机步，托管实现要为约 2.5 GB 的权重读取发出
约 150 次 GPU 调度 —— 大约是一次目标前向的四分之一 —— 耗时约 100 ms。
[`ggml_ops_dflash.cpp`](../../TensorSharp.GGML.Native/ggml_ops_dflash.cpp)
把它变成两张图：

* `TSGgml_DFlashInject` —— `fc` -> RMSNorm -> 每个草稿层
  {k/v 投影、逐头 k norm、NeoX RoPE、`ggml_set_rows` 写环}。
  没有 Q、没有注意力、没有 FFN、没有 LM head。llama.cpp 的 `build_dflash`
  也在同一点提前返回。
* `TSGgml_DFlashDraftBlock` —— `[anchor, MASK x (b-1)]` 过完 5 个草稿块，
  然后是*主干的* LM head（借用，绝不复制）和一次 softmax。

两者都是持久且可捕获的。这里 TensorSharp 与 llama.cpp 分道扬镳：后者在草稿上下文
上**得不到**任何图复用 —— `gf_res_prev` 是 encode 与 decode 共用的一个槽位，而
DFlash 的循环 ENCODER -> DECODER(embd) -> DECODER(token) 永远无法满足 `can_reuse`，
所以这三次调用每步都要重建。llama.cpp 把重建做得很便宜；融合则是彻底避免它。

有两个细节让捕获成为可能：

* **注意力读整个环**，`kv_len = ring_rows + b`，跨步固定。注意力在 KV 轴上是
  置换不变的，所以环的循环顺序无关紧要，而一个按槽位位置映射构建的主机侧掩码
  正好表达了哪些槽位是活的。
* **掩码的截断点是 ANCHOR 的位置，不是 query 的位置。** 部分拒绝之后，环里仍然
  留着草稿模型为 anchor 之后的位置写入的 key；那些行已经过期，块内任何 query 都
  不得看到它们，哪怕自身位置更靠后的那些也不行。逐算子路径天然没有这个问题，
  因为它只会去取 `[winStart, anchor)`。

**设备端 top-1。** llama.cpp 每个起草步都把整块 `[202048, 16]` 概率拉回主机 ——
12.9 MB 过 PCIe 外加一次 320 万元素的扫描，`common_sampler` 随后还要*为块内每个
位置*物化一个 202048 项的数组。内核则以 `ggml_argmax` 加一次 `get_rows` 取回胜出
概率收尾，只返回两个 16 元素张量。argmax 在 softmax 下不变，而胜出概率正是执行器
要累乘的那个置信度。

在做这项工作的 RTX 3080 Laptop 上，融合草稿模型把 decode 从 15.8-16.4 提到
26.0-27.0 tok/s，而起草统计（`drafted=96 accepted=75`，78.1% 接受率）与逐算子
草稿模型逐字节一致 —— 融合改变的是速度，不是任何一个采样出来的 token。
`TS_DFLASH_FUSED=0` 可强制使用逐算子草稿模型做 A/B。

### 每类注意力一张掩码，而不是每层一张

这就是长上下文 prefill 差距的全部原因。

内核过去在逐层循环**内部**分配 F16 注意力掩码，于是一张图携带 52 个掩码张量 ——
以及 52 次主机侧拷贝 —— 尽管其内容只取决于 `(window, n_tokens, start_pos)`，
而所有滑动窗口层共享一个窗口、所有全注意力层共享另一个。真正不同的缓冲区从来
只有两个；decode 重放路径早就证明了这一点：它只构建两个，然后循环上传到全部 52 个
张量里。

在 64K、分块 2048 时，这是 13 x [65536, 2048] + 39 x [4352, 2048] 的 F16 =
**每个分块 3.90 GB 的掩码**，而卡上只有 16 GB 且已被 9.2 GB 权重占用。比常驻更糟的
是，这些掩码全部由*主机*用逐元素的标量循环生成再推过 PCIe —— 而且在 prefill 路径上
走的是同步的 `ggml_backend_tensor_set`，每次拷贝后都要完整 `cudaStreamSynchronize`：
每个分块 52 次停顿，一个 64K 提示就是 1664 次。整个 64K prefill 累计生成并上传了
约 75 GB 的掩码。

llama.cpp 从来没有这个问题：`build_attn_inp_kv_iswa` 每张图只创建两个掩码
（`llama-graph.cpp:3266,3276`），逐层选择只是一次指针挑选（`:3042`）。

共享之后 3.90 GB 变成 273 MB，每个分块两次上传。在 RTX 3080 Laptop 的 64K 上实测：
**prefill 353 -> 486 tok/s，显存峰值 16059 -> 12671 MiB。** 共享是安全的，因为
没有人写掩码 —— `ggml_flash_attn_ext` 把它当 `src[3]`、`ggml_soft_max_ext` 当
`src[1]`，两个 CUDA 内核都以 `const` 绑定。

同时落地的还有三件小事：

* **prefill 路径上掩码改为在 GPU 上生成**，直接写进设备缓冲区，彻底去掉主机填充与
  H2D 上传。因果掩码内核本来就有（`ggml_ops_mask.cu`，为 Gemma 4 验证内核写的）；
  滑动窗口层需要新增 `tsg_cuda_fill_ring_mask_f16`，因为环的列不按位置顺序排列。
  decode 仍走 `decode_input_set_async`，那是 CUDA 捕获安全的路径，而且单行本来就
  很便宜。
* **SwiGLU 变成单个 `GGML_OP_GLU` 节点**，取代 view + 2x `cont` + `silu` + `mul`。
  这样每层少了三个 `[n_ff, n_tokens]` 临时量 —— 每行每层约 479 KB 流量，
  每个 decode token 约 156 次调度。注意 `ggml.h` 对 `ggml_glu` 的注释说门控是
  后一半，这与内核矛盾；以内核为准（`swapped=false` 对*前*一半施加 SiLU），
  parity 测试也证实了这一点。
* **K/V 写入路径上的两次 `ggml_cont` 去掉了。** `ggml_set_rows` 只断言
  `ggml_is_contiguous_rows` 而不是完全连续，而 `0,2,1,3` 的 permute 已经满足 ——
  那两次拷贝是纯开销，每个 token 104 次内核。

已经试过并否决的做法，不要再试：**用更小的 prefill 分块来缩小 SWA 环只会更糟。**
64K 上分块 2048（环 4352）是 475.6 prefill / 16.1 decode；分块 1024（环 3328）是
443.2 / 15.7；分块 512（环 2816）是 421.9 / 15.1（均为 RTX 3080 Laptop、IQ2_XXS）。
decode 在这里不是 KV 带宽瓶颈 —— 两个引擎都只跑到峰值带宽的约 35-39%，因为
IQ2_XXS 的 matvec 是 ALU 瓶颈 —— 所以更小的环没有收益，而更小的分块会损失 GEMM
效率。llama.cpp 的环更小（2560 行）只是因为它默认 `n_ubatch` 是 512，这不是值得
照搬的优势。

### 长提示：分块与 SWA 环

在 16K+ 的提示能跑起来（更别说跑得快）之前，有两件事必须先改。

**prefill 是分块的。** `ForwardCore` 把长于 `TS_MUSE_GLIMMER_PREFILL_CHUNK`
（默认 2048）的输入切块，正如 llama.cpp 在 `n_ubatch` 处切分。不切的话，
16336 token 的提示会构成一张激活量随行数增长的图，在融合与逐算子两条路径上都会
分配失败 —— 16K token 的 KV 不到 1 GB，但一张 16K 行的图需要几十 GB。多模态提示
永远不分块：视觉行按绝对偏移注入，待处理列表要一次性排空。

**滑动窗口层有自己的小环。** 52 层里有 39 层永远不会回看超过 `n_swa`（2048），
按完整上下文给它们分配缓存等于浪费掉绝大部分。64K 时统一的 F16 KV 是 3.5 GB；
llama.cpp 只分配 954 MB，因为 `llama_kv_cache_iswa` 把 SWA 缓存定为
`min(n_ctx, n_swa + n_ubatch)`。在一张已经装着 9.2 GB 权重的 16 GB 卡上，这个差值
就是全部余量 —— 一次 64K 纯文本运行峰值达到 **16384 中的 16059 MiB**，并因内存压力
损失吞吐。

TensorSharp 现在给 SWA 层分配 `pad(n_swa + chunk + 1, 256)` 行（默认分块下是
4352），并按 `position % rows` 索引。64K 时这是 **1049 MB 而不是 3.5 GB
（统一缓存的 29%）**，在笔记本卡上换来 64K 处 +56% 的 prefill（226 -> 353 tok/s）
与 +15% 的 decode（13.4 -> 15.4 tok/s）。

值得知道的细节：

* **内核读整个环，而不是一个子区间。** 环的槽位不按位置顺序排列，因此没有可以
  收窄的连续窗口。读满 `rows` 行让图的形状*固定*，这也是 CUDA 捕获能成立的原因；
  `fill_mg_ring_mask` 负责承载存活性、因果性与窗口。因此 SWA 层多了一个 I64
  写入索引输入（`kv_index_swa`），保存 `position % rows`，而全注意力层仍用原始
  `position`。
* **那个 `+1` 是有意义的。** 恰好取 `n_swa + chunk`（4096 行）时，一个 4651 token
  的提示在*第一个 decode 步*就与 llama.cpp 分叉；4352 行是能精确复现 llama.cpp 的
  最小尺寸。这与 `DFlashConfig.RingRows` 已经使用的余量（`n_swa + block_size + 1`）
  相同。
* **只有在融合内核可用时环才会启用**，因为逐算子注意力把缓存行当作绝对位置。
  如果环已启用而融合前向拒绝执行，逐算子路径会抛异常，而不是悄悄返回错误 logits。
  `TS_MUSE_GLIMMER_SWA_RING=0` 恢复统一尺寸。
* **宽于 `rows - n_swa` 的前向会被拒绝。** 分块从构造上保证文本提示不会超，
  但多模态提示故意不分块（视觉行按绝对偏移注入），因此它是唯一可能提交超宽批次的
  路径 —— 那会把两个活跃位置映射到同一个环槽位，静默污染全部 39 个滑动窗口层。
  `ForwardCore` 选择抛异常。

托管侧同时应用的改动：

* 逐算子 **prefill** 路径上把 `ffn_norm` + gate/up + SiLU-mul + down 合进一张 GGML
  图（`TryFusedDenseFFNProject`；prefill +53-76%，单行 decode -9%，因此只用于
  prefill）。这现在是回退路径。
* 视觉塔保持量化（1.94 GB 而不是 7.4 GB）。
* 逐算子 decode 注意力支持 F32、F16 与块量化 KV 缓存，因此
  `ApplyModelAlignedKvCacheDefault` 选择的 F16 会被尊重。

### CUDA 上填充后的 KV 窗口必须物化

ggml-cuda 的 flash-attention **vec** 内核 —— 也就是
`ggml_cuda_get_best_fattn_kernel` 为单行 query（即每个 decode 步）选中的那个 ——
在 K/V 是"某个更长轴的*截断前缀*"视图时会返回错误结果。扁平缓存上的填充注意力窗口
正是这种形状：一个全注意力层从 `cache_size` 行的张量里读 `[0, window_full)` 行，
于是 KV 头的 stride 会跨过未读的尾部。

一旦看张量就会发现问题一点都不隐晦：在第一个全注意力层（第 3 层），
**共享同一个 KV 头的 16 个 query 头返回了完全相同的输出向量**，绝对误差最大 4.6。
用内核收到的那些 `q`/`k`/`v`/`mask` 张量在主机上重算注意力，能复现连续版本的结果
到 2e-6 —— 输入是对的，内核是错的。误差随后在剩下 49 层里累积成另一条 token 流：
模型不再复述提示词，而是开始结巴（`请详细介绍详细介绍最终最终幻想幻想7`）。

有三件事把它藏住了：

* **滑动窗口层读的是整个环**（`window == rows`），它们的视图本来就是连续的 ——
  只有 13 个全注意力层受影响；
* **prefill 没问题**，因为多于一行 query 会选中 MMA 内核，那个内核确实尊重 stride。
  prefill 的 logits 与逐算子路径吻合到 4e-4 并选出同一个 top-1 token，所以分叉只在
  第一个 decode 步之后才显现；
* **Metal 与逐算子路径都是对的**，这让融合 CUDA 路径显得像个异类，而原因其实与这个
  内核毫无关系。

在窗口上加一次 `ggml_cont` 就能修好。判定条件必须是"窗口是缓存的一个子区间"，
**而不是** `ggml_is_contiguous(k_full)`：`ggml_is_contiguous_n` 会跳过 `ne` 为 1 的
维度，因此在只有一个 KV 头时（`--tp 2` 下每个 rank 正是如此），一个被截断的窗口会
自称连续，守卫就把坏形状放了过去。在判定停止咨询 `ggml_is_contiguous` 之前，
`--tp 2` 会复现出一模一样的错误 token 流。

代价测不出来：50 token 提示上 decode 40.5 → 40.5 tok/s、prefill 465 → 465 tok/s；
12371 token 提示上 prefill 1159 → 1197、decode 38.2 → 37.7（2x RTX PRO 4000
Blackwell）。只有窗口小于缓存时才会走这次拷贝，而那也正是它便宜的时候。

## 6. 张量并行

`--tp 2` 在 GGML CUDA / Vulkan 后端上把模型切到两张 GPU。对 30B 来说 2 就是上限：
它只有 **2 个 KV 头**，而本仓库中没有任何模型会在 `num_kv_heads < tp` 时复制 KV 头。

| 权重 | 切分方式 |
|---|---|
| `attn_q` / `attn_k` / `attn_v` / `attn_gate` | 列并行（按头） |
| `ffn_gate_up` | 列并行，**按段** —— 融合后的 `[gate\|up]` 的两半各自独立切分 |
| `attn_output` / `ffn_down` | 行并行 → AllReduce |
| 每层的四个 norm、`attn_q_norm`、`attn_k_norm` | 复制（QK norm 是逐头的 `[headDim]` 向量；Q norm 还携带折叠进去的 `qk_scale_factor`） |
| `output_norm`、`token_embd`、`output` | 复制；尾部在 rank 0 上执行 |

注意力输出门控与 Q 一起列并行，并在行并行 `o_proj` **之前**、在每个 rank 的区域
**内部**施加。两个 AllReduce 点都落在原始矩阵乘的输出上，也就是 1e-8 post-norm
**之前** —— RMSNorm 是非线性的，在它之后归约会产生看似通顺但错误的输出。
52 层合计每步 104 次归约。

`TSGgml_MuseGlimmerModelForward` 接收一对 `tp_degree` / `tp_plan_out`：在 TP 模式下
它为每个 rank 构建图并返回 `TpRankPlan` 而不是直接执行，由驱动在段边界带着集合通信
运行所有 rank。逐算子的 TP 路径作为回退存在，但慢得多（方案文档实测逐算子 TP 只有
单卡的 0.01–0.04 倍）。

DFlash 投机解码与分页 KV 块快照只走单卡路径；`--tp` 下的多轮复用来自活跃缓存续接。

### 双 GPU

在 **2x RTX PRO 4000 Blackwell 24 GB（PCIe）** 上测量 —— 与上面单卡表格不是同一台
机器，且未在 2026-08-13 这一轮中重测。prefill 512 / decode 64，三次取最好：

| 模型 | | prefill tok/s | decode tok/s | GPU 0 | GPU 1 |
|---|---|---|---|---|---|
| 30B-UD-IQ2_XXS（10.2 GB） | `--tp 1` | 1171 | 40.2 | 9178 MB | — |
| 30B-UD-IQ2_XXS | `--tp 2` | **1569**（1.34×） | **63.2**（1.57×） | 5115 MB | 4063 MB |
| 30B-Q8_0（28.2 GB） | `--tp 2` | 1691 | 34.3 | 15474 MB | 12748 MB |

Q8_0 没有单卡行：28.2 GB 权重在一张 24 GB 卡上根本装不下，`--tp 2` 是唯一能跑起来
的方式。

**正确性。** `--tp 2` 在重复运行间逐字节一致（排除了 rank 工作池里的竞态），
并且与 `--tp 1` 的贪心续写在前 468 / 500 个字符上一致，之后在一个改述点分叉，
两条续写都通顺 —— 这是行并行部分和以不同顺序求和的预期结果。

## 7. 环境变量

| 变量 | 作用 |
|---|---|
| `TS_MUSE_GLIMMER_FUSED` | `0` = 关闭融合整模型内核（逐算子 A/B） |
| `TS_MUSE_GLIMMER_PERSIST` | `0` = 关闭持久 / 可捕获图，每次调用重建 |
| `TS_MUSE_GLIMMER_INGRAPH_EMBED` | `1` = 即使 LM head 未绑定也强制启用图内 embedding 阶段 |
| `TS_MUSE_GLIMMER_DFLASH` | DFlash 草稿模型 GGUF 路径（等同 `--draft-model`） |
| `TS_MUSE_GLIMMER_VENC_F32` | `1` = 把视觉塔反量化为 F32（A/B；约 7.4 GB） |
| `TS_MUSE_GLIMMER_VENC_FUSED` | `0` = 关闭 CUDA 融合视觉块 / flash-attention 路径（诊断用回退） |
| `TS_MUSE_GLIMMER_GELU_TANH` | `1` = 视觉塔改用 tanh GELU 近似而非精确 erf |
| `TS_MUSE_GLIMMER_VENC_TRACE` | `1` = 打印视觉残差流的逐阶段校验和 |
| `TS_MUSE_GLIMMER_LAYER_TRACE` | `1` = 打印*进入*每一层的残差校验和。融合内核与逐算子循环都会输出，因此把融合运行与 `TS_MUSE_GLIMMER_FUSED=0` 运行做 diff 就能把分叉定位到某一层 |
| `TS_MUSE_GLIMMER_LAYER_TRACE_POS` | 从该绝对位置起追踪第一次前向（跳过启动预热，其 prefill 与 decode 位置无关） |
| `TS_MUSE_GLIMMER_LAYER_TRACE_N` | 从那里开始连续追踪多少次前向（默认 1） |
| `TS_MUSE_GLIMMER_LAYER_TRACE_DIR` | 同时把每次追踪的残差以原始 F32 写到 `<dir>/{fused,perop}_S<step>_L<layer>.bin`，用于逐元素 diff —— 只看校验和会漏掉整体很小的分叉 |
| `TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS` | MLX 惰性图的 flush 间隔（默认 4，`0` 关闭） |
| `TS_PREFILL_CHUNK` | `ForwardRefill` 的提示分块大小（默认 2048） |
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | 每次 prefill 前向的 token 数（默认 2048，`0` 关闭分块） |
| `TS_MUSE_GLIMMER_SWA_RING` | `0` = 所有层都按完整上下文分配，而不是给 SWA 层用环 |
| `TS_MUSE_GLIMMER_SWA_ROWS` | 覆盖 SWA 环的行数（诊断用） |
| `TS_DFLASH_FUSED` | `0` = 关闭融合 DFlash 草稿模型（逐算子 A/B） |
| `TS_DFLASH_PERSIST` | `0` = 每步重建 DFlash 图，而不是重放 |
