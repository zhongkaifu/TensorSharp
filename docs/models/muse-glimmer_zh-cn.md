# Muse-Glimmer

[English](muse-glimmer.md) | [中文](muse-glimmer_zh-cn.md)

[← 返回模型索引](README_zh-cn.md)

| 属性 | 值 |
|---|---|
| GGUF 架构标识 | `muse-glimmer`（也接受 `muse_glimmer`） |
| 源码类 | [`MuseGlimmerModel`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.cs)（传统单序列） |
| 投机草稿模型 | [`MuseGlimmerModel.DFlash.cs`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.DFlash.cs) + [`DFlashConfig`](../../TensorSharp.Models/Speculative/DFlashConfig.cs) |
| 视觉编码器 | [`MuseGlimmerVisionEncoder`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerVisionEncoder.cs) |
| 图像预处理 | [`MuseGlimmerImageProcessor`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerImageProcessor.cs) |
| 示例模型 | Muse-Glimmer-30B |
| 模态 | 文本、图像 |
| 思维链 | 支持（聊天模板会输出 `assistant to=self` 推理通道） |
| 工具调用 | 支持（聊天模板中的 ATEM XML 标记）；可使用 skills、代码工具以及服务端的[子智能体委派](../multi_agent.md) |
| 批处理 / 分页前向 | 不支持（传统单序列） |
| 融合整模型内核 | GGML CUDA / Vulkan / Metal / CPU（四者都有持久化 decode 图） |
| 张量并行 | 支持 —— GGML CUDA / Vulkan，最高 `--tp 2`（30B 只有 2 个 KV 头） |

## 快速开始

```bash
# 文本（按你的机器选择后端：ggml_cuda、ggml_metal、ggml_cpu、mlx）
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --input prompt.txt --backend ggml_metal --max-tokens 256

# 图像理解（需要 mmproj）
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --mmproj models/mmproj-Muse-Glimmer-30B-Q8_0.gguf \
  --image photo.png --input question.txt --backend ggml_cuda --max-tokens 300

# DFlash 投机解码（每个 token 取自主干行；除浮点近平局外与普通贪心一致，见第 3 节）
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --draft-model models/dflash-kquant.gguf \
  --spec-draft 15 --input prompt.txt --backend ggml_cuda
```

`--draft-model` 也可以用环境变量 `TS_MUSE_GLIMMER_DFLASH` 指定。这个模型上的投机解码
离不开该草稿模型：没有它时，`--spec`（包括无权重的 n-gram 草稿器）只提供普通解码。

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

验证以贪心方式对齐主干，所以除浮点近平局外，输出的 token 流就是普通贪心 decode 的流：
验证批的 GEMM 与单行 decode 的 GEMM 形状不同，logits 的末位不同，接近平局的位置就可能
翻转。TensorSharp 与 llama.cpp 在同一语料上都表现出这种现象；这是平局翻转，不是验证 bug。

草稿模型与主干验证都作为融合的原生图运行
（[`ggml_ops_dflash.cpp`](../../TensorSharp.GGML.Native/ggml_ops_dflash.cpp)），
在 CUDA / Vulkan / Metal 上是持久化的（CUDA 还会对它们做图捕获）。两张图：

* `TSGgml_DFlashInject` —— `fc` -> RMSNorm -> 每个草稿层
  {k/v 投影、逐头 k norm、NeoX RoPE、`ggml_set_rows` 写环}。
  没有 Q、没有注意力、没有 FFN、没有 LM head —— llama.cpp 的 `build_dflash`
  也在同一点提前返回。
* `TSGgml_DFlashDraftBlock` —— `[anchor, MASK x (b-1)]` 过完 5 个草稿块，
  然后是*主干的* LM head（借用，绝不复制）、一次 softmax，以及**设备端 top-1**
  （`ggml_argmax` 加一次 `get_rows` 取回胜出概率）—— 只下载两个 16 元素的张量，
  而不是整块 `[202048, 16]` 概率。

让持久化草稿模型保持正确的设计点：

* **注意力读整个环**，`kv_len = ring_rows + b`，跨步固定。注意力在 KV 轴上是
  置换不变的，所以环的循环顺序无关紧要；一个按槽位位置映射构建的主机侧掩码
  正好表达了哪些槽位是活的。
* **掩码的截断点是 ANCHOR 的位置，不是 query 的位置。** 部分拒绝之后，环里仍然
  留着草稿模型为 anchor 之后的位置写入的 key；那些行已经过期，块内任何 query 都
  不得看到它们。
* **在 Metal 上，读取草稿 id 之前先排空队列。** Metal 的 `graph_compute` 返回时
  GPU 仍在运行，而共享缓冲区上的 `tensor_get` 只是一次裸 memcpy，因此不同步就读取
  argmax id 会拿到过期的草稿 —— 输出仍然正确（验证会拒绝它们），但接受率会悄悄崩塌。

### 自适应成本调控器

投机永远只是速度优化，因此
[`SpeculationCostGovernor`](../../TensorSharp.Runtime/Speculative/SpeculationCostGovernor.cs)
（由 `SpeculativeExecution` 持有）会分别测量带起草与不带起草时每个*输出* token 的 ms，一旦起草更慢就把草稿模型
**暂停**。当前设计（每一条都是在真实问题里实测出来的）：

* 估计量是**总和之比**（`sum(ticks) / sum(tokens)`），而不是逐步速率的均值 ——
  一个全部被拒绝的步不能与一个输出了 16 个 token 的步同权；
* 每一轮探测都**丢弃两侧各自的第一个样本**（吸收任一侧冷启动的建图开销）；
* **剔除该轮最差的一个投机样本**（吸收探测中途的一次重建）；
* `SpecWinMargin` 为 1.15，暂停会**退避**（第一次判负暂停 32 步，逐次加倍，最多 256）——
  紧跟在 prefill 之后的判定最不可信，判错的代价也最小；
* `Reset()` 会清除判定，因此暂停永远不会泄漏到下一个请求。

置信度下限默认为 `confMin = 0.35`（llama.cpp 这条路径上的 `p_min` 默认是 0）。两种设置
都不是处处更优；让下限自适应（在 `{0, 0.15, 0.35, 0.6}` 上做一个小型 bandit）是自然的
下一步。

`SpecPrefillChunkSize`（环境变量 `TS_DFLASH_PREFILL_CHUNK`，默认 1024）设定 DFlash
prefill 追赶草稿模型时所用的**主干**前向宽度。它会被限制在两个环一次前向能吸收的范围内
（草稿模型的 `RingRows` = 2080，以及主干 SWA 环的 `rows - n_swa`）。调小它会成倍增加
长提示需要付出的完整主干前向次数；历史上硬编码的 128 曾是 DFlash prefill 最大的单项开销。

## 4. 与 llama.cpp 的对齐

`InferenceWeb.Tests/MuseGlimmerParityTests.cs` 用运行同一 GGUF 的 `llama-server`
采集的黄金输出来校验实现（`.parity/gen_ref.py`、`.parity/gen_ref_long.py`）：

```bash
llama-server -m Muse-Glimmer-30B-UD-IQ2_XXS.gguf -ngl 99 -c 8192 --port 8899
python .parity/gen_ref.py      http://127.0.0.1:8899 .parity/ref_text.json
python .parity/gen_ref_long.py http://127.0.0.1:8899 .parity/ref_text_long.json

TS_TEST_MODEL_DIR=<模型目录> TS_TEST_GGML_BACKEND=metal \
TS_MUSE_GLIMMER_BACKEND=GgmlMetal \
dotnet test InferenceWeb.Tests --filter MuseGlimmerParityTests
```

两个测试工具细节，如果重新踩一遍，每个都要花掉一个下午：

* `TS_MUSE_GLIMMER_BACKEND` 接收的是**枚举名**（`GgmlMetal`、`GgmlCpu`、`Mlx`、
  `GgmlCuda`），不是 CLI 的写法 —— 无法解析的值会静默回退到 `GgmlCuda`。
  `TS_TEST_GGML_BACKEND`（`metal`/`cpu`/`cuda`）必须与之一致，因为模块初始化器会在
  第一个测试之前固定进程全局的 GGML 后端。
* 采集黄金数据时，贪心就是 `"temperature": 0.0`，**别的什么都不要加**。给
  llama-server 传 `"samplers": []` 会跳过温度采样器，最后的 dist 抽样就会从原始分布里
  采样 —— 得到看似通顺但**不确定**的黄金数据（两个相同请求返回不同的 token）。

Apple M5 Pro 主机上的结果（2026-08-14，IQ2_XXS，黄金数据来自 llama.cpp b10385）：

| 后端 | 分词器 | 5 条贪心续写（28 token） | 长上下文（5062 token 提示） | DFlash 无损 |
|---|---|---|---|---|
| Mlx | 一致 | 5/5 token 一致 | token 一致 | 5/5 |
| GgmlMetal | 一致 | 5/5 token 一致 | 近平局翻转（见下文） | 5/5 |
| GgmlCpu | 一致 | 3/5 token 一致，另有 2 处近平局翻转，位于第 24/12 个 token | token 一致 | 3/5（翻转的是同样 2 个提示） |

**近平局翻转不是正确性 bug。** 在长上下文的分叉点，llama.cpp 自己的 top-2 logprob 是
' rising' −1.5323 对 ' The' −1.5414 —— 只差 0.009 nat。Metal 选了 ' The'，用不同的措辞
产出了相同的内容（"The population trend was rising"）；逐算子与融合 Metal 路径彼此一致，
而且用本轮优化之前的内核也能复现这次翻转。IQ2_XXS 在长上下文上本来就会留下近平局，
不同后端的内核栈会做出不同的选择 —— 这与两个引擎彼此之间表现出的行为相同。CUDA 主机上
的运行历来与长上下文黄金数据逐 token 一致。

视觉几何检查（`ComputeTargetSize` / `ComputeTokenCount` 对照 `muse_glimmer_grid_size`；
1024x1024 -> 1036x1036 -> 1369 token，336x336 -> 144 token）以及图像描述的近乎逐字一致，
与 CUDA 主机上的验证结果相同，没有变化。

## 5. 性能 —— Apple Silicon（2026-08-14）

在 Apple **M5 Pro**（6 个 P 核 + 12 个 E 核，48 GB 统一内存，启用了 tensor API 的
Metal 4）、macOS 26.6 上测量。`Muse-Glimmer-30B-UD-IQ2_XXS.gguf`（10.0 GB），贪心，
引擎交替执行。llama.cpp `a4a4c51f3`（b10385，2026-08-12）以 Metal 构建，`-fa 1`；
内置 ggml（`8846b79`）与之字节兼容。TensorSharp 的数字来自 `--benchmark`（prefill
tok/s；decode tok/s **包含**主机上的贪心采样），llama.cpp 的数字来自 `llama-bench`
（tg 不含采样），因此 decode 对比略微偏向 llama.cpp。

### ggml_metal 对 llama.cpp Metal

两个引擎在同一会话中背靠背运行（SoC 在长时间会话中会降频，因此跨会话的绝对数字会
浮动几个百分点；会话内的比值是稳定的 —— llama.cpp 自己的 tg64 在不同会话中重复测得
22.22 与 22.29）。

| 指标 | llama.cpp | TensorSharp | TS / llama.cpp |
|---|---:|---:|---:|
| prefill 512 | 427.2 | 413.6 | 0.97x |
| prefill 2048 | 407.7 | 392.1 | 0.96x |
| prefill 16384（整段提示，0→16K） | 区间：407.7（pp2048\@d0）… 286.6（pp8192\@d16K） | 320.9 | 相对区间中点 ≈0.93x |
| decode，~512 上下文 | 22.29（tg64\@d0） | 21.2 | 0.95x |
| decode，~2048 上下文 | ≈21.9（d0…d4096 插值） | 20.7 | 0.95x |
| decode，16384 上下文 | 18.81（tg64\@d16384） | 18.0 | 0.96x |

TensorSharp 的 decode 列*包含*主机贪心采样（llama-bench 的 tg 完全不含采样）；只算
模型的数字约高 0.5%。本轮优化开始时，decode 比值是 0.94x/0.94x/0.94x，且形状随上下文
变差（逐 token 的图重建不随任何东西增长，但 O(context) 的掩码重填会）；持久化/重放移植
是贡献最大的单项 —— 同一二进制、同一会话：2K 上下文时 `TS_MUSE_GLIMMER_PERSIST=0` 的
decode 为 19.5 tok/s，走重放路径为 21.3（+9%）。

### ggml_cpu 对 llama.cpp CPU

llama.cpp 以 `--device none -ngl 0` 运行（在 Metal 构建上，只给 `-ngl 0` 仍会把
batch≥32 的矩阵乘按算子卸载到 GPU）。llama.cpp 默认只用 P 核（本机为 `-t 6`）；
TensorSharp 的 ggml CPU 后端现在默认使用**全部物理核**（这里是 18 个），因为这个负载
会随 E 核扩展 —— llama.cpp 自己在给 `-t 18` 时 tg 从 3.69 升到 7.87，而它的提示吞吐
随 E 核*下降*，我们的则上升。

| 指标 | llama.cpp `-t 6`（其默认） | llama.cpp `-t 18` | TensorSharp ggml_cpu（默认） |
|---|---:|---:|---:|
| prefill 256 | 25.1 | 23.6 | 8.9–9.2 |
| decode（短上下文） | 3.69 | 7.87 | 6.8–8.2 |

decode 是重点：**线程数相同时与 llama.cpp 持平（各次探测 0.86–1.04x），约为 llama.cpp
自身默认配置的 2 倍** —— 起点是逐算子路径 20 分钟都跑不完一次 256/16 基准（每 token
约 940 次同步图提交，每次都新建一个一次性的 4 线程池）。

**prefill 是一个已知未解决的差距（线程数相同时 ≈0.4x）。** 其特征很明确：TensorSharp
在每个线程数（1/6/12/18）下每个 prefill token 的开销 ≈ 每个 decode token 的开销，也就是
说 batch 维度什么都没有摊薄，而 llama.cpp 靠按缓存分块的权重复用拿到 3–7 倍的逐 token
摊薄。原因不是融合图（逐算子路径测得相同），不是 SWA 环，也不是线程数（都已在同一二进制
上用直接 A/B 排除）。它被记为下一项 CPU 工作。

纯托管的 `--backend cpu` 是从不触碰原生代码的正确性参照（`NativeDequant.PreferManaged`），
不是服务后端：它跑这个模型只有 2.7 prefill / 0.2 decode tok/s（IQ2_XXS 在
`ManagedQuantizedOps` 中没有直接的整数点积方案，因此每次点积都要把权重行重新展开成
F32）。真正的 CPU 推理请用 `ggml_cpu`。

### mlx 后端

MLX 后端是正确的，并且现在每个 prefill 分块都留在设备上（见改动 8），但它手写的 IQ
量化矩阵乘内核没有用上 M5 的 tensor API，在两个维度上都是瓶颈。同一台机器、同一 GGUF：

| 指标 | ggml_metal | mlx |
|---|---:|---:|
| prefill 512 | 413.6 | 29.0 |
| prefill 4096（多分块，带状掩码） | ~392 | 27.7 |
| decode，短上下文（真实生成） | 21.2 | 14.2 |
| decode，~5K 上下文（真实生成） | ~20 | 11.7 |

MLX 的 decode 数字是真实生成所走的**流水线贪心**路径（设备端 argmax、链式步骤、零逐
token logits 回读 —— 96 token 的生成中主机拷贝为 0）。`--benchmark` 的 decode 模式测的是
逐算子 MLX 路径，它在深上下文时会退到主机注意力循环，不具代表性。在 Apple Silicon 上，
这个模型推荐使用 `--backend ggml_metal`；`--backend mlx` 是接入 MLX 生态的路径（差距在于
它手写的 IQ 量化矩阵乘内核，而不是它的架构）。

### 这台硬件上的 DFlash

在 M5 Pro 上，短上下文时投机对两个引擎都不划算。同一个 54 token 的提示，256 个贪心 token：

| 引擎 | 普通 decode | DFlash decode |
|---|---:|---:|
| llama.cpp（`--spec-type draft-dflash`） | ~22.3 | 8.2（0.37x） |
| TensorSharp（`--draft-model`） | 20.7 | 13.9（0.67x，接受率 70.9%） |

M5 的 GPU 让验证批与草稿模型自身的前向相对普通 token 而言都很昂贵（构建这项功能所用的
CUDA 主机比例正好相反，在那里 DFlash 是 1.3–5 倍的收益）。在这里 TensorSharp 的退化比
llama.cpp 小得多，并且挂上草稿模型之后，自适应调控器能让起草路径保持在它所能达到的最好
水平约 5% 以内 —— 但如果今天在 Apple Silicon 上在意延迟，请用普通 decode。

### 更早的 CUDA 测量（RTX PRO 6000 Blackwell，2026-08-13）

在下文这一轮优化的前一天测得，此后没有重跑（基准机已下线），因此早于那里列出的所有
改动。单张 **NVIDIA RTX PRO 6000 Blackwell Server Edition**（97,887 MiB，
`CUDA_VISIBLE_DEVICES=0`），`Muse-Glimmer-30B-Q8_0.gguf`（27.6 GiB），草稿模型
`dflash-kquant.gguf`（1.5 GiB）；TensorSharp commit `5098e3f`、内置 ggml `8846b79`，
`--backend ggml_cuda`，对手是以相同 CUDA 架构构建的 llama.cpp master `8e7f22b`
（`-b 2048 -ub 2048`）。两侧都是贪心，生成 128 token，每个点重复两次且两个引擎交替执行；
两个引擎 prefill 的是同一段渲染后的提示词（`TensorSharp.Cli --dump-prompt`，token 数用
`llama-tokenize` 核对）。两次重复的均值，tok/s；比值为 TensorSharp / llama.cpp：

| 提示 token 数 | llama.cpp prefill | TS prefill | 比值 | llama.cpp decode | TS decode | 比值 |
|---|---:|---:|---:|---:|---:|---:|
| 60 | 362 | **459** | 1.27x | 34.7 | **35.0** | 1.01x |
| 501 | 927 | **1135** | 1.23x | **36.2** | 34.3 | 0.95x |
| 2050 | 1132 | **1317** | 1.16x | **35.0** | 33.5 | 0.96x |
| 16126 | **1325** | 1249 | 0.94x | **32.2** | 30.9 | 0.96x |
| 32274 | **1303** | 1211 | 0.93x | **32.1** | 29.9 | 0.93x |
| 64575 | **1256** | 1150 | 0.92x | **32.4** | 29.1 | 0.90x |
| 123931 | **1166** | 1073 | 0.92x | **30.7** | 26.6 | 0.86x |

同一批运行上的 DFlash decode（`--draft-model dflash-kquant.gguf --spec-draft 15`，对手是
llama.cpp 的 `-md … --spec-type draft-dflash --spec-draft-n-max 15 -ngld 99`）；括号内是
两次重复的范围（仅在差距大时给出）：

| 提示 token 数 | llama.cpp | TensorSharp | TS，`--spec-pmin 0` |
|---|---:|---:|---:|
| 60 | 45.5 | **50.9** | 43.5 |
| 501 | 117.5 | 164.6（150-179） | **180.3** |
| 2050 | 24.9 | **43.5**（30-57） | 34.7 |
| 16126 | **80.2** | 55.8（37-75） | 33.2 |
| 32274 | **60.7**（43-79） | 33.8（31-36） | 29.9 |
| 64575 | **66.1** | 48.7（34-64） | 49.1 |
| 123931 | **69.0** | 42.3（30-55） | 59.8 |

投机在两个引擎上都要付出 *prefill* 代价，因为草稿模型的编码器也要过一遍提示词
（tok/s，普通 → DFlash）：

| 提示 token 数 | llama.cpp 普通 → DFlash | TensorSharp 普通 → DFlash |
|---|---:|---:|
| 60 | 362 → 203（0.56x） | 459 → 341（0.74x） |
| 501 | 927 → 495（0.53x） | 1135 → 700（0.62x） |
| 2050 | 1132 → 259（0.23x） | 1317 → 703（0.53x） |
| 16126 | 1325 → 988（0.75x） | 1249 → 826（0.66x） |
| 64575 | 1256 → 985（0.78x） | 1150 → 780（0.68x） |
| 123931 | 1166 → 920（0.79x） | 1073 → 742（0.69x） |

读这几张表时有三点限定。四行长上下文是在 CRLF 归一化之前测的，因此在这四个点上
TensorSharp 对同样的文本多 prefill 了 1.2% 的 token（16322 / 32666 / 65359 / 125412）。
TensorSharp DFlash 那些很宽的范围来自成本调控器：在只生成 128 token 的运行里，prefill
之后紧接着的一次测错的探测就会让草稿模型在大半个运行中处于暂停状态（同一提示、同一二进制
的两次 16K 重复分别是 36.7 与 74.8 tok/s）。此外，长提示来自一个高度重复的合成语料；501 token 那一点在两个引擎上的接受率都是 100%，因此它的
DFlash 数字不代表通用的加速比。

两次 16K 重复唯一的差别就是调控器的判定。当时那一版调控器一旦判负就把草稿模型固定暂停
`ParkedProbeInterval = 64` 步；此后调控器已经重做（见[自适应成本调控器](#自适应成本调控器)）。

| 16K 第几次 | 起草 / 接受 | 验证步数 | 暂停步数 | decode |
|---|---|---:|---:|---:|
| 1 | 64 / 48（75%） | 13 | **67** | 36.7 tok/s |
| 2 | 132 / 103（78%） | 22 | 3 | **74.8 tok/s** |

第 1 次在暂停之后的重新探测里测到投机是 **14.0 ms/token，而普通解码是 37.8**，说明当初的
暂停判错了：prefill 之后最初几步投机要为验证批的形状付一次性的建图开销，而探测采样到的
正是这几步。32K、64K、128K 上被暂停的那些重复都有同样的指纹（`drafted` 卡在 64-84）。
若以未被暂停的那次作为稳态，TensorSharp 的融合 DFlash 达到 **16K 时 llama.cpp 的 94%
（74.8 对 79.8）、64K 时 96%（63.6 对 66.5）**。

`--spec-pmin 0` 那一列并非一律更好：它在 501 与 128K 上赢，在 16K 与 32K 上输得很惨 ——
那里接受率从约 75% 掉到 24-42%，而每个被拒绝的行仍然占用一个验证槽位。2K 那一点是
llama.cpp 的异常：它的 DFlash decode（两次都是 24.9 tok/s）低于它自己的普通 decode（35.0），
DFlash prefill 塌了 4.4 倍；这个提示停在文档中间，因此续写比其他尺寸上"问题 + 回答"式的
提示更难预测。

单卡整进程显存峰值，每 2 秒采样一次（MiB）：

| 提示 token 数 | llama.cpp 普通 | TS 普通 | llama.cpp DFlash | TS DFlash |
|---|---:|---:|---:|---:|
| 501 | 28329 | 29655 | 31881 | 29887 |
| 16126 | 28585 | 30567 | 32191 | 34089 |
| 64575 | 29401 | 32003 | 33007 | 35809 |
| 123931 | 30471 | 33787 | 34641 | 37769 |

普通路径上 TensorSharp 比 llama.cpp 多占 1.3-3.3 GB，加载草稿模型后大约多 3 GB。

这批运行的输出一致性（贪心验证在浮点近平局之外复现普通贪心的输出，见第 3 节）：

* TensorSharp 是**确定性的**：每个配置与自己的重复运行逐字节一致。
* TensorSharp 普通 vs TensorSharp DFlash：在 60 / 501 / 2050 / 16126 上完全一致，
  在 32274 上分叉。
* llama.cpp 普通 vs llama.cpp DFlash：除 2050 外处处一致。
* 跨引擎对比时，两条续写在前 127-636 个字符内一致，之后分开：同样的权重上不同的内核与
  不同的归约顺序。

方法说明。每个上下文都按 llama.cpp → TensorSharp → llama.cpp DFlash → TensorSharp DFlash
的顺序执行，因为先跑完一个引擎的整条阶梯会让另一个偏低。整个测试期间 GPU 一直报告
`HW Power Brake Slowdown: Active`（2280-2347 MHz，功耗 180-270 W、上限 450 W，温度 28-42 C），
这是主机层面的功率制动，对两个引擎一视同仁；大约每二十次运行会有一次在 prefill 和 decode
上同时慢约 40%，且没有任何频率或温度上的痕迹，因此在那台机器上对单次重复的差值要保持怀疑。

### 2026-08-14 这一轮改了什么

下面每一项优化在保留之前都验证过数值中性（同一二进制上，各 A/B 环境变量下的贪心续写
逐字节一致）。

1. **持久化 decode 图现在覆盖 Metal 与 CPU**（`ggml_ops_muse_glimmer.cpp`，以前只有
   CUDA/Vulkan）。Metal 没有 CUDA 图那样的机制 —— 每次提交都要重新编码节点 —— 但重放
   路径仍然去掉了逐 token 的图元数据重建（约 2,000 个节点）、约 790 次张量重新绑定、
   gallocr 的生命周期重新规划、104 次小 norm 重新上传，以及 O(context) 的整类掩码重新
   生成（重放改为每 token 只把该掩码延长 2 字节）。Metal 上 2K 上下文的同二进制归因：
   decode 19.5 -> 21.3 tok/s（+9%）。
2. **Metal 图会经过后端的 `graph_optimize` 重排**（能识别别名、扩大编码器并发集合的
   重排），但**只在持久化构建上**：`ggml_backend_sched` 会为 llama.cpp 自动做这件事，
   直接调用 `graph_compute` 则不会。对每次临时构建都做重排实测是净亏损（重排的开销超过
   一次提交能省下的时间），因此 prefill 分块跳过它。
3. **图内 embedding 现在是 Metal 与 CPU 上的默认行为。** 在统一内存上，量化的
   `token_embd` 从 GGUF mmap 零拷贝绑定，因此独显上的那种取舍（再钉住一个约 1.1 GB 的
   张量）并不存在。这去掉了每个 decode token 的一次主机逐行反量化、一次张量分配与一次
   RMSNorm 调度，以及每个 2048 行 prefill 分块 54 MB 的隐状态上传。
4. **掩码填充改为区间填充并并行化。** `fill_mg_mask` / `fill_mg_ring_mask` 每行写三次
   块填充，而不是逐元素循环，超过 2M 个元素时最多分给 8 个线程。在没有设备端填充内核的
   后端（Metal、CPU）上，64K 上下文下一个 2048 行的分块就是 256 MB 的掩码；过去这是每个
   分块几十毫秒的单线程工作。
5. **共享的 ggml CPU 后端有了真正的多线程。** 裸的 `ggml_backend_cpu_init()` 使用
   `GGML_DEFAULT_N_THREADS`（4），并且**每次 graph_compute 都新建一个一次性线程池** ——
   逐算子路径每个 decode token 要付出约 940 次线程池的创建/回收，而且只用 18 个核中的
   4 个。后端现在固定一个按全部物理核（`hw.physicalcpu`）设定大小的持久
   `ggml_threadpool`（llama.cpp 默认只用 P 核，这对该负载的 decode 来说少拿了 2 倍）。
   `TS_GGML_CPU_THREADS` 可以覆盖。
6. **融合整模型内核现在也在 GgmlCpu 上运行**（它曾是仓库里唯一排除 CPU 的融合内核；
   GPT-OSS 与 Gemma 4 早已包含）。每 token 一张图，取代约 940 次同步的逐算子提交。历史上
   "1024 行融合图在预热时让 CPU 后端崩溃"的问题没有复现 —— 2048 行预热与整套 parity 测试
   都能通过。`TS_MUSE_GLIMMER_FUSED_CPU=0` 可恢复 CPU 上的逐算子路径。
7. **GgmlCpu 保留统一尺寸的 KV 缓存（不用 SWA 环）。** 环要整体读取（槽位不按位置
   顺序），而 ggml-cpu 的 flash-attention 会计算每一个 KV 列，无论是否被掩掉 —— 因此
   52 层中的 39 层在任何深度都要付出完整 4352 行环的注意力开销，而统一缓存的滑动区间
   只有在上下文真的那么长时才需要 `pad256(window + chunk)`。GPU 后端保留环：它们固定的
   图形状正是持久化图（以及 CUDA 捕获）得以成立的前提，而且它们的 flash 内核会跳过完全
   被掩掉的块。
8. **MLX prefill 的每个分块都留在设备上。** 快速 SDPA 路径过去只接受第一个 prefill
   分块；第 2 个及以后的分块会退到一条链：把整个 KV 缓存下载到主机、在 F32 中把 GQA
   展开 16 倍，并为主机侧掩码往返一个 `[32, seqLen, kvLen]` 的打分张量 —— 每层、每个
   分块都如此。`MlxFusedOps.TryPrefillAttentionBanded` 现在**在设备上**构建因果（+SWA）
   带状掩码（两个 arange + 比较 + where，作为数组掩码传给
   `mlx_fast_scaled_dot_product_attention`），而且滑动窗口层把缓存读取收窄到
   `[qStart − window + 1, total)`，长 prefill 就不会去给窗口外的 key 打分。第一个分块仍用
   普通的 `"causal"` 字符串掩码，完全不需要掩码数组。
9. **修复了两个潜伏的 Metal 竞态**（正确性问题，通过审查发现，两者都早已存在）：Metal
   的 `graph_compute` 是异步的，而共享缓冲区上的 `tensor_get` 是一次裸 memcpy，因此
   (a) DFlash 的捕获行与 (b) 草稿模型的 argmax id 都可能在执行中途被读取。(a) 会污染
   草稿模型的编码器特征；(b) 会悄悄验证过期的草稿 —— 输出仍然正确，但接受率崩塌。两条
   路径现在都在读取之前同步。

### 仍在约束设计的工程笔记

* **为什么要融合：** 逐算子前向每个 token 提交约 600–940 个 GGML 算子，每个都有主机
  可见的开销；本仓库中每个达到 llama.cpp 级 decode 的模型，都是每次前向只跑**一张**
  整模型图（`TSGgml_MuseGlimmerModelForward` —— 全部 52 层、最终 norm、LM head、logit
  缩放与软上限）。`TS_MUSE_GLIMMER_FUSED=0` 可强制走逐算子路径。
* **持久、可捕获的图。** decode 图只构建一次且张量地址稳定（用裸 `ggml_init` +
  `ggml_backend_alloc_ctx_tensors`，而不是会按生命周期打包、移动地址的 gallocr）。拓扑
  在步与步之间保持逐字节一致：KV 用 `ggml_set_rows` 写入（写入行号是一个 I64 *输入*），
  读取的是按 256 行对齐填充的窗口，配一个 F16 掩码输入 —— 于是每 256 个 token 才重建
  一次，而不是每个 token 都重建。在 CUDA 上重放还会被图捕获。图池的键是
  `(model, KV holder, n_tokens)`，因此 1 行 decode 与 k 行验证的形状永远不会互相挤出。
* **每类注意力一张掩码，而不是每层一张。** 掩码只取决于 `(window, n_tokens, start_pos)`；
  所有滑动窗口层共享一个张量，所有全注意力层共享另一个（52 个掩码 -> 2 个）。llama.cpp
  一直如此（`build_attn_inp_kv_iswa`）。在 CUDA 上 prefill 掩码由设备内核填充；Metal/CPU
  在主机上填充（已并行化，见上文）。
* **prefill 是分块的**，分块大小为 `TS_MUSE_GLIMMER_PREFILL_CHUNK`（默认 2048），正如
  llama.cpp 在 `n_ubatch` 处切分；一张 16K 行的图需要几十 GB 的激活。多模态提示同样
  分块：`ForwardChunked` 会把待注入的视觉区间重新切分到每个分块上。prefill 走共享的复用 gallocr（按生命周期打包中间结果）；
  decode 不走（稳定地址更重要）。
* **SWA 环**（GPU 后端）：52 层中有 39 层永远不会回看超过 2048，因此它们用一个
  `pad(n_swa + chunk + 1, 256)` = 4352 行、按 `position % rows` 索引的环，而不是完整
  上下文的缓存 —— 64K 时是统一缓存的 29%。那个 `+1` 是承重的（不加它时，一个 4651 token
  的提示在第一个 decode 步就与 llama.cpp 分叉）。内核读整个环（槽位不按位置顺序）；由
  掩码承载存活性。只有在融合内核可用时环才会启用；如果环已启用而融合前向拒绝执行，逐算子
  路径会抛异常，而不是悄悄返回错误的 logits。`TS_MUSE_GLIMMER_SWA_RING=0` 恢复统一尺寸。
* **已回卷的环上，回退深度受余量限制。** 截断缓存只移动写入位置；被回卷覆盖的行
  不会回来，而下一个 query 仍要回看新位置之前完整的一个窗口。因此自缓存上次清空以来序列长度
  一旦超过环的大小，只有满足 `furthest - target <= rows - n_swa - 1` 时才接受回退，其中 `furthest`
  是序列曾达到的最大长度（而不是当前长度：之前的回退可能已把它降回环大小以下）。默认分块下为
  2303 个 token，覆盖引擎 16 token 的 live-cache 回退。更深的回退会被拒绝——
  `CanTruncateKVCache`/`TryTruncateKVCache` 返回否，本轮改为重新 prefill，
  `TruncateKVCache` 抛异常。尚未回卷的环、统一尺寸的缓存以及回退到 0 仍可回退到任意深度
  （`KvBlockTransferRingTests`）。
* **跨请求的前缀复用**以分页家族的方式走 Radix 前缀缓存（默认模式）：缓存的 KV 块加上
  常驻的主缓存（`MuseGlimmerModel.PrefixCache.cs`）。常驻缓存的回退最多 16 个 token，
  并且还必须在上一条所说的环余量之内；带图像的提示不会复用其第一张图像起点之后的任何内容。
* **只有在 ggml 自己的 flash-attention VEC 内核会被选中时，CUDA 才物化填充后的 KV
  窗口** —— 该内核会误读被截断前缀的 K/V 视图（共享同一个 KV 头的 16 个 query 头全都
  返回同一个错误向量）。`kv_window_needs_cuda_flash_attn_copy` 镜像了
  `ggml_cuda_get_best_fattn_kernel`：当 `gqa_ratio >= 2` 时，在 Turing/Ampere 上直接跳过
  拷贝，在 Ada+ 上窗口 ≥ 8192 行时跳过（MMA 内核尊重 stride）。若无条件拷贝，每个 decode
  步要多 26 个 `ggml_cont` 节点，128K 时每 token 高达 3.3 GB 的流量。`TS_KV_FATTN_COPY`
  （`0`/`force`）可以固定任一行为；每次升级 `ExternalProjects/ggml` 时都要重新核对这个镜像
  的启发式。Metal 不需要拷贝 —— 它的 flash 内核完全通过 stride 寻址 K/V。
* **GPU 后端上 KV 缓存在分配时清零**：融合内核读的是*填充后*的窗口，多出来的行用 `-inf`
  掩掉，而 `-inf + NaN` 仍是 NaN，所以那些行必须是有限值。
* **SwiGLU 是单个 `GGML_OP_GLU` 节点**（`swapped=false` 对*前*一半施加 SiLU —— 以内核
  为准，`ggml.h` 的注释不可信，两半弄反时 parity 测试会大声失败）。K/V 写入路径直接在
  `0,2,1,3` permute 上用 `ggml_set_rows`（真正的前提是 `ggml_is_contiguous_rows`；被它
  取代的 `ggml_cont` 拷贝纯属开销）。
* **通过调小 prefill 分块来缩小 SWA 环会让一切更糟**（CUDA 上 64K 实测：分块
  2048/1024/512 -> prefill 476/443/422，decode 16.1/15.7/15.1 tok/s）。在 IQ2_XXS 上
  decode 不受 KV 带宽限制 —— matvec 是 ALU 瓶颈 —— 所以更小的环没有收益，而更小的分块会
  损失 GEMM 效率。

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

`TSGgml_MuseGlimmerModelForward` 接收一对 `tp_degree` / `tp_plan_out`：在 TP 模式下
它为每个 rank 构建图并返回 `TpRankPlan` 而不是直接执行，由驱动在段边界带着集合通信
运行所有 rank。

在 2× RTX PRO 4000 Blackwell 24 GB（PCIe）上测量，prefill 512 / decode 64，三次取最好
—— 这是迄今唯一测过 TP 的主机：

| 模型 | | prefill tok/s | decode tok/s | GPU 0 | GPU 1 |
|---|---|---|---|---|---|
| 30B-UD-IQ2_XXS（10.2 GB） | `--tp 1` | 1171 | 40.2 | 9178 MB | — |
| 30B-UD-IQ2_XXS | `--tp 2` | **1569**（1.34×） | **63.2**（1.57×） | 5115 MB | 4063 MB |
| 30B-Q8_0（28.2 GB） | `--tp 2` | 1691 | 34.3 | 15474 MB | 12748 MB |

`--tp 2` 在重复运行间逐字节一致，并且与 `--tp 1` 的贪心续写在前 468 / 500 个字符上
一致，之后在一个无害的改述点分叉（行并行部分和以不同顺序求和）。Q8_0 在那台机器上没有
单卡行 —— 28.2 GB 装不进一张 24 GB 的卡。

DFlash 投机解码只走单卡路径：`--tp N` > 1 下配置的草稿模型会被拒绝挂载。CLI 会打印警告并按标准解码
服务；服务端则拒绝启动（退出码 2），因为在服务端无法启用的显式 `--draft-model` 属于致命错误——去掉该参数或
不用 `--tp` 运行即可。池化的 KV 块快照在
`--tp` 下同样可用（快照会逐层遍历各 rank 的缓存），因此 `--tp` 下的多轮复用并不只靠
活跃缓存续接。

## 7. 环境变量

| 变量 | 作用 |
|---|---|
| `TS_MUSE_GLIMMER_FUSED` | `0` = 在所有后端上关闭融合整模型内核（逐算子 A/B） |
| `TS_MUSE_GLIMMER_FUSED_CPU` | `0` = 只在 GgmlCpu 上走逐算子路径（2026-08-14 之前的默认行为） |
| `TS_MUSE_GLIMMER_PERSIST` | `0` = 关闭持久化 / 重放的 decode 图，每次调用重建 |
| `TS_MUSE_GLIMMER_INGRAPH_EMBED` | `1` = 在任何后端上强制启用图内 embedding 阶段，`0` = 强制关闭（默认：LM head 与 embedding 表绑定时、Metal 与 CPU 上启用） |
| `TS_MUSE_GLIMMER_DFLASH` | DFlash 草稿模型 GGUF 路径（等同 `--draft-model`） |
| `TS_MUSE_GLIMMER_VENC_F32` | `1` = 把视觉塔反量化为 F32（A/B；约 7.4 GB） |
| `TS_MUSE_GLIMMER_VENC_FUSED` | `0` = 关闭 CUDA 融合视觉块 / flash-attention 路径 |
| `TS_MUSE_GLIMMER_GELU_TANH` | `1` = 视觉塔改用 tanh GELU 近似而非精确 erf |
| `TS_MUSE_GLIMMER_VENC_TRACE` | `1` = 打印视觉残差流的逐阶段校验和 |
| `TS_MUSE_GLIMMER_LAYER_TRACE` | `1` = 打印进入每一层的残差校验和（融合与逐算子路径输出相同格式，因此做 diff 就能把分叉定位到某一层） |
| `TS_MUSE_GLIMMER_LAYER_TRACE_POS` / `_N` / `_DIR` | 第一个追踪的位置 / 追踪多少次前向 / 原始 F32 转储目录 |
| `TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS` | MLX 逐算子惰性图的 flush 间隔（默认 4，`0` 关闭） |
| `TS_MLX_PIPELINED_DECODE` | `0` = 关闭 MLX 流水线贪心 decode 快速路径 |
| `TS_PREFILL_CHUNK` | `ForwardRefill` 的提示分块大小（默认 2048） |
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | 每次 prefill 前向的 token 数（默认 2048，`0` 关闭分块） |
| `TS_MUSE_GLIMMER_SWA_RING` | `0` = 所有层都按完整上下文分配，而不是给 SWA 层用环（GPU 后端；GgmlCpu 始终是统一尺寸） |
| `TS_MUSE_GLIMMER_SWA_ROWS` | 覆盖 SWA 环的行数（诊断用） |
| `TS_DFLASH_FUSED` | `0` = 关闭融合 DFlash 草稿模型（逐算子 A/B） |
| `TS_DFLASH_PERSIST` | `0` = 每步重建 DFlash 图，而不是重放 |
| `TS_DFLASH_PREFILL_CHUNK` | DFlash prefill 追赶草稿模型时每次**主干**前向的 token 数（默认 1024） |
| `TS_KV_FATTN_COPY` | `0` = 从不物化填充后的 KV 窗口（会复现 ggml-cuda flash-attention **vec** 故障）；`force` = 总是物化 |
| `TS_GGML_CPU_THREADS` | 共享 ggml CPU 后端的线程数（默认：全部物理核） |
