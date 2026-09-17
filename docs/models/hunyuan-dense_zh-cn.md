# Hunyuan Dense（`hunyuan-dense`）

[← 返回模型索引](README_zh-cn.md) | [English](hunyuan-dense.md)

TensorSharp 支持腾讯的稠密 Hunyuan 解码器——即 GGUF 中 `general.architecture`
为 `hunyuan-dense` 的检查点，例如 Hy-MT2 翻译系列。在该架构被注册之前，这些官方
Q4 文件会在加载阶段直接失败，而不会退回到某个相近的系列：加载器遇到未知架构时
选择失败关闭，而不是猜测一套计算图。

它是走通用 per-op 执行器的单设备、纯文本路径：没有融合的整模型计算图，没有张量
并行，也没有按层切分。服务端通过连续批处理引擎、以批处理分页前向提供服务（见
[服务](#服务连续批处理)）。

| 属性 | 取值 |
|---|---|
| GGUF 架构 | `hunyuan-dense` |
| 源码类 | [`HunyuanDenseModel`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.cs) |
| 架构插件 | [`HunyuanDenseArchitecture`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseArchitecture.cs) |
| 参考模板 | `tencent/Hy-MT2-1.8B` 的 `chat_template.jinja` |
| 模态 | 仅文本 |
| 思考模式 | 否 |
| 工具调用 | 否——该协议既不渲染工具声明，也不渲染 `role: "tool"` 结果 |
| 批处理 / 分页前向 | **默认启用** —— [`HunyuanDenseModel.BatchedForward.cs`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.BatchedForward.cs)（`IBatchedPagedModel`）；`TS_HUNYUAN_BATCHED=0` 选择 K/V 快照换入换出路径 |
| 投机解码 | 不支持 |
| 多 GPU | 单设备。`MultiGpuLimitation` 会在 stderr 上明说，而不是让多余的 GPU 静默闲置 |
| 后端 | 通过通用 per-op 路径支持全部后端：`cpu`、`ggml_cpu`、`ggml_metal`、`ggml_cuda`、`ggml_vulkan`、`cuda`、`mlx` |

## 运行

```bash
dotnet run --project TensorSharp.Cli -- \
    --model models/Hy-MT2-1.8B-Q4_K_M.gguf \
    --backend ggml_metal \
    --input "Translate to French: the harbour was quiet before dawn."
```

服务端使用同样的模型参数：

```bash
dotnet run --project TensorSharp.Server.Host -- \
    --model models/Hy-MT2-1.8B-Q4_K_M.gguf \
    --backend ggml_cuda --port 5000
```

## 架构

文本计算图与 llama.cpp 的 `hunyuan-vl` 文本塔一致：

```text
tokens -> embedding
  -> N x [ RMSNorm
            -> 融合 QKV -> NeoX RoPE -> per-head Q/K RMSNorm
            -> 因果 GQA
            -> 输出投影 + 残差
            -> RMSNorm -> SwiGLU(gate, up) -> down + 残差 ]
  -> 最终 RMSNorm -> LM head
```

**QK-norm 在 RoPE 之后**，与 Qwen 3.5 的顺序正好相反。搞反这个顺序不会导致加载
失败，只会产出流畅但错误的文本，因此加载器把这两个 norm 视为必需：任何一层缺少
`attn_q_norm.weight` 或 `attn_k_norm.weight` 都会在加载时抛异常。key 与 value 的
头宽必须相等；不相等时抛出 `NotSupportedException`，而不是悄悄 reshape。

### RoPE base 与 NTK alpha

当 GGUF 带有 `hunyuan-dense.rope.scaling.alpha` 时，加载器会在计算任何位置编码
之前套用 llama.cpp 的 Hunyuan 公式：

```text
base = rope_theta * alpha^(dim / (dim - 2))
```

Hy-MT2 的 Q4 文件写的是 `scaling.type = none` 且没有 alpha，因此直接使用发布的
base。启动日志会打印生效的 base、scale 和 RoPE 维度数，走了哪条分支可以直接看到，
不用猜。

### 权重融合与 KV cache

加载时，模型把每层的 Q/K/V 融合为一个 `attn_qkv.weight`，把每层的 gate/up 融合为
一个 `ffn_gate_up.weight`；量化路径与 F32 路径都会做，且仅当三个张量的 GGML 类型
与输入宽度一致时才融合。不满足条件的层保留独立投影；每层的融合标志在加载时预计算
一次，而不是每个 token 重新判断。

每层的 K/V cache 按模型对齐的 KV dtype 分配，从初始分配长度开始，按需翻倍直到配置
的最大上下文。每次扩容都会打印。

### 分词器

`hunyuan-dense` 使用与 DeepSeek V3/V4、JoyAI 词表相同的三遍 Unicode 预分词：先切
最长 3 位的数字串，再切 CJK 连续段，最后套用通用模式。把这几遍合并成一个 alternation
会改变中英混排处的切分边界，所以它们保持分开。

## 服务（连续批处理）

服务端的连续批处理引擎要求模型提供批处理分页前向或 K/V 状态快照。Hunyuan Dense
最初两者都没有，因此 `InferenceEngineHost.TryGetEngine` 返回 null，**每个**对话
请求都是 HTTP 500（`Continuous-batching engine is unavailable for this model`），
启动时的前缀缓存预热也以同样原因失败。现在两者都提供：

- **批处理分页前向（默认）。** 一次 `ForwardBatch` 打包所有正在运行请求的全部待算
  token，通过 slot mapping 把 K/V 写入逐层分页缓冲，并用原生分页注意力内核做按请求
  的因果注意力（`TS_PAGED_ATTN_KERNEL`，与 Mistral 3 相同）。逐层顺序与上文一致：
  先 NeoX RoPE，再逐头 Q/K RMSNorm。块量化 KV cache（`q8_0`/`q4_0`）会拒绝这条
  路径，因为分页缓冲是 F32。
- **K/V 状态快照。** 每一层都是线性 cache 上的完整因果注意力，因此
  `TryExtractKVBlock` / `TryInjectKVBlock` 恢复的内容与全新 prefill 写入的完全一致。
  设置 `TS_HUNYUAN_BATCHED=0`（或使用块量化 KV cache）时，并发请求通过换入换出
  快照轮流使用同一个 cache；结果正确，但按顺序串行服务。

两条路径都支持前缀复用。批处理路径写入的块在模型的分页存储中被复用；快照路径捕获
的块被恢复到线性 cache（见 `KvBlock.HoldsModelPagedKv` / `HoldsSnapshotBytes`）。

## 对话模板

Hy-MT2 的框架始终以 BOS 开头，助手标记只在需要生成提示时追加——它不会被粘在用户
轮次后面。这与 llama.cpp 的 `LLM_CHAT_TEMPLATE_HUNYUAN_DENSE`（Hunyuan-4B-Instruct）
不同，后者不带 BOS，且确实会把标记粘上去。

| 轮次 | 渲染结果 |
|---|---|
| 只有用户 | `<｜hy_begin▁of▁sentence｜><｜hy_User｜>Hello<｜hy_Assistant｜>` |
| 系统 + 用户 | `<｜hy_begin▁of▁sentence｜>SYSTEM<｜hy_place▁holder▁no▁3｜><｜hy_User｜>Hello<｜hy_Assistant｜>` |
| 带助手历史 | 每条历史回答以 `<｜hy_place▁holder▁no▁2｜>` 收尾 |
| 不加生成提示 | 渲染结果以 `<｜hy_place▁holder▁no▁8｜>` 结束 |

该架构优先使用内置渲染器而非 GGUF 中的 Jinja 模板，并且不输出工具声明。因此
Agent Skills 在这里退回到内联指令，与所有没有工具解析器的系列一致。

## 当前限制

- 仅文本。没有接入投影器，`--image`、`--video`、`--audio` 均不适用。
- 没有思考通道，没有工具调用解析器。
- 单设备：没有张量并行、没有按层切分、没有融合整模型计算图、不支持投机解码。
- Hy-MT2-1.8B 是翻译模型，有自己的输出习惯：对 `What is 17 + 25? Reply with only
  the integer.` 回答 `42.`；在发布翻译夹具上，`structured_json` 与 `||` 分隔词表
  两类 prompt 会被原样抄回而不是翻译。这是模型本身的行为，与服务路径无关。在
  `ggml_cuda` 上运行 `validate-release-translation.py --concurrency 1,4 --repeats 3`，
  两条服务路径的规律相同：`zh_en`、`en_zh`、`fr_en`、`long_translation` 全部通过
  （每项并发 1 为 3 个请求，并发 4 为 12 个），`delimiters` 全部失败。
  `structured_json` 在并发 1 时 3 个全部失败，并发 4 时批处理路径 12 个中有 3 个、
  快照路径 12 个中有 4 个完成了翻译。模型在这里处于近乎持平的状态（`"Hello"` 对
  `"你好"`），批处理改变的 kernel 形状足以让结果翻转。同一个 GGUF 在 llama.cpp（`llama-server`，CUDA）上
  同样把 `delimiters` prompt 原样抄回；它在并发 1 时翻译了 `structured_json`，而决定
  结果的那个 token 上前两名的对数概率为 -0.63（`你好`）与 -0.79（`Hello`）。
