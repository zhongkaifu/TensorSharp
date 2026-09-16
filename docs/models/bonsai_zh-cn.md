# Bonsai Q1_0 模型

[← 返回模型索引](README_zh-cn.md)

TensorSharp 已针对两个本地的、纯文本 Bonsai GGUF 完成验证。它们名字与量化相同，
但 transformer 架构并不相同：8B 文件是稠密 Qwen 3 解码器，27B 文件则是带
GatedDeltaNet 循环层的稠密 Qwen 3.5 混合架构。

| 属性 | Bonsai 8B | Bonsai 27B |
|---|---|---|
| 文件 | `Bonsai-8B-Q1_0.gguf` | `Bonsai-27B-Q1_0.gguf` |
| GGUF 架构 | `qwen3` | `qwen35` |
| 源码类 | [`Qwen3Model`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.cs) | [`Qwen35Model`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.cs) |
| 声明参数量 | 8.2B | 27B |
| 层数 | 36 层全注意力 | 48 层 GatedDeltaNet + 16 层全注意力 |
| GGUF 中的上下文 | 65,536（由 16,384 经 YaRN 扩展） | 262,144 |
| 词表 | 151,669（`gpt2` / `qwen2`） | 248,320（`gpt2` / `qwen35`） |
| 思考模式 | 内置模板总是以一个空的 `<think></think>` 块开始助手回复；它没有 `enable_thinking` 开关 | 内置模板支持 |
| 工具调用 | 内置模板与 TensorSharp 的 ChatML 解析器均支持 | 内置模板与 TensorSharp 的 Qwen 3.5 解析器均支持 |
| 模态 | 仅文本（验证过的产物不附带投影器） | 仅文本（验证过的产物不附带投影器） |

## 确切的本地产物

这些文件的 GGUF 元数据里没有发布方 / 来源仓库 URL 或 license。2026-09-16 已核对
发布方 Hugging Face 文件记录：[Bonsai 8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf/blob/48516770dd04643643e9f9019a2a349cf26c5dbd/Bonsai-8B-Q1_0.gguf)
与 [Bonsai 27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf/blob/f10afb355f104535e3e3e98cf7ab7795c72bd292/Bonsai-27B-Q1_0.gguf)
的 SHA-256 均与下表完全一致。两个发布方模型卡均声明 Apache-2.0。这一来源信息来自
发布方记录，不改变 GGUF 内嵌元数据，也不代表新的运行时已通过验证。

| 文件 | 确切字节数 | SHA-256 | 张量构成 |
|---|---:|---|---|
| `Bonsai-8B-Q1_0.gguf` | 1,158,654,496 | `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` | 254 个 Q1_0 + 145 个 F32 张量 |
| `Bonsai-27B-Q1_0.gguf` | 3,803,452,480 | `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0` | 498 个 Q1_0 + 353 个 F32 张量 |

下载固定版本并在使用前校验：

```bash
hf download prism-ml/Bonsai-8B-gguf Bonsai-8B-Q1_0.gguf --revision 48516770dd04643643e9f9019a2a349cf26c5dbd --local-dir ./bonsai
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --revision f10afb355f104535e3e3e98cf7ab7795c72bd292 --local-dir ./bonsai
cd bonsai
shasum -a 256 Bonsai-8B-Q1_0.gguf Bonsai-27B-Q1_0.gguf
```

`Q1_0` 是 GGML 张量类型 41：每 128 个值一块，存一个 F16 scale 和 128 个 1 bit
符号位（18 字节，即每权重 1.125 bit）。TensorSharp 在 GGUF 读取器、原生 GGML 绑定
与托管回退路径中都识别该布局，不会把它误当成 256 值的 K-quant 块。

## 架构

### Bonsai 8B：稠密 Qwen 3

该文件声明隐藏层宽度 4,096、32 个 query head、8 个 KV head、head 宽度 128、
FFN 宽度 12,288、RMSNorm epsilon `1e-6`、RoPE base 1,000,000，以及由原始 16,384
上下文出发的 YaRN factor 4。

```text
tokens -> Q1_0 embedding
  -> 36 x [ RMSNorm
            -> Q/K/V -> per-head QK RMSNorm -> YaRN RoPE
            -> causal GQA (32 Q heads / 8 KV heads)
            -> output projection + residual
            -> RMSNorm -> SwiGLU(gate, up) -> down + residual ]
  -> RMSNorm -> Q1_0 LM head -> logits
```

GGUF 中 Q、K、V、gate、up 是分开存储的。加载时 TensorSharp 为优化计算图组装出
QKV 与 gate/up 的后备缓冲区；原始的逻辑投影与托管回退路径仍然保留可用。

### Bonsai 27B：稠密 Qwen 3.5 混合架构

该文件声明隐藏层宽度 5,120、FFN 宽度 17,408、共 64 层。每第四层是全注意力
（按人类计数为第 4、8、…、64 层），因此是 16 层注意力加 48 层循环层。

```text
tokens -> Q1_0 embedding
  -> repeat 64 layers:
       layers 1-3 of each group:
         RMSNorm -> GatedDeltaNet recurrent update -> projection + residual
       layer 4 of each group:
         RMSNorm -> gated causal GQA -> projection + residual
       all layers:
         RMSNorm -> dense SwiGLU(gate, up) -> down + residual
  -> RMSNorm -> Q1_0 LM head -> logits
```

全注意力使用 24 个 query head 与 4 个 KV head，宽度均为 256。它的 Q 投影还会额外
产出一个 sigmoid 输出门。文本位置使用 MRoPE 分段 `[11, 11, 10, 0]`，RoPE base
10,000,000。每个 GatedDeltaNet 层有 16 个 K group、48 个 V head、128 宽的 K/V
状态、6,144 的内部宽度，以及一个四抽头因果卷积。只有那 16 层全注意力会增长 KV
缓存；其余 48 层携带固定大小的卷积与循环状态。

完整的 Qwen 3.5 前向公式与状态生命周期见 [Qwen 3.5 / 3.6 卡片](qwen35_zh-cn.md)。

## TensorSharp 推理路径

### 8B

- 多 token 输入走 [`ggml_ops_qwen3_prefill.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen3_prefill.cpp)：embedding、全部 36 个 block、直接 KV 写入、最终 norm 与末 token LM head 是同一张原生计算图。长提示词在符合条件的 Metal 快路径上按 512 token 分块；中间分块在提交最后一层 KV 后即结束，只有最后一块返回 logits。`startPos` 让续写与分块边界等价于一次性提示词。
- 单 token 的 Metal decode 使用 [`ggml_ops_qwen3_decode.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen3_decode.cpp) 中的常驻整模型计算图。它把量化 embedding、全部层、KV 更新、最终 norm 与 LM head 都留在设备上，并按补齐后的注意力桶重放。
- 原生 prefill 接受 F32、F16、Q8_0 与 Q4_0 KV 缓存。遇到不支持的几何形状、混合投影存储、张量并行或被关闭的快路径时，会干净地回落到既有的模型 / 逐层实现。
- [`Qwen3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.BatchedForward.cs) 提供分页多序列接口。当 KV 设置为块量化时，它会主动拒绝自己的 F32 分页缓冲路径，好让调度器保留更快的按序列路径，而不是悄悄把缓存撑大。

### 27B

- GGML CUDA、Vulkan 与 Metal 的 prefill 使用 `TSGgml_Qwen35ModelVerify`——一张同时包含两种层类型、循环状态更新、注意力 KV 写入、稠密 FFN 与末 token head 的整模型计算图。Bonsai 的 Metal 几何默认按 512 token 分块，循环状态跨块常驻设备。
- decode 使用常驻的 `TSGgml_Qwen35ModelDecode` 整模型计算图。全注意力 KV 与 GatedDeltaNet 的卷积 / delta 状态一同推进，避免每个 token 上百次托管 / 原生调度。
- 服务端连续批处理对 Qwen 3.5 系模型默认开启，并为每个序列持有独立的 KV / 循环状态。只有在需要明确做回退对比时才使用 `--no-continuous-batching`。

## CLI 与 Server

在仓库根目录把提示词放进 `prompt.txt`，然后任选一个产物。F16 KV 是质量最高的缓存；
Q8_0 与 Q4_0 更省内存。对 27B 文件而言设置 `MAX_CONTEXT` 尤其有用——它标称的 262k
窗口远大于多数本地内存预算。

```bash
# Apple Silicon 上的 Bonsai 8B
MAX_CONTEXT=16384 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Cli -c Release -- \
  --model /path/to/Bonsai-8B-Q1_0.gguf --backend ggml_metal \
  --input prompt.txt --max-tokens 256 \
  --temperature 0.5 --top-k 20 --top-p 0.85

# Bonsai 27B；加 --think 可显示它的推理流
MAX_CONTEXT=32768 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Cli -c Release -- \
  --model /path/to/Bonsai-27B-Q1_0.gguf --backend ggml_metal \
  --input prompt.txt --max-tokens 256 --think \
  --temperature 1.0 --top-k 20 --top-p 0.95
```

同样的模型路径也可用于兼容 OpenAI 的服务端：

```bash
MAX_CONTEXT=16384 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Server.Host -c Release -- \
  --model /path/to/Bonsai-8B-Q1_0.gguf --backend ggml_metal \
  --host 127.0.0.1 --port 5000

curl http://127.0.0.1:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Bonsai-8B-Q1_0.gguf","messages":[{"role":"user","content":"Explain KV caching briefly."}],"max_tokens":128}'
```

把 8B 路径换成 27B 文件，并在内存允许时调高 `MAX_CONTEXT`。在对应平台上可使用
`ggml_cuda`、`ggml_vulkan` 或 `ggml_cpu`；下文中 8B 的常驻 decode 专用测量数据针对
的是 Metal。

## TensorAgent 旁加载

TensorAgent 内置的这两个条目被有意标记为 `SideloadOnly`；空的发布方 URL 永远不会
退化成一次网络下载。

1. 让设备的 Files 选择器能访问到那个确切的 GGUF。
2. 打开 **Models**，找到 **Bonsai 8B** 或 **Bonsai 27B**，点 **Import**。
3. 选择对应文件名。TensorAgent 会先复制到暂存文件，校验字节数与 SHA-256，通过后才
   原子地发布并选中它。错误或中断的文件无法替换掉已经校验过的导入结果。

目录 ID 为 `bonsai-8b-q1-0` 与 `bonsai-27b-q1-0`；两者都要求 12 GB 设备档位。它们
在目录中的预算分别是 16,384 与 32,768 token，并请求 Q8_0 KV。用户在 Settings 中的
可见选择优先于目录请求；当前默认是 Q4_0，F16/Q8_0/Q4_0 均可选。

## 正确性与性能

8B 的原生路径在那个确切的哈希钉住文件上，与当前 llama.cpp 构建做过对比。对于九个
token 的输入 `785,3974,13876,38835,34208,916,279,15678,5562`，TensorSharp 一次性
F16 prefill 得到的 logits 为 13.423851（token 13）、12.851368（token 1）与
12.267226（token 1189），llama.cpp 对应为 13.4242、12.8526 与 12.2681。逐步 decode
保持相同的 top token 序列。F16、Q8_0 与 Q4_0 KV 缓存的裸冒烟测试都产出相同的前两个
贪心 token `13, 576`；基于模型的 Metal 批处理与回退路径一致性测试也通过。

M5 Pro 上的成对 Metal 运行使用同一 GGUF、F16 KV、512 token 物理分块，且计时 decode
内部不含采样 / argmax：

| 测试 | TensorSharp | llama.cpp | 比值 |
|---|---:|---:|---:|
| pp128 | 1,436.59 tok/s | 1,372.94 tok/s | 1.046x |
| pp512 | 1,508.56 tok/s | 1,517.48 tok/s | 0.994x |
| pp2048 | 1,343.23 tok/s | 1,334.25 tok/s | 1.007x |
| tg128 | 142.48 tok/s | 142.39 tok/s（均值） | 1.001x |

27B 的运行用前后各一次新的 llama.cpp 测量做了包夹，以抵消持续 27B 负载带来的较大
热漂移。下表使用紧邻之前与紧邻之后两次 llama.cpp 最佳成绩的中点：

| 测试 | TensorSharp | 包夹后的 llama.cpp | 比值 |
|---|---:|---:|---:|
| pp128 | 387.71 tok/s | 383.03 tok/s | 1.012x |
| pp512 | 410.12 tok/s | 397.30 tok/s | 1.032x |
| pp2048 | 398.16 tok/s | 381.66 tok/s | 1.043x |
| tg128 | 41.48 tok/s | 36.15 tok/s | 1.147x |

原始的 llama.cpp 包夹值依次为 382.35/383.71、408.11/386.48、395.40/367.92 与
36.10/36.20 tok/s。同时给出两端而不是掩盖机器漂移，是为了让对比可审计。

不依赖模型的快速测试覆盖 Q1_0 行 ABI、托管 / 原生反量化一致性、架构路由、ChatML
行为、目录钉值与暂存导入：

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  --filter FullyQualifiedName~BonsaiCompatibilityTests
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj -c Release \
  --filter FullyQualifiedName~BonsaiCatalogTests
```

## 有用的覆盖项

默认值就是实测过的路径。下列开关主要用于内存策略、诊断或 A/B 对比：

| 设置 | 作用 |
|---|---|
| `MAX_CONTEXT=N` | 硬性限制并预先分配可用上下文，而不是把 GGUF 中的长度当作上限。 |
| `KV_CACHE_DTYPE=f16\|q8_0\|q4_0` | 选择缓存精度。F16 保真度最高；块量化更省内存。 |
| `TS_PREFILL_CHUNK=N` | 覆盖提示词分块宽度（Bonsai 在 Metal 上的实测默认值：512）。 |
| `TS_QWEN3_MODEL_PREFILL=0` | 关闭 8B 的整模型原生 prefill 计算图。 |
| `TS_QWEN3_FUSED_LOGITS_DECODE=0` | 关闭 8B 那张从 embedding 直达 logits 的专用 Metal 计算图；通用 Qwen 3 路径仍保留。 |
| `TS_QWEN3_MODEL_DECODE=0` | 连通用的整模型 Qwen 3 decode 一起关闭，强制走逐层回退。 |
| `TS_QWEN3_FD_PERSIST=0` | 重建而不是重放 8B 专用 decode 计算图（诊断用；更慢）。 |
| `TS_QWEN35_PREFILL_VERIFY=0` | 关闭 27B 的整模型 prefill/verify 计算图。 |
| `TS_QWEN35_FULL_DECODE=0` | 关闭 27B 的常驻整模型 decode 计算图。 |
| `TS_QWEN35_BATCHED=0` | 关闭 Qwen 3.5 的分页 / 连续批处理以做 A/B。 |
| `TS_GGML_ASYNC_COMPUTE=0` | 全局关闭 GGML 异步提交以便诊断。 |
| `TS_GGML_PHASE_TIMING=1` | 打印原生计算图的 build/bind/allocate/upload/compute/download 计时。 |

原生源码里那些拆分投影的旋钮属于工程诊断手段，不是推荐的调优项：在 Bonsai 8B 的
对比中，融合的 QKV 与融合的 gate/up 存储更快，并且仍是默认值。
