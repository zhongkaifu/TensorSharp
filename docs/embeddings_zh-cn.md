# 嵌入模型

[English](embeddings.md) | [中文](embeddings_zh-cn.md)

TensorSharp 通过 `TensorSharp.Server` 与 `TensorSharp.Server.Host` 托管 GGUF BERT / XLM-RoBERTa 句向量编码器。这是当前源码新增的功能；较早的发布归档不一定包含。它回应了[讨论 #183](https://github.com/zhongkaifu/TensorSharp/discussions/183)中把代码向量存入 SQLite、按语义检索代码的需求。

## 模型与下载

| 模型 | 文件 | 向量维数 | 上下文上限（含特殊 token） | 池化 |
|---|---|---:|---:|---|
| Snowflake Arctic Embed L v2.0 | `snowflake-arctic-embed-l-v2.0-q8_0.gguf` | 1024 | 8192 | CLS |
| all-MiniLM-L6-v2 | `all-MiniLM-L6-v2-Q8_0.gguf` | 384 | 此 GGUF 为 512 | 均值 |

下载验证使用的固定修订版：

```bash
mkdir -p models/embeddings
curl -fL --retry 3 -o models/embeddings/snowflake-arctic-embed-l-v2.0-q8_0.gguf \
  https://huggingface.co/fisher046/snowflake-arctic-embed-l-v2.0-Q8_0-GGUF/resolve/2b05c46c74499a1a8e5075cabc6f58490aa6f2c1/snowflake-arctic-embed-l-v2.0-q8_0.gguf
curl -fL --retry 3 -o models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf \
  https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-Q8_0.gguf
```

两者均采用 Apache-2.0 模型许可证；Snowflake 文件约 635 MB，MiniLM 约 25 MB。验证清单 `docs/validation/embeddings-2026-09/models.json`（本地验证记录，未提交到 Git）记录了校验和与 GGUF 元数据。

当前支持范围是 `general.architecture=bert`，且具备受支持的分词器与池化元数据的 GGUF。解码器嵌入模型、重排序器、稀疏向量和多向量检索属于其他功能。

也可使用包含固定下载与启动参数的 [`config/embedding-snowflake.json`](../config/embedding-snowflake.json) 或 [`config/embedding-minilm.json`](../config/embedding-minilm.json)。宿主配置加载器下载指定修订版并验证 SHA-256。

## 启动嵌入服务

使用原生 GGML 时，按[开发说明](../DEVELOPMENT.md)构建原生 GGML 库，然后构建宿主。Apple 芯片示例：

```bash
bash TensorSharp.GGML.Native/build-macos.sh
dotnet build TensorSharp.Server.Host -c Release
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/embeddings/snowflake-arctic-embed-l-v2.0-q8_0.gguf \
  --embeddings --backend ggml_metal --embedding-threads 8 \
  --host 127.0.0.1 --port 5000 --no-webui
```

使用 `cpu` 运行 100% 纯 C# 推理，无需原生推理库；使用 `ggml_cpu` 运行原生 CPU 内核；构建时启用 CUDA 后可用 `ggml_cuda`。实际测试的硬件与性能范围见验证报告 `docs/validation/embeddings-2026-09/README.md`（本地验证记录，未提交到 Git）。直接 `cuda`、MLX 和 Vulkan 的嵌入路径尚未实现。

`--embedding-context-size N` 缩小每条输入的上限；`0` 使用模型元数据。每个进程常驻一个编码器；应用同时需要聊天和嵌入时，在不同端口启动两个服务。启用 `--embeddings` 后，对生成类路由（`/v1/chat/completions`、`/v1/responses`、`/v1/systemone`、`/v1/videos/generations`、`/api/generate`、`/api/chat`、`/api/chat/ollama`、`/api/models/load`，以及 `/api/image-generate`、`/api/image-edit`、`/api/video-generate` 与它们的 `/stream` 形式）的 POST 请求返回 HTTP 400 `This server hosts an embedding model. Use /v1/embeddings or /api/embed.`。显式指定本机没有的 `--backend` 时，启动以退出码 2 结束，并打印 `error: model load refused: Backend 'X' is not supported on this machine.`。

C# API 默认使用纯 C# 后端；需要原生 CPU 时须显式指定：

| 宿主 `--backend` | 库 `EmbeddingModelOptions.Backend` | 执行方式 |
|---|---|---|
| `cpu` | `CPU`（默认） | 100% 纯 C# CPU |
| `ggml_cpu` | `GGML_CPU` | 原生 GGML CPU |
| `ggml_metal` | `METAL` 或 `GGML_METAL` | 原生 GGML Metal |
| `ggml_cuda` | `CUDA` 或 `GGML_CUDA` | 原生 GGML CUDA |

以下命令构建并运行纯 C# CPU 宿主，不构建原生推理库：

```bash
dotnet build TensorSharp.Server.Host -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf \
  --embeddings --backend cpu --embedding-threads 8 \
  --host 127.0.0.1 --port 5000 --no-webui
```

`--embedding-threads N` 配置 `cpu` 与 `ggml_cpu` 的 CPU 执行线程。
托管后端中，`0` 选择四个线程；正数表示包含调用线程在内的计算线程总数。

面向网络用户部署时，`--host 0.0.0.0` 监听网络接口，并接入部署环境已有的 HTTPS 和鉴权网关。`/v1/models`、`/api/tags`、`/api/show` 返回当前模型及嵌入能力；请求的 `model` 使用文件基名，如以下示例。

## 兼容 OpenAI 的 API

```bash
curl http://127.0.0.1:5000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"snowflake-arctic-embed-l-v2.0-q8_0","input":["query: find a function that reads files","def read_file(path): return open(path).read()"],"encoding_format":"float"}'
```

响应为 OpenAI `list`，含按输入顺序编号的 `embedding` 对象与 `usage`（`prompt_tokens`、`total_tokens`）。向量作 L2 归一化。`input` 接受一个字符串、字符串数组、一个整数 token 数组或多个整数 token 数组。token 数组必须已包含本模型所需的特殊 token，ID 必须来自本模型的分词器。

`encoding_format` 为 `float`（默认）或 `base64`（小端 float32）。`dimensions` 截取向量前若干维并重新归一化。Snowflake 经过 Matryoshka 训练，支持缩减到 256 维；对没有此训练的模型任意缩减维数可能降低检索质量。

OpenAI 超长输入会被拒绝；空输入、混合类型、未知模型名、非法 token ID 与不支持的编码格式也会报错。每次请求最多 2048 条输入，总计最多 262144 个输入 token。嵌入及模型信息查询的 JSON 请求体上限为 16 MiB，包括 chunked 传输；超限返回 HTTP 413。

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="local")
result = client.embeddings.create(
    model="snowflake-arctic-embed-l-v2.0-q8_0",
    input=["query: read a file", "def read_file(path): return open(path).read()"],
    dimensions=256,
)
vectors = [item.embedding for item in result.data]
similarity = sum(a * b for a, b in zip(*vectors))
```

## 兼容 Ollama 的 API

```bash
curl http://127.0.0.1:5000/api/embed \
  -H 'Content-Type: application/json' \
  -d '{"model":"snowflake-arctic-embed-l-v2.0-q8_0","input":["query: read a file","def read_file(path): return open(path).read()"],"truncate":false,"dimensions":256}'

curl http://127.0.0.1:5000/api/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"snowflake-arctic-embed-l-v2.0-q8_0","prompt":"query: read a file"}'
```

`/api/embed` 返回 `model`、`embeddings`、`total_duration`、`load_duration`（纳秒）与 `prompt_eval_count`。`truncate` 默认 `true`；设为 `false` 可拒绝超长输入。截断保留最后的分隔 token。旧版 `/api/embeddings` 接受一个 `prompt`，返回一个 `embedding`。

服务保持启动时的模型常驻；请求中的 `keep_alive` 与生成 `options` 不会重新配置或卸载模型。

## 并发请求

嵌入服务立即执行第一个请求。每次编码器调用结束后，按 FIFO 顺序合并已等待的完整请求，每次调用最多合并 64 条输入序列与 4096 个 token，不使用批处理定时器或人为等待。超出任一合并上限的完整请求单独执行，包括超过 4096 token 的单条序列；模型仍进行自己的微批处理。这些合并上限不改变每个 API 请求最多 2048 条输入、262144 token、16 MiB JSON 的接收限制。

每个调用者保留自己的输入行顺序、token 用量、维数与编码格式。排队期间已取消的请求会跳过；取消一个调用者会及时结束该调用者的等待，其他调用者继续执行。只有合并调用中的所有调用者均取消时，才取消整次编码器调用。释放服务会结束等待中的请求，并在支持时请求取消当前计算、等待其完成。

## 检索质量

Snowflake 查询添加 `query: ` 前缀，文档不添加。服务不会自动添加，因为同一个批次可能同时包含查询与文档。把归一化向量与文件路径、分块文本、模型/版本标识一起存储；单位向量的点积即余弦相似度。更换模型、分词器、维数或分块策略后应重建索引。

GGUF 的位置容量不等于长文档检索质量保证。MiniLM 上游 sentence-transformers 配置的默认长度小于此 GGUF 的 512 位置容量。代码可按函数/类边界分块，并用真实业务查询评估召回结果。

### 分词一致性

XLM-R 使用 GGUF 内嵌的 SentencePiece 字符映射与全局最优 unigram 分词；BERT 使用 WordPiece、Unicode NFD 与元数据中的大小写/去重音设置。测试对比两个模型各 30 条独立 HuggingFace/llama.cpp 用例，以及 13 条 Snowflake HTTP 参照用例。

[分词参照文件](../InferenceWeb.Tests/Fixtures/EmbeddingTokenizer/huggingface-tokenization.json)（13 条 Snowflake 参照用例位于同目录的 `snowflake-tokenization.json`）保留了上游差异：Snowflake GGUF 不含字面量 `<mask>`，且按元数据去除多余空白；MiniLM 将垂直制表和换页符视为空白。对于韩文分解及印度文字间距元音，TensorSharp 保留 HuggingFace 的完整 Unicode 规范化行为；llama.cpp 的简化实现会丢掉部分字符。

## C# API 与实现

```csharp
using TensorSharp.Models.Embeddings;

using var model = EmbeddingModel.Load("models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf",
    new EmbeddingModelOptions { Backend = "CPU", Threads = 8 });
EmbeddingBatchResult result = await model.EmbedAsync(new[] { "read a file", "open a document" });
float[] vector = result.Embeddings[0];
```

`Backend = "CPU"` 使用托管 BERT/XLM-R 编码器，与原生后端共享分词、池化、归一化及 API 约定，不加载原生推理库。需要原生 CPU 内核时，使用 `Backend = "GGML_CPU"`。

### 托管 CPU 执行

每个模型拥有一个常驻工作线程池，按配置提供包含调用线程在内的计算线程数（`Threads = 0` 选择四个线程）。token 与位置向量读取、投影、激活量化、注意力、归一化和 GELU 共用该池。工作线程短暂自旋后休眠，等待新任务；释放模型会停止并等待这些线程退出。嵌入运算不初始化全局共享 CPU 池，也不通过 .NET 线程池调度并行循环。

量化权重保持紧凑存储。在支持有符号点积指令的 ARM 处理器上，Q8_0 投影使用 C# SIMD 内核，每个块同时计算四个 token 行与四个输出列。加载时重排原始 int8 权重，将半精度缩放因子缓存为 float32，并释放原始投影字节数组。激活按 Q8 块量化，保留原始缩放与舍入到偶数规则，缩放后的块结果通过融合乘加（FMA）累加。单行尾部使用专用内核，避免在四行块中重复计算。其他指令集和权重格式使用托管量化内核。存储类型相同时合并 Q/K/V 投影；归一化与偏置向量在加载时解码一次。

x86 托管路径通过 [`TensorComputePrimitives`](../TensorSharp.Core/TensorComputePrimitives.cs) 中由运行时选择的 `Vector<float>` 运算支持 AVX，并在 [`ManagedQuantizedOps`](../TensorSharp.Models/ManagedQuantizedOps.cs) 中提供显式 AVX2 / AVX-512 量化点积。例如 Q8_0 投影仅在 `Avx512F.IsSupported` 与 `Avx512BW.IsSupported` 同时为真时选择 `VecDotQ8_0Q8_0Avx512`，然后在 `Avx2.IsSupported` 时选择 `VecDotQ8_0Q8_0Avx2`，其余硬件使用回退代码。注意力也在使用融合向量运算前检查 FMA 支持。四 query × 十六列的长注意力内核专用于 ARM；其他处理器使用便携向量宽度实现。当前嵌入耗时证据来自 Apple Silicon，尚未测量 x86 性能。

投影共享由实际 token 组成的紧凑批次。FP32 注意力始终按序列隔离，一次计算四个 query 行。每层将 key 转置一次，使同一通道的 token 连续排列。每条 SIMD 通道累加一个相邻 key 的分数；ARM 使用选定 query 系数的 FMA 指令，其他处理器采用可用的便携向量宽度。key 行跨度带有尾部 padding，以允许最后一次向量读取，但仅写入实际 token 的分数。value 按通道块、token 与 SIMD 通道打包（ARM 上每块四个通道），四个 query 通过 FMA 共用数据块，复用概率读取并直接在向量通道中累加各输出通道。head 维度不满足块宽度对齐时，使用转置 value 与四 query 点积回退。短序列与回退路径的每个工作线程保存四行临时分数，其空间随序列长度线性增长；激活数组在请求间复用。CLS 与末 token 池化仅在最后一层注意力及前馈阶段计算所需的 query 行；均值池化使用全部实际 token。

当批次中存在长度不少于 1024 token 的序列，且 head 支持 value 打包时，托管编码器使用 64 query × 128 key 分块。ARM 内层内核同时计算四个 query 与十六个 key 或 value 通道，将 query/概率向量复用于四组 SIMD 列向量；其他处理器保留便携内核。在线 FP32 softmax 在 key 块之间维护每个 query 的运行最大值、指数和及加权 value 累加器，最后归一化。每个概率块的减最大值、指数与求和融合在一次遍历中。每个 head 的相邻 query 块复用紧凑 K/V 块，工作线程临时空间由块大小和 head 维度限定；任务按 head 排列。不满足对齐条件的 head，以及 CLS/末 token 池化最后一层选定的输出行，保留回退路径；较短的批次按 query 组排列。

分块保持相同的数学注意力公式和 FP32 精度，但改变浮点归约顺序。[标量编码器测试](../InferenceWeb.Tests/EmbeddingModelTests.cs)验证长序列混合批次与全部池化模式，绝对误差不超过 `2e-6`；该检查验证数值一致性，不代表逐位相同。

无原生推理库宿主检查（`docs/validation/embeddings-2026-09/managed-native-free.json`，本地验证记录，未提交到 Git；可用 `benchmarks/EmbeddingBench/native_free_smoke.py` 重新运行）删除宿主中的自定义原生资源后运行两个下载模型，并在推理后检查实际加载的库。

### 原生 GGML 执行

原生后端使用 TensorSharp 自己的 GGML 集成，量化权重常驻并复用完整编码器图；加载时合并量化 Q/K/V 投影，保留 GGUF 权重原始字节。注意力为双向。GPU 将不同长度的序列打包为一次前向，通过分块对角掩码隔离各序列；等长序列使用独立批维。原生 CPU 将投影 token 紧凑打包，注意力序列则独立 padding，以保持数值稳定。结果恢复到输入顺序；CLS 池化跳过末层未参与输出的前馈行，均值池化排除 padding。原生 CPU 注意力保持 float32 K/V 与对齐形状。短注意力序列直接读取 K/V 视图；注意力长度不少于 1024 时，将每个 head 的 K/V 连续打包一次，仍保持 F32 精度。执行计划与图内存一起缓存。模型图状态串行访问；取消在原生批次之间及在途设备计算完成后检查。

原生 CPU 投影权重在后端支持时采用优化后的缓冲区布局，保持 GGUF 量化格式。Metal 在分配图内存之前进行图优化，使融合操作与张量生命周期一致。

另有独立 NumPy 前向验证（[`eng/embedding-reference.py`](../eng/embedding-reference.py)；结果位于 `docs/validation/embeddings-2026-09/numpy-oracle/`，本地验证记录，未提交到 Git），直接从反量化后的 GGUF 权重计算完整编码器，并分别对照 TensorSharp 的纯 C# CPU、原生 GGML CPU、Metal，以及 llama.cpp 的 CPU/Metal 输出，覆盖两个模型与两种引擎执行顺序。

本实现参考了 llama.cpp 的 BERT 图、GGUF 布局、UGM/WordPiece、池化与无 KV 执行，vLLM 的池化模型分离和逐输入元数据，以及 SGLang 的请求校验、批处理与 float32 base64 编码。

参见[可复现 HTTP 基准](../benchmarks/EmbeddingBench/README.md)与验证结果 `docs/validation/embeddings-2026-09/README.md`（本地验证记录，未提交到 Git）。这些测试使用相同 GGUF 对比 llama.cpp，不能替代完整的 MTEB 评测。

### 并发 API 客户端基准

[`concurrency_bench.py`](../benchmarks/EmbeddingBench/concurrency_bench.py)测量多个客户端发起的独立单条输入请求。它复用已记录基准中的命令与输入，依次运行两个引擎，每个客户端保持一个 HTTP 连接。默认先进行十秒运行时预热，再由八个客户端每轮各自顺序发送四次请求。

```bash
python3 benchmarks/EmbeddingBench/concurrency_bench.py \
  --base-results docs/validation/embeddings-2026-09/minilm-managed/results.json \
  --clients 8 --requests 32 --warmup 3 --rounds 10 \
  --output /tmp/minilm-managed-concurrency --require-performance
```

`--base-results` 接收 `embedding_bench.py --output DIR` 写出的 `results.json`；示例路径是已记录的运行（本地验证记录，未提交到 Git）。重复测试使用新的输出目录；添加 `--tensorsharp-first` 可反转引擎执行顺序。输出记录包含 JSON 解析的逐请求延迟、整轮延迟、每秒请求数、p95、向量、token 计数与二进制哈希。向量验证在每轮计时结束后进行。性能门槛比较与 llama.cpp 的整轮耗时中位数，允许最多 5% 的额外耗时。报告在顶层 `vectors` 表中只存一次完全相同的已解析向量；用 `report["vectors"][request["vector_ref"]]` 读取请求对应的向量。该测试补充单请求与批量输入基准。

来源：[Snowflake 模型卡](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)、[MiniLM 模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)、[OpenAI embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create)、[Ollama embed API](https://docs.ollama.com/api/embed)。
