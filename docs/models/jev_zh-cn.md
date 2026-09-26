# Jev 决策推理

[← 返回模型索引](README_zh-cn.md) | [English](jev.md)

TensorSharp 使用 DiffusionGemma GGUF 提供与 Jev 兼容的 `POST /v1/systemone` 端点。请求提供
状态（state）与类型化问题；响应包含布尔概率（`noul`）、分类选择与期望得分。状态可以是文本、
图像或两者兼有：图像以内联方式放在请求体中，并作为状态的一部分被读取，因此可以针对一张图片
做类型化判定（见[图像输入](#图像输入)）。
`jev-latest` 与 `jev-preview` 是指向已加载 DiffusionGemma 模型的 API 别名，不是独立的检查点，
也不是专有的托管 Jev 模型。

实现遵循 [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) 中的带种子结构化读取
方法：为每个问题在答案 canvas 中放一个 token 槽位，并在一次去噪步之后读取所请求标签的 logits。
它不做输出采样、self-conditioning、JSON 生成，也不做 commit 前向。这与
[LocalJev](https://github.com/githubnext/localjev) 不同，后者让聊天端点把概率值生成为 JSON，并在
输出格式错误时重试。[Jev 开发者文章](https://huggingface.co/blog/sora-2/how-to-use-the-jev-ai-model-a-step-by-step-develop)
介绍了类型化判定的用例与托管 API；它不是模型权重或数值一致性的规范。

改编自上游的编译器提示词 / 模板逻辑的出处见
[`TensorSharp.Chat/Jev/NOTICE.md`](../../TensorSharp.Chat/Jev/NOTICE.md)，Apache-2.0 许可证会随
构建、发布与打包输出一起提供。审阅过的参考版本：LocalJev
`3f23e36e1a3bff46c7e83e8e3781d3512bc82021`；vLLM 检出
`0eb42dbbfce96477f4ca174980f4d68336fb1971`，其中包括
[位于提交 `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8` 的 `structured_server.py`
原型](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py)。

## 启动服务端

随附的 Q4_K_M 检查点不需要额外的 tokenizer 文件。图像输入需要视觉塔，配置会下载一次
（2.8 GB）；纯文本判定不需要它。在命令行上传 `--mmproj none` 会让服务端只跑纯文本，也会跳过
这次下载，因为命令行上的 `--mmproj` 会在解析之前丢弃配置中的条目。在仓库根目录下，用 PowerShell：

```powershell
$env:TENSORSHARP_MODELS = 'C:/Works/models'
$env:DIFFUSION_VRAM_HEADROOM_MB = '4096'
$env:MAX_CONTEXT = '4096'
dotnet run --project TensorSharp.Server.Host -c Release -- --config config/jev-diffusiongemma-q4.json
```

该[配置](../../config/jev-diffusiongemma-q4.json)绑定回环地址的 5000 端口并使用 `ggml_cuda`。
CPU 执行可覆盖为 `--backend ggml_cpu`，受支持的 Mac 上可用 `--backend ggml_metal`。这些只是执行
选项，并不表示每个后端都做过基准测试。本地文件不存在时，配置会从 Hugging Face 上的
`unsloth/diffusiongemma-26B-A4B-it-GGUF` 下载 `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`，之后的启动
直接复用。它默认使用仓库的 `models` 目录；把 `TENSORSHARP_MODELS` 设为绝对路径即可使用其他位置。
同一服务端上的普通聊天端点仍然可用。

该检查点已发布的 GGUF 全部是纯文本的，也没有发布过 mmproj，因此配置直接从上游
`model-00011-of-00011.safetensors` 分片加载视觉塔——它的全部 356 个张量都在这个分片里。该文件只
下载一次，之后复用。没有它时，服务端仍会回答纯文本请求，对图像请求返回 HTTP 503，而不是用填充行
作答。同一个视觉塔也服务于该服务端上的普通 DiffusionGemma 聊天。

这份配置方案在 16 GiB CUDA GPU 上为激活保留 4 GiB 显存。更大的保留量会让常驻的权重更少，但可以
避免较长提示下的严重换页。请针对设备与负载调整 `DIFFUSION_VRAM_HEADROOM_MB`；模型默认值为
2048 MiB。4096 token 的准入上限并不保证该规模的每种 schema 与提示都能放进设备内存。

在启动进程之前设置 `MAX_CONTEXT`，以限制分词后的提示与答案 canvas 的总长度。不设置这个环境变量时，
使用模型 GGUF 中的上下文上限。该设置只能通过环境变量提供；配置中的 `max-tokens: 256` 限制的是普通
聊天的生成长度，而不是 Jev 的输入长度或 canvas 宽度。请按可用内存选择上下文上限：检查点声明的上下文
并不保证某台设备能够执行。超长的 Jev 请求会被拒绝，而不是被截断。

```powershell
Invoke-RestMethod http://127.0.0.1:5000/v1/systemone -Method Post `
  -ContentType 'application/json' -InFile docs/examples/jev-ticket.json |
  ConvertTo-Json -Depth 12
```

在仓库根目录下等价的 curl 请求：

```bash
curl http://127.0.0.1:5000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @docs/examples/jev-ticket.json
```

如果终端已经位于 `docs/examples`，请改用 `--data-binary @jev-ticket.json`。`@` 让 curl 读取文件内容；
没有它，curl 会发送字面文本 `jev-ticket.json`，服务端在推理之前就返回 `Request body must be valid
JSON.`。

## 请求示例

[工单示例](../examples/jev-ticket.json)在 prompt prefill 之后用一次共享的 canvas 前向回答三个问题：
把工单分派到某个部门、判断是否为故障，以及给严重程度打分。下面是一个只用标准库的更小的 Python 示例：

```python
import json
from urllib.request import Request, urlopen

body = {
    "model": "jev-latest",
    "state": "I was charged twice for the same subscription this month.",
    "questions": {
        "billing": {
            "type": "noul",
            "instructions": "Is this a billing issue?"
        }
    },
    "samples": 1,
    "seed": 42
}
request = Request("http://127.0.0.1:5000/v1/systemone",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=180) as response:
    answer = json.load(response)
print(answer["answers"]["billing"]["noul"])
```

在进程内使用的 .NET 应用可以引用 `TensorSharp.Chat`，使用同样经过校验的请求与执行门控：

```csharp
using System.Text.Json;
using TensorSharp.Server;
using TensorSharp.Server.Jev;

using var service = new ModelService();
// Pass the vision shard as mmProjPath to accept requests that carry images;
// null loads the text-only path, which refuses them.
service.LoadModel("C:/Works/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
                  mmProjPath: null, backendStr: "ggml_cuda");
using var json = JsonDocument.Parse(File.ReadAllText("docs/examples/jev-ticket.json"));
object response = await service.JevAsync(JevRequest.Parse(json.RootElement));
Console.WriteLine(JsonSerializer.Serialize(response));
```

底层调用方可以使用 `DiffusionGemmaModel.ReadStructured(promptTokens, seedCanvas, positions,
tokenIds)`，其中 positions 是从零开始的 canvas 下标，`tokenIds` 的每一行列出对应问题的标签。访问模型
期间要一直持有 `GpuComputeLock`。在这一层，图像需要调用方自己处理：展开提示中的占位符，在读取之前用
`DiffusionGemmaModel.SetVisionEmbeddings` 安装编码后的 span，请求结束时调用 `ClearVisionEmbeddings`。
服务层会替你处理分词、提示渲染、图像 span、上下文检查、canvas 构建、序列化与生命周期安全。

`state` 可以是字符串、JSON 对象或数组。问题名称原样返回。`choice` 的 `criteria` 是从选项键到描述的
映射，`score` 的 `criteria` 是有序的描述列表，`noul` 可选地提供 `true` 与 `false` 的描述。得分使用从零
开始的下标；例如三个档位得到 0 到 2 之间的期望得分。

| 请求字段 | 默认值 | 含义 |
|---|---|---|
| `instructions` | 无 | 可选的请求级说明，加在问题之前的系统文本中 |
| `images` | `[]` | 至多 8 张内联图像，每张为 base64 或 `data:` URL |
| `samples` | `"auto"` | `1` 到 `32` 次独立的带种子读取，或自适应读取 |
| `auto_max` | `4` | 触发自适应不确定性时的总读取次数，至多 `32` |
| `auto_threshold` | `0.1` | 条件标签熵阈值，单位为 nat |
| `seed` | `42` | 在同一 .NET 运行时 / 后端上可复现的 canvas 噪声 |
| `steps` | `1` | 只支持一步结构化读取 |
| `think` | `0` | 该端点不包含思考生成 |
| `chunk_rows` | 服务端 canvas 上限 | 可选的每个问题分块的最大 canvas 宽度，`8` 到 `4096` |
| `chunk_prompt` | `"own"` | 每个分块只包含自己的问题，或用 `"shared"` 重复全部问题 |

自适应模式先做一次读取，只要任一问题的条件熵超过阈值，就总共做 `auto_max` 次读取。固定次数的读取
按分布取平均。需要固定的最小工作量时使用 `samples: 1`；示例中显式设置了它。启用 prompt 缓存时，多次
读取在 GGML CUDA 与 Metal 上复用同一份 prompt K/V。每个模型只保留一份 prompt 缓存与标签投影；切换
state 或 schema 会替换该缓存。

请求最多支持 64 个问题，每个问题 2 到 26 个备选项。问题 ID 必须为 1 到 128 个字符，不能包含冒号、控制
字符或首尾空白。选项名称与得分描述可以包含多个 token：编译器会把它们映射为短标签，并验证每个标签在完整
答案模板中只占一个 token。无效模板与上下文溢出会返回校验错误。条件性问题依赖、顺序问题串联、额外的去噪
步、思考生成以及音频或视频输入都会被明确拒绝，并各自说明原因。

由于图像字节以 base64 编码放在请求体内，服务端把请求体上限设为 8 MiB；`TS_JEV_MAX_BODY_MB`（1 到 64）
可在启动时设定其他上限。`TS_JEV_MAX_CANVAS` 默认为 64 个 token（同时受检查点 canvas 宽度限制）；超出的
schema 会被切成多个分块。`TS_JEV_MAX_PENDING` 默认允许 32 个已接纳请求，包括正在执行的请求。超出的请求
收到 HTTP 529，并带 `Retry-After: 1`。无效请求收到 HTTP 422，未知模型名收到 HTTP 404，模型不可用或不是
扩散模型时收到 HTTP 503。格式错误的 JSON 收到 HTTP 400，超大请求体收到 HTTP 413，非 JSON 媒体类型收到
HTTP 415。重新加载模型与关闭服务都会等待正在进行的推理结束后再释放权重。取消操作在托管层之间与原生派发
之间检查；它无法中断已在执行的 GPU kernel。

## 图像输入

请求可以在 `images` 中携带图像，这是一个有序数组，至多 8 项。每一项是 `data:` URL 或裸 base64：

```json
{
  "model": "jev-latest",
  "state": "Dashcam frame captured as the vehicle approaches the intersection.",
  "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
  "questions": {
    "signal": {
      "type": "choice",
      "instructions": "Which lamp of the traffic light is lit?",
      "criteria": { "red": "The top lamp is lit", "yellow": "The middle lamp is lit", "green": "The bottom lamp is lit" }
    },
    "stop": { "type": "noul", "instructions": "Must the vehicle stop before entering the intersection?" }
  },
  "samples": 1,
  "seed": 42
}
```

[红绿灯示例](../examples/jev-traffic-light.json)就是上面这个请求，并已嵌入一张小的合成图片，可以直接
发送。图中亮的是绿灯，而 state 文本只说车辆正在接近路口，因此只有读取像素才能得到答案：在该检查点上，
去掉图像的同一请求会以很高的置信度回答 `red` 与“必须停车”。

```bash
curl http://127.0.0.1:5000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @docs/examples/jev-traffic-light.json
```

附加你自己的文件只需要 base64 编码：

```python
import base64, json
from pathlib import Path
from urllib.request import Request, urlopen

body = {
    "model": "jev-latest",
    "state": "Photo submitted with an expense report.",
    "images": ["data:image/jpeg;base64," + base64.b64encode(Path("receipt.jpg").read_bytes()).decode()],
    "questions": {
        "legible": {"type": "noul", "instructions": "Is the total amount legible?"},
        "kind": {"type": "choice", "instructions": "What kind of document is this?",
                 "criteria": {"receipt": "A purchase receipt", "invoice": "An invoice", "other": "Anything else"}},
    },
    "samples": 1,
    "seed": 42,
}
request = Request("http://127.0.0.1:5000/v1/systemone",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    answer = json.load(response)
print(answer["answers"]["legible"]["noul"], answer["answers"]["kind"]["choice"])
```

同样的请求也可以在进程内执行：`JevRequest.Parse` 解码图像，`ModelService.JevAsync` 运行它们。服务端
宿主会把 `ModelService.MediaStorage` 指向它已经管理的上传目录；未设置该属性的进程内调用方会得到系统
临时路径下的一个目录，并在日志中打印一次。

### 图像如何进入读取

每张图像在用户轮次中渲染为一个 `<|image>` 标记，位于 state 文本之前，并在上下文检查之前展开为
`[BOI]` + 软 token 行 + `[EOI]`。行数是编码器针对该图片的实际输出，而不是常数：每张图片都会被放大或
缩小到 280 行预算内、边长为 48 像素倍数的最大画布，因此行数取决于宽高比而不是图片大小（192x384 的
红绿灯示例被放大到 528x1104，产生 253 行）。每张图像大约按
`MAX_CONTEXT` 中的 282 个提示 token 预算，另加一句说明附件的系统文本。编码后的行在预填充该提示的前向
之前才安装到模型上。对 canvas 来说过宽的 schema 会被切成分块，每个分块的提示都会在各自的偏移处再次携带
这些图像，所以 `n` 个分块要对图像行做 `n` 次 prefill；同一分块的重复读取（固定 `samples` 或自适应扩展）
与文本一样复用其 prompt K/V。`diagnostics.images` 报告被回答的请求收到了几张图像，`usage.input_tokens`
统计包括软 token 行在内的展开后提示。

解码后的字节以内容寻址的方式写入服务端的上传目录，因此受 `--upload-max-mb`、`--upload-quota-mb` 与
`--upload-ttl-hours` 管理；跨请求重复发送的同一张图片只存一个文件，其 embedding 只编码一次并被缓存
（`TS_MM_EMBEDDING_CACHE_MB`，默认 512）。每个软 token span 内部的图像注意力在滑动窗口层上是双向的，
在全局层上是因果的；`DIFFUSION_IMAGE_BIDIRECTIONAL=0` 让它在所有层上都变为因果。

### 哪些输入会被拒绝

像素必须放在请求体内。远程 URL 从不抓取，文件系统路径从不读取（HTTP 422），因为两者都会让推理服务的
客户端触及服务端的网络或文件；聊天端点出于同样原因拒绝两者。multipart 请求体以 HTTP 415 拒绝；音频与
视频以 HTTP 422 拒绝，并指出该检查点缺少相应的塔；不是有效 base64、不是可识别的图像容器（PNG、JPEG、GIF、
BMP、WebP、TIFF、HEIC）或解码后大于 16 MiB 的条目以 HTTP 422 拒绝。带有容器魔数但无法解码的字节同样
被拒绝，并附上解码器给出的原因（`images: PNG does not start with IHDR`），而不是作为服务端错误。在未
加载视觉塔的服务端上，带图像的请求收到 HTTP 503，而不是用填充行读出的答案。图像字节计入 8 MiB 的请求体
上限；超过上限的请求体收到 HTTP 413，存储限制则按上传策略声明的状态码返回（超过单文件上限为 413，超过
配额为 507）。

视频帧只能作为单独的图像发送：上游针对该检查点的视频特征路径会抛出 `NotImplementedError`，词表中也
没有视频的开始 / 结束标记对。音频完全不受支持——该检查点没有音频权重，因此其 tokenizer 继承来的
`<|audio>` id 背后什么都没有。

## 概率语义

对每个问题，模型计算其允许标签的 logits `z`，在模型最终的 logit softcap 之后以温度 1 返回
`softmax(z)`。这些概率是以所列答案为条件的，而当请求携带图像时，也以同一提示中编码后的图像行为条件。
`noul` 的值是 true 标签的概率。选择题取概率最高的选项；得分题取其档位下标的概率加权平均。
选择题与得分题的 `confidence` 是最大的条件概率，与 vLLM 原型一致。LocalJev 使用的是 1 减去归一化熵，
因此两者的置信度数值不应直接比较。诊断信息会说明条件熵的语义，并给出读取次数、canvas 宽度以及重复
读取的一致率 / 标准误。单次读取没有经验误差估计。

自适应熵规则不同于 vLLM 原型针对其报告的词表条目计算的熵。TensorSharp 不会从较小的条件分布中凭空
推出全词表的概率质量或 argmax 诊断。种子控制的是 TensorSharp 内部的可复现性；它不保证 Python/NumPy
与 .NET 对同一个数值种子生成相同的噪声。

稀疏输出投影只计算答案位置与所请求的词表行。它的条件概率在数学上等价于从全词表分布中选出相同标签再
重新归一化；矩阵乘法形状不同可能带来很小的浮点差异。它不计算分配给全部允许标签的总概率质量，也不计算
全词表熵。

量化、问题措辞、答案顺序以及读取次数都可能影响结果。条件置信度并不是经过经验校准的正确概率。选择置信度
阈值之前，请在目标负载的留出样本上评估。

## 执行与验证

答案 canvas 按 schema 确定大小，在模型最大 canvas 宽度之内向上取整到 16 token 的边界。多个问题共享
transformer 前向。输出头只作用于所请求的标签行，而不分配完整的 canvas × 词表 logits 张量。`ggml_cuda`
与 `ggml_metal` 使用现有的 DiffusionGemma prompt K/V 与融合 decode 路径；其他后端（包括 `ggml_vulkan`、
`mlx` 与 `cuda`）使用统一的 prompt 加 canvas 前向。
模型执行锁使其与普通扩散聊天请求串行访问共享的 GPU 状态。

GGML CUDA 默认使用融合的 prompt 注意力，把注意力运算保留在一张原生图内，以减少中间传输与显式的 KV head
展开。DiffusionGemma 路径保持精确的序列范围与现有的矩阵乘法精度策略。它使用物化的注意力分数，没有 flash
attention，也没有常驻的注意力图缓存，因此 prompt 内存仍按平方增长。启动前设置
`DIFFUSION_FUSED_PREFILL_ATTN=0` 可恢复逐算子的参考路径。其他 GGML GPU 后端可用 `1` 选择开启；这并不
代表它们已经过验证。

可复用的验证工具位于 `eng/`；生成的证据应放在已忽略的 `artifacts/jev/` 中。比较以相同请求独立运行的
多个 `/v1/systemone` 端点的方法见基准工具的 `--help`。一个小型冒烟集检查集成与明显的提示条件作用；它
不能证明生产环境的校准或广泛的模型质量。

图像路径由不需要模型的协议测试覆盖——`images` 条目可以是什么、会因什么被拒绝、内容寻址存储，以及
“每个分块提示读取它自己安装的 span”这一规则——另外还有扩展冒烟测试的 `--image` 用例，它需要一台加载了
视觉塔的运行中服务端，并筛查一张含义明确的合成图片。这项筛查证明像素到达了读取，而不是图像理解基准，
也说明不了视觉判定上的校准。在为图像输入设置置信度阈值之前，请记录一次独立评估。

```powershell
# Pure protocol/math tests (no model or GPU required).
dotnet test InferenceWeb.Tests -c Release --filter 'FullyQualifiedName~Jev&Requires!=Models&Requires!=Cuda&Requires!=Mlx'

# Python harness checks use mock responses, without model inference.
python eng/tests/jev-benchmark-tests.py

# Real GGUF sparse/full projection comparison, including deterministic reuse.
$env:TS_TEST_MODEL_DIR = 'C:/Works/models'
$env:TS_TEST_BACKEND = 'ggmlcuda'
$env:TS_TEST_GGML_BACKEND = 'cuda'
dotnet test InferenceWeb.Tests -c Release --filter 'FullyQualifiedName~JevStructuredReadTests'

# End-to-end quality smoke, HTTP contract and concurrent isolation.
python eng/jev-benchmark.py --endpoint tensorsharp=http://127.0.0.1:5000 --samples 1 --concurrency 1,2 --out artifacts/jev/http

# Adaptive/multiple reads, chunking, and concurrent ordinary-chat isolation.
python eng/jev-extended-smoke.py --endpoint http://127.0.0.1:5000 --chat --chat-seed 0 --chat-prompt 'What is the capital of France? Answer in one short sentence.' --chat-expected Paris --output artifacts/jev/extended

# Image input, against a server started WITH the vision tower. --image posts the
# traffic-light example and requires its two unambiguous answers; the refusal of
# non-inline image input is checked with or without the flag.
python eng/jev-extended-smoke.py --endpoint http://127.0.0.1:5000 --image --output artifacts/jev/image

# Stop the server first: this probe loads its own copy of the weights.
dotnet run --project eng/JevProbe -c Release -- --model C:/Works/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggmlcuda --iterations 5 --warmup 1 --widths 16,64,256 --output artifacts/jev/projection.json
```

基准原有的带标注样例已检入
[`TensorSharp.TestMatrix/Inputs/jev`](../../TensorSharp.TestMatrix/Inputs/jev/decisions.json)。若要提出
有意义的质量 / 校准结论，请记录一次独立评估。延迟分位数只统计成功的请求；失败会单独报告，并计为错误的
判定。加上 `--max-p95-ms` 和 / 或 `--min-requests-per-second`，可对每个被测分组强制施加明确的预算。不加
这些选项时，冒烟测试通过并不代表性能通过。使用 `--background-words 0,512` 可加入一组更长上下文的测试，
`--state-bust` 可在重复测量时改变 state。被比较的各端点之间，这些设置、上下文上限与预热次数都要保持一致。

若要在不启动第二个服务端的情况下与未修改的 LocalJev 引擎比较，请安装 Bun，检出固定版本的 LocalJev，
然后针对运行中的 TensorSharp 服务端执行前台比较器：

```powershell
git clone https://github.com/githubnext/localjev artifacts/jev-reference/localjev
git -C artifacts/jev-reference/localjev checkout 3f23e36e1a3bff46c7e83e8e3781d3512bc82021
bun eng/jev-localjev-compare.ts --url http://127.0.0.1:5000 --model diffusiongemma-26B-A4B-it-Q4_K_M.gguf --localjev artifacts/jev-reference/localjev --limit 3 --repeats 1 --server-max-tokens 256 --out artifacts/jev/localjev-comparison
```

它导入原始的 `Engine`，计时中包括提示构建、推理、JSON 校验与重试，但省略了 LocalJev 额外的 HTTP 桥接
一跳。两条路径收到相同的 state / 问题，但内部提示不同。请记录服务端实际的 token 上限；随附配置把普通
聊天限制为 256 个 token。报告保留上游的结束原因、重试次数与用量，因此被截断或失败的 JSON 响应都可见。

比较结果必须报告硬件、权重与量化、输入长度、读取次数、并发度、预热、失败以及延迟分布。通过 TensorSharp
聊天端点运行的 LocalJev 测量的是生成 JSON 这种方式在该后端上的开销；它不是与 oMLX 的比较。vLLM 原型发布
的 DGX Spark 结果与 LocalJev 的 M5 Max 结果不能作为 Windows RTX GPU 上同硬件的速度比较。
