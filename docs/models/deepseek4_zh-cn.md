# DeepSeek V4 Flash（`deepseek4`）

[← 返回模型索引](README_zh-cn.md) | [English](deepseek4.md)

DeepSeek V4 Flash 是一个 284B 参数的 MoE 模型（256 个路由专家，top-6 + 1 个共享
专家），配备一套全新的长上下文注意力栈：每层一个仅 128 token 的原始滑动窗口，
外加对 4:1（CSA）或 128:1（HCA）块压缩 key 的**压缩注意力**，其中 CSA 层由
lightning indexer 选出 top-512 压缩行。残差经由 4 路 hyper-connection 流动，混合
矩阵按 Sinkhorn 归一化。官方宣称上下文：1M token（YaRN ×16）。

## TensorSharp 如何运行它

DeepSeek V4 有三套整模型执行器：

- **`--backend cuda`**：一个 **Direct CUDA 整模型引擎**
  （`TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs`），完全不依赖 ggml。
  量化权重从 GGUF 分片直接流式写入按设备的竞技场，并按层切分到所有可见 GPU，
  因此大于单卡显存的模型可以由多张卡共同承载。
- **GPU 后端**（`--backend ggml_cuda` / `ggml_vulkan`）：下文描述的原生 ggml
  执行器。ggml 只为 CPU 与 CUDA 提供了 DeepSeek-V4 的四个架构专属算子（三个
  hyper-connection 算子与 lightning indexer）。在其他任何后端上，`hc_pre` / `hc_post`
  改由批量 `mul_mat` 拼出——它们本来就是沿 stream 轴的一次收缩——因此整层都留在
  加速器上；`hc_comb`（对 4x4 矩阵做 20 次迭代的 Sinkhorn）与 lightning indexer 仍走
  调度器的 CPU 回退，这就是 Vulkan 与 CUDA 之间剩下的差距。该分解在 Vulkan 上让
  prefill 提升 34%、decode 提升 14%，由加载时的 `ggml_backend_supports_op` 探测自动
  选择（`TS_DSV4_HC_NATIVE=0/1` 可做 A/B）。
- **`--backend ggml_cpu`**：同一个原生 ggml 执行器，只用一个 CPU 计算设备，架构
  专属算子走各自的标量 CPU 内核。它确实跑在 CPU 上（此前在有 CUDA 的机器上会悄悄
  改用 GPU），并在 stderr 上说明一次；要用 GPU 请传 `--backend ggml_cuda`。V4 在这个
  后端上没有测过 CPU 吞吐，本卡片的所有数字都来自 GPU 执行器。
- **`--backend cpu`**：一个 **100% 纯 C# 的整模型执行器**
  （`TensorSharp.Models/Models/DeepSeek4/DeepSeek4CpuExecutor.cs`）——完全没有
  原生依赖。它直接从内存映射的 GGUF 分片提供量化权重，并用托管 SIMD 代码执行
  每一个算子（带 Sinkhorn 混合的 hyper-connection、CSA/HCA 块压缩器、lightning
  indexer、带 sink 与逆 RoPE 的共享 K 注意力、带 hash 层的 sqrt-softplus MoE
  路由）。

两套 TensorSharp 原生执行器都构建在**共享的张量栈**之上，而不是各自私有的
重新实现：每个缓冲区都是由 `IAllocator` 支撑的 `Tensor`（GPU 上是
`CudaAllocator` 池，CPU 上是 `CpuAllocator`），线性层走各后端已有的那一个量化
matmul 路由（`CudaQuantizedOps.AddmmResidentToFloat32` /
`ManagedQuantizedOps.AddmmQuantizedToFloat32`），通用数学则使用 `Ops`
（`Ops.RMSNorm`、`Ops.SiLUMulClamp`……）。只有真正属于 DSV4 特有的计算——
hyper-connection、块压缩器、lightning indexer 与 top-k、sink 注意力，以及分组
MoE 内核——才留在 DeepSeek V4 的文件里。

原生整模型执行器（`TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp`）：

- 直接加载（分片的）GGUF，并把权重**按层切分到所有可见 CUDA GPU** 上——大于
  单卡显存的模型由所有卡共同承载（128 GiB 的 IQ4_XS 构建需要 2×80GB）。这是
  **默认行为，不需要任何开关**：把 DSV4 放到多张卡上的并不是 `--tp`，它也不会在
  层内部做切分。在这个系列上 `--tp N` 只是把按层切分限制在 N 张卡，作用与
  `TS_DSV4_NGPU` 相同；两者都设置时以 `TS_DSV4_NGPU` 为准。
- 在设备上持有全部 DSV4 KV 状态：原始 SWA 环、CSA/HCA 压缩 K 缓存、lightning
  indexer 缓存，以及压缩器状态环。
- 通过 `ggml_backend_sched` 把 prefill/decode 的每个 ubatch 作为单张 ggml 计算图
  执行，使用 flash attention（512 维共享 K=V 头 + attention sink）、融合的
  hyper-connection 算子，以及按形状签名的计算图缓存，因此稳态 decode 直接重放
  已捕获的 CUDA 图。

C# 一侧（`TensorSharp.Models/Models/DeepSeek4/DeepSeek4Model.cs`）负责 GGUF
元数据、`joyai-llm` BPE 预分词器、DeepSeek V4 聊天模板
（`<｜User｜>…<｜Assistant｜></think>`，`--think` 会改为打开 `<think>`）与采样。
模板会渲染工具声明，`DeepSeek4OutputParser` 负责解析 DSML 工具调用，因此 V4 可以
使用基于工具的功能：skills、代码工具（`--code-exec`），以及服务端的
[子智能体委派](../multi_agent.md)（在服务端的对话路径上默认开启）。本模型没有发布
任何委派相关的实测结果。

### 装下模型：按显存切分，以及按需开启的 MoE CPU 卸载

两个 GPU 引擎——原生 ggml 执行器与 Direct CUDA 引擎——都按每张卡**此刻实际空闲**的
显存来确定按层切分，而不是按字节平均分配。由此带来两点。

**切分与预算成比例。** 每张卡都被填到其自身空闲显存的同一比例，扣除为调度器计算
缓冲区保留的运行时余量，再扣除切分能够精确计算的 KV 缓存与压缩器状态环。原生执行器
在每次加载时估算这份余量：2 GiB 加上 lightning indexer top-k 的临时开销与一个 ubatch
的激活，因此它随 `MAX_CONTEXT` 与 `TS_DSV4_UBATCH` 增长。Direct CUDA 引擎每张卡固定
保留 2048 MiB。`TS_DSV4_VRAM_RESERVE_MB` 可覆盖两者。接了显示器的卡、或已经跑着别的
进程的卡，会按比例分到更少的层，而不是像按字节平均切分那样直接 OOM。

**路由专家只在你要求时才溢出到系统内存。** V4 Flash checkpoint 的 91% 是路由专家
权重（UD-Q8_K_XL 下 151 GiB 中占 137 GiB），因此它们是唯一有足够调节范围的旋钮——
但这里的卸载默认**关闭**，与其他所有架构一样。在显存**确实**够用的机器上悄悄开启它
是错误的取舍：它会把几十 GiB 的专家挪到 CPU，白白损失大部分 decode 吞吐。

当模型装不下时，加载器会直接说明，并给出能装下的数字，而不是加载到一半因显存不足
而中止：

```
[dsv4] not enough VRAM: 150.7 GiB of weights plus this context's KV caches
       against 152.9 GiB free across 7 device(s). Re-run with --n-cpu-moe 9
       (moves the routed experts of the first 9 layer(s), 28.7 GiB, to system
       RAM) or --cpu-moe to offload every layer.
```

注意这个例子说明的问题：单看权重似乎装得下，实际却不行，因为每张卡还要放它的 KV
缓存与计算余量（每张卡不少于 2 GiB，7 张卡至少 14 GiB）。

随后 `--n-cpu-moe N` / `--cpu-moe` 会把前 N 层的路由专家（`ffn_gate_exps` /
`ffn_up_exps` / `ffn_down_exps`）留在主机内存中，并在 ggml CPU 后端上运行它们的
`mul_mat_id` 链；路由器、norm、注意力栈与始终激活的共享专家留在加速器上，因此每个
被卸载的层在每个方向上只有 `[n_embd, n_tokens]` 的激活需要过总线。卸载的层数多于
所需，就是拿 decode 速度换显存。

`--backend cuda` 用自己的内核走同一道接缝：被卸载层的专家由 `ManagedQuantizedOps`
做乘法，加权求和的收尾仍是设备上的 `moe_scatter_add`，因此一层驻留还是卸载可以互换。
它的主机 FFN 把每个选中专家的投影在每个阶段合并为一次并行派发——若按（专家，投影）
逐个派发，花在调度器上的时间比算术本身还多（每 token 226 → 105 ms）。

3×RTX A6000 48 GB 实测（UD-Q8_K_XL，151 GiB 权重对 139 GiB 显存，双路 Xeon Gold
6342，受 23.8 CPU 的 cgroup 配额限制；`ggml_cuda` 运行期间另有一个进程占用约
15 GiB）：

| 指标 | `ggml_cuda`（卸载 8 层） | `cuda`（卸载 9 层） | `ggml_vulkan`（卸载 7 层） |
|---|---|---|---|
| Decode | 14.7 tok/s | 8.9 tok/s | 6.5 tok/s |
| Prefill（1024–2048） | 206 tok/s | — | 138 tok/s |
| Prefill（8.7K） | 195 tok/s | 43.6 tok/s | 73 tok/s |
| 每个卸载层的主机开销 | ~5.4 ms/token | ~8.3 ms/token | ~5.4 ms/token |

Vulkan 剩下的 decode 差距并不来自专家卸载——而是 `hc_comb` 与 lightning indexer 每层
一次的 CPU 往返，也就是分解时刻意保留不动的那两个算子（把 20 次迭代的 Sinkhorn 拆开，
产生的细碎派发比它消除的边界开销还多；而展开 indexer 会物化一个
`[n_kv, n_head, n_tokens]` 的打分张量，在 1024 token 的 prefill 分块下达 2.2 GB）。

### DSpark 与 MoE CPU 卸载同时使用

两个引擎都能在 MoE 被卸载的同时运行草稿器，自动切分也会计入草稿器自身的显存——加载
草稿器恰好会让多一层的专家移到主机。投机是否仍然划算取决于主机 MoE 有多快，因为 B 个
token 的验证批要把 B 行的专家都过一遍主机 MoE：

| 引擎 | 无 DSpark | + DSpark | 接受率 |
|---|---|---|---|
| `ggml_cuda` | 14.4 tok/s | **16.7 tok/s（1.16x）** | — |
| `cuda` | 8.9 tok/s | 6.2 tok/s（0.70x） | 51% |

因此在专家被卸载时，请把 `--draft-model` 与 `--backend ggml_cuda` 搭配使用；在 Direct
引擎上，托管的主机矩阵乘足够慢，多出的验证行花掉的时间超过了被接受的 token 省下的时间。

每个被卸载的层每 token 要读取约 80 MB 的 MXFP4 专家块，因此加载器只卸载**最少**的
层数，正是这一点让它仍然可用。**请按进程实际可用的 CPU 数来设定主机线程池，而不是按
`nproc`**——见下文的 `TS_CPU_MOE_THREADS`。

## 用法

```bash
# --model 指向分片 GGUF 的第一片
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend ggml_cuda --chat
```

```bash
# Direct CUDA 引擎（不用 ggml）：权重流式写入按设备的竞技场，
# 并按层切分到所有可见设备
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend cuda --chat
```

多轮对话跨轮复用 KV 缓存（纯追加）；如果提示词回退了历史，则自动重新 prefill
（压缩缓存无法截断）。

```bash
# 纯 C# CPU 推理（无原生库；需要足够内存放下缓存，
# 权重由内存映射的分片提供）
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend cpu --chat
```

## DSpark 投机解码

DeepSeek 随模型一起发布了 **DSpark** 支持模块（checkpoint 中的 `mtp.*`）：三个
DSV4 块读取主干在第 40-42 层的 hidden states，每步提议**一整块**未来 token；
另有一个 Markov 头让块内每个位置以其前一个 token 为条件，以及一个置信度头预测
每个位置被接受的概率。随后主干用**一次**批量前向验证整块，并只保留它自己的
采样器本来也会抽到的最长前缀——因此投机是加速手段，而不是质量变化。

它作为独立的草稿 GGUF 通过 `--draft-model` 加载，在**两个 GPU 引擎**上对单序列
（solo）生成生效：`--backend cuda`（Direct CUDA）与
`--backend ggml_cuda`（原生 ggml 执行器）。`ggml_vulkan`、`ggml_cpu` 与 `cpu` 对该
架构没有投机路径，配置了草稿器时会打印警告。没有草稿器时也就无从投机：单独的
`--spec`（包括 `--spec-type ngram`）在这个系列上只提供普通解码。

所有单序列 CLI 生成路径都会用到它：一次性 `--input`、`--multi-turn-jsonl`，以及
`--interactive` 聊天 REPL（后者会把被接受的块逐 token 流式输出，并跨轮复用缓存
前缀）。每轮的统计行会报告投机的效果，例如
`spec=window5/accepted330of502(66 %)`。

```bash
TensorSharp.Cli --model DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model DeepSeek-V4-Flash-0731-DSpark.gguf \
    --input prompt.txt --max-tokens 200 --temperature 0

# 交互式聊天，4 张 GPU
TensorSharp.Cli --model DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model DSpark-drafter-Q2K-Q8-0731.gguf \
    --interactive --think --tp 4 --max-tokens 20000
```

在 CLI 上，验证会用本次运行所配置的采样器抽取每一行——`--temperature 0` 下是
argmax，否则就是对话采样器——因此投机可与 REPL 中的 `/temp`、`/top-k`…… 组合。
在带惩罚项的采样器下有一点需要注意：DSpark 一次提出整块草稿，所以验证所施加的
重复/存在/频率惩罚并不会施加到该提案上，接受率会随着带惩罚的历史增长而下降。
带图像或音频附件的轮次则完全不使用投机——那些 embedding 只有普通 prefill 能注入。

### 在服务端（TensorSharp.Server.Host）上

同一个草稿器也可用于 HTTP API。用 `--draft-model` 传入即可 —— 指定草稿器本身
就会启用投机（显式 `--no-spec` 可否决）：

```bash
TensorSharp.Server.Host --model DeepSeek-V4-Flash-...-00001-of-00005.gguf \
    --backend ggml_cuda --tp 4 \
    --draft-model DSpark-drafter-Q2K-Q8-0731.gguf
```

引擎同样会用**该请求自己的采样器**抽取每一行验证结果，因此投机可与任意
采样设置组合，输出就是该采样器本来会产生的结果。惩罚项只影响接受率，而且影响
很小：同一提示词在 `repeat_penalty` 1.1 与 1.0 下分别是 31.3 与 32.1 tok/s
（接受率 66% 对 57%——差异来自生成的文本本身，而非惩罚项）。

有两个引擎层面的限制值得了解。投机是在一次全新的完整 prefill 时按请求装配的，
并且**只服务单序列**：一旦有第二个请求在途，planner 就会记录
`PerSequenceFused; rejected: SpecPerSequence: multi-sequence step`，并由 DSV4 的
per-sequence slot 以正常 decode 速度服务这一批。并发是安全的，只是不再投机。

4×A40 实测（`--tp 4`，300 token 的 OpenAI chat completion）：

| 配置 | tok/s |
|---|---|
| 无草稿器 | 25.1 |
| `--draft-model …` | **31.3 – 32.1（1.25–1.28×）** |

`--spec-pmin` 的默认值会匹配所加载的草稿器——块级草稿器为 0.35，逐 token 草稿头
为 0.15——因此无需调参。显式设置的值仍然优先；启动日志会报告当前生效的门限
（`pMin=0.35, draft=block(5)`）。

两个引擎以不同方式实现同一套算法。**ggml** 引擎把草稿器构建为模型计算图中额外的
三层：主干图捕获目标特征、对其做投影，并在同一趟中提交草稿器的 key ring；起草
本身是一张缓存的计算图，其 Markov 链（每个块位置一次
`get_rows` → `mul_mat` → `argmax`）完全在设备上运行。**Direct CUDA** 引擎用自己的
内核驱动同样的阶段，并通过主机把目标特征喂给草稿器。C# 投机核心中没有任何与
后端相关的代码。

### 获取草稿器

草稿器**不在**目标 GGUF 里：DeepSeek V4 的所有 GGUF 转换都会丢掉 `mtp.*` 张量
（`DeepSeek-V4-Flash-0731-UD-Q8_K_XL` 只有 `blk.0`-`blk.42`，别无其他）。你可以
下载预构建的草稿器 GGUF（[模型下载](../../MODEL_DOWNLOADS_zh-cn.md)中列出了三个；
各发布者对张量与元数据的命名不同，加载器都能识别），或者从上游的
**safetensors** checkpoint 自行转换。

任何 `config.json` 中带 `dspark_block_size` 的 DeepSeek V4 发布版都携带该模块：
`deepseek-ai/DeepSeek-V4-Flash-0731`、`deepseek-ai/DeepSeek-V4-Flash-DSpark`、
`deepseek-ai/DeepSeek-V4-Pro-DSpark`。只需要含 `mtp.*` 的那几片（最后三片，
约 11 GB，而整个仓库约 340 GB）：

```bash
REPO=deepseek-ai/DeepSeek-V4-Flash-0731
B=https://huggingface.co/$REPO/resolve/main
mkdir -p dspark-src && cd dspark-src
curl -sLO $B/config.json
curl -sLO $B/model.safetensors.index.json

# 含 mtp.* 的分片（0731 版本为 model-000{46,47,48}-of-00048）
python - <<'PY' | while read f; do curl -sLO $B/$f; done
import json
wm = json.load(open("model.safetensors.index.json"))["weight_map"]
print("\n".join(sorted({v for k, v in wm.items() if k.startswith("mtp.")})))
PY
```

然后转换（分词器与其余 45 片始终不会被打开）：

```bash
python eng/dsv4-dspark-to-gguf.py --checkpoint dspark-src \
    --out DeepSeek-V4-Flash-0731-DSpark.gguf --expert-type q2_k
```

checkpoint 中的路由专家以 FP4 加每 32 个元素一个 E8M0 scale 存储，除 nibble 顺序
外与 GGUF 的 MXFP4 布局一致，因此 `mxfp4` 可以无损重打包（约 11 GB）；
`--expert-type q2_k` 大致再减半。

**草稿器大小是一个真实的权衡。** 它的权重在每个投机步骤都会被重新读取，因此更大
的草稿器必须用更高的接受率把带宽成本赚回来。在 4×A40 上，这一效应在 Direct CUDA
引擎上很明显（5.6 GB 2-bit：34.0 tok/s / 接受率 69%，对比 10.9 GB MXFP4：
30.9 / 65%），在 ggml 上则落在运行间噪声范围内（120 token 样本上，5.6 GB、7 GB
与 10.9 GB 构建分别是 31.1 / 31.5 / 31.8 tok/s，接受率随体积从 63% → 66% → 68%）。
先从小的开始；只有当接受率成为瓶颈时才往上换。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--draft-model <path>` | 无 | DSpark 草稿器 GGUF（环境变量 `TS_DSV4_DSPARK`） |
| `--spec-draft <N>` | 块大小（5） | 每步最多起草的 token 数 |
| `--spec-pmin <p>` | `0.35` | 保留某个起草位置所需的最小**累积**接受概率（置信度头各位置估计值的乘积）；`0` 表示从不设阈 |

`--spec-pmin` 是真正关键的旋钮：在这个稀疏 MoE 主干上，验证批中多一行
大约相当于四分之一个 decode 步（每一行都会拉入自己那套专家），因此在前缀接受率
估计低于约 0.35 之后继续起草，期望收益为负。调低会起草更多、回滚更多；调高则更
频繁地退回普通 decode。

草稿器需要把它那三个目标特征层放在持有输出头的那张卡上；层切分会自动为它预留
空间，若切分无法满足则加载会给出明确报错（减少 GPU 数量，或去掉
`--draft-model`）。切分以**单卡最大负载**为优化目标，并按各设备固定常驻项的
实际落点计入——embedding 表在第一张卡，输出头与整个草稿器在最后一张卡。若改为把
这些均摊到所有设备（即朴素的按字节比例切分），在加载了草稿器时会让第一张卡多出
约 1.7 GB 的权重，足以让长提示词的 prefill 计算缓冲区放不进显存。

4×A40 46 GB 实测（DeepSeek-V4-Flash-0731 UD-Q8_K_XL，贪心，生成 200 token，
5.6 GB 草稿器）：

| 指标 | `cuda` 基线 | `cuda` + DSpark | `ggml_cuda` 基线 | `ggml_cuda` + DSpark |
|---|---|---|---|---|
| Decode | 26.0 tok/s | **34.0 tok/s（1.31×）** | 26.4 tok/s | **37.1 tok/s（1.41×）** |
| Prefill（15K 提示词） | 962 tok/s | 955 tok/s | 952 tok/s | 954 tok/s |
| 接受率 | — | 69% | — | 69% |

多轮 decode 获益最大（样例对话第三轮达到 2.2×，接受率 93%）：延续既有上下文的
轮次，正是草稿器最有把握的地方。

同一台机器，`--interactive --think --tp 4` 配 7 GB 的 Q2K-Q8 0731 草稿器，五轮
对话：短回答、长篇解释、追问总结，然后是一份 10K token 的文档与关于它的两个问题。

| 轮次 | 基线 | + DSpark | 接受率 |
|---|---|---|---|
| 1 短回答（53 token） | 25.6 tok/s | **44.4 tok/s（1.73×）** | 87% |
| 2 长生成（512） | 26.4 tok/s | **39.6 tok/s（1.50×）** | 66% |
| 3 追问（470） | 26.4 tok/s | **45.3 tok/s（1.72×）** | 76% |
| 4 10K token 文档（214） | 25.3 tok/s | **51.0 tok/s（2.02×）** | 85% |
| 5 关于它的第二个问题（156） | 25.4 tok/s | **49.3 tok/s（1.94×）** | 82% |

Prefill 保持持平（10K 提示词上 831 对 835 tok/s）。接受率——以及由此而来的加速
比——在下一批 token 最可预测的地方最高：从上下文中的文档里回答一个问题，胜过
自由发挥的散文。

在 200 token 生成与 15K 上下文两次运行中，贪心输出都与非投机基线**逐字节一致**。
投机只是重新排列了被接受 token 的算术顺序，因此足够长的运行仍可能在某个批量验证
行恰好落在近似平局时与顺序 decode 产生分歧——这与区分 prefill 和 decode 的那种
批量 vs 顺序漂移是同一回事。

为什么是 1.3× 而不是更多：验证批中每多一个 token，就要把它自己那套路由专家拉过
显存（主干是 6 选 256 的稀疏结构），因此无论起草多便宜，一行验证都要花掉约四分之
一个完整 decode 步。置信度门限正是让这笔交易保持正收益的关键。

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `MAX_CONTEXT` | 65536 | 上下文窗口（缓存随之伸缩；元数据允许 1M） |
| `TS_DSV4_UBATCH` | `cpu` 为 512，其他为 1024 | Prefill 微批大小 |
| `TS_DSV4_NGPU` | 全部 | 按层切分所使用的 GPU 数量（GPU 后端） |
| `TS_DSV4_VRAM_RESERVE_MB` | 按每次加载估算（至少 2048）；`cuda` 上为 2048 | GPU 后端：覆盖每张卡为调度器计算缓冲区保留的显存。不设置时，ggml 执行器取 2 GiB 加上 lightning indexer top-k 的临时开销与一个 ubatch 的激活，随 `MAX_CONTEXT` 与 `TS_DSV4_UBATCH` 增长；`--backend cuda` 固定为 2048。调低可少卸载几层专家；长提示词分配计算图失败时调高 |
| `TS_N_CPU_MOE` / `TS_CPU_MOE` | 0（关闭） | 路由专家留在系统内存中的前置层数（等同于 `--n-cpu-moe` / `--cpu-moe`）。默认关闭；放不下的模型会被拒绝，并给出能装下的层数 |
| `TS_CPU_MOE_THREADS` | 全部可用 CPU（卸载时） | ggml 执行器上主机专家矩阵乘的工作线程数。开启卸载时，线程池取 `hardware_concurrency` 再按亲和性掩码与 cgroup CPU 配额收紧——不像其他 MoE 架构那样减半，因为 DSV4 每个被卸载的层每 token 读取的专家字节远多于它们，且在那之后仍能继续扩展。`--cpu-moe-threads N` 可覆盖它，继承自启动环境的 `TS_CPU_MOE_THREADS` 优先级最高。请按配额而不是 `nproc` 来设：在 23.8 CPU 的配额下，96 线程比 23 线程实测慢 **25 倍**。在服务端部署时要给其他线程留出余地：在 95 CPU 的配额下，gemma-4-26B-A4B（不是 DSV4）上的共享 MoE 线程池在 71 线程时为 8.2 tok/s，64 线程时为 20.7，因此在那种环境下请用 `--cpu-moe-threads` 设一个低于配额的值 |
| `TS_DSV4_LOAD_THREADS` | 16 | `--backend cuda`：流式写显存加载器的读取线程数 |
| `TS_DSV4_LOAD_STATS` | 0 | `--backend cuda`：1 = 打印各阶段加载耗时 |
| `TS_DSV4_STAGED_EXPERTS` | 1 | `--backend cuda`：0 = 逐 token 的专家内核（用于 A/B） |
| `TS_CUDA_BF16_MATVEC` | 1 | 0 = 单行 BF16 投影改用 cuBLAS 而非专用 matvec（也接受 `TS_DSV4_BF16_MATVEC`） |
| `TS_DSV4_FA` | 1 | Flash attention（GPU 后端，自动探测） |
| `TS_DSV4_PERF` | 0 | 1 = 打印 tok/s 与 DSpark 起草阶段耗时，2 = 每个 ubatch 的分阶段计时 |
| `TS_DSV4_DSPARK` | — | DSpark 草稿器 GGUF 路径（等同于 `--draft-model`） |
| `TS_DSV4_DSPARK_CAPTURE` | 1 | 0 = 跳过草稿器的目标特征捕获（A/B 开关；此时草稿会过期并被拒绝） |
| `TS_DSV4_THREADS` | 全部核心 | CPU 执行器的工作线程数 |
| `TS_DSV4_MMAP` | 1 | CPU 执行器：0 = 加载时把全部权重拷入内存（并行读取；模型位于网络文件系统时适用） |
| `TS_DSV4_BUFFER_SHARDS` | — | CPU 执行器：以逗号分隔的 1 基分片序号，指定拷入内存的分片（其余仍用 mmap） |
| `TS_DSV4_MLOCK` | 1 | CPU 执行器：尽力对映射分片做 mlock（需要 memlock rlimit） |
| `TS_DSV4_SPINPOOL` | 0 | CPU 执行器：1 = 常驻自旋工作线程池（独占机器上有帮助，在共享 CPU 配额下反而更慢） |

## 性能（2×A100 80GB，IQ4_XS）

| 指标 | TensorSharp | llama.cpp（同一台机器） |
|---|---|---|
| Prefill（3.3K 提示词） | ~500 tok/s | 574（pp512）/ 634（pp4096） |
| Decode @3.3K 上下文 | ~33 tok/s | 40.3（tg128） |

长上下文召回（4.6K 与 15K token 的 needle 测试）能精确取回植入的事实——压缩注意力
路径在 128 token 的原始窗口之外同样被真实使用。
