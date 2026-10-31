# DeepSeek V4.1 Flash（`deepseek41`）

> **多 GPU 模式选择：** 整层放置使用 `--layer-split N`，支持的张量并行使用 `--tp N`。未配置两种模式时默认单设备。在 `ggml_cuda` 上（按层切分或 TP）、或在 `cuda` 上配合 `--layer-split N`（Direct CUDA 引擎的 stage，见[后端](#后端)）再加 `--tp-node-id` 与 `--tp-peers` 可跨节点运行，N 为每个节点的本地 GPU 数。配置及限制见[多节点推理](../multi-node-inference.md)（英文）。下面的历史测量保留原有拓扑，不构成新增分布式按层切分路径的验证。

[← 返回模型索引](README_zh-cn.md) | [English](deepseek41.md)

TensorSharp 为 V4.1 提供了**运行在 `ggml_cuda` 上的专用推理计算图**，并带有可选的
原生视觉编码器。它复用 DeepSeek 的整模型加载器与调度器，注意力、Engram、残差连接
与对话处理则是 V4.1 专属的。本卡片描述已实现的路径及其限制。涉及模型质量与性能的
结论需要以[验证报告](../deepseek41_validation.md)中记录的实测产物为准。
同一套计算图也能在 `--backend ggml_cpu` 上加载，但那条路径是为了在没有 GPU 时运行
和检查该架构，而不是拿来对外服务——见
[在 ggml CPU 后端上运行](#在-ggml-cpu-后端上运行)。还有第二条不需要 GPU 的路径，而且
完全不带 ggml、也不依赖任何原生库：`--backend cpu` 用纯 C# 的
`DeepSeek4CpuExecutor` 跑 V4.1，用托管代码实现同一套计算图，并对齐 PyTorch 参考实现。
两者都是正确性与可移植性通道，而不是服务通道——见[后端](#后端)。

[官方模型](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)声明的架构是
`DeepseekV41ForCausalLM`。它的文本网络有 40 层、hidden size 5120、64 个 query head
（每个 512 个分量）、384 个路由专家 top-6、一个共享专家，专家中间维为 2304，声明的
上下文长度为 1,048,576 token。V4.1 与 V4 的差异会影响每一次前向，把 GGUF 的架构名
改成 `deepseek4` 是无效的。

## 下载修复后的 Q2_K/Q5_K 检查点

新下载请使用
[smalinin/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/tree/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5)
的 `Q2_K-Q5/`，固定 revision 为 `d1de55c19f95172c882906cc83c0e55932d26a63`。
十个分片放在同一目录，并把第一个分片交给 TensorSharp。文件共 **335,382,014,624 字节
（312.349 GiB）**：主干与专家混合使用 Q2_K/Q3_K，Engram 表为 Q5_K，80 个 mHC
矩阵保留 F32，四个 Engram 门控张量保留 BF16。

原始 `vcruz305` 七分片 Q2_K 把这 84 个敏感张量也量化成了 Q2_K，因此不再推荐用于
质量验证。[发布者的修复报告](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/blob/2c525d63b9ba5319185c93637f00d70fea55b44f/Q2_K/Q2_REPAIR_REPORT.md)
说明了受影响的张量。给旧分片补充 Engram 元数据不能修复张量精度。

已检查修复版全部 1,046 个张量的名称与形状、敏感张量的存储类型，以及分词器/Engram
元数据的一致性。下载的十个分片均已通过完整文件 SHA-256 校验。`ggml_cuda` 上
`--layer-split 2` 与实验性路由专家 TP（`--tp 2`）的有限普通/DSpark
HTTP 检查均已通过。四个服务各完成三个文本检查与一个图像 OCR/颜色检查，均以 EOS
结束并正常退出。另行进行的普通/DSpark 文本与图像配对检查，在两种模式下均逐一匹配全部
24 个 token ID 与 `max_tokens` 结束原因，DSpark 实际参与解码且进程正常退出。这些有限
续写与 HTTP 的 EOS 检查分开记录。两个图像配对中 DSpark 都更慢。简短的重复文本预热对照
也已完成：每种模式一次预热、三次计时，每对输入 69 token、输出 24 token，token ID 与
结束原因完全一致，DSpark 实际参与解码且进程正常退出。
**这些小样本、受磁盘换页限制的检查不构成通用质量、性能、提速、跨节点执行或完整模型 TP 验证。**
下方历史结果不能作为这套量化的验证结果。本地证据保存在
`docs/validation/model-matrix-20260927/deepseek41/`（不提交）。每次重新下载仍须按下方
哈希校验文件。

**GGUF 已包含 Engram。** TensorSharp 直接读取 GGUF 元数据中的 token 映射、哈希乘数、
桶质数与偏移以及 padding ID，并使用同一检查点中的 Engram 权重张量。文本推理无需生成
Engram、无需单独的 Engram 文件，也无需另行下载分词器或配置文件。

加载器会对照 Engram 张量维度校验内嵌布局，并拒绝缺失或无效的常量；不再回退读取旧的
附属文件，也不再提供 Engram 路径覆盖。Engram 表的设备放置与页预热仍作用于学习到的
权重，不会生成哈希常量。

在存放模型的机器上，从仓库根目录执行：

```bash
python3 -m venv /workspace/dsv41-tools
/workspace/dsv41-tools/bin/python -m pip install huggingface_hub
/workspace/dsv41-tools/bin/hf download smalinin/DeepSeek-V4.1-Flash-GGUF \
  --revision d1de55c19f95172c882906cc83c0e55932d26a63 \
  --include "Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-*.gguf" \
  --local-dir /workspace/models/deepseek41-q2-q5
```

`hf download` 会保留 `Q2_K-Q5/` 子目录。推理前须校验每个完整文件；以下 SHA-256
来自固定 revision 的发布者 LFS 标识。此操作会读取全部 312.349 GiB，请在计时测试之外执行：

```bash
(cd /workspace/models/deepseek41-q2-q5/Q2_K-Q5 && sha256sum --check - <<'SHA256'
8126b49dfcfde02cb3db24b6f98d56b031f0a34eaca2509fc7b2b9d362cf0ac8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf
22bb293aee509a348ce32a739e006fa41f2348c6bcfafa3be76a3ee079eadf96  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00002-of-00010.gguf
db894848b4f14d42c39e18faa907c737cd4850f2fcb9deff9d5ebd86f580aaf7  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00003-of-00010.gguf
655a3400f2c092d6e3c11b8b18bf319b29563e59b960337115266574321953e3  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00004-of-00010.gguf
4b9378c6819d1130517e8719026b34e5257f1f73bd1d50cf6f30243982100bac  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00005-of-00010.gguf
04f161084d82032c65247c02e6169784a757be7db9e068b0baba6833125b6bb8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00006-of-00010.gguf
b1b1bf3cfbbc7388ce49b5ced69c42a13c7c5c5d3d609ac86902897a08d3c88a  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00007-of-00010.gguf
fae7f35123ae3557034a541507bb9fc24fb62c2e16ff8441c32e8e0477743d5d  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00008-of-00010.gguf
781fd69e9dd17c09676865830523a2d077a97544d6a004429aab95d2570f8534  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00009-of-00010.gguf
fa5affb1f971cd6e7effad5684b73780472b486a46330cd7c5776f87c38fea97  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00010-of-00010.gguf
SHA256
)
```

任一哈希不匹配时不要继续推理。`eng/dsv41-verify-download.py` 是原七分片布局的历史
校验器，不能校验这套十分片文件。两张 Q5_K Engram 表共约 125.889 GiB；在 RAM 限额
57.74 GiB 的主机上，即使不计 CPU 专家也无法全部驻留。此类主机应使用
`TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=0`，并记录分页开销。CPU 专家
卸载层数须依据实际设备容量选择，同时为可选草稿器与视觉编码器预留空间。

**历史基准的来源：** 本卡片中的早期结果与整文件 SHA-256 记录
`docs/validation/deepseek41/checkpoint-sha256.json`（本地验证记录，未提交到 Git）对应
原 `vcruz305` 发布的 revision `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` 及其旧版
第一个分片。较晚的七分片示例使用 `58d8ac86298fdf85a2440defee08b1abcad32e45`；
旧 Q4_K_M 的放置记录也须单独看待。保留的历史启动命令指向这些旧文件，而非修复版。
不能把它们的哈希、246.35 GiB 大小、约 60 GiB Engram 预热、质量结果或吞吐率归给
Q2_K-Q5。每次新验证都须记录 revision 与全部分片哈希。

## Unsloth 的 DeepSeek-V4.1-Flash-1017 GGUF

[unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/60ad3651641b5c97679c807e064ab1d7a3ca1455)
为 llama.cpp 打包了 1017 检查点。TensorSharp 在 `--backend ggml_cuda`（以及 `ggml_cpu`）上按发布原样加载
本轮已验证的 `UD-IQ2_XXS` 文件；其布局与上文有四处不同，均由原生加载器处理：

- Engram 表是一个张量 `per_layer_token_embd.weight`（Q5_K，独占一个分片），由
  `deepseek41.ple.*` 元数据描述。加载器把它切成逐层的表，并像 `engram.py` 一样，
  用 token 映射压缩原始的 `engram.pad_token_id`。
- `blk.N.engram_gate.weight` 存的是 Engram 的 `q_weight * k_weight` 乘积，这也是参考
  实现使用这两个权重的唯一方式。
- DSpark 草稿器 `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` 是 `dflash` 文件。它的噪声
  token、Markov 秩、块大小、专家数与目标层来自目标模型的 `deepseek41.dspark.*`
  元数据，与之不符的草稿器会被拒绝。
- 视觉塔以 llama.cpp 的 `mmproj-BF16.gguf` / `mmproj-F16.gguf`（`clip`，projector
  `deepseek41v`）发布，`--mmproj` 可直接加载；图像 token 的路由偏置取自文本分片
  （`blk.N.exp_probs_b_vl.bias`）。

托管执行器 `--backend cuda` 与 `--backend cpu` 以同样方式读取这种布局的 Engram 部分：合并表中每层的切片
成为所在分片的视图（不复制任何数据），`engram_gate` 缩放 query，key 保持纯 RMS 归一化。1017 的路由
专家仍使这些文件无法在 `cpu` 上运行。`cuda` 在专家留在显存中时能解码全部类型：`UD-IQ2_XXS` 文件的 10 层
截断版本已在它上面按[逐阶段 oracle](#对照-ggml_cuda-的逐阶段-oracle) 运行过，完整文件也已在两个各有两张 A40 的节点上对照 `ggml_cuda` 运行过（[完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行](#完整-1017-在-cuda-上的-2-节点-x-2-a40-运行)）；见[后端](#后端)。本轮已完成的加载与分布式对照覆盖 `ggml_cuda` 与 `cuda` 上的 `UD-IQ2_XXS`，其他量化版本需要单独验证。修订 `60ad3651…` 的
`UD-Q4_K_XL` 含 Engram 类型 43，测试所用未修改的 ggml `3612c0af…` 不支持该类型，
实际加载失败；详见[验证报告](../deepseek41_validation.md)。

每次调用只传一个 `--include` 模式（`hf` 2.2.0 在一次调用中给多个模式时只匹配到了
根目录文件）：

```bash
REPO=unsloth/DeepSeek-V4.1-Flash-1017-GGUF
REV=60ad3651641b5c97679c807e064ab1d7a3ca1455
DIR=/workspace/DeepSeek-V4.1-Flash-1017-GGUF
for pattern in "UD-IQ2_XXS/*" dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf mmproj-BF16.gguf; do
  hf download $REPO --revision $REV --include "$pattern" --local-dir $DIR
done
(cd $DIR && sha256sum --check - <<'SHA256'
22ad51c224e499883c1a753ad35719af44c296bbc8bcb9c931de6bdf950fa3aa  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf
6880d0483490504a530b72686d79c92daa46e58ca5b71763bbfa3778d8a6131a  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00002-of-00006.gguf
04b3ec7947255efaf789da77557da9180e93128948644828798d07b8a6e7a60a  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00003-of-00006.gguf
318762f39cd7ebf03e3d7b30feaf938fab0464a9c2d6c0e0e99fc030bbfcd820  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00004-of-00006.gguf
34fa48fbe2237b1b71a4f18a04bbe10946f6b1c76b7bbff21c76da630da677b6  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00005-of-00006.gguf
406103485a7a7193337adee38b7178d3b636940aaa3ca033bc91e0d3687e3031  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00006-of-00006.gguf
6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a  dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9  mmproj-BF16.gguf
SHA256
)
```

`mmproj-F16.gguf`（`dad8ad13f1d9a25757e121592d7b4f8ca83620d1172048004998b7c0b964b18e`）
同样可以加载；其矩阵会转换回 BF16，除落入 F16 次正规数范围的权重外都是精确的。
UD-IQ2_XXS 分片共 300.5 GB，其中 Q5_K Engram 表占 125.9 GiB。在五张 46 GB 的 A40 上，
这张表无法驻留显存，只能主机映射，因此主机（或容器）内存上限必须容纳它；
`TS_DSV41_ENGRAM_WARM=1` 会在加载时就把它读入，而不是在最初几个请求中按页调入。

```bash
MAX_CONTEXT=32768 TS_DSV41_ENGRAM_WARM=1 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model $DIR/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  --draft-model $DIR/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --mmproj $DIR/mmproj-BF16.gguf \
  --backend ggml_cuda --layer-split 5 --host 127.0.0.1 --port 5000
```

### 2026 年 10 月 22–23 日 UTC 的测量

该轮测试使用五张 46 GB A40（CUDA 12.8 工具链，驱动 595.91），主机为 96 核、503 GB
内存，容器限额为 233.8 GiB（包含页缓存），模型存放在 NFS 上。参照为 unsloth 的 llama.cpp
（`be8365722ea432809818fa21a4ecebfac2e7d298`，`-ngl 99 -sm layer -lm dio`）。
TensorSharp 使用未修改的 ggml `da8e2a54b919557b816b1c2beb432ccb766f7ce0`；
参照 fork 使用其树内的 ggml。每次加载前都清空页缓存。
TensorSharp 的测量使用 `ae60abf6` 加当时尚未提交的测试工作区改动，原生库 SHA-256
前缀为 `04bc9dd748378492`。实现随后提交为 `7f26e2d1`；这些结果并非在该提交或某个
公开发布版本上重新测得。复查修复后的原生构建 `8b910b29e88d6891` 只确认了默认 TP 下
2,048 个随机 token 的测试单元（prefill 1,046.3、decode 36.2 tok/s），没有重跑完整的服务端矩阵。

TensorSharp 使用默认设置，其中包括 YOCO
有界重放（近似 prefill，见下文）；“精确”行设置了 `TS_DSV41_YOCO_KEEP=0`。TensorSharp 的合成
测试数值取三次运行中的最好一次，llama.cpp 的数值为 llama-bench 的均值（TensorSharp 自身的均值在
2,048 与 8,192 个 token 时低 0.4-5%）。把命令中的
`--layer-split 5` 换成 `--tp 5` 即启用路由 MoE 张量并行，在这台机器上更快。合成测试行使用
随机 token；服务端测试行向两个服务端发送 wikitext-2 摘要提示（首行各不相同），贪心解码、
关闭思考，上下文均为 32,768 个 token（llama.cpp 的四并发行使用 `-c 65536 -np 4`）：

| 测量项 | TensorSharp `--layer-split 5` | TensorSharp `--tp 5` | llama.cpp |
| --- | ---: | ---: | ---: |
| Prefill，512 / 2,048 / 8,192 个随机 token（tok/s） | 500.3 / 844.3 / 906.2 | 779.8 / 1,046.7 / 1,082.7 | 227.6 / 284.8 / 310.8 |
| 同上，精确 prefill（tok/s） | 352.7 / 463.5 / 444.3 | 516.4 / 562.9 / 540.8 | |
| 深度 2,048 / 8,192 处的 decode（tok/s） | 34.5 / 34.2 | 36.4 / 36.1 | 32.6 / 31.9 |
| 服务端 prefill，约 1.8K / 7.3K token 提示（中位数 tok/s） | 697.4 / 762.8 | 960.5 / 1,000.7 | 174.6 / 225.7 |
| 同上，精确 prefill（中位数 tok/s） | 405.5 / 412.7 | 563.9 / 549.7 | |
| 上述提示之后的服务端 decode（中位数 tok/s） | 33.8 / 34.0 | 35.5 / 35.7 | 29.6 / 33.0 |
| 使用 DSpark 草稿器的服务端 decode（均值 tok/s） | 44.1 / 43.5 | 55.0 / 51.2 | 41.3 / 40.8 |
| 四个并发、约 1.8K token 的请求，各生成 256 个 token（整批耗时 s） | 27.3-33.1 | 21.2-25.5 | 43.3-46.8 |

服务端每个提示词尺寸都先舍弃一次预热，再测量三个请求。两个较长工作负载的输入分别为
1,747–1,812 与 7,181–7,375 个 token；两个引擎使用相同的原文片段与 token 数，每个请求
生成 256 个 token。每次请求首行各不相同，以避免前缀复用。Prefill 速率使用各引擎内部的
提示词计算计时（TensorSharp 的 `prompt_eval_duration` 与 llama.cpp 的 `timings.prompt_ms`），
并非流式 HTTP TTFT。TP 的精确与近似结果分别为：

| 单请求工作负载 | TensorSharp TP，精确 prefill | TensorSharp TP，默认 YOCO | llama.cpp 按层切分 |
|---|---:|---:|---:|
| 约 1.8K token 提示，prefill 中位数（tok/s） | 563.9 | 960.5 | 174.6 |
| 约 7.3K token 提示，prefill 中位数（tok/s） | 549.7 | 1,000.7 | 225.7 |
| 约 1.8K token 提示之后的 decode 中位数（tok/s） | 35.4 | 35.5 | 29.6 |
| 约 7.3K token 提示之后的 decode 中位数（tok/s） | 35.6 | 35.7 | 33.0 |

这里“精确”表示关闭 YOCO 截断，并不表示两个引擎逐位一致。在这两个工作负载上，精确
TP prefill 是参照速率的 3.23× / 2.44×；默认 YOCO 为 5.50× / 4.43×，但会改变输出分布。
llama.cpp 的服务端记录来自较早的一轮测试；三个约 1.8K 提示的 prefill 速率为 69.8、174.6
与 183.8 tok/s，其中的停顿没有进一步调查。这些小样本中位数不能作为普适加速比。

四并发行只测了默认 YOCO。三批完整请求的耗时中位数为按层切分 28.26 秒、TP 22.61 秒、
llama.cpp 44.52 秒，每批共生成 1,024 个 token。TensorSharp 的并发单请求 decode 计时会分摊
共享批次时间，因此不用于跨引擎吞吐对比。DSpark 行使用均值；接受率依赖生成的文本，而
YOCO 可能改变文本。

在较早的一次 1017 CLI 运行中，一个简短代码提示（58 个 token，生成 512 个；按层切分、精确 prefill）的草稿器接受率为 80%，速度 60.2 tok/s
（llama.cpp `--spec-type draft-dspark` 每次起草 3 个 token 时为 51.6 tok/s）。在相同 token
上做 teacher-forced 的 wikitext-2 困惑度：4,608 token 窗口，前 2,304 个 token 作为提示
prefill（超过稀疏 prefill 门控），其余逐 token 打分，共 10 段。llama.cpp 为 2.4697；
TensorSharp 在默认设置下按层切分为 2.4970、`--tp 5` 为 2.4978，精确 prefill 时为 2.4820 与
2.4811。相对 llama.cpp 分布的平均 KL 散度默认为 0.0172，精确时为 0.0160（top-1 一致率
96.2% / 96.3%）。TP 与按层切分之间的平均 KL 散度在精确 prefill 时为 0.0005，默认设置下为 0.0004。
在同一份共计 23,030 个打分 token 的样本上，默认 TP 的困惑度比精确 TP 高 0.67%，比
llama.cpp 高 1.14%；精确 TP 比 llama.cpp 高 0.46%。这只是 wikitext-2 的质量样本，不能作为
通用质量保证。在 YOCO 与 TP 工作之前，只含布局
支持、不含任何提速改动的构建为 2.4752 与 0.0167（精确 prefill）；按段配对，最终精确构建与它的
困惑度差异不显著（t = 1.02），KL 散度更低（t = -3.01）。原始记录是 `docs/validation/deepseek41-1017/` 下的本地验证记录（未提交到
Git）。

## 准备可选的视觉伴随文件

使用 unsloth 的 1017 GGUF 时无需准备：`--mmproj mmproj-BF16.gguf` 对图像的编码与用官方
1017 检查点准备的伴随文件逐位相同（`mmproj-F16.gguf` 因 F16 对约 0.02% 权重的舍入而略有
差异）。下面的伴随文件用于不带视觉塔的转换。
smalinin 发布的 GGUF 转换过程丢掉了视觉塔、aligner、可学习的图像分隔符以及逐层的视觉路由
偏置。官方发布把约 970 MB 的视觉/aligner 权重放在一个独立分片里，因此可以在不下载
原始文本权重的情况下准备它们：

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2 gguf
/workspace/dsv41-tools/bin/python eng/dsv41-prepare-vision.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5 \
  --parent-model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --repository deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

它会在文本分片旁边生成 `deepseek41.vision.gguf` 与
`deepseek41.vision.json`。`--parent-model` 提供 GGUF 的分词器指纹；分词器一致时，
已有的视觉伴随文件仍然兼容。
该脚本下载独立的视觉分片，以及分隔符/路由张量所需的少量字节范围，并保留 BF16/F32
存储。它会用 LFS 的 SHA-256 校验完整的独立视觉分片，并记录分隔符/路由权重各自的
范围哈希；它不会下载或校验完整的原始文本分片。
伴随文件溯源记录 `docs/validation/deepseek41/vision-companion.json`（本地验证记录，未提交到 Git）
包含全部 306 个张量、来源 revision、字节范围与输出摘要。
原生加载器在挂载前会检查父模型的分词器指纹与模型维度。要启用图像，在下文的服务命令
中加上 `--mmproj /workspace/models/deepseek41-q2-q5/Q2_K-Q5/deepseek41.vision.gguf`。
没有挂载伴随文件时，图像能力保持关闭。

视觉部分默认使用稠密 F32 注意力，与官方视觉塔的注意力算术一致。在 Ampere 及更新的
NVIDIA CUDA 上，BF16 矩阵输入会保持 F32 累加与 F32 输出，直到加偏置并做 BF16 激活
舍入为止。`TS_DSV41_VISION_BF16_GEMM=0` 选择诊断用的 F32 提升矩阵路径。
`TS_DSV41_VISION_FA=1` 选择更快的、使用 F16 中间量的 flash attention；在真实图像的
参考对比中，这条路径的特征差异更大。这些选项只影响图像编码器，与文本注意力无关。
精确边界与实测取舍见
[验证报告](../deepseek41_validation.md#independent-vision-and-mixed-modality-reference)。

做数值排查时，`TS_DSV41_VISION_TRACE_DIR=/绝对路径` 会写出 F32 的 patch、block、norm
与 projector 输出。开启 trace 会保留中间张量并增加设备传输，正常推理与基准测试时请
保持不设置。

编码器使用 32 层双向 transformer、二维旋转位置、带 padding 的 3×3 空间 merge，以及
两层 projector。它输出完整的图像 span，包括可学习的起止标记与逐行的 newline embedding。
图像 token 使用视觉 MoE 偏置，抑制 Engram 注入，并在图像 span 处打断 Engram 的 n-gram。
图文混排的 span 可以跨越 prefill 微批边界。已有的视频抽帧走同一条图像路径。
CPU/CUDA 数值 fixture 与精确预处理检查均已通过。
完整检查点的 CUDA 按层切分运行通过了并发 1/4 下的 25 个图像/视频请求，以及一个长文本
之后的单独图像请求。最终的 routed-TP 配置同样通过了并发 1/4 下的全部 25 个图像/视频
请求。严格的编码器 parity 单独跟踪。
官方配置没有提供音频解码器。V4.1 会拒绝带音频的请求（包括图像+音频的混合请求），
而不是忽略音频。Chat Completions、Responses 与 Web UI 都返回 HTTP 400。
OpenAI 解析器在读取音频负载之前就拒绝音频部分（包括缺失或格式错误的音频），因此后续
的音频附件不会把先前已上传的图像留在半途状态。

OpenAI chat 端点接受包含 base64 图像 data URI 的 `image_url` 部分，支持多张图像以及
历史轮次中的图像。它的 V4.1 `video_url` 扩展会通过既有的视频解码器对 base64 的 MP4、
WebM 或 MOV 采样。例如，消息的 content 数组里可以包含：

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1,"max_frames":3}}
```

每个采样帧成为一个图像 span，并带上以帧序号除以探测到的帧率算出的源时间（秒）。
对可变帧率的片段，时间标签是近似值。
帧保持原始顺序；这是抽帧，不是原生的时序编码器。`fps` 必须大于 0 且不超过 60，
`max_frames` 取 1–64。默认值取自 `VIDEO_SAMPLE_FPS` 与正的 `VIDEO_MAX_FRAMES`，
否则为 1 fps 与 16 帧。超过上限的帧会在整个片段上均匀采样。远程 HTTP 图像/视频 URL
不会被拉取，请发送 data URI。对 V4.1 而言，`/v1/chat/completions`（`image_url`）与
`/v1/responses`（`input_image`）都会在开始流式输出或写入任何图像之前，以 HTTP 400
拒绝远程或格式错误的图像 URL 以及无效/空的图像 base64。
合法的图像 data URI 保持既有的解码与上传路径。展开所有图像之后，文本上下文预算依然
适用。
最终的 routed-TP 主机（托管侧 stage 3,651，原生 `6b3b5ab3…`）通过了全部八项
图像拒绝检查 `docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-image-input-rejections.json`
与四项 Responses 音频拒绝检查 `docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-audio-input-rejections.json`（本地验证记录，未提交到 Git）。
无论流式还是非流式请求，对远程图像 URL、格式错误的图像 base64、以及形状合法或格式
错误的音频部分，都返回 JSON 400 且不产生 SSE。更早的 CPU-offload 主机的八项图像检查
另行保留。

## 准备可选的 DSpark 伴随文件

V4.1 需要 V4.1 草稿器：即这里准备的 `deepseek41-dspark` 产物，或 unsloth 随 1017 GGUF
发布的 `dflash` 文件（见[Unsloth 的 DeepSeek-V4.1-Flash-1017 GGUF](#unsloth-的-deepseek-v41-flash-1017-gguf)）。
不能使用 V4 草稿模型。以下官方 revision 将 DSpark
放在分片 44–46 中（7,933,129,808 字节），无需下载原始文本模型：

```bash
/workspace/dsv41-tools/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --include config.json model.safetensors.index.json \
    model-00044-of-00048.safetensors model-00045-of-00048.safetensors \
    model-00046-of-00048.safetensors \
  --local-dir /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash

(cd /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash && sha256sum --check - <<'SHA256'
8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879  config.json
74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8  model.safetensors.index.json
9a6b39fb88a2510487a8efaef77aa7864e8061f6b62c95a0f010e9dd538f3b05  model-00044-of-00048.safetensors
0cc9d5f6ca3a2158ccc63ce2c70c76aeda8177d54913340481af566680329eb5  model-00045-of-00048.safetensors
e625902027b9d23d416f8818c665fab4704e0b96dc1bc778321601b700475a9d  model-00046-of-00048.safetensors
SHA256
)
```

五项校验全部通过后，再转换伴随文件：

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2
mkdir -p /workspace/models/deepseek41-dspark
/workspace/dsv41-tools/bin/python eng/dsv4-dspark-to-gguf.py \
  --checkpoint /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash \
  --expert-type mxfp4 \
  --out /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
```

已审计的转换产物包含 81 个张量（包括三个视觉路由偏置），大小为 7,940,628,416 字节。
来源标识与转换校验已经完成。真实 DSpark 草稿器已在 `ggml_cuda` 双 GPU 按层切分与
实验性路由专家 TP 下通过初步文本/图像 HTTP 检查，图像回答完整结束，服务正常退出。
大量磁盘换页下的通用质量与吞吐仍未获验证。另行进行的 24-token 文本/图像配对检查，
在两种模式下均匹配普通解码的 token ID 与 `max_tokens` 结束原因，DSpark 实际参与解码
且进程正常退出；这些结果仅涵盖有限续写，不声明提速、跨节点执行或完整模型 TP 通过。
在使用修复版十个文本分片的启动命令中，加上
`--draft-model /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --spec`。
对比普通解码时仍加载同一伴随文件，改用 `--no-spec`。草稿器驻留在输出头所在 GPU，
会影响设备放置；视觉编码器还需要额外空间。

## 运行已实现的路径

需要安装 .NET 10 SDK、CMake、C++ 编译器，以及 `PATH` 上带 `nvcc` 的 CUDA 工具链。
从仓库根目录构建。此示例使用 A40 的 CUDA 架构 8.6；换其他 GPU 时，两处架构值都要改。
命令指向修复后的下载；仍须在实际硬件上验证容量与推理行为：

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES=86 \
  bash TensorSharp.GGML.Native/build-linux.sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
  -p:CudaArch=compute_86 -p:TensorSharpSkipGgmlNative=true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_CPU_MOE_THREADS=32 TS_DSV4_UBATCH=256 \
  TS_DSV41_ENGRAM_WARM=0 \
  TS_DSV41_COMPACT_RAW_GATHER=0 KV_CACHE_DTYPE=f16 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=256 TS_SCHED_SOLO_PREFILL_CHUNK=8192 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cuda --layer-split 8 --port 5000
```

宿主构建会把原生库复制到服务端 DLL 旁边。上面是显式指定微批与调度器设置的启动示例，
并非修复版检查点的实测配置；验证报告中的历史配置使用不同权重，部分参数也不同。
稀疏 prefill attention 不需要任何
开关：它在这条路径上默认开启，`TS_DSV41_SPARSE_FA=0` 可将其关闭。`TS_DSV4_UBATCH=256`
固定该示例的宽度；不设置则由加载器自行选择（见[后端](#后端)）。`TS_CPU_MOE_THREADS` 要按可用的
CPU 配额来选，并为每次运行记录下来。即便是纯 GPU 放置也要在启动环境里设置它：原生的
CPU 图工作与主机侧归约仍会影响延迟。当前 CLI 也接受 `--cpu-moe-threads N`；两者都给
时请填相同的值，因为原生加载器优先采用为正的环境变量值。

在没有显式线程设置时，纯 GPU 的 V4.1 加载使用调用方的 `TS_DSV4_THREADS`，默认上限为
32；CPU 专家卸载则改用探测到的可用 CPU 并行度。更早的原生构建收不到 CLI 的线程覆盖。
具体来说，首次实测 TP 运行中的 `--cpu-moe-threads 48` **并没有**配置 48 个原生线程：
根据加载器源码，在没有原生环境变量覆盖、也没有 VM 覆盖探测的情况下，其原生线程池推断
为默认的 32。那台原始主机没有暴露线程池宽度的读取接口，所以 32 并非在那里直接测得。
请把这个基线与后续显式设置线程数的实验区分开。

`--layer-split 8` 表示**用八张 GPU 做按层切分**。启动诊断会说明采用的放置模式。
TensorSharp 默认按可用显存分配整层。用 `CUDA_VISIBLE_DEVICES` 精确指定本次运行使用的设备。

`--tp 8 --backend ggml_cuda` 会在这八张 GPU 上启用 **routed-MoE 张量并行**；`--tp` 支持 2 到 8 的每个度数（含 3、5、6）。
该度数作为按模型参数传给原生加载器，且必须等于它选中的 GPU 数。`--tp` 与 `--layer-split` 不能组合。

在这种模式下，路由专家的 gate/up 矩阵沿 FFN 中间维切分，down 矩阵按输出行切分，并在所有
选中的 GPU 上并发执行。先按 F32 精度拼接完整 SwiGLU 激活，再计算 down 的各输出行，最后
拼接互不重叠的输出。这保留完整的点积归约次序，避免舍入差异被后续激活量化放大。量化块从不被
切开，不均匀的条带在各层之间轮换以平衡存储。注意力、共享专家与各类 cache 仍保持按层
放置。该张量并行路径不切分注意力；在 `ggml_cuda` 上可通过现有的
`--tp-node-id` / `--tp-peers` 分布式组跨节点切分路由专家。
`--tp N` 表示每个节点的 GPU 数量，各节点必须相同（1–8），全局支持 2–16 个 rank。
该范围是执行器容量，仍受 GGUF 量化块形状限制：2304 宽的 K 量化 down 行只有九个
256 元素块，超过九个全局 rank 无法形成非空分片，因此两节点各八卡的布局会被拒绝。
已完成的跨节点 fixture 验证覆盖两节点各一张或两张 GPU。
跨节点模式每个 GPU 驻留 MoE 层进行两次 F32 激活与输出行拼接，网络延迟会影响解码速度。
跨节点时这一交换由主机驱动：下文描述的设备侧交换只用于本地 TP，各 rank 的条带经主机 F32
缓冲区与每层两次阻塞集合通信传输（数据量见[当前限制](#当前限制与张量并行工作)）。
分布式请求使用一个镜像序列并串行执行，不支持序列槽、前缀保留/截断、视觉或 DSpark，
也不能与 `--layer-split` 合用。请参阅[英文部署说明](deepseek41.md#distributed-routed-moe-tensor-parallelism)
与[固定 token 验证流程](../../eng/validation/DeepSeek41DistributedProbe/README.md)。

在 `ggml_cuda` 上，路由专家的交换驻留在设备侧（`ggml_ops_deepseek41_tp_cuda.cu`）。每个 TP
层在本层 GPU 上是一个 launch 节点和一个 gather 节点：launch 暂存本层激活并下发到每个
rank，本层 GPU 随后在各 rank 运行期间计算共享专家，gather 再等待各 rank 的输出行。每次
传输都由各 GPU 自己的拷贝引擎经可移植的 pinned 主机内存中转，只用 CUDA event 排序，因此
主机在层内从不等待，并且每个 rank 由一个线程并发下发。它不需要点对点访问；某些 PCIe
主机（包括下面的五卡 A40 VM）声明了点对点访问，但实际不可用。输出与它所取代的主机驱动
节点逐位相同，并在每一层与按层切分的 MoE 逐位相同（`GgmlOpsDsv41TpTest`，包括在真实
检查点上的 `--model-dir`）。某个 rank 失败（例如临时缓冲分配失败）时，该层输出变为 NaN，
请求失败：已经下发工作的 rank 会被排空，不会出现永久等待，下一个请求照常运行
（`deepseek41-tp-resident-failure-cuda`）。

在五张 A40 上用 unsloth 的 UD-IQ2_XXS 文件，`--tp 5` 在默认 YOCO 重放下以 1,046.7 tok/s
prefill 2,048 个随机 token（`TS_DSV41_YOCO_KEEP=0` 时为 562.9），decode 为 36.4 tok/s；
`--layer-split 5` 为 844.3（463.5）与 34.5。改为设备侧交换之前，`--tp 5` 的 prefill（精确）
为 402.3 tok/s，decode 为 15.2 tok/s。在约 1.8K 与约 7.3K token 的真实文本服务端提示上，其
prefill 中位数为 960.5 与 1,000.7 tok/s。在 teacher-forced 的 wikitext-2（4,608 token 窗口）
上，TP 与按层切分的下一 token 分布之间平均 KL 散度在精确 prefill 时为 0.0005、默认设置下为
0.0004（中位数 0，top-1 一致率 99.8%），而两者与 llama.cpp 之间为 0.016 与 0.017。unsloth 的 llama.cpp 构建在这个模型上没有可用的张量切分：
`-sm row` 拒绝加载（`device CUDA0 does not support split buffers`），`-sm tensor` 在为第一个提示
批次分配计算图时中止，因此上面的对比针对的是它的按层切分。decode 与按层切分接近：在一次 nsys
decode 剖析中，每层仍在一张 GPU 上的 Q8_0/Q6_K 稠密投影（注意力、共享专家与输出头）占 GPU 内核
时间的 31%，而在这个量化下每个 token 的六个路由专家在五张 GPU 上合计占 21%。下面的主机驱动
测量早于这项改动；另见[实测放置配置](../deepseek41_validation.md#full-checkpoint-routed-moe-tp)。

TP 每层上传之前，按 `TS_DSV4_LOAD_THREADS`（默认 16，每线程 64 MiB 临时缓冲）
顺序读取当前层三个路由专家张量的文件范围，跳过已驻留的页面，避免 rank 跨行读取时触发
大量网络文件缺页。只准备当前层，不预读整个检查点；`TS_DSV4_WARM_PREAD=0` 保留此阶段
原有的直接映射读取。

2026 年 9 月 29 日，在六张 A40 的 VM 上以同一构建、同一 TP6 放置，按
`TS_DSV4_WARM_PREAD=0,1,1,0` 的模型启动顺序测得加载时间（不含内核预热）为
**546.26、149.61、157.35、510.88 秒**。关闭/开启预读的中位数分别为
**528.57/153.48 秒**，描述性比值为 **3.444×**。每次启动前，都要求 120 个路由专家
张量中 183.25 GiB 的完整页面在客户端内核页缓存中的驻留数为零。配置、检查点、源码、
原生库及托管运行时均保持一致；129,280 个首轮预填充 logits 和单 token 贪心检查全部
逐位相同，进程均正常退出。第四次缓存准备起初残留 135 页，因而没有启动模型；另行记录
的续跑在第一次尝试中满足同样的零驻留条件，提供了第四个观测值，原失败记录仍保留。
这是每种设置两次启动、途中中断的对照，并非连续的 ABBA 试验。部分边界页及 MooseFS
用户态、网络和服务端缓存未受控，因此不能把该比值称为冷存储或通用加速比。

没有 TensorSharp 融合 CUDA 后端时（`TS_DSV4_FUSED=0`），TP 节点在 CPU 后端上由主机驱动
各 rank，并在每个阶段同步。在这条路径上，当 NCCL 选择 `NCCL_P2P_DISABLE=1` 时，不超过
16 个 token 的批次使用可复用 pinned 主机激活缓冲进行拼接，更大批次使用私有 F32 NCCL
拼接（首次使用时才创建）。该阈值来自六张 PCIe A40 的
配对实测，不会用于启用 P2P 的配置。`TS_DSV41_TP_HOST_TOKENS=0` 强制使用可用的设备
拼接，便于对比；0 到 4096 的整数设置主机拼接阈值。`TS_GGML_TP_F32_NCCL=0` 对所有
批次使用主机回退。两条路径均保持 F32 位模式。设备拼接沿 rank 的 CUDA stream 顺序
执行，省去 gate/up 后单独等待；异常返回前仍等待每个 rank 完成。所选放置与传输方式的
整模型吞吐仍需实测。

较早的 2026 年 9 月 29 日 UTC 主机驱动传输验证使用六张 PCIe A40、修复版十分片 EngramQ5/Q2_K 检查点、
4096 上下文、F16 KV、关闭预热的主机 Engram 表，以及 `NCCL_P2P_DISABLE=1`。
每种放置启动一次进程，执行五轮相同固定输入：512 个 prefill token、128 个 decode token。
候选原生库 `473ee64d…` 通过运行时文件完整性检查；ggml 上游保持 `353b63b4…` 且未修改。
三次运行的完整 128-token 非计时贪心序列均与原始按层切分基线相同。

该主机驱动实验早于上面的设备驻留交换，不能用于选择当前最快的放置方式。原始计时行保留在
[验证报告](../deepseek41_validation.md#full-checkpoint-routed-moe-tp)中。

在该实验中，TP 的容器内存约为 255 GiB，按层切分约为 70 GiB。其中文件缓存约为
252 / 68 GiB，匿名内存约为 1.6–1.7 / 0.8–0.9 GiB。TP 上传临时缓冲有界，但已上传的
路由权重页面当时仍保留在可回收的文件缓存中：`TS_DSV4_LOAD_DROP_CACHE` 当时仅作用于普通
按层上传。自 2026-10-23 起，TP 上传路径遵循同一规则：当源页面与主机映射无法同时放入主机
内存限额时，每层分片上传后即释放其源页面（日志 `[dsv41-tp] expert source page cache: ...`）。因此，显存足够并不代表主机内存限制也足够。
观测到的 255 GiB 也不代表最低内存要求。这几次顺序加载的源页面驻留状态不同，不能作为
受控的加载延迟对比。

当前按输出行切分的实现通过了两张和六张 A40 的数值 fixture，包括双卡 F32、BF16、F16、
Q2_K、Q3_K、Q4_K、Q6_K，以及六卡真实修复版检查点的专家权重。输出行分片现在按 128、64
或 32 行对齐：此前的 2 行回退让 256 宽 F16 fixture 在三卡和五卡时的 down 投影走 ggml 以
F16 累加的回退路径（相对 L2 `2.8e-3`），这正是那两个 CUDA 用例失败的原因。在五卡 A40 VM
上，CUDA fixture 在 2、3、4、5 个 rank 下全部通过，设备驻留节点与 launch/gather 节点对
都与主机驱动节点逐位相同。CPU fixture 覆盖 2 到 8 个
rank；验证 VM 只有六张 GPU，未执行七卡与八卡 CUDA 测试。最终 `473ee64d…` 运行时
使用修复版十分片 Q2_K/Q5 检查点通过六卡 HTTP 验证：16 个文本响应与未修改的按层
切分基线完全相同，四个严格工具场景、三个图像场景全部通过，129,280 个首轮 prefill
logits 逐位相同。DSpark 与 ngram 在文本、图像场景都保持全部 96 个贪心 token 及结束
原因一致。草稿/接受/校验步数分别为：两个 ngram 场景均 49/49/10，文本 DSpark
83/73/19，图像 DSpark 40/22/13；DSpark 分别实际执行七次与九次回滚。HTTP 与投机
测试均正常退出，运行时文件未改变。这些计数证明实际执行了投机路径，不代表投机
吞吐加速。这些检查不能说明完整检查点能装进两张或四张 A40。
当历史七分片 Q2_K 的 Engram 表使用主机映射时，同步预热会在就绪之前占用约 60 GiB 主机页缓存；
驻留 GPU 的表跳过这一步。冷加载与预热时间要与热态吞吐分开记录。

在 CUDA 上，量化 gate/up 与 down 输出行分片走 TensorSharp 自有的量化分片 kernel
（`ggml_ops_matmul_quant_strip.cuh`、`tsg_matmul_id_quant_pair`）：它只读取本 rank 的权重分片，
但保留未切分发射的 stream-k 划分与归约顺序，因此每个分片的 gate/up 行与完整张量逐位相同。
在此之前，ggml 的批量 MMQ 路径对分片的 F32 求和分组与未切分发射不同，down 投影的 Q8 激活
重新量化又把它放大成检查点形状 Q2_K/Q3_K 在 16 token 时的失败（相对 L2 `3.9e-5`，完整权重
容差为 `1e-5`）。`GgmlOpsDsv41TpTest` 仍以严格的完整权重参考及其原始容差为通过标准，同时
记录同设备按分区求值的结果，`--cuda 1 --quant-strip-only` 检查 gate/up 逐位相等以及 scratch
增长/失败恢复。非对齐的分片形状仍走 ggml 路径。已记录的微基准中窄分片每次 MoE 调用慢
15-36%；这描述的是此前切分 down 归约维的实现，并非当前的两次拼接实现；见
`docs/validation/qualification-2026-09-16/numerical-tp-chosen-r1/README.md`（本地验证记录，未提交到 Git）。

如果权重与上下文放不下，加上 `--n-cpu-moe N` 把前 N 层的路由专家留在主机上，或者用
`--cpu-moe` 卸载全部路由专家。注意力、路由与共享专家仍在 GPU 上。
[Engram 表的放置](#engram-表放在哪里)单独选择；使用主机映射时，每个输入批次只读取并
传输选中的 embedding 行。CPU MoE 卸载与按层切分都已实现，但它们在你所用硬件与上下文下的
吞吐需要实测。与 `--tp N` 组合时，
被 CPU 卸载的前置层保留完整的 CPU 专家，其余层使用路由专家分片。

原生版本 `6b3b5ab3…` 显式把共享专家的 gate/up/down 投影指派到该层所在设备。这修正了
更早的调度器放置问题：在经过 CPU 卸载或 TP routed 分支之后，共享的 gate/up 计算可能被
送到 CPU。相关的放置与数值检查
（`docs/validation/deepseek41/shared-expert-placement/README.md`，本地验证记录，未提交到 Git）
在两张 GPU 上通过了 597/597。更早的完整检查点放置基准仍保留其原始二进制与结果。
使用修正后放置的最终 CPU-offload、routed-TP 与按层切分配置均已完成，见放置记录
`docs/validation/deepseek41/final-placements/README.md`（本地验证记录，未提交到 Git）。

要做热态的 CPU 卸载基准，请在模型加载之后、开始计时请求之前，单独把被卸载的专家页读入：

```bash
/workspace/dsv41-tools/bin/python eng/dsv41-warm-experts.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --layers 4 \
  --report cpu-expert-warming.json
```

这需要 `gguf` Python 包，并会读取所选各层的全部三个路由矩阵。它的 I/O 时间要单独记录。
这些页仍可被回收；Engram 预热本身不会预热 CPU 专家权重。

### Engram 表放在哪里

默认情况下两张 Engram 表**驻留 GPU**：各自加载到拥有对应层的设备上，计算图用
`get_rows` 直接在量化表上取行。主机完全不读行，因此 CPU 上不做任何反量化，只有行号
跨越链路。启动会打印
`Engram lookup: GPU-resident tables, gathered in-graph`。

之所以能这样，是因为 TensorSharp 保持检查点自身的量化格式。一行 256 个值的 Q2_K 只有
84 字节，因此两张 3.84 亿行的表合计 60.1 GiB。vLLM 与 SGLang 把同样的行存成 FP8 值加
每 32 个元素的块 scale，一行 264 字节——这也是它们默认使用主机表加 FP8 gather kernel
的原因。

放置策略是自动且保守的：这些表会计入按层切分的装箱；如果把它们放上 GPU 会逼出任何
路由专家的 CPU 卸载，它们就改为主机映射，并在启动时说明。设置
`TS_DSV41_ENGRAM_DEVICE=0` 强制主机映射，`=1` 要求驻留 GPU、放不下就失败。
`TS_DSV41_ENGRAM_THREADS`、`TS_DSV41_ENGRAM_WARM` 与 `TS_DSV41_ENGRAM_RANDOM` 只对
主机映射有效；在 GPU 驻留路径上设置了它们，启动时会给出提示。

驻留 GPU 可以省去主机表预热，但容量代价取决于检查点。上面的 1017 五卡 A40 测量使用
主机映射的 Q5_K 表，旧八卡 A40 的 Q2_K 吞吐不能预测该布局。较早的放置检查中，主机映射
（预热或不预热）及 GPU 驻留三条路径在 temperature 0 下产出的文本完全一致。

那是 argmax 层面的稳定，而不是逐位一致：ggml 的 CPU 与 CUDA Q2_K 反量化计算同一个
表达式，但设备端可能把乘减合并成 FMA，从而相差最多一个 ulp。如果某次运行必须与 CPU
oracle 逐位一致，请使用 `TS_DSV41_ENGRAM_DEVICE=0`。

### 主机映射的 Engram 表

以下选项仅在表留在主机上时生效——GPU 装不下，或显式设置了
`TS_DSV41_ENGRAM_DEVICE=0` 时会发生这种情况。

在网络存储上，首次访问稀疏的 Engram 行会主导 prefill 与 decode 延迟。每个 token 在每张
表上要哈希出 24 个行号，任何不在页缓存中的行都是一次存储往返：在八卡 A40 上的网络挂载
Q4_K_M 权重上，这意味着**每个 1024-token 的 prefill 分块要花 1435–2565 毫秒做输入准备**，
而页面驻留后只需 56–81 毫秒。decode 的输入准备冷态为每 token 5–15 毫秒，热态为 0.9 毫秒。

因此只要表在主机映射上且页面能够常驻，**预热就是默认行为**。它在模型开始服务之后于独立
线程上运行，而不是在加载期间，所以启动时间不变，预热进行期间请求照常工作（只是更慢）；
启动时会打印预热的数据量，完成时再打印一次。每次模型加载都会把整份权重读过页缓存，从而
挤掉上一次运行的 Engram 页面，这就是为什么每次加载之后都要重做，而不是每台机器做一次。

预热只读这些表，不创建私有副本或 pinned 分配，页面仍可被操作系统回收。当映射的主机权重
再加 8 GiB 装不进探测到的主机/cgroup 内存额度、或 `MemAvailable` 本来就留不住这些表时，
预热会被跳过并给出诊断。

预热用 `pread` 以 64 MiB 为块读取每张表，每个读取线程（`TS_DSV4_LOAD_THREADS`，默认 16）
负责表中一段连续区间，并跳过 `mincore` 报告页面已全部驻留的块，因此表仍在缓存中时重新
加载几乎不读任何数据。过去的做法是每 4 KiB 页触碰映射的一个字节，而在网络文件系统上每一次
这样的缺页都是一次受挂载点预读上限约束的同步读（A40 虚拟机的 MooseFS 挂载为 128 KiB）：
七卡 A40 通道同步预热 103 GiB 的 Q4_K_M 表用了 311.3 秒。在同一台虚拟机上实测，对一段已
驱逐的 8 GiB 表区间，`pread` 为 2.38–2.54 GiB/s，逐页触碰为 0.63–0.69 GiB/s；测量方法见
[加载时间](#加载时间)。`TS_DSV4_WARM_PREAD=0` 让下面两种形式都恢复逐页触碰。同步模式下的
读取错误会让加载失败并给出文件与偏移；后台模式下只打印一行日志，模型照常服务。

| `TS_DSV41_ENGRAM_WARM` | 行为 |
|---|---|
| 未设置（默认） | 模型开始服务后在后台预热；完成时打印 `[dsv41] warmed ... Engram pages in ...s (background, ...)` |
| `1` | 与此前一致，在加载期间同步预热；启动时间增加的就是读表的时间（七卡 A40 通道上逐页触碰 103 GiB 用了 311.3 秒；按实测的 2.24–2.54 GiB/s `pread` 速率计算约为 41–46 秒的读取） |
| `0` | 从不预热 |

稀疏读取的映射建议（`MADV_RANDOM`）只在预热结束之后才施加，无论采用哪种预热形式：该建议
会关闭预热本身所依赖的预读。`pread` 预热只把页面放进页缓存，并不把它们映射进进程；此后
查表第一次触碰某一行是一次次缺页（minor fault），而不是一次存储读取。在这样预热过的表上，
经 `MADV_RANDOM` 映射随机读取 2,000 个 144 字节的行，平均 0.0037–0.0056 毫秒，没有主缺页。

`TS_DSV41_ENGRAM_THREADS=1..32` 控制常驻查表工作线程数，默认取 16 与硬件线程数中的
较小者。prefill 与 decode 都使用并行取行：单个 token 在每张 Engram 表上要选 24 行互不
相关的数据。设为 1 会让读取串行化。并行取行避免了 decode 期间串行的缺页等待，且不改变
embedding 数值。执行器在同一个工作池任务中交错读取两张表，工作线程完成后再逐表上传行。
staging 以 64 MiB 为单位分组；单张更大的表沿用此前的一表分配上限。

在 Linux 上，并行取行会自动为映射的 Engram 区间请求随机访问建议，以减少不必要的预读。
该提示在可选的整表预热之后应用，除共享的边界页之外，不改变其他张量区间的策略。设置
`TS_DSV41_ENGRAM_RANDOM=0` 可关闭它，`=1` 可强制开启。不设置时，单线程配置保持默认的
映射策略。不支持的平台与被 OS 拒绝的提示都是非致命的。源码对比与配对的 scratch 文件
结果见 Engram 调查 `docs/validation/deepseek41/cli-gpu-execution/README.md`（本地验证记录，未提交到 Git）。

### 后端

`--backend ggml_cuda` 是服务路径：只有它为这个架构的融合算子提供了内核。

`--backend ggml_cpu` 在一个 CPU 设备上运行整个模型，这些融合算子退回到标量实现。
它的存在是为了在没有 GPU 时运行和检查该架构，而不是用于服务：这个检查点每层每个 token
要从 246 GiB 中读出 384 个路由专家中的 6 个。

`TS_DSV41_ALLOW_NON_CUDA_GPU=1` 额外允许 `ggml_vulkan` 与 `ggml_metal`。在那里，普通
计算图跑在 GPU 上，只有架构专属算子落到 CPU 后端，每出现一次就要一次主机往返。它需要
显式开启而不是自动生效，因为那道拒绝原本挡住的是静默回退到"第一个被枚举出来的 GPU"，
而不是一次明确的请求；启动时会逐个列出它作用到的设备。在你的硬件上实测之前，请把它
当作可移植性与正确性通道。

`--backend cpu` 根本不是 ggml 后端：它是纯 C# 的 `DeepSeek4CpuExecutor`，不依赖任何
原生库，也不需要 GPU，并且实现了完整的 V4.1 计算图——比例 1 与比例 2 的块压缩器、
共享的压缩 cache 与索引器 cache（含 lightning indexer 的 top-k）、候选块剪枝、
Engram 表、延迟 hyper-connection 门控、共享专家，以及检查点里训练好的 cache 量化
（原始行 FP8 E4M3、索引器 MXFP4、压缩 cache NVFP4）。它以 atol=rtol=2e-5 对齐独立的
PyTorch 参考实现 `eng/dsv41-reference.py`，并要求贪心 argmax 完全一致，覆盖一次性
prefill、分块大小 1/3/5/8 与 reset——不过那是 fixture 规模的对齐，而不是在已发布权重上
的等价。Direct CUDA 引擎 `--backend cuda` 同样用自己的内核、不经 ggml 运行 V4.1。它在一个
用 C# 生成、形状仿照 1017 的检查点（`InferenceWeb.Tests` 的 `Dsv41SyntheticModelBuilder`）上对齐
`ggml_cuda` 与 `ggml_cpu`（两者都使用精确 prefill，`TS_DSV41_YOCO_KEEP=0`）：llama.cpp 的合并
Engram 布局、1017 的 IQ1_S/IQ2_S/IQ2_XXS gate+up 与 IQ2_S/IQ3_XXS/IQ2_XS/IQ2_XXS/Q2_K down 专家
（专家宽度 768，行只按 2 字节对齐）、Q6_K/Q8_0 注意力、Q5_K/Q8_0 共享专家、BF16 压缩器与索引器，
提示长度足以让稀疏选择和候选掩码生效。脚本是 150 个 token 的 prefill（三个 64 token 的 ubatch）、
15 步解码、在奇数位置 165 追加 21 个 token（一次多行调用补完某个解码步开始的压缩块）、再解码 2 步，
然后 reset，重新运行提示与前 3 步解码，结果必须与第一次逐位相同。依次强制每条路由专家路径
（tensor core、寄存器暂存、warp-per-row、per-slot），每个 logit 向量与原生结果的相对 L2 都在 2e-2
以内且 argmax 相同：实测对 `ggml_cuda` 最大 8.2e-3，对 `ggml_cpu` 最大 9.7e-3。这个检查点经过调理，
使两个正确的引擎能满足这样的界限：在全尺寸更新的随机权重上，两个 ggml 后端彼此就相差 9e-2，因为路由、
稀疏选择以及 FP8/NVFP4/MXFP4 cache 舍入会把舍入差异放大。它只写入很小的注意力与共享专家更新，所以
logits 对注意力路径、以及每次调用最后一行以外的所有行都只是粗略可见：完全去掉稀疏选择只让 logits 变化
1.6e-2。因此各阶段还逐行对齐原生执行器自己的转储（`TS_DSV41_TRACE_DIR` 对 `TS_DSV4_CUDA_TRACE_DIR`）：

- 每个路由专家块的输出：在每条强制路径上、每次调用的每一行、每一层，相对 L2 不超过 0.1 加上它所接收
  的残差流误差的两倍（实测对 `ggml_cuda` 最大 4.5e-2）。超出的行必须是路由翻转，即 Direct 引擎自己
  第四与第五个专家选择相差不超过 2e-2（对任一后端，1128 行中各有一行，余量 3e-3 到 5e-3）。它选中的
  专家必须是它自己的路由得分加选择偏置后的前四名；偏置决定了其中 183 行的选择。
- 稀疏注意力。两边都把索引器的 query 与 key 舍入到训练时的 4 bit cache 格式，所以两个引擎的得分每个
  token 平均相差约 6%，约一半 token 的选择相差一两行；每个引擎必须精确完成的是基于自己得分的算术。
  每个索引源打分的行（全部可见行；候选层之后，是它最好的四个 8 行块，且查询自己所在的块排第一）以及
  它选出的 16 行，在两个引擎上都按这些规则检查，而 ggml 的转储表明这些规则正是它的算术。得分按 token
  并按平均值对齐 ggml，压缩器的 latent 逐行对齐（最大 4.3e-3），注意力输出在所有选择与 ggml 相同的
  行上对齐（最大 3.4e-2）。
- Engram 阶段：逐 token 的投影行（解码时 2e-7；Direct 引擎 prefill GEMM 读取 F16 激活时 5.4e-3），
  以及每个引擎实际施加的门控更新，相对于残差流（解码时 2.2e-5，多行调用时 4.8e-4）。

每项阶段检查都能抓到一个让所有 logit 向量仍在 2e-2 以内的人为注入错误：分组专家路径在第 3-5 层丢掉
繁忙专家第 16 个之后的成员；消费层在另一个源的压缩 cache 上做注意力；候选层多保留一个块；Engram 门控
在多行调用中使用下一个 token 的 key；压缩器丢掉奇数位置调用补完的块。把 IQ1_S 解码器的 delta 符号取反，
logits 本身就会失败（每个向量偏离 3.9e-2 到 7.7e-2，在脚本较早的 16 步形式上测得）。
`eng/validation/compare-dsv4-traces.py` 可逐阶段比对 `TS_DSV4_CUDA_TRACE_DIR` 与
`TS_DSV41_TRACE_DIR` 的转储（它映射原生的 `v41.NN.<stage>` 命名，包括压缩器的 latent，并推导出 Direct
引擎折叠进内核的 Engram 门控），定位模型第一次偏离的位置。和 `ggml_cpu` 一样，这两条都是正确性与可移植性通道，而不是服务通道；
细节见[在 ggml CPU 后端上运行](#在-ggml-cpu-后端上运行)的末尾。`--backend mlx` 仍然被
拒绝。

| `--backend` | Engram GGUF 布局 | 单节点多 GPU | 跨节点（`--tp-node-id`/`--tp-peers`） | 定位 |
|---|---|---|---|---|
| `ggml_cuda` | TensorSharp 布局与 llama.cpp 布局（unsloth 1017） | `--layer-split N`、路由专家 `--tp 2..8` | 按层切分（每节点 1–8 张 GPU）与路由专家 TP（全局 2–16 个 rank） | 服务 |
| `ggml_cpu` | 两种 | 不适用（一个 CPU 设备） | 拒绝 | 正确性、可移植性 |
| `cuda` | 两种 | `--layer-split N`；拒绝 `--tp` | 按层切分（每节点 1–8 张 GPU，driver 运行最后一个 stage）；拒绝 TP | 正确性、可移植性 |
| `cpu` | 两种 | 不适用 | 拒绝 | 正确性、可移植性 |
| `ggml_vulkan`、`ggml_metal`（`TS_DSV41_ALLOW_NON_CUDA_GPU=1`） | 两种 | 一张 GPU | 拒绝 | 可移植性 |

两个托管执行器 `cuda` 与 `cpu` 读取任一种 Engram 布局，并与原生加载器完全一样地判定布局（存在
`deepseek41.ple.layers` 且不存在 `deepseek41.engram.layer_ids` 即为合并布局）。TensorSharp 的逐层布局是
`deepseek41.engram.*` 元数据（层号、乘数、素数、偏移、已压缩的 pad id），以及 `blk.N.engram_embd`、
`engram_wkv`、`engram_q`、`engram_k` 张量，也就是 `eng/dsv41-fixture.py` 写出的布局。unsloth 的
DeepSeek-V4.1-Flash-1017 GGUF 使用的 llama.cpp 合并布局是 `deepseek41.ple.*` 元数据、单个
`per_layer_token_embd.weight` 表、`engram_kv` 与融合的 `engram_gate`（q*k）。对这种布局，逐层偏移
被重定位到该层在表中的切片上，原始的 `engram.pad_token_id` 像 `engram.py` 一样经 token 映射压缩。每个
切片随后作为该层的 `blk.N.engram_embd.weight` 暴露：它是存放该表的分片中的一段行区间，原地映射或流式
读取。合并表条目本身被移除，因此 `TS_DSV4_MMAP=0` 不会把 126 GiB 缓冲两次（在仿 1017 的 fixture 上，纯 C#
执行器只以两个行视图的形式缓冲一次该表，结果与映射读取完全相同）。`--backend cuda` 与 `ggml_cuda` 一样放置
这些表：层切分在不额外卸载专家的前提下放得下就放进显存，否则使用主机映射；`TS_DSV41_ENGRAM_DEVICE=0` 强制
主机映射（1017 的 62.9 GiB 切片在 46 GB GPU 上就走这条路径），`=1` 要求驻留显存。Direct 引擎以前忽略这个
变量。强制主机映射时，fixture 的 logits 与显存中的表逐位相同。`engram_gate` 缩放归一化后的
query，key 的增益为全 1，这正是原生计算图的算术。在改写为合并布局的 fixture 上，两个执行器的 logits
与携带同一融合乘积的逐层副本逐位相同。1017 的 `deepseek41.attention.indexer.candidate_top_k_blocks`
被这两个执行器与原生加载器一样读作候选预算，因此 ggml 与托管执行器按同一预算剪枝（同时含有
`candidate_top_k` 的文件两者必须一致）。

托管执行器仍会在映射任何权重之前根据文件头拒绝混用两种布局或只带合并布局一部分的文件：没有
`deepseek41.ple.layers` 的 `per_layer_token_embd.weight` 表或 `deepseek41.ple.*` 键、与 TensorSharp 的
`deepseek41.engram.layer_ids` 并存的合并表，或与逐层表并存的 `ple` 元数据。报错为
`This DeepSeek V4.1 GGUF has ... never a mix or a part of one, so it cannot tell which rows belong to which
Engram layer.`；此前不完整的情况会在执行器内部以 `Missing or invalid DeepSeek V4.1 GGUF metadata:
deepseek41.engram.pad_id` 失败。

Direct CUDA 引擎还会在上传任何数据之前，按实际执行位置检查每个路由专家张量。它的 GPU 专家
内核与 GLM-5.3-Flash、Qwen3.8 的 Direct 引擎共用，在所有路径上（逐 token 解码内核，以及 prefill
的 tensor core、寄存器暂存与逐行 warp 内核）实现了 Q8_0、Q6_K、Q5_K、Q5_1、Q4_K、Q3_K、Q2_K、
IQ4_XS、IQ4_NL、IQ3_S、IQ3_XXS、IQ2_S、IQ2_XS、IQ2_XXS、IQ1_S 与 MXFP4。这覆盖了 1017
`UD-IQ2_XXS` 文件的全部路由专家类型：gate/up 为 IQ1_S（10 层）、IQ2_S（15 层）与 IQ2_XXS（15 层），
down 为 IQ2_S（15）、IQ3_XXS（15）、IQ2_XS（6）、IQ2_XXS（3）与 Q2_K（1）。用 `--n-cpu-moe`
卸载的层在托管主机 matmul 上运行，它支持的类型集合不同（支持 IQ2_XS、IQ3_XXS 与 Q2_K，不支持
IQ1_S、IQ2_S、IQ2_XXS 与 Q3_K），因此该文件的专家在 `cuda` 上必须留在显存中。无法在层切分所放
位置运行的张量会按名称被拒绝，例如 `blk.0.ffn_gate_exps.weight is IQ1_S (offloaded by --n-cpu-moe)`；
此前整个检查点会先流式上传到显存，直到第一个经过该层的 token 才以一个裸类型编号失败。显卡放不下这样的
文件时，显存不足的拒绝信息会说明任何卸载都无济于事，并给出第一个挡路的张量（`... and
blk.0.ffn_gate_exps.weight is IQ1_S, which the host matmul behind --n-cpu-moe does not implement`）；
此前它会建议一个 `--n-cpu-moe` 值，而引擎随后又会拒绝该值。IQ1_S 与 IQ2_S 解码器，以及 IQ2_XXS 的
tensor core 与寄存器暂存解码器，是较新的内核代码。在由更旧源码构建的内核模块（尚未重新生成的已提交
PTX 基线，没有 nvcc 的机器会加载它）上，引擎会按名称拒绝 IQ1_S 与 IQ2_S 专家，
`blk.N.ffn_gate_exps.weight is IQ1_S, and the loaded tensorsharp_dsv4_kernels module was built from
kernel source older than its IQ1_S expert decoder`，而不是把它们当作 Q8_0 读取；IQ2_XXS 专家则在这种
模块已有的逐 token 与逐行 warp 内核上运行。检查针对的是设备实际加载的模块，并在上传任何数据之前完成。在此之前，该引擎
自带一份更窄的类型列表，会拒绝 IQ1_S、IQ2_S、IQ2_XS 与 IQ3_XXS。这些内核已对同一字节的 ggml
反量化结果做过检查，引擎在上面仿 1017 的 fixture 上对照过 `ggml_cuda`；1017 自身的权重在 10 层截断版本上
逐阶段检查过（[逐阶段 oracle](#对照-ggml_cuda-的逐阶段-oracle)），并以完整模型在 2 节点 x 2 A40 上运行，最终 logits
通过了对照 `ggml_cuda` 的门禁（[完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行](#完整-1017-在-cuda-上的-2-节点-x-2-a40-运行)）。
那次运行中所有专家都留在显存里；这些专家的其他量化版本尚未运行过。

分布式路由专家 TP 只支持 `ggml_cuda`（在 `cuda` 上会在联系任何对端之前被拒绝）。分布式按层切分也能在
Direct CUDA 引擎上运行：`--backend cuda --layer-split N --tp-node-id K --tp-peers ...`，CLI、服务端（driver）
与 `eng/validation/DeepSeek41DistributedProbe` 使用同样的参数。每个节点是流水线的一个 stage：

- 切分点由各分片文件头中的字节数决定，从不取决于空闲显存，因此每个节点算出的结果相同：对每层的权重、
  缓存以及切分所需的缓存副本做 `LayerSplitPlan.CreateWeighted`，embedding 计在第一张 GPU，head 计在最后
  一张。1017 在 2 节点 x 2 GPU、64K 上下文下为 (12, 22, 31)，与 `ggml_cuda` 相同。随后每个节点只检查自己的
  部分（权重、缓存、它持有的缓存副本、embedding 或 head）能否放下；某个节点放不下时，它拒绝加载并指明 GPU
  与字节数，其他节点也在上传任何权重之前随之拒绝。不会挪动任何层。
- 每个节点只描述、映射、预读和上传自己的层（拥有第 0 层时还有 embedding，拥有最后一层时还有 head）。
- 有多张 GPU 的节点在第一张 GPU 上接收边界状态，并经自己的节点内边界把后面的 GPU 要读取的内容继续传下去：
  1017 在 2 x 2 下，节点 0 在第一张 GPU 上收到第 20 组的缓存行与候选掩码，第二张 GPU 上的第 31–39 层读取它们。
  每个进程内边界都会检查：对边界之后仍要读取的每个组，所传递的恰好是本 ubatch 封闭的块（无论是本地提交的
  还是接收到的）；边界之后的层用候选掩码剪枝时，掩码必须处于激活状态。这与网络接收端校验载荷的方式相同。
- 每个 ubatch 的 stage 边界携带隐藏流、延迟的超连接门控、该 stage 为边界之后仍要读取的每个组所提交的
  压缩与索引器缓存行、已发布的稀疏选择以及候选掩码（可见的列），与进程内切分在两张 GPU 之间传递的内容
  一致。载荷是自描述的：头部含布局哈希（维度、压缩比、来源层、放置、ubatch、上下文）、位置与序号，
  接收端先校验它们，并按位置重新计算已提交的行区间，之后才把任何数据送到 GPU。1017 的 21|22 边界上，
  深度 2K 时每个解码 token 约 87 KB（64K 时 151 KB），每个 256 token 的预填充 ubatch 为 21–37 MB。载荷必须
  能装进一个传输帧（2 GiB）：会超出的预填充宽度与上下文（例如 1M 上下文下 2048 行：仅候选掩码就有 2 GiB）
  会在启动时被所有节点拒绝。
- 每个节点对检查点文件头以及每个会改变引擎数值的设置（`TS_DSV4_UBATCH`、`MAX_CONTEXT`、
  `TS_CUDA_QMM_F16GEMM_*`、`TS_CUDA_Q80_*`、`TENSORSHARP_CUDA_MOE_FUSION`、`TS_DSV41_ENGRAM_DEVICE` 等）
  求摘要，再对放置与内核模块（GPU 加载的镜像的 SHA-256、构建的版本与源码修订、边界线格式版本以及解码器）
  求摘要，任何不一致都会让所有节点拒绝加载。
- driver（节点 0）运行最后一个 stage，因此 logits 从不经过网络；`cuda` 上拒绝
  `TENSORSHARP_LAYER_SPLIT_DRIVER_STAGE=first`，也拒绝 `--n-cpu-moe`/`--cpu-moe` 与 DSpark 草稿器。与
  `ggml_cuda` 一样只服务一个镜像序列：没有槽位、截断或前缀复用。除非设置 `TS_DSV41_ENGRAM_DEVICE=1`，
  Engram 表保持为主机映射，且每张主机映射的表只由一个节点持有（见
  [`cuda` 上的 Engram 读取](#cuda-上的-engram-读取)）。
- 某个节点在 forward 中途失败时会中止整个网络：其他节点立即让请求失败，而不是等到传输超时；之后所有节点
  都必须重启。
- 设置 `TS_DSV4_ASYNC_SEND=1` 时，发送节点不再等一个预填充 ubatch 的边界数据发出后才开始同一次 forward 的
  下一个 ubatch：它把数据拷贝到 pinned 内存的操作排入队列，在下一个 ubatch 的首个 GPU 的层排队之后再发送，因此
  节点内各 GPU 同时处理相邻的 ubatch，而不是轮流工作。`=2` 改为在下一个 ubatch 自己发送时才发送，而不是在它的
  第一次节点内交接时。每次 forward 的最后一个 ubatch 都在 forward 返回前发出，请求之间没有在途数据；decode 步
  （一个 ubatch）不变。未设置或 `0`（默认）时与此前一样同步发送每份数据。延迟发送在每个 stage 都跑在自己的 GPU
  上实测之前保持为可选：在一张共用的 GPU 上看不出它的收益，而从另一张 GPU 的 CUDA 上下文等待它的拷贝事件这一
  路径也从未运行过（所有测试设备共用同一张物理 GPU 的上下文）。各模式下数据内容、顺序与每个内核都相同，因此
  logits 逐位一致，各节点也无需在此设置上一致。

覆盖范围：在一张 GPU 上的两节点回环（每个节点是进程内的一个 stage，经真实 TCP 通信），使用 1017 形状的
合成 fixture，把切分点分别强制放在 KV 来源层之后、候选来源层之后和 Engram 层之后，多 ubatch 预填充、18 次
解码、一次追加以及 reset 后重新预填充的每个 logit 都与单进程引擎逐位相同；以两个进程运行
`DeepSeek41DistributedProbe` 也是如此，服务端的启动路径驱动一个 worker 时同样如此。测试中，有多张 GPU 的节点
以多个逻辑设备运行（每个逻辑设备有自己的分配器、流、事件与内核模块；逻辑设备 d 运行在编号为 d mod 可见 GPU 数的
物理 GPU 上）：在一个压缩比为 1 的组被四层读取的合成拓扑上，2 个各有 2 张 GPU 的节点（接收到的行与掩码经
节点 0 的节点内边界继续传递）以及 3 个各有 1 张 GPU 的节点（中间节点接收并继续发送它们）也逐位相同，单进程在
2 到 4 张 GPU 上的切分同样如此。延迟发送也按同一标准检验：`TS_DSV4_ASYNC_SEND` 取 1、2、0 时，一段包含
1 到 5 个 ubatch 的提示（奇偶个数、整倍数与余数）、分三块输入的提示、decode 步、被拒绝的截断与 reset 的脚本，
在 2 个各 2 张 GPU 节点的三个切分点上、在 3 个各 2 张 GPU 的节点（中间节点既接收又发送）和 3 个各 1 张 GPU 的
进程内节点上，以及经 TCP 的 3 个各 1 张 GPU 的节点上都逐位相同；15 个 ubatch 的提示也是如此。这些运行共用一张
GPU，不衡量速度。只有一张 GPU 时，所有逻辑设备共用它；在有两张 A40 的主机上，2 个各有 2 张 GPU
的节点那次运行把每个节点的两个设备分别放在两张物理 GPU 上（两个节点在同一进程内共用它们）。这些运行都使用合成
fixture。目前在 `cuda` 上分布式运行过的唯一真实检查点是两个各有两张 A40 的 pod 上的完整 1017，它对照的是 `ggml_cuda`，
而不是单进程 `cuda` 运行（这里没有能单独放下它的主机）；见
[完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行](#完整-1017-在-cuda-上的-2-节点-x-2-a40-运行)。

在 `ggml_cuda` 上，V4.1 的 attention 运行 TensorSharp 自有的 F32 内核（ggml 的 CUDA flash
attention 会把 Q 与 softmax 权重收窄到 F16，而 cache 量化可能放大丢失的这些位）。**prefill
默认是稀疏的**，前提是调用既宽又长：超过 8 个 query、至少 2,048 个 key（原始窗口环加上可见的
压缩行）的分块，每个 query 只经由掩码压缩内核关注自己的滑动窗口与索引器选中的行，最多
128 + 512 个 key。其余情况保持稠密内核：单 token decode 与每次 DSpark verify（6 行）使用
分 key 内核，因此 verify 提交的 cache 行仍与 decode 完全一致；更短的 prefill 使用分块
（tiled）内核，因此 attention 始终低于 2,048 个 key 的提示词逐位不变。可见 key 多于上限的行
会退回全行扫描，所以这个上限不会丢掉任何 key。

在单张 A40 上实测（`GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill 512 33536 64 5`，
它通过生产环境的门控与该变量选择内核）：64 个头、512 个 query、33,536 个 key 时，稀疏每次
**34.1–34.3 ms**，分块为 **1,547–1,549 ms**。与分解的 F32 参考相比，稀疏内核的最大绝对误差为
1.1e-7（相对 L2 7.4e-7），分块内核为 8.9e-8（4.8e-7）。稀疏 query 也与同一调用中的其他 query
无关——query 0 单独计算、在 9 个 query 中、在 512 个 query 中逐位相同——因此提示词的结果不取决于
prefill 如何切成 micro-batch。（在 YOCO 有界重放下，把提示拆成多次 prefill 调用则不同：每次调用
保留各自的最后 140 个位置，见下文。）关闭有界重放时，只要每次调用都结束在 micro-batch 边界上
（除最后一次外，每次调用的长度都是 prefill micro-batch 宽度 `TS_DSV4_UBATCH` 的整数倍），拆成几次
调用也不影响结果：一个 micro-batch 的 key 范围（因而它走稀疏还是分块内核）只取决于它自己的位置。
若某次调用结束在 micro-batch 中间——调用比 micro-batch 短，或前缀复用、回退从任意位置继续——其后的
位置会被分进不同的 micro-batch，而投影与 MoE 对 micro-batch 宽度并不不变，因此这样的拆分与整段
prefill 只在 F32 舍入（经模型放大后）的范围内一致。早期版本把每个 micro-batch 都补齐到整次调用末尾
所在的桶，因此同一提示一次调用 prefill 与按 256 token 分次 prefill 会让部分 micro-batch 走不同内核，
留下不同的 cache（`Dsv41PrefillSplitTests`）。修复保留了分次的结果：整段提示前面的 micro-batch 现在
走分块内核，而早期版本走稀疏内核，因此一旦某个 micro-batch 过去的补齐宽度达到 2,048 个 key（micro-batch
为 256 时从 1,025 个提示 token 起），其 cache 与 logits 就可能与早期版本不同。`TS_DSV41_SPARSE_FA=0` 恢复分块 prefill。

CUDA 用 MQA 内核执行这些压缩调用：V4.1 的 64 个头共用一个 K=V 头，因此一个 block 每个 key 分块
只加载一次、供四个头共用，而不是每个头各加载一次。这使稀疏开销在 1,024 个 query 时恒为
16.1–16.3 ms（`--benchmark-sparse 1024 KEYS 64 5 640`；512 × 8,960 个 key：8.0 ms，此前为
34.4 ms，相对 L2 5.2e-7），而分块内核每个 key 约增加 9 µs（1,792 / 2,560 / 6,656 个 key 时为
15.9 / 22.5 / 63.8 ms）。交叉点约在 1,800 个 key，因此门控从 8,192 降到 2,048 个 key。

这个自有门控（超过 8 个 query、至少 2,048 个 key、F32 压缩内核）与 ggml flash-attention
内核的门控不同，后者由非自有的 attention 路径使用（非 CUDA GPU、CPU 后端）。在那里
`TS_DSV41_SPARSE_FA=1` 仍然只在单个 query 或至少 16,384 个 key 时启用 ggml 的掩码压缩 flash
attention，其 F16 运算与 CPU oracle 的相对 L2 实测最高 7.8e-4；该提示保持显式开启。稀疏
attention 减少的是 attention 计算量，并不消除共享压缩 cache 在 GPU 之间的拷贝。按层切分时，
这些拷贝现在只是每一步新写入的行，写进每个读取该 cache 的 GPU 上的副本
（`TS_DSV41_KV_MIRRORS`，默认开启），而不是每次读取时搬运选中的行或整个可见 cache。两种方式
logits 逐位相同；五张 A40 在 2K 上下文的 decode 从 32.6 提到 34.5 tok/s。

YOCO 有界重放（DeepInfer 的 YOCO prefill 截断）默认开启，取 DeepInfer 的 140
（`TS_DSV41_YOCO_KEEP`）。从第一个 ratio 为 1 的层（40 层中的第 20 层）起，每一层读取的都是
第 20 层由其 attention 输入写入的压缩 cache，因此这些层只处理一次 prefill 调用的最后 140 个
位置，每个重放位置的滑动窗口被截到重放范围之内。这是近似。在五张 A40 上用 unsloth 的
UD-IQ2_XXS 文件（`--layer-split 5`，随机 token），512 个 token 的 prefill 由 352.7 提到
500.3 tok/s，2,048 个由 463.5 提到 844.3，8,192 个由 444.3 提到 906.2（`--tp 5`：562.9 ->
1,046.7 与 540.8 -> 1,082.7）。在 4,608 token 窗口的 teacher-forced wikitext-2 上（prefill
2,304 个 token，其余逐 token 打分，10 段），困惑度由 2.4820 变为 2.4970（+0.6%；按段 t = 4.85），
相对 llama.cpp 的平均 KL 散度由 0.0160 变为 0.0172。代价集中在提示之后最先生成的 token 上：
前 128 个的 KL 散度为 0.026（精确 prefill 为 0.015），到 512 个为 0.018（0.015），其后为
0.016（0.016）。在 2,560 token 窗口（prefill 1,280 个）上，困惑度由 2.4867 变为 2.5193。在约
6.3K 与约 19K token 提示的 10%、50%、90% 深度处埋入的口令句，开启与关闭时六次都能取回。
`TS_DSV41_YOCO_KEEP=0` 让每一层处理每个位置。decode 步、DSpark 验证、含图像 token 的调用
与单 token 的 prefill 微批始终让每一层处理每个位置：`TS_DSV4_UBATCH=1` 时默认值提示重放不生效，
显式设值则拒绝加载。回退到先前某次调用跳过的位置（比该次调用的末尾早 13 个 token 以上，且在
其起点之后）会重新 prefill 而不是复用 cache，并在日志中提示一次。服务端以每次最多 4,096 个
token（并发时更少）的调用 prefill 长提示，每次调用各自保留最后 140 个位置。布局无法重放的
检查点在默认值下以一条提示改走精确 prefill，显式设值则在那里被拒绝；滑动窗口宽于 132 个
token 的检查点默认至少保留窗口 + 8。只有 ggml 执行器实现了它：`--backend cuda` 与
`--backend cpu` 会提示一次其 prefill 为精确计算，并拒绝显式的非零值。

默认上下文分配上限为 65,536 个 token，除非提供 `MAX_CONTEXT`。`TS_DSV4_UBATCH` 控制前向
微批。不设置时，V4.1 在 ggml GPU 后端上由加载器选择：1024、512 或 256 中所需路由专家 CPU
层数不多于 256 的最宽者，记录为 `[dsv4] prefill ubatch: N (auto; ...)`（见
[为图保留的设备内存](#为图保留的设备内存)）。CPU 执行器与 direct CUDA 引擎保持 256。任何
显式值都原样使用；`TS_DSV4_UBATCH=256` 恢复此前固定的默认值。模型声明的窗口更大，并不意味着
某个具体的 GPU 配置能分配或高效服务那么长的上下文。

### `cuda` 上的 Engram 读取

直连引擎由执行器自己读取主机映射的 Engram 表（`Dsv41EngramReader`、`GatherEngramRows`）。下面的设置都不会
改变结果：每一行都经由同一个主机反量化函数写到同一位置。

- **并行取行**（默认开启）。一个 ubatch 在一张表上的行按 (token, 列) 并行反量化，一次
  `TS_DSV41_ENGRAM_THREADS` 行（1–32，默认 min(16, 核数)），因此 decode 一步每张表的 24 行一起读取，而不是
  一行接一行。S9 只在 token 之间并行，decode 的每个 token 串行读取 48 行（冷启动时节点 1 每个 token 花
  83.5–94.5 ms）。这些行由读取器自己的线程（`TS_DSV41_ENGRAM_THREADS` - 1 个）和发起读取的线程读取，从不使用
  .NET 线程池，因为繁忙服务器的请求线程可能把线程池占满。`TS_DSV41_ENGRAM_THREADS=1` 精确恢复 S9 的读取方式
  （在 .NET 线程池上按 token 并行、每个 token 的各列依次读取），并在未显式设置时关闭访问建议。
- **预读**（可选，`TS_DSV41_ENGRAM_PREFETCH=1`；未设置或 `0`：关闭）。每次前向开始时，读取器的线程按层请求的
  顺序（先是 ubatch 0 的两张表，再是 ubatch 1……）读取本节点各表在这次前向中要取的每一行所在页的一个字节，同时
  主机继续下发各层。它只对历史的副本做哈希（`Dsv41EngramData.PreviewHashes`），只读不写。取行总是优先：正在
  预读的线程在读下一行之前先加入取行，因此读取器在途的行数从不超过 `TS_DSV41_ENGRAM_THREADS`。reset 与失败的
  前向会停止预读；页缓存逐出会停止预读，并等到没有线程还在读为止。默认关闭，因为它在冷的网络存储上能带来多少
  收益、以及为长提示词预读各行会给放不下这些表的 cgroup 带来多少代价，都还没有在完整模型上测量过。
- **`MADV_RANDOM`。** 表以随机读方式映射，缺页只读一页，而不是一整个预读窗口（S9 实测没有它时每读一行带入
  55–60 KB 页缓存，ggml 为 3–4 KB）。`TS_DSV41_ENGRAM_RANDOM=0`/`1`（未设置时线程数大于 1 即开启），仅 Linux。
  下面的预热用 `pread` 读取，不受这一映射建议影响。
- **持有关系**（跨节点层切分，主机映射的表）。每张表由一个节点持有：由它映射、预热并取行
  （`Dsv4StagePlacement.EngramOwners`）。`TS_DSV41_ENGRAM_OWNER` 未设置或为 `local`（默认）时，与 S9 一样由
  运行该表所在层的节点持有；`balance` 按表序号依次把表分给当前持有表字节最少的节点，相同时留给读它的节点。
  1017 在 (12, 22, 31) 下两张表都在节点 1 上被读取；使用 `balance` 时节点 0 持有第 14 层的 63 GiB 切片，每个
  节点各持有一张，这是每个节点的表都能放进其 93.1 GiB cgroup 的唯一放置方式。由别的节点持有的表在每次前向开始、
  任何边界之前传过来：持有节点用自己那份历史副本对这次前向做哈希（每个节点都能看到每次前向的 token），取出
  各行，发送行号与原始量化字节，每个 token 24 x (4 + 176) = 4,320 B（2,048 token 的提示词为 8.8 MB，分成至多
  16 MiB 的片发送）。读取节点先用自己的哈希核对行号，再用与本地取行相同的函数反量化，因此结果与 `local` 逐位
  相同。在双节点实测之前 `balance` 保持为可选：读取节点要等整个前向的行都到达后才能开始它的第一层，而持有节点
  不会预读这张表。持有关系参与启动时的一致性校验，配置摘要按含义比较该模式（未设置与 `local` 一致；只有一个
  节点设为 `balance` 会被拒绝）。服务器上提示词按至多 4,096 token 分块到达，所以一条消息至多约 18 MB；CLI 的
  提示词整体传输。超过 32 MiB 的消息缓冲（长 CLI 提示词的）在交换后释放，读取节点保存的行副本在下一次只需要
  其一半以下的前向时缩回。
- **预热额度。** `TS_DSV41_ENGRAM_WARM` 含义不变（未设置：后台；`0`：不预热；正数：交出模型之前预热完）。
  节点只预热自己持有的表，并在权重上传之后进行，使用 `TS_DSV41_ENGRAM_WARM_THREADS`（1–64，默认 16）个线程，
  每个线程用自己的文件句柄以 64 MiB 为块 `pread`。读取之前先核对额度：cgroup 上限（v1 为
  `memory.limit_in_bytes`，v2 为 `memory.max`）减去其中的匿名与 shmem 内存再减 8 GiB，以及主机
  `MemAvailable` 减 8 GiB。放不下的表不预热，日志给出具体数字。S9 只按 `MemAvailable` 计算，而 VM 上它是
  整机数值，曾让预热塞满 93.1 GiB 的 cgroup。每个块先用 `mincore` 检查，只读不在内存中的页，因此被驱逐后的
  再次预热只需补回丢失的部分。对没有写权限的进程（Linux 5.2 起），`mincore` 会把文件的每一页都报告为在
  内存中；这样的进程（非 root、只读文件）会读取每个块。在默认的持有关系下，1017 的节点 1 持有两张表
  （126 GiB），其 cgroup 放不下，因此拒绝预热，它的行一直是冷读取，一次读 16 行。页缓存逐出（基准测试的
  `cold` 行）会停止仍在运行的后台预热。
- **权重页缓存。** `TS_DSV4_LOAD_DROP_CACHE` 与原生加载器含义相同（见 [加载时间](#加载时间)）：未设置时，
  当本节点上传的字节数加上它持有的表再加 8 GiB 超过其 cgroup 上限（否则为 `MemTotal`）时，加载器读完每个
  权重分块后就释放其页缓存；`0` 从不释放，`1` 总是释放；仅 Linux，位于 tmpfs 的分片不处理。1017 的节点在
  93.1 GiB 的 cgroup 中上传约 76 GiB。使用 `balance` 时每个节点持有 63 GiB，两个节点都会释放，下一个进程读取
  权重时不再把已预热的切片挤出去；在默认的持有关系下，节点 1 持有 126 GiB 并释放，节点 0 不持有表，不释放。
  内核只按整个 folio 释放，所以跨两个分块的 folio 可能留下。

这些设置都还没有在完整的 1017 检查点上运行过：上面的 S9 数字描述的是引入它们之前的引擎，它们在那里的效果尚未
测量。默认只开启逐位一致且低风险的部分（并行取行、`MADV_RANDOM`、预热额度与加载时的释放规则）；预读、
`balance` 与延迟发送在双节点实测之前保持为可选。

测试：`Dsv41EngramReaderTests`（并行取行与 S9 方式在 1、7、256 个 token 和 2–32 个线程下逐字节一致；读取只在
读取器自己的线程上进行，从不使用 .NET 线程池；预读繁忙时取行优先，在途行数不超过 `TS_DSV41_ENGRAM_THREADS`；
预读被停止并等待完成；取行失败；预读用的哈希与前向自身的哈希一致；fixture 引擎在 S9 方式、默认读取器以及另外
两种设置下逐位一致）、`Dsv41CudaDistributedTests.Engram`（由 1017 文件头算出的持有关系；表由另一节点持有的
双节点回环与单进程、`local` 及未设置的默认值逐位一致，且消息缓冲被释放；配置摘要把未设置读作 `local`；某个
节点持有设置不同时启动即被拒绝；持有节点在发出行之前失败）、`Dsv41CudaDistributedTests.PerfSwitches`（每个
开关单独开启、全部开启以及默认值，在单进程、经 TCP 的双节点和双节点各两张 GPU 上都与全部关闭逐字节一致）、
`Dsv41CudaDistributedTests.Eviction`（逐出会先停止预读）以及 `MappedTableWarmTests`
（cgroup v1/v2 解析、额度，以及在 Linux 上的 `mincore` 再预热、页缓存释放和访问建议；
`eng/validation/wsl-host-page-cache-tests.sh` 在 WSL 下运行它们）。

### 对照 `ggml_cuda` 的逐阶段 oracle

`eng/validation/dsv41-stage-oracle.py` 在真实权重上逐阶段检查 `--backend cuda`。它读取同一组 token
的两份 trace：一份来自 direct 引擎（`TS_DSV4_CUDA_TRACE_DIR`），一份来自 `ggml_cuda`
（`TS_DSV41_TRACE_DIR`，配合 `TS_DSV41_YOCO_KEEP=0`）。它以 float64 从 GGUF 权重和各引擎自己在该阶段的
trace 输入重新计算每个阶段，因此评判的是每个引擎在该阶段新增的舍入，而不是它从前面继承的误差。

它的 gate 要求：

- direct 引擎在每个阶段的误差不超过 ggml 的 1.25 倍，逐次调用与整体都如此（F32 阶段不超过 1e-6）；
- 提交的 cache 行与训练格式的重放相同，只允许平局；
- 每个稀疏选择与路由选择都是引擎自身分数的精确 top-k，误差在推导出的 F32 界之内；
- ggml 自身的误差须符合 ggml-cuda 运算的精度模型，两个引擎都须比扰动变体更符合 oracle 的 attention
  与 MoE 语义。设置这些双向检查，是为了让 oracle 与 direct 引擎共有的错误同样失败。

规则、局限，以及如何采集和评判一对 trace，见
[README-dsv41-stage-oracle.md](../../eng/validation/README-dsv41-stage-oracle.md)（英文）。长提示通过
`dsv41-trace-sink.py` 按层窗口采集。

2026 年 10 月 30 日，它在两张 A40 上针对 1017 `UD-IQ2_XXS` 文件的 10 层截断版本运行。该截断版本保留第
0–3、20–24 与 36 块，候选预算降到 65 个块，以使剪枝生效。提示长度为 68、773、2,003 与 4,204 个 token，
各含 4 步 decode。

- direct 引擎的每项检查在每个长度上都通过。
- gate 在前三个长度上通过。
- 在 4,204 个 token 时，gate 有一项针对 ggml 自身运算的检查失败：第 4 层上，精度模型对 ggml `attn_out`
  的重放距离为其误差的 0.405，限值为 0.40。原因是 ggml 用 float32 角度旋转它提交的压缩行，而 oracle
  用精确角度重建这些行。

这次运行覆盖单进程 `--layer-split 2` 下的十层，并非完整模型的 gate；完整模型的 gate 见下一节。

### 完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行

2026 年 10 月 30 日，完整的 1017 `UD-IQ2_XXS` 文件（unsloth 修订 `60ad3651…`，300.5 GB）在两个 pod 上用
direct 引擎运行，每个 pod 有两张 46 GB A40：`--backend cuda --layer-split 2 --tp-node-id {0|1} --tp-peers ...`，
driver 运行最后一个 stage，`--n-cpu-moe 0`。构建为 TensorSharp `501add66`（与 `main` 在 `9e0d9926` 的源码树
相同）加上添加本节的改动，测量时该改动尚未提交。除文档以及验证脚本 `eng/validation/run-multinode-case.py` 与其
测试外，该提交的代码与所测构建完全相同。ggml 为未修改的 `facdc44a`；CUDA 13.2，驱动 595.91。direct 引擎的内核
由 pod 上的 nvcc 13.2 针对 sm_86 编译，因此没有使用已提交的 PTX（没有 nvcc 的机器会加载它）。两个节点使用同一个
原生库。
两个引擎都使用 F16 KV、`TS_DSV4_UBATCH=256`（direct 引擎对 V4.1 的默认值）、`MAX_CONTEXT=65536`、
`TS_DSV41_ENGRAM_DEVICE=0`（两个 Engram 切片都以主机映射放在节点 1 上）、`TS_DSV41_ENGRAM_WARM=0` 与
`TS_DSV41_YOCO_KEEP=0`，因此 `ggml_cuda` 运行的是精确 prefill，也是 direct 引擎唯一的 prefill 方式。每次运行
两个节点都记录了切分点 (12, 22, 31)：节点 1 持有 0–21 层与 embedding，节点 0 持有 22–39 层与 head。节点之间
经 SSH 隧道通信，与[当前限制](#当前限制与张量并行工作)中 `ggml_cuda` 的测量相同。

**数值。** `DeepSeek41DistributedProbe` 以 teacher-forced 方式运行三段文本（分块 256，128 步 decode）：取自本仓库的
两段英文文档和一段 C# 源码。它保存每个 logit 向量：整段 prefill、decode、分块 prefill、分块之后的 refill 以及 reset。每个向量与同一 pod 上用逐
micro-batch 稀疏注意力修复记录的 `ggml_cuda` 参考的同一步比较。限值是 ggml 自身下限的三倍，该下限为 2K
文本上 ggml 的 ubatch 128 对比 ubatch 256：

| 文本（提示 token 数） | 向量数 | 平均 KL | p95 KL | Top-1 一致率 | Decode PPL 比值（127 个目标） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 限值 | | <= 0.02626 | <= 0.1020 | >= 0.8422 | 0.9623–1.0377 |
| ggml 自身下限（2K 英文） | 266 | 0.00876 | 0.0340 | 0.9474 | 0.9874 |
| 2K 英文（1,875） | 266 | 0.00935 | 0.0346 | 0.9774 | 0.9939 |
| 8K 英文（8,534） | 292 | 0.01957 | 0.0670 | 0.9418 | 1.0136 |
| C# 代码（1,507） | 264 | 0.00918 | 0.0429 | 0.9545 | 0.9868 |

三段文本都通过。8K 文本用掉平均 KL 允许值的 74.5%，且没有 8K 自己的下限可供对比。2K 探针重跑一次逐位
相同（266 个向量中 266 个）。每个探针内部，最后一个分块与整段 prefill、每步 refill 与对应的 decode、reset 与
第一次 prefill 都逐位相同。

**HTTP 套件。** 服务端先在 `cuda` 上、再在 `ggml_cuda` 上运行，构建与设置相同。两者都运行了下文 2 x 2
`ggml_cuda` 测量所用的套件（`eng/validation/run-http-quality-suite.sh`：问答、JSON、多轮与工具调用），
`/v1/chat/completions` 上流式与非流式的 Paris/42 冒烟测试，以及
`benchmarks/engine_comparison/validate_deepseek41_tools.py` 中的十个 V4.1 工具策略场景（DSML 工具调用，
逐个请求）。两者通过的是相同的 26 项中的 25 项。两者失败的都是 `tool_round_trip`：答案是正确的 JSON，但包在
代码块里，严格检查会拒绝。

**吞吐。** 两个引擎在同一会话中运行相同的基准（随机 token，decode 128 个 token，每个进程运行四次），
每种负载按 `cuda`、`ggml_cuda`、`ggml_cuda`、`cuda` 的顺序交错。除热态进程外，每个进程之前都在两个节点上
逐出 Engram 的页缓存；热态进程重复前一个冷态进程的 token 集合。下表为每个引擎两个进程第 2–4 次运行的中位数
（六次运行），单位 tok/s：

| 负载 | `cuda` prefill / decode | `ggml_cuda` prefill / decode | `cuda` / `ggml_cuda` |
| --- | ---: | ---: | ---: |
| 512 token，冷态 | 98.7 / 8.0 | 152.9 / 27.1 | 0.65 / 0.30 |
| 2,048 token，冷态 | 137.6 / 9.7 | 207.9 / 27.8 | 0.66 / 0.35 |
| 2,048 token，热态 | 547.5 / 28.6 | 445.2 / 33.2 | 1.23 / 0.86 |
| 8,192 token，冷态 | 306.1 / 17.7 | 218.5 / 27.7 | 1.40 / 0.64 |
| 20,480 token，冷态 | 335.8 / 16.6 | 206.0 / 27.2 | 1.63 / 0.61 |

Engram 行在页缓存中时，direct 引擎的 prefill 比 `ggml_cuda` 的精确 prefill 快 23%，decode 为其 86%。行在存储上
时，它的 decode 只有 8–18 tok/s。

**吞吐下限。** 这次首个完整模型运行只有一个吞吐目标，即一个下限：在相同的 Engram 状态下，2,048 token 的 prefill
至少为 `ggml_cuda` 的 0.6 倍，2K 与 8K 上下文处的 decode 至少为 0.8 倍。prefill 达到了下限（冷态 0.66 倍，热态
1.23 倍）。只要 Engram 行起始为冷态，decode 就达不到：2,048 token 时为 0.35 倍，8,192 token 时为 0.64 倍。热态下
2,048 token 时达到下限（0.86 倍）；没有 8K 的热态数据。

剖析结果指向 Engram 行的收集，这是最可能的原因。在拥有 Engram 层的节点上，direct 引擎从映射的表中收集行时只在
token 之间并行（`DeepSeek4CudaExecutor.GatherEngramRows`）。decode 时只有一个 token，因此它在一个线程上从两张表
各读 24 行，每个未缓存的行都是一次串行的存储往返；`ggml_cuda` 用 16 个 I/O 线程读取它们。在 2,048 token 上使用
`TS_DSV4_PERF=2` 测得：direct 引擎的节点 1 每个 decode token 冷态耗时 84–95 ms、热态 20 ms；`ggml_cuda` 在该节点
上的输入准备冷态每个 token 5–7 ms，计算 15 ms。direct 引擎不会逐阶段计时节点 1 的工作，因此这个原因是由代码与
这些计时推断的，并非直接测量。计划中的修复，也是下一项性能改动，是在行之间（而不仅在 token 之间）并行收集。

8K 与 20K 的冷态中位数也偏向 direct 引擎，但原因并非速度。direct 引擎把读过的行留在映射和缓存中：在 20,480 token
时，节点 1 的进程常驻内存增长到 89 GB，100 GB 的容器达到上限。之后在新 token 上的运行会找到更多已缓存的行，
它的 8K prefill 从第 1 次运行的约 140 tok/s 升到第 4 次运行的最高 405 tok/s。`ggml_cuda` 在容器中保持在 19 GB
以下，后续运行的提升较小（8K：第 1 次 148–181 tok/s，第 2–4 次 188–228）。每个进程的第 1 次运行使用相同的
token；只看第 1 次运行，`cuda` / `ggml_cuda` 在 512、2,048、8,192 与 20,480 token 上的 prefill 比值为 0.61、
0.63、0.86 与 0.95，decode 比值为 0.29–0.53。

这些数字需要两个引擎处于相同的 Engram 状态，也不是服务能力的结论：每一行都是经 SSH 隧道的单个序列。`ggml_cuda`
的 `TS_DSV41_YOCO_KEEP=0` 行是精确 prefill；它默认的 YOCO 重放（近似 prefill）这里没有运行。

**显存。** direct 引擎在加载时分配工作集并一直保持，上下文为 65,536 token。在节点 0 的两张 GPU 与节点 1 的两张
GPU 上，它在每次探针与基准中（512 到 20,480 token）分别使用 40,539 / 41,111 / 40,433 / 39,791 MiB，在服务端中
最多再多 31 MiB。最高的是节点 0 的第二张 GPU，它持有第 31–39 层与 head。同一会话中 `ggml_cuda` 在那里的峰值为
41,611（512 token）到 43,149 MiB（20,480 token）。每张 GPU 在每个整刻钟会出现一次来自 TensorSharp 之外的
267 MiB 分配；算上它，实测最坏情况为 direct 引擎 41,384 MiB（在服务端中）、`ggml_cuda` 43,416 MiB，分别比
44 GiB 低 3,672 与 1,640 MiB。

**未覆盖：** YOCO 重放（direct 引擎没有），以及同一会话中以默认 YOCO 重放运行的 `ggml_cuda` 吞吐基线；路由专家
TP；多于一个序列（分布式运行只服务一个镜像序列，因此没有槽位、批量解码、前缀复用或回退）；其他 KV 类型、上下文、
ubatch 宽度或分块大小；放在 GPU 上或预热的 Engram；不设置 `TS_DSV4_UBATCH`（这些运行显式设置了默认值 256）；
长于 8,534 token 的提示上的数值（基准到 20,480 token，但不检查 logits）；8,192 与 20,480 token 的热态吞吐；
约 1.8K 与 7.3K token 真实文本提示的服务端请求（吞吐行使用随机 token）；完整模型上的已提交 PTX（只有 fixture 与
内核测试在 A40 上用它运行过）；`--n-cpu-moe`；以及 1017 的其他量化版本。

### 每张 GPU 一个后端

DeepSeek 架构专属的算子（压缩器、注意力前后处理、MoE 路由与归约、带 clamp 的 SwiGLU、
hyper-connection 门控、top-k 掩码）以 `GGML_OP_CUSTOM` 节点发出，由一个 TensorSharp
后端执行。该后端**包裹**所在 GPU 的 CUDA 后端，并在 `ggml_backend_sched` 中取而代之：
它同时声明 CUDA 设备的算子与缓冲类型以及自己的，把普通节点以图视图的形式转交 CUDA，
并在同一条流上启动融合内核。

如果改成把两个后端并列注册，计算图就会在每次后端切换处被切开，而 V4.1 的一层大约要
切十四次。一张 decode 图曾被切成 565 个平均约六个节点的 split，调度器还会在每个边界上
做一次阻塞式主机同步。包裹方式把它降到每张 GPU 一个 split。下面较早的后端调度诊断隔离
了该机制，并非 1017 的完整模型基准：

| | 每张 decode 图的 split 数 | Decode 计算 |
|---|---:|---:|
| 后端并列 | 565 / 577 | 26.6-27.1 ms |
| 包裹式后端 | 8 | 23.2-23.5 ms |

如预期，prefill 不受影响：一个 prefill split 本身就有毫秒级的工作量，逐边界同步在那里
只是噪声。设置 `TS_DSV4_FUSED=0` 可让这些算子完全退回到原生 CUDA 内核。

启动时会报告每个已初始化的计算设备，以及路由专家的 CPU 卸载层数。使用
`--backend ggml_cuda` 且不带任何 CPU 卸载选项时，全部 40 层都在 CUDA 设备上运行。
`auxiliary CPU worker pool` 这条消息描述的是调度器的主机线程池，并不表示在做纯 CPU
推理。Engram 仅在表使用主机映射时才从主机查表。按层切分会让相邻的层落在相邻的 GPU 上，因此仅凭
单卡利用率低并不能断定发生了 CPU 回退。`TS_DSV4_PERF=2` 报告输入准备与图计算耗时；
`TS_DSV4_PERF=3` 还会记录调度器实际的后端切换。它们是诊断模式，其日志开销会影响吞吐。
见 CLI 执行调查 `docs/validation/deepseek41/cli-gpu-execution/README.md`（本地验证记录，未提交到 Git）。

`TS_DSV41_COMPACT_RAW_GATHER=1` 为稀疏的单 token decode 启用原始窗口压缩。它先在持有
这些行的 GPU 上把 128 行可见的原始行 gather 起来，再搬到持有共享压缩缓存的 GPU。被
掩码的重复行把原始前缀补齐到 256 行，使合并后的 768 行 K 张量满足 CUDA 512 分量注意力
的对齐要求。物理 ring、prefill 路径以及不做 gather 的 decode 均保持不变。该选项默认
关闭。在约 8k 提示词下的限定 Q2_K 对比中，并发 1 与 4 的持续 decode 都提升了 13.7%；
严格的 flash 算术差异与完整测量设置见验证报告。

### token 批量 decode

多个序列同时 decode 时，它们的 token 会进入**同一张图**，而不是各自一张。批量化共享
权重读取与 kernel 发射，只有触及某个序列自身状态的部分才按槽位分叉，对 V4.1 而言就是滑动
窗口环、压缩器状态、lightning indexer 的选择以及 attention 本身。Engram 查表完全不需要
分叉——它暂存的行本来就是每 token 一列，槽位不过是其中一列，只是用该槽位自己的历史来
哈希。

收益上限来自路由：每个 token 各自从 384 个专家里挑 6 个，因此槽位之间的路由专家读取并不
重叠，只有稠密投影、共享专家和输出头是共享的。实测并发工作负载请看上面的 1017 整批
耗时，旧 Q4_K_M 吞吐扫描不能作为当前模型的并发估计。

较早的批量正确性诊断也说明了为什么需要检查输出。批量化会改变 GEMM 形状，所以批量步与单独步并非逐位一致，logits 上接近平手的位置可能选出
不同的 token。单独 decode 在 6/6 条贪心提示上复现了自己的输出；批量步在 2–3/6 上与单独文本
一致。同一条提示分别以批宽 2 和批宽 4 运行——同一条代码路径，只是 GEMM 更宽——分歧比例相同，
所以改变输出的是"哪些请求恰好同处一步"，而不是各槽位的接线。需要串行路径及其确定性时设
`TS_BATCHED_FUSED_DECODE=0`。

四个各藏一个不同秘密、长度 10,836 token 的并发文档，得到 4/4 正确答案，且没有任何一份答案
包含其他槽位的秘密——这正是用来检验各槽位的环、压缩缓存与稀疏选择确实彼此独立的检查。

### 为图保留的设备内存

`TS_DSV4_VRAM_RESERVE_MB` 覆盖分层切分打包器在每张卡上留出的空余。默认值按 indexer 的
top-k 临时量、一个微批的激活以及 2 GiB 底线计价。过多的保留量可能把更多路由专家层挤到
主机上；请检查所选检查点的放置日志，不要套用旧 Q4_K_M 的吞吐估计。

这个保留量与原始滑动窗口环都随 prefill 微批增长，所以在 `TS_DSV4_UBATCH` 未设置时，加载器
会为每个候选宽度分别计算一次切分。上下文 65,536 时，256 / 512 / 1024 的默认保留量为每张卡
2,318 / 2,588 / 3,128 MiB，窗口环为 512 / 768 / 1,280 行。它选择所需路由专家 CPU 层数不多于
256（或本次运行本就要付出的显式 `--n-cpu-moe`）、且不会把驻留 GPU 的 Engram 表挪回主机的
最宽候选。更宽的分块让每个 prefill token 更便宜：一层 Q4_K_M 形状的路由专家层
（`GgmlOpsDsv4MoeWidthBench`，均匀 top-6 路由，单张 A40）驻留 GPU 时在 256 / 512 / 1024 个
token 下每个分块耗时 35.3–35.7 / 36.9–37.2 / 38.9–39.1 ms，1024 时每 token 便宜 3.6 倍；在 32 个
主机线程上为 112.7–117.9 / 185.0–189.1 / 349.9–353.3 ms（每 token 0.44–0.46 对 0.34 ms）。但多一层
主机专家的代价要由每个 decode token 承担，所以从不做这种交换。日志行会给出宽度，并在放弃更宽宽度时说明原因。低于 256
所需层数的 `--n-cpu-moe` 会被拒绝并给出该数字（`Re-run with --n-cpu-moe N`）。2048 不在候选
之列：保留量只在 1024 上验证过（一次 57,424 token 的 prefill 在 3,072 MiB 保留量下峰值时仍剩
1,522 MiB），而 ggml 的设备 OOM 会直接结束进程。所选宽度通过 `TSGgml_Dsv4UBatch` 导出，推测
解码的 prefill 按同一宽度分块。

用 Q4_K_M 发行版的张量大小，以及它在七卡 A40 上的运行所反推的每卡预算（加载后空闲内存加上
加载放置的字节，每卡 45,091–45,123 MiB）来计算：上下文 65,536 时三种宽度都需要 6 层路由
专家 CPU 卸载，因此加载器选择 1024；上下文 131,072 时同样的预算下每种宽度也都是 6 层。这些
是计算出的方案（`GgmlOpsDsv4UbatchPlanTest`），不是实测加载。

图缓存现在同时按字节数和条目数设限。一个条目的计算缓冲区随其形状增长，而处于不同位置的
并发序列会产生很多不同形状：过去四个并发的 10.8k token prefill 会占满全部十二个条目并把
某张卡的显存用尽，而这不是可恢复的错误——ggml 的分配器在重新分配前先释放旧缓冲区，因此
一次失败的 reserve 会留下空指针，进程直接死掉，而不是让该请求失败。现在在构建新条目之前
会先释放最久未使用的条目，直到每张卡都能再放下一个与已缓存的最大条目同样大的图，外加一个
底线。`TS_DSV4_GRAPH_CACHE_HEADROOM_MB` 设定这个底线（默认 1024）；设为 `0` 回到纯条目数
上限。

### 多轮 KV 复用

第二轮渲染出的提示词并不是第一轮缓存的延续。普通对话会丢弃上一轮 assistant 的推理
内容，因此渲染结果恰好在上一轮 `<｜Assistant｜>` 之后一个 token 处与缓存分叉：缓存里
是 `<think>`，渲染出的是 `</think>`。分叉点之前的内容仍然一致，而从第三轮起，那就是
上一轮的**整个**提示词。

因此原生执行器支持部分复用：`TSGgml_Dsv4Truncate` 把槽位的 head 回退到匹配前缀的末尾，
只前向新的后缀，于是每轮的 prefill 取决于最新的回答，而不是整段对话。两个条件约束它
（见 `dsv41_truncate.h`）：

* **对齐。** 目标位置必须是最大压缩比（已发布检查点为 2）的整数倍，使任何压缩块都不
  跨越新的 head；调用方把复用长度向下对齐，最多损失一个 token。
* **深度。** 原始滑动窗口存放在 `pad64(n_swa + n_ubatch, 256)` 个位置的环里（已发布
  检查点为 512，窗口 128），所以从当前状态最多只能回退 385 个位置。生成回答会让 head
  远远越过下一轮需要的提示词边界，因此每个槽位都保留一份**回退检查点**：在每次多
  token 前向结束（即提示词边界）时，对两个按模寻址的环（原始窗口与压缩器状态）做影子
  拷贝，decode 步刻意不移动它。它每层每个槽位占 `n_embd_head x ring_raw x 2` 字节
  （已发布检查点约 21 MiB）；`TS_DSV41_REWIND_CHECKPOINT=0` 将其关闭，此后超出活动环的
  回退会被拒绝。

较早的一项缓存正确性诊断使用八张 A40 与 Q4_K_M（`--n-cpu-moe 2`，贪心，提示词后接两轮 `continue`，
`TS_KV_DEBUG=1`），时间是 2026-09-11，当时 CLI 仍自行规划复用（`KVCache.PlanReuse`，
`TS_KV_DEBUG` 打印的正是它）。自 2026-09-17 起 CLI 与服务器都走引擎的 Radix 前缀缓存
（见下文），`TS_KV_DEBUG` 在那里不再打印任何内容。分叉点正好落在策略预期的位置——两轮中
缓存在那里都是 token 128821（`<think>`），而渲染结果是 128822（`</think>`），恰在
`<｜Assistant｜>` 之后一个 token：

| 轮次 | 提示词 token | 匹配前缀 | 计划 |
|---:|---:|---:|---|
| 1 | 38 | 0（冷启动） | Reset |
| 2 | 2,056 | 37，对齐到 36 | PartialReuse |
| 3 | 5,841 | 2,055，对齐到 2,054 | PartialReuse |

第 2 轮只能收回一个常量——system 块、问题与 assistant 头——因为该点之后的缓存是带着
推理内容的第一轮回答，而提示词里已经没有它了。第 3 轮收回第 2 轮的整个提示词，此后的
占比会继续增长：匹配前缀随对话增长，而需要重新前向的后缀始终只有一个回答那么长。两次
回退都远超活动环（第 3 轮为 6,529 个位置），由检查点提供。

回退被拒绝是正常结果而不是错误：调用方会 reset 并重新 prefill。普通 `deepseek4` 完全
不做截断（它的压缩器块相互重叠，对齐 head 不足以保证正确），V4.1 的 direct CUDA 与纯 C#
执行器也不做（它们没有检查点）。关闭 `--think` 时这一切都不需要：没有推理内容被丢弃，
渲染结果就是缓存的纯追加，复用无需回退。

跨请求时，这种复用由 Radix 前缀缓存驱动——它是默认模式，CLI 与服务器走的都是这条路径。
结束的一轮作为模型的主缓存（primary）常驻。下一轮思考模式的对话把它回退到整段上一轮回答
之前来保留它，树在准入时询问模型槽位能否回退到那么远（`CanRewindPrimary`，读取
`TSGgml_Dsv4SlotCanReuse`：活动环或提示词边界检查点）。被拒绝就完整 prefill，准入日志行会
说明原因，例如 `Radix prompt reuse for …: 0/2056 tokens; 2056 token(s) to prefill
(rewinding the cached conversation is declined by the model).` 放置方式不影响规划：
`--tp`、`--layer-split` 与单卡的规划相同。复用之后至少前向两个提示词 token 的一轮会留下
新的检查点（能力字段 `MinTailPrefillTokens`），因此重新生成的一轮不会让它之后那一轮失去复用。

以上成立的前提是：这一轮单独运行，且紧随其后的是同一会话的下一轮——CLI 正是这样运行的。
与其他请求重叠过的一轮运行在按请求分配的槽位上，结束时即被释放；而引擎为任何其他请求执行的
第一步都会丢弃常驻的主缓存。因此在会话相互重叠的服务器上，除非上一轮结束时没有其他请求在运行、
且在它之前没有其他请求被接纳，思考模式的一轮会重新 prefill 它的提示词。

2026-09-29 之前，树的捐赠（donation）规则拒绝任何超过 16 个 token 的回退，这一次也不例外，
因此第一轮之后的每一轮思考对话都没有任何复用（CLI 中显示 `kvPlan=Prefill`；最早在
`--tp 6` 下被报告）。该规则的用意是把较深的缓存状态留给之后的请求，而不是交给一个只共享
其开头的请求。但拒绝并不能保住主缓存——引擎执行的下一步无论如何都会丢弃它——所以该规则
不再约束主缓存。`DeepSeek41ThinkingTurnReuseTests` 用真实的 V4.1 对话模板让这样一段对话
走完引擎。匹配前缀按构造与上表相同，但上表尚未在真实模型上通过引擎重新测量。

已结束请求的原生槽位也会被保留，使多个会话（包括相互重叠的会话）
都能继续而无需完整重新 prefill。一个被保留的槽位至多服务之后的一个请求；树允许它**自己的**
会话在槽位能做到的范围内越过 16 个 token 的规则回退它（`CanMaterialize`，同样的检查点判断）：
思考模式的一轮总要回退越过上一轮回答，按那条规则被保留的槽位将无法服务任何思考轮次。其他会话
永远到达不了带作用域的槽位，因此无法取走它。若原生侧拒绝保留某个槽位（预算或设备余量不足），
结束的那一轮不会被复用，也不会被登记——2026-09-29 之前，失败的尝试会留下一个已被清空却仍被
登记的主缓存，精确延续的请求随后会从它解码。能够回退的
原生执行器始终开启保留，只在加载了 DSpark 草稿器时不生效；保留槽位的预算由
`TS_DSV41_RETAINED_CACHE_MB`（默认 2048）决定，不是正数的值保持默认。2026-09-29 在 6x A40（`--tp 6`，
Q2_K）上用四个重叠的三轮 Web UI 会话（`eng/validation/parallel-multiturn-webui.py`）实测：没有保留时
每一轮复用 0 个 token，因为每个结束的槽位在该会话下一轮到来之前就已释放；有保留时每个会话每一轮都
复用了自己上一轮的提示——开启思考时，第 2 轮复用 167-210 个 token 中的 54-62 个（被丢弃思考内容
之前的部分会重新渲染），第 3 轮复用第 2 轮的整段提示；关闭思考时分别为 175-200 中的 171-196 与
307-437 中的 287-417。两种模式下八个会话全部通过答案检查。

原生 V4.1 请求各自拥有独立的 KV 槽位。因此调度器按"每个允许运行的请求一个上下文"来
计算它那份仅含元数据的块池。调度器默认允许 16 个运行中的请求；示例显式设置
`TS_SCHED_MAX_RUNNING_SEQS=4` 表示四个活动槽位。在上下文 65,536、块大小 256 时，它们
的自动记账容量是 1,024 个块。这既不会预分配额外的 GPU 缓存，也不会放大单个请求的上下文
上限。原生槽位在分配时仍然需要足够的设备内存。显式设置为正的 `TS_SCHED_NUM_BLOCKS`
仍是一个硬性的总量记账上限；把它设得低于活动请求的合计需求会导致提示词重算。

对四个并发 prefill，`TS_SCHED_MAX_BATCHED_TOKENS=4096` 允许在一个全 prefill 的调度步中
每个请求处理 1,024 个 token。一旦有请求开始 decode，`TS_SCHED_PREFILL_CHUNK` 就会限制
其余每个 prefill，其默认值为 256。`TS_SCHED_SOLO_PREFILL_CHUNK` 控制单独一个请求的情形，
其配置默认值为 8,192，并受批处理 token 总上限（默认 4,096）约束。
把两个 chunk 上限都显式设为 1,024，可以避免一个很短的首个请求改变其他请求的 prefill
大小，但更长的混合步骤也会拖慢正在 decode 的请求。做性能对比时，请把这些调度器设置与
`TS_DSV4_UBATCH` 一并记录。

[基准矩阵](../../benchmarks/engine_comparison/benchmark_config_deepseek41.json)
使用上述保守配置，并继承其各后端环境中未列出的设置。要复现该基线，请在启动矩阵前显式
设置 `TS_CPU_MOE_THREADS` 与 `TS_DSV41_COMPACT_RAW_GATHER=0`。若使用本卡片中的模型目录，
请把 `BENCH_DSV41_GGUF` 指向第一个分片；矩阵默认的目录名与此不同。它的文本场景不能替代
验证报告中单独的严格工具、JSON、图像/视频与推理检查。

### 加载时间

权重由一组读取线程（`TS_DSV4_LOAD_THREADS`，默认 16）按 `TS_DSV4_LOAD_CHUNK_MB` 大小的
分块（默认 64）流式上传到 GPU。每个线程只走任务列表中的**一段连续区间**。在网络文件系统上，
这一点比加载器的其他任何细节都重要：任务按（分片，文件偏移）排序，如果像这个加载器过去那样
从共享游标分发，每个文件描述符读完一个分块就要跳过 `线程数 x 分块`，默认参数下是 1 GiB。
预读是按描述符进行的，因此十六条读流没有一条是顺序的。

下面较早的加载器 I/O 诊断使用八卡 A40 与 MooseFS 上的 Q4_K_M 发行版，并非 1017 启动
时间的测量。测试为冷态（每次运行前把全部 414 GiB 分片逐出
页缓存，两种顺序各交替两次）：

| 任务顺序 | 权重上传，294.8 GiB | 模型总加载时间 |
|---|---:|---:|
| 每线程一段连续区间 | **141–147 s**（2.0–2.1 GiB/s） | **144–155 s** |
| 共享游标（`TS_DSV4_LOAD_CONTIGUOUS=0`） | 360–377 s（0.78–0.82 GiB/s） | 363–382 s |

`TensorSharp.Runtime/GgufReader.cs:330` 为托管侧 GGUF 预取记录了同样的机制
（"~3x slower on MooseFS"）。区间按**字节数**而不是任务数切分，因为一个张量的最后一个分块
不是满块；读完自己区间的线程从进度最落后区间的**尾部**窃取任务，让被窃取者继续向前读。

有三件事看起来像是解法，但在这台机器上实测都不是：

* **页锁定 staging 缓冲区。** 主机到设备的拷贝只占 87 s 线程时间，而 `fread` 占 5,539 s——
  只是加载工作量的 1.6%。pinning 能让拷贝本身快几倍，但对整个加载几乎没有影响。
* **更多读取线程。** 在这个文件系统上吞吐量并不随线程数单调增长；
  `TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs:744` 记录了 16 线程 2.4 GB/s，
  96 线程只有 1.0 GB/s。
* **对主机专家预取使用 `MADV_WILLNEED`。** 29.7 s 与 31.3 s，而普通的逐页缺页遍历平均
  27.7 s，也就是说并没有更好。

上传之后的两趟读取同样用 `pread`。驻留主机的专家（`--n-cpu-moe`）预取与 Engram 预热（见
[主机映射的 Engram 表](#主机映射的-engram-表)）把各自的张量合并成文件区间，按字节切成每个
线程一段连续区间（`TS_DSV4_LOAD_THREADS`），每个线程用自己的描述符按 64 MiB 块读取，并跳过
`mincore` 报告页面已全部驻留的块。它们过去是每 4 KiB 页触碰映射的一个字节。在网络文件系统上，
每一次这样的缺页都是一次受挂载点预读（`read_ahead_kb`，A40 虚拟机上为 128 KiB）约束的同步读，
所以七卡 A40 通道（`--n-cpu-moe 6`，Q4_K_M）预取 48.2 GiB 专家花了 129.7 s，预热 103 GiB
Engram 表花了 311.3 s。

在那台虚拟机上用 `GgmlOpsDsv4FileWarmBench`（在 Linux 上随原生测试一起构建）实测：16 线程，
8 GiB 区间，每组测量前逐出并确认 `mincore` = 0，重复三到五次，各组交替进行：

| 读取 8 GiB | Engram 表，分片 00002 @ 20 GiB | 专家，分片 00003 @ 9,002,135,936 |
|---|---:|---:|
| `pread`（默认） | 2.38–2.54 GiB/s | 2.24–2.47 GiB/s |
| 预取的逐页遍历，256 MiB 段（`TS_DSV4_WARM_PREAD=0`） | 0.62–0.66 GiB/s | 0.68–0.74 GiB/s |
| Engram 的逐页遍历，8 MiB 块（`TS_DSV4_WARM_PREAD=0`） | 0.63–0.69 GiB/s | 0.65–0.68 GiB/s |
| 同一区间再次预热，已驻留（所有块均跳过） | 64–159 GiB/s | 139–187 GiB/s |

`pread` 只填充页缓存，并不填充进程的页表，而逐页遍历两者都做了；第一次 prefill 又会密集地
读取专家。因此预取在每个块进入缓存后还会逐页读一个字节：在已驻留的页面上，这趟遍历在 16
线程下耗时 0.004–0.006 s/GiB，而对同样区间调用 `madvise(MADV_POPULATE_READ)` 需要
0.019–0.023 s/GiB。主机映射的 Engram 表不预先填充进程页表，因为每次只读取其中几行。

在这个挂载点上，预读提示无法替代读取。对一段已逐出的 8 GiB 区间调用 `MADV_WILLNEED`、
`POSIX_FADV_WILLNEED` 或 `readahead(2)`，十秒后都只有 128 KiB（0.0015%）驻留。不要再尝试。

预取或同步 Engram 预热中的读取错误会让加载失败，并给出分片与偏移。`TS_DSV4_WARM_PREAD=0`
原样恢复两种逐页遍历。

除非设置 `TS_HOST_MOE_PIN=1`，被卸载的专家不会被页锁定。被卸载层的路由专家的每个节点都被
指定到 CPU 后端（`build_moe_host`），而 `ggml_backend_sched` 从不覆盖这种指定，因此它的
op-offload 规则永远不会把这些权重流式送到 GPU：跨总线的只有 `[n_embd, n_tokens]` 的激活，
锁定的专家页从来不是 DMA 的来源。在七卡 A40 通道（`--n-cpu-moe 6`）上，锁定 48.2 GiB 让加载
多花 20.4 s，并让这些页面在同样承载页缓存的 cgroup 中无法被回收。现在加载时会打印一行，
说明被卸载的专家保持可分页。`TS_HOST_MOE_PIN=1` 恢复页锁定以及
`page-locked ... GiB of host experts` 日志；`TS_HOST_MOE_PIN=0` 仍对所有架构关闭锁页。
其他 MoE 架构的 prefill 确实会流式传输被卸载的专家，默认仍然锁页。

加载器不会再读已上传的分块，但上传之后紧接着就要读主机映射的权重，并在整个运行期间从页缓存
提供它们，而页缓存计入 cgroup。因此默认情况下，只有当上传字节数加上主机映射的权重再加 8 GiB
超过主机额度（cgroup 上限）时，每个分块上传到设备后才释放它的页缓存；否则保留，额度未知时也
保留。加载时会打印这一判断及其数值：

```text
[dsv4] load page cache: dropping each uploaded chunk's page cache (automatic: 263.0 GiB upload + 151.2 GiB host-mapped + 8.0 GiB headroom exceeds the 326.9 GiB allowance; TS_DSV4_LOAD_DROP_CACHE=0 overrides)
```

这正是七卡 A40 通道的数值：414 GiB 的读取装进 326.9 GiB 的 cgroup。它的加载日志中专家预取为
0.37 GiB/s、Engram 预热为 0.33 GiB/s，大约只有同样的逐页遍历在该虚拟机上、cgroup 约一半占用
时实测速率（见上表）的一半。释放无法让上传本身变快——上传已经以存储速率读取——而且在该挂载上
每个已驻留的 64 MiB 分块要花 5.9–7.3 ms（`GgmlOpsDsv4FileWarmBench --drop-cost`，263 GiB
约合 25–30 s 线程时间）。预期收益在上传之后的阶段，尚未在完整加载上测量；检验方法是用默认值的
冷加载对比 `TS_DSV4_LOAD_DROP_CACHE=0` 的冷加载。规则之所以保持有条件，是因为每次都释放会让
完全驻留在 GPU 上的检查点每次重新加载都变冷。`TS_DSV4_LOAD_DROP_CACHE=0` 从不释放（此前的
默认），`=1` 总是释放。在八卡 A40 机器上，`=1` 没有改变上传的读取线程时间（5,374 s 对比
5,539 s，处于运行间波动之内），加载结束时页缓存约 39 GiB 而不是约 330 GiB。

自己计时时有一点要注意：如果机器的页缓存已经塞满了这份检查点，加载可能比从空缓存开始**更慢**，
因为 cgroup 在第一次读取之前就已到达上限，之后的每次读取都要与回收竞争。请同条件比较——先把
分片逐出。

### 在 ggml CPU 后端上运行

`--backend ggml_cpu` 选择加载器的纯 CPU 分支：只有一个 CPU 计算设备而不是枚举出的
加速器，所有层都在它上面，所有 V4.1 专属算子都走
`ggml_ops_dsv4_fused_cpu.cpp` 里的标量 CPU 实现而不是 CUDA 内核。启动会打印
`compute devices initialized: 1 CPU device(s)` 与
`routed-expert placement: all 40 layer(s) on the explicitly selected CPU device`。

**这是一条正确性与可移植性通道，不是服务通道。**每解码一个 token，都要在 40 层里
各读出所选检查点 384 个路由专家中的 6 个，而且跑在通用核心上。
那些参考内核每个节点只跑一个 worker（V4.1 的 quantize、candidate-score 与
candidate-mask 内核是例外），而[每张 GPU 一个后端](#每张-gpu-一个后端)里那个包裹式
后端是 CUDA 对象，在这里根本没有编译进来，所以那一节的数字在这里一个都不适用。
请把这个后端当作"在没有 CUDA 设备的地方运行该架构"的手段——用来检查改动、用来把
CUDA 结果与主机结果对照，或者在一台本来无法承载该模型的机器上把它跑起来。不要把它
挂到服务端点后面，也不要把它当成吞吐数字来引用：本卡片没有给出任何 CPU 吞吐，因为
从未测过。

除 macOS 之外，服务端的默认后端就是 `ggml_cpu`，所以在有 GPU 的机器上省略
`--backend` 就会选到这条路径。以前这种情况会直接拒绝并提示 `ggml_cuda`；现在它可以
加载，并且在读取任何权重之前先在 stderr 上说明一次
（`[dsv41] --backend ggml_cpu: DeepSeek V4.1 will run on ONE CPU device...`）。
如果你在一台有 GPU 的机器上看到这一行，那你想要的是 `--backend ggml_cuda`。

CPU 路径确实有一致性证据，但它是逐算子的，而不是整检查点的。验证报告的
[fixture 对比](../deepseek41_validation.md#independent-numerical-reference)
记录了在本地 CPU 上、以 `atol=rtol=2e-5` 对独立 oracle 的 41/41 项逐元素检查，最大
绝对误差 5.1633e-6，贪心 token 41/41 一致。这覆盖了融合算子与 fixture 规模的计算图。
完整检查点的 CPU 运行尚未测量，因此真实权重上的 CPU/CUDA parity 在此并未确立。

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cpu --port 5000
```

CPU 后端与 CUDA 一样，直接读取内嵌的 Engram 元数据。下载当前 GGUF 分片即可，无需
另行准备 Engram。

`TS_DSV4_THREADS` 设置计算线程数，默认上限为 32。那个上限是为 GPU 运行选的——在那里
这些线程只做辅助性的主机工作；而在纯 CPU 运行里它们就是整个引擎，所以请把它设为本次
运行真正可以使用的核心数。`TS_DSV4_UBATCH`（这里未设置时固定为 256，而 `ggml_cuda`
会自动选择宽度）与 `MAX_CONTEXT`（默认 65,536）在这里消耗的都是主机内存而不是显存。

那些以 GPU 命名的选项在这里的行为如下：

- `--tp N` 是把路由专家的维度切分到多张 GPU 上。与 `ggml_cpu` 组合时，会在打开
  检查点之前被拒绝，而不是被忽略。
- CPU 后端会拒绝 `--tp N` 与 `--layer-split N` 的多 GPU 请求。
  那条警告是为 GPU 主机写的，写的是 "Running on ONE GPU"；在这个后端上请读作"一个
  CPU 设备"。
- Engram 表保持主机映射：上文描述的 GPU 驻留放置需要有设备可放。因此
  `TS_DSV41_ENGRAM_DEVICE=1` 会在打开检查点之前被拒绝，而不是被接受后忽略；除 `0`
  与 `1` 以外的取值同样如此；`=0` 指的正是这条路径本来就在用的主机映射，会被接受。
  由于表是主机映射的，[主机映射的 Engram 选项](#主机映射的-engram-表)——
  `TS_DSV41_ENGRAM_WARM`、`TS_DSV41_ENGRAM_THREADS`、`TS_DSV41_ENGRAM_RANDOM`——
  在这个后端上全部有效。
- `TS_DSV4_VRAM_RESERVE_MB` 会在往这个唯一设备上装层之前，从它的空闲内存里扣除。
  这里那个设备就是主机，所以它的默认保留量（至少 2 GiB；在这条路径使用的 256 ubatch、
  65,536 上下文下约 2,318 MiB，见[为图保留的设备内存](#为图保留的设备内存)）是从系统
  内存中扣除的；而模型装不下
  时加载器的拒绝信息是按显存措辞的（"not enough VRAM ... Re-run with
  `--n-cpu-moe N`"）。那条建议本身仍然正确——见下文。
- `TS_DSV41_COMPACT_RAW_GATHER` 是在 CUDA 上测的，默认关闭；它改的是计算图而不是内核，
  因此在这里也能触发，但这条路径上开启它的效果从未测过。保持关闭。
- `TS_DSV41_SPARSE_FA` 在这里没有任何作用。它设置的是 flash attention 的 `n_kv_max`
  边界，而 ggml 的 CPU flash-attention 内核只读取算子的前三个参数
  （`ggml-cpu/ops.cpp`），从不读那一个。设置它是静默无效，而不是变慢或出错；这里没有
  可供开启的掩码压缩 CPU 内核。

以下来自阅读加载器源码而非实测：层权重分配在所选的计算设备上，而这条路径上那就是主机，
因此它们是私有的匿名分配，而不是可回收的 GGUF 映射。这也是在开始一次漫长加载之前值得
做的一个决定——`--cpu-moe`（或 `--n-cpu-moe N`）会把这些层的路由专家移进加载器的主机
上下文，而后者*确实*由 GGUF 映射提供，于是内核可以回收这些页面，而不是把进程 OOM 掉。
当权重加上本上下文的各类 cache 装不下时，加载器建议的也正是这一条，尽管它嘴上说的是
"VRAM"。纯 CPU 完整检查点加载的常驻内存占用与加载时间都尚未测量。

视觉伴随文件会跟随文本模型落到同一个后端上。在这条路径存在之前，它是用硬编码的
`CUDA` 加载的——那会把一张 GPU 拉进一次明确要求纯 CPU 的运行里；现在，没有 ggml 注册
名的后端会按名字被拒绝，而不是被硬着头皮尝试。

`--backend cpu` 是纯 C# 执行器（`DeepSeek4CpuExecutor`），它如今除 V4 之外也实现了
V4.1 的计算图：比例 1 与比例 2 的压缩器、共享的压缩 cache 与索引器 cache、候选剪枝、
Engram 表、延迟 hyper-connection 门控，以及训练好的 cache 量化。
`InferenceWeb.Tests.Dsv41CpuExecutorTests` 以 atol=rtol=2e-5 把它对齐到
`eng/dsv41-reference.py`，覆盖一次性 prefill、分块大小 1/3/5/8 与 reset。那道门禁是
fixture 规模的——一个五层、hidden 256、16 token 的 F32 合成模型——因此它确立的是与参考
实现在架构层面的一致，而不是在已发布的 246 GiB Q2_K 权重上的等价；而且不设置
`TS_DSV41_FIXTURE_DIR` 指向 fixture 目录时，这些测试会显式跳过并说明缺少该变量。与
`--backend ggml_cpu` 不同，它不接受视觉伴随文件：那个编码器是原生 ggml 组件，在这里
`LoadVisionEncoder` 会抛异常，所以图像与视频输入不可用。原生加载器的 Engram 与注意力
开关——`TS_DSV41_ENGRAM_WARM`、`_THREADS`、`_RANDOM`、
`TS_DSV41_SPARSE_FA`、`TS_DSV41_COMPACT_RAW_GATHER`——在这里全部无效。Engram 配置直接来自 GGUF，
两种 Engram 布局都支持（见[后端](#后端)）；
`TS_DSV4_THREADS` 在这个后端上默认取
`ProcessorCount`，而不是 min(核数, 32)；`TS_DSV4_CPU_TRACE_DIR` 写出的逐张量文件与
`eng/dsv41-reference.py --output` 写出的同名，于是两个目录可以逐张量对拍。和
`--backend ggml_cpu` 一样，它是正确性与可移植性通道，而不是服务通道：完整检查点在它
上面的吞吐、加载时间与常驻内存占用都没有测过。

Direct CUDA 引擎 `--backend cuda` 也用自己的内核、不经 ggml 运行 V4.1，读取两种 Engram 布局
（见[后端](#后端)）。它具有逐序列槽位、批量解码、会话状态保留，并通过 `--layer-split N` 在单个节点或跨节点
按整层放置；路由专家 `--tp N` 仍是 `ggml_cuda` 模式。它的数值检查：
[`Dsv41CudaEngineParityTests`](../../InferenceWeb.Tests/Dsv41CudaEngineParityTests.cs)
在仿 1017 的 fixture 上把整个引擎与 `ggml_cuda` 和 `ggml_cpu` 对照（见[后端](#后端)）；
`eng/validation/dsv41-stage-oracle.py` 在真实权重上把每个阶段与 float64 oracle 对照
（[逐阶段 oracle](#对照-ggml_cuda-的逐阶段-oracle)）；
[`Dsv4ExpertKernelTests`](../../InferenceWeb.Tests/Dsv4ExpertKernelTests.cs)
把合成量化专家投影与上游反量化结果对比；
[`Dsv41CudaSlotTests`](../../InferenceWeb.Tests/Dsv41CudaSlotTests.cs)
把槽位隔离、批量解码、环形缓存回退与保留状态，和同一 Direct 引擎的全新运行对比，容差为最大
logit 的 1e-5。批量解码步对每个序列稠密投影的计算与该序列自己的单步逐位相同：最多 16 行时，Q2_K 到
Q6_K、Q8_0、IQ2_XXS、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、MXFP4 与 BF16 权重使用逐行计算与单行一致的内核，
覆盖 1017 的全部稠密类型：Q5_K（13 层的共享专家）、Q6_K、Q8_0 与 BF16（其压缩器与索引器）
（[`Dsv4DenseRowInvarianceTests`](../../InferenceWeb.Tests/Dsv4DenseRowInvarianceTests.cs)）。
F16 与 F32 权重（cuBLAS 按行数选择内核）、旧的 Q4_0 到 Q5_1 类型、IQ2_S 与 NVFP4 不在此保证之内。BF16
投影在这样的批量步中逐行计算，每行读取一次权重：在 RTX 3080 Laptop GPU 上，16 行时耗时是一次 cuBLAS
调用的 6 到 14 倍。
这些是 fixture 规模、逐阶段、内核与状态一致性的检查。完整检查点的门禁另行进行：完整 1017 在 2 节点 x 2 A40 上
对照 `ggml_cuda` 运行，并跑了 HTTP 套件（[完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行](#完整-1017-在-cuda-上的-2-节点-x-2-a40-运行)）。
模型/设备门控测试需要对应夹具与 CUDA 硬件，不可用的场景会跳过。已经验证了什么、还有什么挡着，见 CUDA 后端说明
`docs/validation/deepseek41-cuda-backend/README.md`（本地验证记录，未提交到 Git）。`--backend mlx`
仍然被拒绝。

## 前向计算图与状态

原生计算图使用四条残差流以及 V4.1 的延迟 hyper-connection 混合。第 1 层与第 14 层会
加入由确定性 token n-gram 哈希选出的 Engram 特征。token 归一化与桶布局来自 GGUF 元数据；
各序列槽位保留各自的 token 历史。

每个注意力块都包含一个 128 token 的原始滑动窗口。前两层没有压缩注意力，接下来的 18 层
压缩比为 2，最后 20 层为 1。压缩 KV 源与索引器选择按官方的因果 encoder-decoder 拓扑
共享。query 投影、cache 量化、逆 RoPE 与分组输出 LoRA 都遵循 V4.1 的计算图。索引选择
使用 lightning indexer 与候选块过滤。MoE 使用共享专家加上归一化后的选中路由专家输出。

该实现复用了原生的量化矩阵乘、`mul_mat_id`、注意力、hyper-connection 内核、
per-sequence 槽位与计算图缓存。V4.1 的激活量化与候选过滤有专门的算子。它不调用纯 C#
或直接 CUDA 的 V4 执行器。

有用的源码位置：

- [架构门控](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek41Architecture.cs)
  与[托管驱动](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek4Model.cs)。
- [原生加载器与调度器](../../TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp)
  与 [V4.1 计算图](../../TensorSharp.GGML.Native/ggml_ops_deepseek41.inc)。
- [Engram 元数据加载器](../../TensorSharp.GGML.Native/dsv41_engram_gguf.h)
  与[哈希与配置校验](../../TensorSharp.GGML.Native/dsv41_engram.h)。
- [对话渲染器](../../TensorSharp.Runtime/ChatTemplate.DeepSeek41.cs)
  与[输出解析器](../../TensorSharp.Runtime/DeepSeek41OutputParser.cs)。

## 对话、工具与 JSON

V4.1 使用显式 BOS 与 `<｜System｜>` 框架，以及带空格的 DSML 标签，例如
`<｜DSML｜ calls>` 与 `<｜DSML｜ invoke name="tool">`。V4 不带空格的 DSML 格式与之
不兼容。渲染器处理 system、user、developer、assistant 与 tool 历史；并行工具结果按其
来源调用 ID 重新排序。字符串型工具参数保留空白字符，未完成的 invoke 不会被派发。
在完整的 DSML invoke 内部，解析器还接受在 Q2_K 上观察到的朴素 `<parameter name="...">`
与 `</parameter>` 变体。无法识别或格式错误的参数标记会被拒绝，而不是被转换成空参数
或残缺参数。OpenAI chat 端点在渲染下一轮时会保留传入的工具调用、推理内容与工具结果 ID。

V4.1 的 OpenAI chat 端点用请求级的 DSML 语法约束已声明的工具调用。`tool_choice: "auto"`
让普通回答不受约束，只有在模型开启 calls 块之后才激活。开启思考时，激活还会等到
`</think>` 之后，这样推理中被引用的工具语法不会启动一次调用。`required` 要求必须调用
一个客户端声明的函数；指定函数名则把该次调用限定为该函数。`none` 阻止产生工具调用，
`parallel_tool_calls: false` 把一个 calls 块限制为一次 invoke。内部的 skill 轮次会拿到
全新的语法状态。其他模型系列保持各自既有的策略行为。

工具语法会强制约束已声明的函数名、必填参数、可省略参数、原始类型、原始类型的
enum/const，以及递归定义的对象/数组。它按 schema 顺序发出参数与对象属性。没有声明
属性的嵌套开放对象接受任意 JSON map。对于声明了属性的对象，即使 schema 允许额外键，
生成时也只发出这些声明过的属性。没有声明属性的函数参数 schema 保留"无参函数"的约定。
带类型的 `additionalProperties` schema 不受支持，会被显式拒绝。声明的整数 `enum`/`const`
取值必须是有符号 64 位范围内的 JSON 整数字面量；其他编码或取值在生成之前就被拒绝。
普通数值参数保持既有的 Int64-或-double 解析行为；下文的无损参数保证针对的是字符串，
而不是任意精度的 JSON 数字。
不支持的断言——包括类型联合、schema 组合子、pattern、数值范围与字符串/数组长度限制
——都会在生成之前返回 HTTP 400。这一受支持子集适用于工具参数；JSON 响应 schema 使用
另一套既有的编译器。

普通字符串可以使用训练过的原始 `string="true"` 表示。包含 DSML 保留分隔符的字符串仍可
通过 `string="false"` 加 JSON 字符串来表达：诸如 `\u003c` 这样的 JSON 转义能在不闭合
外围工具标记的前提下保留精确的解码值。当已解析的调用被渲染进后续工具历史时（包括嵌套
JSON 中的字符串与键），同样的保护依然有效。普通历史的格式化保持不变。原始字符串还保留
了 `<param`、`</param`、`<invoke` 与 `</invoke` 这几个标签族，使得写错的工具标签无法把
后续回复整段吞成参数文本。包含这些前缀的字面字符串使用同一套无损 JSON 替代方案；普通
的 XML（如 `<x>`）与比较符号仍是合法的原始文本。严格解析器保持不变。语法测试确立的是
语法与参数往返；完整检查点上的工具选择与准确率单独测量，并保留未加约束时的失败基线。

推理渲染器使用参考实现默认的 effort 50。普通对话会丢弃过去的推理内容；启用工具的对话
则保留。缓存的原始 assistant token 无法覆盖这条历史策略。开启思考时，JSON 语法约束在
`</think>` 之后开始；否则从第一个输出 token 就开始。提示词与解析器测试确立的是格式
兼容性，而不是模型层面的工具选择、推理质量或 JSON 任务准确率。

由于 V4.1 的协议声明了这种延迟触发的语法，Chat Completions 端点允许在开启思考的同时
使用 `response_format`。这个组合要求启用 JSON 语法约束。
若要在一次工具往返之后再请求 JSON 最终答案，请保留工具历史与工具目录，并发送
`tool_choice: "none"`。进行中的工具生成与 `response_format` 仍然互斥。校验只看 assistant
的 content 通道；只有推理内容不算最终答案。
这些工具策略与"思考 + JSON"保证适用于 `/v1/chat/completions`。既有的 `/v1/responses`
接口不支持同样的 V4.1 工具历史往返或"推理加 JSON"组合。

由于 V4.1 会渲染工具声明并解析 DSML 调用，它同样可以使用 skills、代码工具
（`--code-exec`），以及服务端的[子智能体委派](../multi_agent.md)（在对话路径上默认开启）。
V4.1 没有发布任何委派相关的实测结果。

对 V4.1 而言，达到 `TS_THINKING_BUDGET` 会发出训练过的 `</think>` token，并在原有
`max_tokens` 限制之内继续写最终答案。当请求的输出额度不少于 512 token 时，默认预算为
75%。额度更小时没有自动的思考预算；显式设置为正的 `TS_THINKING_BUDGET` 仍然生效。
设为 `0` 关闭该预算。收尾 token 本身也占一个输出 token。这一转换有托管层测试覆盖；
完整检查点上的思考工作流在下文单独测量。
在这条 V4.1 策略生效期间，如果推理进入被检测到的循环，重复守卫也会请求同一个正常的
收尾转换。最终答案中的重复仍会停止生成。关闭请求级或调度器级的重复守卫会同时关闭这个
提前转换；取消、EOS 与原有的输出上限仍具有优先级。

## 当前限制与张量并行工作

- `ggml_cuda` 是 V4.1 的服务后端。`ggml_cpu` 用标量 CPU 实现加载同一套原生计算图，
  作为正确性与可移植性通道，没有实测吞吐；见
  [在 ggml CPU 后端上运行](#在-ggml-cpu-后端上运行)。`cpu` 运行纯 C# 的 V4.1 执行器，
  已按 2e-5 对齐 PyTorch 参考实现；`cuda` 用 Direct CUDA 引擎自己的内核运行 V4.1，
  它在仿 1017 的 fixture 上对照 `ggml_cuda`，并在 1017 的 10 层截断版本上逐阶段检查
  （[逐阶段 oracle](#对照-ggml_cuda-的逐阶段-oracle)），并在 2 节点 x 2 A40 上以完整 1017 对照
  （[完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行](#完整-1017-在-cuda-上的-2-节点-x-2-a40-运行)）——两者都是正确性与可移植性通道，而不是服务通道。`mlx` 会在读取权重
  之前失败，而不会把 V4.1 的权重塞进并未实现它的计算图。
- `cuda` 与 `cpu` 读取两种 Engram 布局，包括 unsloth 的 DeepSeek-V4.1-Flash-1017 GGUF 所用的
  llama.cpp 合并布局。这一点已对真实的 1017 文件头、以及端到端地对改写为合并布局的 fixture 做过
  检查。尚无 1017 权重在 `cpu` 上运行过；在 `cuda` 上，`UD-IQ2_XXS` 文件的 10 层截断版本已在两张 A40 上以
  `--layer-split 2` 按[逐阶段 oracle](#对照-ggml_cuda-的逐阶段-oracle) 运行过。1017 `UD-IQ2_XXS` 的路由专家留在
  显存中时可以通过加载时检查（GPU 专家内核能解码全部类型；早于 IQ1_S 与 IQ2_S 解码器的内核模块会
  按名称拒绝它们），而 `--n-cpu-moe` 背后的主机 matmul 不支持 IQ1_S、IQ2_S 与 IQ2_XXS。因此在
  `cuda` 上运行完整文件需要能放下全部专家的 GPU（单台主机，或借助分布式按层切分跨节点）。目前有一次这样的
  运行，在 2 节点 x 2 A40 上、单序列、使用下表的设置；它覆盖了什么、没有覆盖什么，见
  [完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行](#完整-1017-在-cuda-上的-2-节点-x-2-a40-运行)。
- 分布式路由专家 TP 只在 `ggml_cuda` 上运行；分布式按层切分在 `ggml_cuda` 与 `cuda` 上运行。`cpu`、
  `ggml_cpu` 以及需显式开启的非 CUDA ggml 后端会在联系对端之前拒绝 `--tp-node-id`/`--tp-peers`，
  `cuda` 配合 `--tp` 时也是如此。
- 分布式路由专家 TP 使用主机驱动的交换，而不是本地 TP 的设备侧交换。每个 GPU 驻留的 MoE 层、
  每个 token，每个节点都要经跨节点传输进入两次阻塞的 F32 AllReduce：拼接后的专家激活
  2304 x 6 个值（54 KiB），以及互不重叠的输出行 5120 个值（20 KiB）。所有层的专家都在 GPU 上时，
  每个解码 token 需要 80 次集合通信、约 3 MiB 负载（`--n-cpu-moe` 层则在每个节点上各自计算），
  因此解码速度受网络往返而非 GPU 限制，本卡中的本地 TP 数据不适用。注意力、共享专家、Engram
  与卸载到 CPU 的专家层在每个节点上完整计算。
- 当主机映射的 Engram 表大于主机或容器内存上限时，它们永远无法常驻。1017 `UD-IQ2_XXS` 的表
  为 125.9 GiB，大于 100 GB 的 cgroup；在 46 GB 的 A40 上，每层约 63 GiB 的切片也放不进 GPU。
  此时后台预热会被跳过（映射的主机权重加 8 GiB 超出配额），每个 token 在两张表中各读 24 行，
  都是随机缺页。在网络存储上，每个未缓存的行都是一次存储往返：
  [主机映射的 Engram 表](#主机映射的-engram-表)中的测量为冷态每个解码 token 5–15 ms 输入准备、
  每个 1,024 token 的 prefill 分块 1.4–2.6 s，热态分别为 0.9 ms 与 56–81 ms。页缓存计入容器
  上限，行会不断被逐出，代价不会随会话消失，随机 token 基准则始终是冷态。分布式按层切分时只有
  拥有两个 Engram 层的 stage 承担这一代价；分布式 TP 时每个节点都要承担。
- 实测环境：两台 RunPod 实例，每台两张 46 GB A40（sm_86，驱动 595.91，CUDA 12.8），容器内存上限
  100 GB，模型放在实例的网络存储上，节点间走 SSH 隧道（`eng/validation/multinode-ssh-tunnel.sh`；
  链式 all-reduce 10 KiB 0.61 ms，1 MiB 2.6 ms），F16 KV，`TS_DSV4_UBATCH=256`，
  `TS_DSV41_ENGRAM_DEVICE=0`，`TS_DSV41_ENGRAM_WARM=0`（Engram 表放不进 cgroup）。两种放置都在
  `--n-cpu-moe 0` 下放得下：按层切分每节点 77.9 GiB（节点 1 持有 0-21 层与 Engram，节点 0 持有
  22-39 层），TP 每节点 80.8 GiB。下表为第 2-4 次运行的中位数，随机 token（Engram 冷态）：

  | 2 节点 x 2 A40，`ggml_cuda` | Prefill tok/s | Decode tok/s |
  | --- | ---: | ---: |
  | 按层切分，512 / 2,048 token | 148 / 163-179 | 24.2 / 25.2-25.9 |
  | 按层切分，20K 上下文处的 decode | | 26.4 |
  | TP，精确 prefill，2,048 token | 56-58 | 9.4-9.7 |
  | TP，默认 YOCO，2,048 token | 139 | 9.2 |

  页缓存热态下的首次运行达到 406 / 31.7 tok/s，接近本地五卡 A40 的数据：在这种存储上，冷态 Engram
  读取主导按层切分的速度；TP 受网络限制（见上）。分布式 TP 与按层切分在 3,400 token 的 teacher-forced
  探针上（82 个向量）logits 逐位一致；加上下面的选择修复后，在 23,357 token 的探针上（126 个向量）
  也逐位一致，且每种放置多次运行之间逐位一致。HTTP 套件（问答、JSON、多轮、工具调用）在两种放置下
  均通过，唯一例外是一项严格检查拒绝了包在代码块中的正确 JSON 答案。
- 2026 年 10 月 30 日，同样的两个 pod（此时为 CUDA 13.2）用同一构建、同一会话运行 direct 引擎
  （`--backend cuda --layer-split 2`，分布式）并对照精确 prefill 的 `ggml_cuda`，每个冷态进程之前逐出 Engram。
  第 2–4 次运行的中位数，tok/s，prefill / decode：

  | 2 节点 x 2 A40，按层切分，精确 prefill | `cuda` | `ggml_cuda` |
  | --- | ---: | ---: |
  | 2,048 token，Engram 冷态 | 137.6 / 9.7 | 207.9 / 27.8 |
  | 2,048 token，Engram 热态 | 547.5 / 28.6 | 445.2 / 33.2 |
  | 20,480 token，开始时 Engram 冷态 | 335.8 / 16.6 | 206.0 / 27.2 |

  只要 Engram 行起始为冷态，`cuda` 的 decode 就达不到这次运行 0.8 倍 `ggml_cuda` 的下限：2,048 token 时为 0.35 倍，
  8,192 token 时为 0.64 倍（2,048 token 热态为 0.86 倍）。最可能的原因是这些行：拥有它们的节点在 decode 时在一个线程
  上逐个读取，该节点每个 decode token 冷态耗时 84–95 ms，热态 20 ms。计划中的修复是并行收集这些行。长提示的数字
  得益于 direct 引擎保留在缓存中的行。数值门禁、HTTP 套件、完整矩阵与覆盖范围见[完整 1017 在 `cuda` 上的 2 节点 x 2 A40 运行](#完整-1017-在-cuda-上的-2-节点-x-2-a40-运行)。
- 索引与候选选择使用 `ggml_argsort_top_k`。ggml-cuda 的 radix top-k（宽行上的 `ggml_top_k`）按原子
  操作到达顺序处理并列值与输出顺序，因此索引行超过 4,096 时 gather 解码每个 token 的 key 求和顺序都不同，
  超过 16,384 行时候选选择可能保留不同的并列块：长上下文的结果每次运行都不同，分布式 TP 下两个节点
  （各自执行每次选择）还会逐渐偏离。排序是确定性的，在 20K 上下文下没有可测的 decode 开销。
- 默认只使用一张 GPU。`--layer-split N` 选择整层放置；`--tp N` 启用 routed-MoE 张量并行，
  使用 F32 激活与输出拼接。`ggml_cuda` 的现有分布式组支持跨节点路由专家 TP，
  注意力不切分。分布式模式使用单个镜像序列，不支持槽、前缀保留/截断、视觉或 DSpark。
- 并发请求拥有隔离的序列槽位。在带 CUDA 融合后端的原生执行器上（`--backend ggml_cuda`），
  它们的 decode 步作为一张按 token 批处理的计算图执行（见[token 批量 decode](#token-批量-decode)）。
  `--backend ggml_cpu`、`TS_DSV4_FUSED=0`、加载了 DSpark 草稿器或设置
  `TS_BATCHED_FUSED_DECODE=0` 时，仍走逐槽前向调用。
- V4.1 的 DSpark 投机解码属于实验性功能。加载器只在 `ggml_cuda` 与 `ggml_cpu` 上接受
  `deepseek41-dspark` 或 unsloth 的 `dflash` 草稿器（`--draft-model`），在其他执行器上拒绝，
  V4 的草稿模型也会被拒绝。合成测试（`DeepSeek41DsparkIntegrationTests`）以及真实草稿器在
  `ggml_cuda` 双 GPU 按层切分与实验性路由专家 TP 下的初步文本/图像 HTTP 检查已通过；
  大量磁盘换页下的通用质量与吞吐仍未获验证。另行进行的 24-token 文本/图像配对检查，
  在两种模式下均匹配普通解码的 token ID 与 `max_tokens` 结束原因，DSpark 实际参与解码
  且进程正常退出。加载草稿器期间，按 token
  批量 decode 与保留缓存都不生效。没有草稿器时，`--spec`（包括 `--spec-type ngram`）
  只提供普通解码。
- K/V cache 在每个执行器上都是 F16，`KV_CACHE_DTYPE=q8_0` / `q4_0` 会在**加载时被拒绝**
  （`DeepSeek41Architecture.ValidateLoad` 在打开检查点之前抛 `NotSupportedException`）；
  显式的 `f32` 会在 stderr 上被告知并按 `f16` 报告。以前它是被静默接受的：原生计算图照样
  分配 F16 cache，而 `KvCacheDtype` 却报告 `q8_0`。这些 cache 不是共享家族交给 ggml
  flash attention 的逐层 K/V 张量（后者在其向量内核的 64/128 宽 head 尺寸下能读
  q8_0/q4_0 K/V）：它们是 MLA 潜变量行（K 兼作 V，宽度为潜变量宽度，CUDA flash-attention
  内核只接受 F16），存放在滑动窗口环、压缩行与索引器行、回退检查点影子以及 DSpark 草稿环
  里，由 `TSG_DSV4_FUSED_ATTN_PREP` / `TSG_DSV4_FUSED_COMPRESS` 写入，由
  `TSG_DSV4_FUSED_KGATHER`、compact / TP gather 和检查点复制以 F16 行直接读取，没有任何
  反量化步骤 —— ggml CUDA、ggml CPU、Direct CUDA 与纯 C# 执行器都是如此。要量化它们就得
  把上述每个内核重新定型；何况检查点自带的训练时 cache 量化（原始行 FP8 E4M3、索引器行
  MXFP4、压缩行 NVFP4）已在每次 F16 存储之前施加，q8_0 块并不会进一步缩小 cache 的信息量。
  上面的启动示例传 `KV_CACHE_DTYPE=f16`，这是唯一什么都不改变的取值。
- 图像/视频输入需要单独准备的视觉伴随文件。编码器与图文计算图有 CPU/CUDA fixture 覆盖，
  并在按层放置与 routed TP 下做过完整检查点的媒体检查。真实图像的 BF16 特征对比超出了
  小规模 fixture 的逐元素容差；见验证报告。目前没有经过验证的音频推理路径。
- 要做同权重对比，需要一个兼容的 llama.cpp V4.1 推理运行时。上面链接的 GGUF 仓库的补丁
  只增加了转换支持。缺少参考实现并不能确立质量或性能的对等。
- 完整检查点的数值 smoke 能产出预期 token，但未通过严格的 F32 输入 oracle 对比（相对
  L2 0.146216，最大绝对误差 2.708920）。量化激活的算术与该参考不同；
  保留的分阶段分析（`docs/validation/deepseek41/smoke18-reference/README.md`，
  本地验证记录，未提交到 Git）未能完全归因最终的差异。贪心结果一致不等于严格的数值一致。

把张量并行扩展到注意力，需要 rank 局部的计算图、权重分片与 cache 状态，并在进入下一个
非线性残差算子之前对注意力输出做一次归约。现有的 GLM 支持
（[ggml_ops_glm_dsa.cpp](../../TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp)）提供了可复用
的块对齐权重切分，以及带设备集合通信的 rank 局部计算图执行。切分 query head 与分组输出
投影时，V4.1 的八个输出组应保持完整；单个 KV head 与共享的索引器状态可以先做复制。

已实现的专家切分使用不等宽的块对齐分区。2304 的中间维包含九个 256 元素的 K-quant 块：
两个 rank 可取 1280 与 1024，四个 rank 可取 768、512、512、512。gate/up 张量对每个专家
都沿该中间维切分；down 张量按输出行切分，每个 rank 读取完整的隐藏激活，各行直接拼接而不求和。共享专家与 CPU 卸载的
输出只计一次。当前的 ggml CUDA 不再暴露旧的 split-buffer 接口；这条路径显式构建并执行
rank 局部的专家计算图。

短提示、长提示、JSON、工具往返、agent 工作流、并发、放置以及既有模型回归检查的完整
流程，见[验证协议](../deepseek41_validation.md)。未支持的场景保持明确的未验证状态。

较早的一次托管验证阶段**在本地与指定 VM 上都通过了 3,651/3,651 项测试**，没有跳过，其中包括在
两台主机上都运行的 84 项音频/图像/API 专项检查。覆盖范围包括工具历史分隔符的往返、合法的
部分 Unicode token、畸形 UTF-8 的拒绝，以及在开始流式输出之前对图像/音频请求的校验。
之前的图像验证阶段在两台主机上都通过了 3,635/3,635 项测试，其中包括本地 91 项专项检查。
每个 Runtime DLL 都与其已验证的 3,597 项测试的原始 tag-family 序列化阶段保持一致。最初的
匹配放置基线使用的是更早的 3,566 阶段主机；那些测量以及中间 3,582 阶段的检查都仍然保留。
推理测试工具另外在本地与 VM 上通过了 33 项单元测试；其当前范围列在
[验证报告](../deepseek41_validation.md)中。
第一次图像验证的 VM 尝试暴露了一个测试准入竞争
（`docs/validation/deepseek41/retained-cache-admission/README.md`，本地验证记录，未提交到
Git）；同步化后的测试类与完整测试通道在两台主机上都通过，原有断言保持不变。
确切的命令、排除项、计数器、哈希以及保留下来的间歇性测试失败
（`docs/validation/deepseek41/managed-correctness/README.md`，本地验证记录，未提交到 Git）
与完整检查点的质量和性能结果分开记录。

较早的按层与 routed-TP 正确性配置使用原生 `6b3b5ab3…` 与托管侧 stage 3,651。
按层切分通过了 **138/138 项推理用例**；routed TP 通过了 **129/130**。
按层计划还包含八个并发长上下文用例：

| 场景组 | 按层切分 | Routed TP |
|---|---:|---:|
| 短提示、JSON、schema、历史与默认并行的工具工作流 | 30/30 | 29/30 |
| required、named、none、串行与并行工具策略 | 30/30 | 30/30 |
| 思考工作流 | 4/4 | 4/4 |
| 非流式工作流 | 4/4 | 4/4 |
| 图像与抽样的视频帧 | 25/25 | 25/25 |
| 中文与 Unicode JSON | 10/10 | 10/10 |
| 单独配置的串行工具工作流 | 10/10 | 10/10 |
| 持续 decode | 15/15 | 15/15 |
| 约 8k 与 32k 输入 token 的长文检索 | 2/2 | 2/2 |
| 每种长上下文尺寸下的四个并发请求 | 8/8 | 不在该配置中 |

这些是旧权重与旧二进制的正确性记录，并非 1017 检查点的性能估计。原始计时与并发长上下文
统计仍保留在 `docs/validation/deepseek41/full-checkpoint/`（本地验证记录，未提交到 Git）中；
本卡片的性能对比请使用上面的 1017 测量。

之前失败的 named-thinking 与 thinking-agent 用例在两个最终配置中都已通过。在 routed TP 下，
仍有一个默认并行的 agent 请求在拿到发票结果之前，就与 `read_invoice` 一起发出了参数全为
零占位值的 `calculate_total`。严格检查器拒绝了它；该组因此从第一次 TP 配置的 30/30 变为
29/30。串行策略单独成功并不能抵消这次失败。一次本地的提示词/语法/解析器诊断
（`docs/validation/deepseek41/parallel-tool-dependency/README.md`，本地验证记录，未提交到
Git）保留了模型发出的调用，没有发现任何迫使多出这次调用或产生零值的缺陷。最终的四层
CPU 卸载配置另外通过了 28/30 个默认并行质量用例，保留了两次过早发出依赖调用的失败。
按层配置的 30/30 结果并不能抵消另外两种放置的失败。原生代码与启动设置也有变化，因此
这些结果既不能单独归因于语法改动，也不能证明质量的全面提升。上文的十二项 HTTP 输入
拒绝检查与推理计划相互独立。确切的报告与其余对比记录在最终放置记录
`docs/validation/deepseek41/final-placements/README.md`（本地验证记录，未提交到 Git）中。

既有模型的检查同样保留了回归。最终的 75 用例对比
（`docs/validation/deepseek41/existing-model-regressions/final3651-native6b3/README.md`，
本地验证记录，未提交到 Git）通过了 39/75 个用例，并且在那一轮中相对其配对参考没有引入新的
失败；单独的 Unicode JSON 覆盖通过了 15/15。随后重复进行的 JSON 对比
（`docs/validation/deepseek41/json-performance/completed-r2/README.md`，本地验证记录，
未提交到 Git）暴露了同一请求上的又一次失败，并记录到 Qwen3.5 的首 token 延迟变慢，
尽管其短回答的 decode 更快。这些结果与前一轮“没有引入新失败”的观察分开记录，并不能证明
完全不存在回归。

Qwen3.5 较短的交替对照（`docs/validation/deepseek41/json-performance/qwen35-alternating/README.md`）同样
显示最终延迟变慢。之后一次 72 请求的对照
（`docs/validation/deepseek41/json-performance/qwen35-solo72/README.md`；以上两项均为本地
验证记录，未提交到 Git）固定了原生库，所有回答都通过，也没有复现这次变慢。这些诊断没有
促成任何生产代码修复；不一致的结果及其局限仍记录在验证报告中。
