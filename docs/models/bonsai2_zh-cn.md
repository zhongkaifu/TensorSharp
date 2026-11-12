# Bonsai2 27B

[← 返回模型索引](README_zh-cn.md) | [English](bonsai2.md)

Bonsai2 使用稠密 Qwen 3.5 混合架构，并带有 PRISM 的带符号 Hadamard 旋转与自定义的
低 bit GGUF 编码。仅仅把它的权重
当作普通的三值数来解释，得到的是一个错误的网络：投影输入与 embedding 输出还必须施加
元数据声明的变换。

## 本地产物与架构

| 产物 | 字节数 | SHA-256 |
|---|---:|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |

配套文件是 `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` 与
`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`。给出图像时，CLI 会在模型旁边查找它们
（优先 BF16）；服务端只通过 `--mmproj` 加载投影器。对比图像质量或内存占用时，用
`--mmproj` 显式选择其中一个。

两个语言模型文件都声明 `general.architecture=qwen35`、64 层、隐藏层宽度 5,120、
FFN 宽度 17,408、词表 248,320，以及 262,144 token 的上下文。其中 48 层是循环
GatedDeltaNet 层，16 层是全注意力层。全注意力使用 24 个 query head 与 4 个 KV head，
宽度均为 256；GatedDeltaNet 使用 16 个 key group 与 48 个宽度为 128 的 value head。
底层的注意力、循环状态、RoPE、视觉编码器以及聊天 / 工具协议见
[Qwen 3.5 卡片](qwen35_zh-cn.md)。

```text
token ID -> low-bit embedding row -> inverse signed Hadamard
  -> 64 hybrid Qwen 3.5 layers:
       RMSNorm -> attention or GatedDeltaNet -> residual
       RMSNorm -> dense gated FFN -> residual
       (declared low-bit projections rotate their input activations)
  -> RMSNorm -> signed Hadamard -> low-bit output head -> logits
```

## 存储与变换

GGUF 读取器把 PQ2_0 识别为张量类型 142，把 PTQ1_0 识别为 143。TensorSharp 自有的原生
代码把它们的块打包无损展开为上游 GGML 的 Q2_0 块。这样保留了所表示的权重值、无需重新
量化；量化数据体积对 PQ2_0 约增加 6%，对 PTQ1_0 约增加 29%。因此加载后的内存占用不等于
文件大小。原始 GGUF 从不被改写，TensorSharp 也不为引入发布方专有的张量类型而给 ggml
打补丁。

`prism.hadamard.*` 元数据规定了变换版本、归一化的 Sylvester-Walsh-Hadamard 变换、输入轴、
块大小、显式的 ±1 符号，以及接受正向或逆向变换的权重名。TensorSharp 在加载权重之前校验
完整的约定，并拒绝未知的变换布局。这些产物使用 1,024 个元素的块。

对投影输入 `x`，每个块计算 `H(Dx)/sqrt(block_size)`，其中 `D` 含存储的符号，`H` 是
未归一化的 Hadamard 矩阵。embedding 行使用逆序，即 `D(Hx)/sqrt(block_size)`。分组 GDN
元数据还会在变换之前，把 SSM 输出投影的输入重排为发布方的 head 顺序。融合的 QKV 与融合的
gate/up 权重保留其原始投影的变换。

实现入口：

- [BonsaiHadamardMetadata.cs](../../TensorSharp.Runtime/BonsaiHadamardMetadata.cs)
  校验元数据，并实现托管的逆向 embedding 变换。
- [ModelBase.Bonsai.cs](../../TensorSharp.Models/ModelBase.Bonsai.cs) 在模型加载期间
  负责转码与变换注册。
- [QuantizedWeight.Bonsai.cs](../../TensorSharp.Models/Weights/QuantizedWeight.Bonsai.cs)
  负责为权重缓存键注册变换，并在释放时注销。
- [bonsai_quant.cpp](../../TensorSharp.GGML.Native/bonsai_quant.cpp) 在不修改上游 GGML
  的前提下实现精确的块解码 / 转码。
- [ggml_ops_bonsai.cpp](../../TensorSharp.GGML.Native/ggml_ops_bonsai.cpp) 把带符号变换
  接入 TensorSharp 的原生计算图路径。

## 运行模型

当前的集成要求单设备 GGML 后端。纯托管 CPU、direct CUDA、MLX 与张量并行配置都会被
拒绝，而不是悄悄省略旋转。针对具体设备的端到端验证与这一加载资格是两回事；见下文的
验证流程。

[TensorAgent](../../TensorAgent/README.md) 把 Bonsai2 列为实验性的 `bonsai-2-27b-ptq1-0`
目录条目。它从
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/b072e1d3b35a0a630cece372c2127528e0994386)
（Apache-2.0）下载 PTQ1_0 文件，以及作为可选投影器的 Q8_0 mmproj，并校验它们的哈希。该条目只向
16 GB 设备档位提供：重打包为 Q2_0 后，5.95 GB 的下载会占用约 7.7 GB，再加上 K/V 缓存与计算缓冲区，
就超出了 12 GB iPhone 所能给出的内存。PQ2_0 持有相同的三值权重，重打包后的大小也大致相同，因此
该条目选用更小的 PTQ1_0 下载。Bonsai2 在 iOS 上的执行尚未验证。

请显式设置上下文上限，而不是按标称的整个 262k 窗口分配：

```sh
MAX_CONTEXT=4096 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Cli -c Release -- \
  --model /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --backend ggml_metal --input prompt.txt --max-tokens 128 \
  --temperature 0

MAX_CONTEXT=4096 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Server.Host -c Release -- \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --mmproj /path/to/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --backend ggml_metal --host 127.0.0.1 --port 5000
```

在单个 `ggml_metal` 或 `ggml_cuda` 设备上，模型运行在共享解码器运行时（`Qwen35RuntimeModel`）上，
这也是那里所有稠密 Qwen 3.5 检查点所走的路径：GGML 构建每一步的计算图时，凡是权重带有正向旋转的乘积，
其输入都会先被旋转（Q/K/V 与 gate/up 共用同一个旋转后的输入），嵌入行则经过逆变换。`ggml_cpu` 与
`ggml_vulkan` 仍使用 Qwen 3.5 系列实现，它把同样的变换插入自己的融合计算图。旋转会增加计算量，
重新打包后的权重也比发布方的自定义格式占用更多内存；性能必须在实际负载上测量。

## 可复现的验证

[Bonsai2 验证流程](../../eng/validation/bonsai2.md)覆盖精确的原始 token 参照、单个 /
并行流式请求、算术、结构化 JSON、工具调用、两个投影器以及解析器测试。生成的证据应放在
已忽略的 `docs/validation/bonsai2/` 或 `artifacts/` 中。

发布方基线是独立的 `PrismML-Eng/llama.cpp` 检出，版本
`bdc23b56b4458b9f1655aec5287f3ab56ee8daaa`。无法加载 PQ2_0/PTQ1_0 的上游 llama.cpp
构建不能作为有效的性能对比。TensorSharp 所用的未修改上游 ggml 版本需另行记录。

可用的验证主机是一台 48 GiB 统一内存的 Apple M5 Pro。它的 Python 环境没有安装 Torch、
vLLM 或 SGLang，也没有 CUDA 设备。查看过的 vLLM 版本
`88aa0d287dd3abac89741bbea349350a4d49194e` 与 SGLang 版本
`1d59ce7c9063edec20fd3f5a49a504b8c12acd23` 包含 Qwen 3.5/GatedDeltaNet 实现，但不支持
PQ2_0/PTQ1_0 格式。它们的融合投影、分块循环 prefill 与隔离的序列状态模式为本实现提供了
参考；这些本地检出并未提供实测的 Bonsai2 吞吐基线。本主机上的测试也没有确立任何与它们的
并行吞吐相当的结论。

### 实测的集成覆盖（2026-09-22）

使用未修改的上游 ggml `179b60f27b1019d42da01ac532cabdb8f73ba8b7`，两种格式在 Metal
上都精确复现了发布方基线的四个 32 token 贪心提示。四个并发序列也与其串行输出完全一致，
共 31 个融合 decode 步，没有回退步。每种格式都通过了 21 项 HTTP 单独 / 并发对比，以及算术、
JSON、工具调用与图像颜色检查。图像检查使用 PQ2 + Q8_0 投影器与 PTQ1 + BF16 投影器。
这些是功能冒烟测试，不是全面的质量评估。

额外的 CPU 模型检查只覆盖 PQ2（两个提示，各 8 个 token）。较长的 Metal 提示覆盖 1,623 个
输入 token 与以 EOS 结束的 2 个输出 token。两者都不能证明标称的 262k 上下文可用。
CUDA、Vulkan、iOS 与张量并行的模型执行均未验证。

| 格式 | 指标（tokens/s） | TensorSharp | 发布方 llama.cpp |
|---|---|---:|---:|
| PQ2_0 | Prefill 512 | 364.05 | 384.44 |
| PQ2_0 | Decode 64 | 25.50 | 26.96 |
| PQ2_0 | HTTP，并发 4 | 28.40 | 32.57 |
| PTQ1_0 | Prefill 512 | 364.49 | 357.67 |
| PTQ1_0 | Decode 64 | 25.53 | 26.31 |
| PTQ1_0 | HTTP，并发 4 | 28.17 | 16.15 |

纯模型速率是上下文深度为零时三次预热后运行的均值；HTTP 速率是三次运行的中位数，每个
请求生成 64 个 token。两个引擎都使用 F16 KV 与 512 token 的物理 prefill 批；服务端每个
请求的上下文预算为 2,048 token。两个引擎分别运行，没有热状态遥测，也没有交替重复。
因此较小的差异不能作为普遍性能优势的证据。

完整的性能目标仍未达成：纯模型 decode 在 PQ2 上慢 5.4%，在 PTQ1 上慢 3.0%，PQ2 并发 4
的 HTTP 吞吐低 12.8%。vLLM/SGLang 吞吐未测量。已忽略的本地报告
`docs/validation/bonsai2/REPORT.md` 保留了命令、全部样本、确切的产物标识与测试局限。
更大范围的原生 CPU 测试套件还有一项 DeepSeek41 容差失败，在未修改的 TensorSharp HEAD
上同样复现；它不计为通过的测试。

### 该次运行之后的 CUDA 状态

之后修复了带符号 Hadamard 路径上一个仅出现在 CUDA 上的缺陷。旋转 matmul 会让后端对
输入运行它自己的快速 Walsh-Hadamard 变换；ggml-cuda 的 kernel 不检查输入类型，把 F16 行
当作 F32 读取。在逆向方向上，这在 A5000 上相对稠密 oracle 产生了 16.0 的最大绝对误差，
而 CPU 后端是正确的。
[ggml_ops_bonsai.cpp](../../TensorSharp.GGML.Native/ggml_ops_bonsai.cpp) 现在会在该节点
之前把变换输入扩宽为 F32。模型激活本来就是 F32，因此模型自身的路径不会多出节点。

[`bonsai2-reference-comparison.sh`](../../eng/validation/bonsai2-reference-comparison.sh)
是面向 A5000 CUDA 主机的参照引擎对比脚本。它记录上游 llama.cpp 对 PQ2_0/PTQ1_0 的
拒绝信息，在独立目录中按固定提交构建 PrismML 分支，同时断言 TensorSharp 的 ggml 检出
保持未修改，在同一 GGUF 上测量纯模型与 HTTP（并发 1 与 4）吞吐，运行一次投影器冒烟测试，
并把输出写入已忽略的 `artifacts/`。它没有任何已提交的结果，因此如上所述，CUDA 模型执行
仍未经验证。

### 共享解码器运行时（2026-11-12）

在 M5 Pro 上使用未修改的上游 ggml `cd6d2b75f5d3e4b2f3fe8e8fbc6c8b84ac80e74a`，PTQ1_0 文件（SHA-256 见上）
在 `ggml_metal` 上由共享运行时与系列实现给出相同的贪心文本，两者各交替运行两次，F16 KV：

| 提示 | 指标 | 运行时 | 系列实现 |
|---|---|---:|---:|
| 55 token | Prefill（毫秒） | 357-370 | 376-402 |
| 55 token | Decode，128 token（token/秒） | 27.4 | 27.2-27.4 |
| 3,609 token | Prefill（毫秒） | 11,512-11,533 | 11,592-11,636 |
| 3,609 token | Decode，109 token（token/秒） | 26.1 | 26.2 |

合成的 PRISM 检查点（旋转后的 PQ2_0 投影、输出头与嵌入，以及分组的 value 头）在
`Qwen35RuntimeTests.PrismRotationsMatchQwen35Model` 中比较两条路径的 logits：`ggml_metal` 上最多相差
1.1e-5，`ggml_cpu` 上 2.0e-2（其激活经 BF16 舍入）；若省略旋转，第一行相差 1.2。该测试在 `ggml_cuda`
（RTX 2000 Ada）上同样通过；真实检查点尚未在 CUDA 上运行。
