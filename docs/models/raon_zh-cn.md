# KRAFTON Raon-1 nano（`raon1nano`）

[← 返回模型索引](README_zh-cn.md) | [English](raon.md)

TensorSharp 可运行 [KRAFTON/Raon-1-nano](https://huggingface.co/KRAFTON/Raon-1-nano)，使用 KRAFTON 在
[KRAFTON/Raon-1-nano-gguf](https://huggingface.co/KRAFTON/Raon-1-nano-gguf) 发布的 GGUF（由 KRAFTON llama.cpp 的
`raon-1-nano` 分支转换）。该架构并非 Qwen 3：模型自带代码（`model_type: raon1_nano`，`trust_remote_code`）是一个
24 层的混合解码器，由以下部分组成：

- **mHC：** 每个 token 4 条超连接流，在每个注意力块与 FFN 块前后各有一次可学习的读出与写回混合。
- **PLE：** 每个 token、每层一份 256 维的逐层嵌入（259,840 × 6,144 的表），在 FFN 之后加到每条流上。
- **RDA：** 18 层循环线性注意力，各含因果卷积、带低秩且有下界衰减的门控 delta 规则状态，以及独立的写入与擦除门。
- **MLA：** 6 层多头潜在注意力（从 0 计第 3、7、…、23 层），吸收式、无位置编码、带 sigmoid 输出门；缓存每个
  token 保存 576 维的键与 512 维的值。

该家族在共享解码运行时上仅以整模型前向图运行（`RequiresGraphs`），复用已有 IR 算子：RDA 用 `GatedDeltaNet` 与
`SsmConv`，MLA 用 DK 576 / DV 512 的 `FlashAttnExt`。不需要新的 ggml 算子，也不修改 ggml。

| 属性 | 值 |
|---|---|
| GGUF 架构 | `raon1nano` |
| 源码类 | [`RaonRuntimeModel`](../../TensorSharp.Models/Models/Raon/RaonRuntimeModel.cs)（[计算图](../../TensorSharp.Models/Models/Raon/RaonRuntimeModel.Graph.cs)） |
| 架构插件 | [`RaonArchitecture`](../../TensorSharp.Models/Models/Raon/RaonArchitecture.cs) |
| 模态 | 仅文本 |
| 思考模式 | 支持。`--think` 以 `<think>\n` 开启；否则保留模板的闭合空块 `<think></think>\n\n`。 |
| 工具调用 | 模板的 Qwen 3.5 风格 `<tool_call><function=…>` 块，由 Qwen 3.5 输出解析器解析；尚未做端到端验证。 |
| 后端 | `cpu`（托管）、`ggml_cpu`、`ggml_metal`、`ggml_cuda`、`ggml_vulkan`；`cuda` 与 `mlx` 在加载时拒绝。 |
| 批处理 | 经共享运行时支持连续批处理、检查点与推测解码校验 |
| 多 GPU | `ggml_cuda` 与 `ggml_vulkan` 上的 `--layer-split N`：共享运行时的整层阶段，hc 流与该步的每层嵌入（阶段携带量）从一个阶段传到下一个阶段，每层的 RDA 状态位于其所在阶段（`RaonRuntimeTests.PlacementMatchesOneDevice`，模拟设备：与单设备的 logits 一致）。张量并行与混合放置仍声明为缺口：该家族的降低不追踪单个 rank 的分片。 |
| MTP | `*_MTP` 文件可加载，但其 NextN 块被跳过（不加载、不运行） |

## 运行

```bash
echo "What is the capital of France? Answer in one word." > prompt.txt
dotnet run --project TensorSharp.Cli -- \
    --model models/raon-1-nano-Q8_0.gguf --backend ggml_cuda \
    --input prompt.txt --max-tokens 256 --no-skills
```

KRAFTON 建议：一般用途 temperature 0.1、presence penalty 1.5；智能体用途 temperature 1.0。

## 实现要点

GGUF 存储的是折叠后的布局，计算图不会再次应用这些常数：

- 超连接写回的 ×2 系数已折入 `attn_output` 与 `ffn_down`。
- `exp(A_log)` 已折入 `ssm_f_b` 与 `ssm_dt.bias`。
- `inp_gate` 已除以 4，因此 PLE 门读取的是各流之和而非均值。
- 每个位置的 `hc_*_chart` 打包了流混合映射 K（17 → 16）、行缩放 r 与偏移 c0，加载时拆为 F32 常量。

其他细节：

- **MLA 缩放：** 运行时施加 1/√576，而模型使用 1/√192，故查询预先乘以 √(576/192)。
- **PLE 表：** token 表留在主机端；每步只在主机上反量化该步 token 的行，并作为图输入上传。
- **PLE GELU：** 由精确算子（tanh、mul、scale）组合而成，因为 ggml-cpu 的 `GELU` 读取 fp16 查找表。

## 验证

[`RaonParityTests`](../../InferenceWeb.Tests/RaonParityTests.cs) 将运行时与 KRAFTON 自带建模代码在 CPU 上以 float32 运行
的结果对比（Transformers 5.16.1，torch 2.15.0）。夹具为
[`Fixtures/Raon/hf_goldens.json`](../../InferenceWeb.Tests/Fixtures/Raon/hf_goldens.json)，由
[`eng/validation/raon-hf-goldens.py`](../../eng/validation/raon-hf-goldens.py) 生成：五个对话提示（英文、算术、韩文、
代码、故事）各 32 token 的贪心续写，以及十条分词语料。测试要求 `TS_TEST_MODEL_DIR` 中有 `raon-1-nano-Q8_0.gguf`，
否则跳过。

`raon-1-nano-Q8_0.gguf` 上的结果（RTX 5090，Windows，2026-11-22）：

| 通道 | 教师强制 argmax 一致 | 首位置 top-10 最大 \|Δ log p\| |
|---|---|---|
| `ggml_cuda` | 105/105 | 0.09–0.28 nats |
| `ggml_cpu` | 105/105 | 0.14–0.39 nats |
| `cpu`（托管） | 105/105 | 0.13–0.46 nats |

作为参照：Transformers 自身在权重与激活均按 Q8_0 舍入时，与其 float32 结果相差 0.09–0.24 nats；KV 缓存在 f16 与
f32 之间切换会使最差的提示移动约 0.26 nats。

KRAFTON 的其他量化文件在 `ggml_cuda` 上的结果（`TS_TEST_MODEL_DIR` 指向该文件；界限沿用 Q8_0 的）：

| 文件 | 教师强制 argmax 一致 | 首位置 top-10 最大 \|Δ log p\| |
|---|---|---|
| `raon-1-nano-F16.gguf` | 104/105（唯一不一致处是 0.005 nats 的近似平局） | 0.03–0.07 nats |
| `raon-1-nano-Q4_K_M.gguf` | 98/105 | 0.50–5.2 nats；超出 Q8_0 的界限，未与 llama.cpp 的 Q4_K_M 对比 |
| `raon-1-nano-Q4_K_M_MTP.gguf` | 与 Q4_K_M 相同（NextN 块被跳过） | 与 Q4_K_M 相同 |

F16 表明计算图复现了参考前向。Q4_K_M 偏差较大符合 4 位权重的预期，但这里只与 float32 对比过；其贪心回答保持连贯
（例如带步骤的 `17 × 23 = 391`）。

所有提示与语料的分词结果与 Hugging Face `tokenizers` 完全一致（含韩文、CJK、emoji、NFC 组合与特殊 token）；GGUF
对话模板的渲染结果与 `apply_chat_template` 完全一致。

[`RaonRuntimeTests`](../../InferenceWeb.Tests/RaonRuntimeTests.cs) 在合成 F32 夹具
（[`RaonSyntheticModelBuilder`](../../InferenceWeb.Tests/RaonSyntheticModelBuilder.cs)）上覆盖服务机制：

- 批处理槽位与各序列单独运行的一致性；
- 引擎并发请求；
- 检查点（克隆与导出，GGML 上含设备拷贝）；
- 贪心推测解码与普通贪心解码的一致性；
- 分块预填充与单块的一致性；
- 固定的 GGML 后端与托管后端的一致性。

这些测试在 `ggml_cpu` 与 `ggml_cuda` 两种固定后端下均通过；尚未在 Metal 或 Vulkan 上运行。
