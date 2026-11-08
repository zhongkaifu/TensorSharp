# Qwen 3.8 Flash Next（`qwen4exp`）

[← 返回模型索引](README_zh-cn.md) | [English](qwen38-flash-next.md)

> **多节点执行：** `ggml_cuda` 下，在 `--layer-split N` 或 `--tp N` 后加
> `--tp-node-id` 和 `--tp-peers`，其中 N 是每个节点的 GPU 数量。所有节点需使用相同
> 的检查点、构建及数值配置。启动顺序和限制见[多节点指南](../multi-node-inference.md)。
> 下文历史上的单机测量不代表多节点性能。

Qwen3.8-Flash-Next 是一个混合型 MoE：GatedDeltaNet 递归层与全注意力层交错
（其中一部分全注意力层挂在 Qwen Sparse Attention 的 indexer 后面），再加上一个
PLE n-gram 嵌入块、×4 hyper-connection 流以及 512 专家的 MoE。GGUF 架构 id 是
`qwen4exp`。权重：
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
（每个量化档一个子目录、均为多分片；`--model` 指向 `-00001-of-` 那一片；图像输入
需要匹配的视觉 projector，例如 `mmproj-BF16.gguf` 或 `mmproj-F16.gguf`：CLI 在给出
`--image` 时会从模型旁边自动加载，服务端则需要显式传
`--mmproj`）。

## TensorSharp 如何运行它

在 GGML 后端上，整个 token（几乎）只跑一张图——嵌入、PLE（在图内）、全部 48 层、
最后的 mixer 以及 LM head——并配一个按形状索引的可复用 ggml 图缓存
（span 放弃时改走逐层融合 kernel，后者再逐算子回退）。视觉沿用
Qwen3.5-VL 塔，位置用 (T,H,W) IMRoPE；支持多图与多轮图像会话，并在轮次之间复用
KV（GDN 递归无法回退，因此只有当新 prompt **恰好扩展**已缓存前缀时才复用；见[保留前缀复用](#保留前缀复用)）。radix 复用可以越过相同的图像或视频 span：完整状态保存 M-RoPE 缓存间隙与 QSA 位置历史，键同时校验媒体身份、span 边界与 token。结束的主缓存留在原处供下一轮精确续接；只有其他请求需要替换它时，才转换为保留 holder。

思考模式可以开启或关闭。关闭时，助手轮次以已发布模板输出的闭合空块 `<think>\n\n</think>` 开头，重放历史时也保留这一确切后缀，因此缓存前缀仍能匹配。

已发布的 Qwen4Exp 视觉塔最终 patch merger 使用 GELU(erf)，与 transformer block 的
`gelu_pytorch_tanh` 分开，符合
[Transformers v5.16.1 参考实现](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1705)。
所提供 GGUF 路径保留既有的 tanh merger 默认行为。在加载 projector 前设置
`TS_Q4E_VISION_MERGER_ERF=1`，可单独为 merger 选择参考实现的 erf，而 block 激活不变。
交付构建的 base 模型默认路径通过 12/12 项图像检查，16 个回答均复现先前 tanh 运行。
完整 erf 试验通过 10/12；随后同一构建的 C=1 OCR 对照中，tanh 通过 2/2，erf 通过
1/2，把蓝色 `9364` 识别为 `9334`。这些限定范围的检查支持保留兼容默认值，尚未解释
识别差异的原因，也不能确立更广泛的激活函数质量结论。详见
[图像验证指南](../../eng/validation/qwen38-parallel-vision.md)。

## 工具调用与 Agent 工作流

`qwen4exp` 通过 Qwen ChatML 输出解析器返回结构化工具调用。支持 `<tool_call>`
内的 JSON 与 `<function=...><parameter=...>` 格式，以及流式分片和思考模式。
调用方工具以带调用 ID 的 OpenAI `tool_calls` 返回，结束原因是 `tool_calls`；
配置启用后，内置技能与代码工具由服务端 Agent 循环执行。

XML 参数按工具声明的 schema 解析：字符串 `123`、`true` 与 JSON 源码不会变成
数字、布尔值或对象。解析器仅移除两侧各一个格式换行，保留源码缩进与额外空行；
参数或 JSON 字符串内的 `</tool_call>` 不会提前结束调用。不完整的参数或函数不会
成为可执行调用；EOS 时已有完整正文、仅缺外层结束标记的恢复行为保持不变。

使用 `--skills-dir` 启用技能目录，`--skills-allow-exec` 启用技能脚本，
`--code-exec` 启用工作区文件与 shell 工具；编辑工具为 `apply_patch`。
执行与沙箱配置见 [Agent Skills](../agent_skills.md)。

由于该系列会渲染工具声明并带有这个解析器，它在服务端也可以使用
[子智能体委派](../multi_agent.md)（在对话路径上默认开启，与 skills 和 `--code-exec` 无关）。
`--no-multi-agent`（或请求中的 `multi_agent: false`）可将其关闭；CLI 没有子智能体。
该系列没有发布任何委派相关的实测结果。

可复用验证脚本：`eng/validation/validate-qwen38-tool-calls.py` 检查通用 API 的
流式/非流式、思考开/关与工具结果回传；`validate-release-agent-workflows.py`
配合 `eng/validation/fixtures/skills` 检查技能发现、读取、脚本、shell、代码生成
与读取/编辑/运行。`verify-agent-code-artifacts.py` 对最终源码使用额外输入独立执行。
若验证服务器显式关闭沙箱，两个执行验证脚本需传 `--sandbox-off`，报告只证明功能
执行成功，不证明沙箱隔离。完整命令见[英文版](qwen38-flash-next.md#tool-calling-and-agent-workflows)。

2026-09-19 使用提供的 UD-IQ4_XS 模型，在三张 NVIDIA A40 上验证
（`ggml_cuda`，按 15/16/17 层切分，关闭 MTP）；构建使用未修改的 upstream ggml
`456172ec733a135778adcd32d00e576a58232e45`：

- 160 项托管回归测试通过，无失败或跳过。
- 12 项常规通用工具用例全部通过：天气、数字字符串/JSON 源码、多行 Python
  （含尾部换行），分别覆盖流式与思考开/关，并用实际调用 ID 完成工具结果回传。
- 12 项技能/shell/代码工作流所需工具均成功执行；4 份最终生成或编辑的程序用额外
  输入独立执行通过。分离工具调用前的说明文字后，严格最终回答检查为 9/12 通过；
  另外 3 项在正确值外添加反引号或说明，仍记为失败。
- 4 项包含工具标记字面量的压力用例均在调用完成前以 EOS 结束，端到端仍失败；
  完整 XML/JSON 字面量调用已通过解析器回归。日志未暴露最终采样 token 的 ID，
  工具标记 token 本身不属于 EOS。

VM 禁止 user namespace，执行验证显式关闭了沙箱。本轮仅覆盖一种量化与串行请求，
不验证沙箱隔离、MTP、其他设备或性能。完整请求、SSE、产物、构建信息及失败证据保存在
已忽略的 `docs/validation/qwen38-tool-calling/` 目录。

## 视频输入

视频以 OpenAI Chat Completions 的 `video_url` content part 送达模型，内容是 base64 的
MP4、WebM 或 MOV data URI（不会抓取远程 URL）：

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1,"max_frames":8}}
```

`fps`（0 < fps ≤ 60）与 `max_frames`（1–64）可选；默认值取 `VIDEO_SAMPLE_FPS` 与正的
`VIDEO_MAX_FRAMES`，否则为 1 fps 与 16 帧，更长的片段会在全长上均匀采样。服务端把片段解码成
有序的帧，每帧带其源时间（帧序号除以探测到的帧率，变帧率片段为近似值），之后由 Qwen-VL 的
视频布局接手：

- **帧对（temporal pair）。** 视觉塔的 patch embedding 有两个时间切片
  （`v.patch_embd.weight` 与 `.weight.1`），所以连续帧两两合并，与 Qwen-VL processor 堆叠帧
  的方式完全一致；奇数帧的片段会重复最后一帧补齐最后一对。processor 以 2 fps 采样，所以它合并的
  两帧相隔 0.5 秒；TensorSharp 只在两帧相距不超过 `QwenVideoFrames.MaxPairedFrameGapSeconds`
  （0.575 秒）时才把它们配成一对。更稀疏的帧——默认的 1 fps，或按 `max_frames` 分散到长片段上的
  帧——是不同的画面，因此各自独占一个时间 patch（像静态图一样重复该帧），并保留自己的时间标签。
  把这种帧配对会把它们混在一起：卡片 17、42、86 的三帧 1 fps 片段读成 `["12", "47", "86"]`，
  每帧一个 patch 时读成 `["17", "42", "86"]`。代价是每个采样帧一个 patch，而不是每两帧一个。每一对单独编码（参考实现的视觉塔
  只在一个时间 patch 内做注意力），得到与一张静态帧相同的合并 patch token 数。整段片段按同一个
  视频像素预算（`Qwen35ImageProcessor.VideoMinPixels` / `VideoMaxPixels`）整体缩放，因此同一片段的
  每一对共用一个网格；即使用最小网格也放不进该预算的帧数会被拒绝。
- **提示词布局。** 模板把视频 part 渲染成 `<|vision_start|><|video_pad|><|vision_end|>`，整个外层
  span 会被替换——与 Qwen3-VL processor（transformers v4.57.1 `processing_qwen3_vl.py`）一致，片段外
  不再多包一对分隔符——为每对一个 `<t seconds><|vision_start|><|video_pad|>…<|vision_end|>` 块，
  `t` 是该对的平均源时间（保留一位小数）。由于帧对的标签丢失了两帧各自的时间（且 `max_frames`
  上限可能选中不相邻的帧），这些块之前会有一行文本
  `Sampled video frame times in chronological order: 0, 1, 2 seconds.`，列出全部采样源时间；逐对的
  视觉 token 布局不变。一条消息里的两个 `video_url` part 渲染成两段片段。同一条消息里的静态图保留
  各自的 `<|image_pad|>` span，按附件顺序排列。
- **编码缓存。** 帧对的 embedding 以两帧路径加片段尺寸为键缓存，任一帧文件的大小或时间戳变化
  （而不只是较晚那一帧）都会使其失效。
- **位置。** 每一对都像一张静态图那样定位，其 (T, H, W) 坐标从该对所处的运行位置起算——这
  就是 Qwen3-VL `get_rope_index` 把视频网格拆成逐对条目的规则——因此相邻帧对拿到严格递增的
  时间轴 M-RoPE id，中间的时间标签文本推进位置流，片段之后的文本从最后一对的网格之后继续。
  QSA indexer 的位置历史、MTP 草稿追赶以及片段之后的旋转/cache gap 记录的都是同一套坐标，
  所以投机解码与保留前缀和目标模型一致。

它不是什么：帧是采样得到的，不是由时间编码器解码；帧时间是采样到的源时间，而非重新对齐到
2 fps 的流。通过 Web UI 上传的视频帧不带源时间，仍然按静态图处理，每帧一个 span，与以前一样。
完整 checkpoint 的检查是 `benchmarks/engine_comparison/validate_deepseek41_media.py` 的
`video_order` / `video_timestamp` 场景，对象是挂了 `mmproj-BF16.gguf` 的 Qwen3.8 服务。

## 思考预算

开启思考时，思考内容一旦达到 `TS_THINKING_BUDGET`（`max_tokens` 不小于 512 时默认为其 75%），服务端和交互式 CLI 都会闭合思考块，答案随后在原 `max_tokens` 内生成。`</think>` 是单个训练过的 token（248069），宿主会先写入 Qwen 官方发布的交接句（"Considering the limited time by the user, I have to give the solution based on the thinking directly now."）。2026-09-29 之前该系列没有闭合 token，思考达到预算的轮次会以**空答案**结束（`finish_reason` 为 `thinking_budget`）：在 4x A40（`--tp 4` 与 `--layer-split 4`）上以 `max_tokens` 2000 运行四个并发的三轮会话，通过 0/4，失败的都是空答案轮次。加入交接句后，同样的 `--tp 4` 运行通过 4/4：交接句闭合了四个轮次，每一轮都有答案，每个会话都复用了上一轮（第 3 轮复用 1591-3509 个提示 token 中的 1564-3482 个）。

## 连续批处理

并发请求通过**逐序列状态持有者**（per-sequence state holders）来服务：每个在飞请求
各自拥有自己的注意力 KV 与 QSA indexer 缓存、GDN 卷积与 delta-net 状态、PLE 卷积
历史与 n-gram 窗口、M-RoPE 缓存间隙与坐标历史，以及固定下来的 kernel 描述符。
采样和 logits 属于各自的请求；第二个请求覆盖模型的共享输出缓冲区之前，执行器会复制
首个请求借用的 solo logits。不可变的 PLE 卷积系数由模型持有，覆盖所有 holder 的生命周期。

**并发 decode 默认使用融合 arena。** 单设备 GGML CPU、Metal 或 CUDA 后端，在 F16 KV
缓存和 span 状态已初始化、几何受支持时，两个或更多就绪的 decode 请求一起执行一张可复用的
ggml 图。路由专家卸载到主机时，执行会在后端图分段之间跨越主机专家接缝。
Metal 使用共享 ggml 图；CUDA graph capture 是另一项后端机制。
Attention/QSA 与 GDN 保留各自的 slot 状态。投影、路由 / 共享专家及语言 head 位于共享图中；
后端数值行为依赖行几何的算子保留 solo 的归约方式，并不要求每个节点都使用多行 kernel。
每条 attention lane 使用自己 solo 路径的填充 KV 窗口和 mask，不使用最长请求的窗口。
Router softmax 同样保留 solo 的逐行归约，因为 Metal 按批量总行数改变线程数量会改变路由权重。
混合调度步会批量处理就绪的 decode 子集，同时通过各自 holder 执行新请求的
prefill 分块。请求加入、退出、缓存增长及返回 solo 路径时，先刷新或退役对应的 arena slot，
再由另一条路径读取其状态。图像请求完成媒体 prefill 后也能加入批量 decode；压缩后的
旋转位置与 QSA 坐标历史随自己的 holder 保存。

张量并行、按层切分、其他 KV dtype，以及 span / 后端几何不可用的配置，在支持时仍使用
逐序列融合路径。GPU arena 要求上游 flash attention 支持对应的 head 几何；CPU 使用相应
attention 回退。参与批量执行的每个 holder 必须已有初始化的权威状态、位置相符，并能容纳
下一行缓存。需要增长的 holder 先走 solo 路径，再加入批量执行。
只有一个就绪 decoder 时使用 solo 融合 decode。
`TS_BATCHED_FUSED_DECODE=0` 可作为轮询对照。原生执行失败会使受影响的请求失败，
不会把递归状态可能已经部分推进的批量步骤改为 solo 重试。实现位于 TensorSharp 自有代码，
上游 ggml 保持未修改；源码支持不等于所有后端都已证明性能提升或真实模型质量。

可复用检查：[并发文本与请求隔离](../../eng/validation/qwen4exp-concurrent-http.md)，以及
[并发图像内容、附件顺序与历史](../../eng/validation/qwen38-parallel-vision.md)。图像指南包括
固定 revision 的 projector 来源和哈希。生成报告应保留在被忽略的 `docs/validation/` 或
`artifacts/`；失败、跳过或不可用的模型 / 设备场景不计入验证通过。

### 本机 Metal 验证，2026-10-04

两个指定 checkpoint 均在 Apple M5 Pro、48 GiB 统一内存上实测，上游 ggml 保持未修改，
revision 为 `353b63b439f27ab2cc19dac97ab1681ba6d2d084`。
[原生 probe](../../eng/validation/qwen4exp-batched-decode-probe.md) 覆盖宽度 2、3、4，
64 个 teacher-forced decode 步和两次重复。保留的 4,728 条 prefill / decode / 续接对照
均测得 max |Δlogit| = 0，贪心 token 差异为 0；未另行检查原始 logit 的逐位一致性。
合并两次重复后的原生 decode 测量如下：

| Checkpoint | 融合 decode tokens/s | 相对轮询提升 | 交付默认图像检查 |
| --- | ---: | ---: | ---: |
| Base UD-Q2_K_XL | 23.86–25.61 | 17.37–50.77% | 12/12 通过 |
| Uncensored IQ2_XXS | 18.70–26.82 | 15.04–35.36% | 4/12 通过，整套失败 |

匹配的两个并发 HTTP 主题回答和精确 marker，在批量开启与轮询配置间一致，运行日志也记录了
融合执行。两次重复合并后，主题吞吐在 base 上提升 7.01%，uncensored 上提升 8.18%；
uncensored marker 提升 0.71%。Base marker 结果有快有慢：先运行并发且 prefix cache
关闭的对照慢 3.08%；交付构建中先串行、再并发且 prefix cache 开启的一次对照快 11.705%。
后者所有匹配请求都报告 0 个 cached prompt token；运行顺序并不能确立缓存命中或预热的
因果解释。这些测量不能支持一致加速或全场景无回归承诺。

交付的 tanh 默认图像检查复现先前默认路径的全部回答：base 完成 16 轮，uncensored
完成 14 轮；C=1/C=2 的 8 对 base 和 7 对 uncensored 已完成轮次均精确一致。
Uncensored 单图 OCR 把蓝色 `9364` 读成 `9324`，两种附件顺序检查也失败。
它的两个蓝图后续请求因首轮 OCR 错误而未发送，不计入测试通过。并发答案一致因此不等于
图像质量验收通过。两张卡片的 OCR / 颜色 / 顺序 / 历史检查，以及文本相关性和 marker
检查，只覆盖有限的质量范围。

证据保存在被忽略的 `docs/validation/qwen38-parallel/` 和
`artifacts/validation/qwen38-locality-{base,uncensored}-native/`。原生、成对 HTTP 和交付
图像运行的托管构建身份不同；这些阶段的最终原生库保持相同。原生速率不含 HTTP / prefill，
HTTP 速率包含 admission 与 prefill。模型文件超过物理内存，分页、进程 / 缓存状态和短时
测量均限制性能结论。本次结果不覆盖其他设备、CUDA / TP、更广泛的事实或视觉质量、长上下文
性能。图像工具比较拼接后的内容和请求哈希，未验证原始 SSE framing 或 completion ID。

**Prefill 的分块形状仍是独立的数值限制。** 调度器对单独一个请求用
一个大块 prefill（`TS_SCHED_SOLO_PREFILL_CHUNK` 与 `TS_SCHED_MAX_BATCHED_TOKENS` 中较小者），对并发
请求则按步预算分份，而本模型的 logits 可能依赖分块大小。历史 CUDA 检查在 UD-Q2_K_XL、三 GPU 按层切分上用
`benchmarks/ChunkParityProbe` 对一个 19,121 token 的 prompt 实测：重复同样的 4096 分块逐位一致
（max |Δlogit| 为 0）；1024 与 512 token 的分块让 logits 最多偏移 1.3，并在近似平局处翻转贪心解码——
最早在第 9 个输出 token，top-2 差值为 0.002（标题的第一个词）。保持分块形状相同就消除了这一效应：
一个 2,928 token 的 prompt，每个请求都一次 prefill 完（`TS_SCHED_PREFILL_CHUNK=4096`、
`TS_SCHED_MAX_BATCHED_TOKENS=16384`），4 路并发 × 3 轮的输出与单独运行逐字节一致（512 token，12/12）。
这些结果验证的是先前轮询路径的特定负载与配置，不能视为新 arena 或所有并发场景的验证。
CUDA 上不承诺 prefill 形状的宽度不变性，
所以 fixture 的分块与整段关卡给差异设上界，而不是要求逐位一致；见 [保留前缀复用](#保留前缀复用)。

## 保留前缀复用

`Qwen4ExpModel.RetainedCache.cs` 为 `qwen4exp` 提供与 Qwen 3.5、DeepSeek V4 路径相同的保留 holder 复用：

- 符合条件的已结束主缓存以**存活状态**登记在 radix 树中（`DeferPrimaryConversion`）。下一轮精确续接
  直接使用同一缓存，不分配替代缓存；树中的标记不增加保留状态字节。只有其他请求需要替换主缓存时，
  才尝试转换为 holder。转换被拒绝或失败时，新请求正常重新 prefill。
- 结束的会话的整个逐序列 holder 会被**保留**，并为恰好扩展它的下一轮重新设键。什么都不移动：以该
  holder 为键的原生状态条目、它的缓存图以及草稿头的私有 K/V 都留在原处。
- 所有聊天共享的 prompt 结尾处的状态会被**检查点**为以主机为准的深拷贝（注意力 K/V、QSA 原始 key
  与位置、GDN/PLE 递归状态、私有 MTP 状态），并**克隆**进每个新聊天。缺少权威原生状态时，克隆会
  拒绝执行，而不是拷贝陈旧的主机种子。
- 复用**仅限精确前缀**（`IExactFusedCacheReuse`）：新 prompt 没有逐 token 复现到最后一个的 holder
  不是它的延续，任何部分匹配都会重新 prefill。
- 精确复用可跨越相同媒体。holder 与检查点克隆保存 `MropeCacheGap`：分块结束后为
  `KV length - 1 - last T`，后续标量 token 的旋转位置为 `KV index - gap`。它等于 Qwen 3.5
  rotary delta 的负值，也保留视频偏移的符号。媒体内容、span 边界或会话 scope 改变时，不能使用该会话状态。
- 额外的保留 holder 与检查点共用预算 `TS_Q4E_RETAINED_CACHE_MB`，受实测内存余量限制；
  `0` 或无法解析的值拒绝这些额外状态，但不关闭现有存活主缓存的精确复用。未设置时，模型准入使用当前
  实测余量的一半，无法测得余量时才使用 4096 MB。radix 树在引擎创建时按空余内存的一半确定默认设备/
  状态上限，还按当前余量扣除运行请求的预留量检查；主缓存标记不作为额外 holder 再次计费。树负责驱逐，
  模型拒绝放不下的 holder，树可以释放较旧的 scoped 状态后重试。
- 主缓存转换先分配空替代缓存，再发布已移动的 holder。分配失败时，原有 KV 与递归状态保持完整，
  不发布半成品 holder。
- 替换前，树可测量主缓存转换后的 holder 大小，超过绝对配置或模型上限时直接拒绝；Qwen4Exp 也先检查
  保留预算，避免无效的替代分配。分配后的准入仍重新检查内存余量。测量测试要求估算与实际转换大小相同，
  并在不分配 tensor 的情况下拒绝无效长度与已借出的 holder。
- 它需要完整的 GGML token-span 路径（每一份逐序列状态都驻留在设备上并以 holder 为键），以及原生
  条目可以精确拷贝的 GDN 状态布局。保留在按层切分下可用；检查点在按层切分下被接受，在张量并行下
  被拒绝。

证据（合成 fixture，不代表训练模型的验收或性能）：
`Qwen4ExpRetainedCacheTests` / `Qwen4ExpRetainedCachePolicyTests` 覆盖保留 A/B/A、检查点克隆、投机重绑定、
预算拒绝后由 owner 释放、缺失状态拒绝以及 QSA 首次/重置增长。引擎测试比较精确图像续接与冷启动贪心输出，
并要求媒体或 scope 改变时复用为零。分配失败测试确认原生续接状态不变且不发布 holder；
`Qwen35MRopeReferencePositionTests` 还将 Qwen4Exp 分块位置、有符号 gap 与续接 decode 对照六组独立
SGLang fixture。这些夹具覆盖 QSA/GDN/PLE/MTP 状态，不评估训练得到的视觉编码器。
`DeferredPrimaryCacheTests` 覆盖无需替代分配的主缓存续接、实际替换、零额外保留预算与转换失败后的冷启动输出一致性。
真实双 GPU 按层切分测试仍受设备条件限制，在此次单 GPU 运行中跳过。在单卡 CUDA 上，一次 4 token 的目标验证与 4 次单 token 前向逐位相同
（`TeacherForcedTargetVerify_…`），以 2–4 为块提交的 32 个 teacher-forced token 在每一行上都与标量解码
逐位相同（`RepeatedTargetBlocks_…`）——见 [验证行使用单 token kernel](#验证行使用单-token-kernel)。
16 token 的 prefill 再接 4 个 token，在 CUDA 上与一次 20 token 的 prefill 并不逐位相同，因为 prefill 的
kernel 按批宽度选择：`SharedPrefixChunking_…` 在 CUDA 上把差异上界设为 1e-2（实测 logits 相差 1.7e-4 到
4.4e-4；它当初要抓的陈旧种子缺陷让 logits 偏移了 0.3155），并且只允许在 top-2 差值不超过实测差异两倍的
近似平局处改变贪心结果（2026-09-17 在 A40 上实测）。

## 共享 MTP 头的投机解码

图像请求也可使用学习得到的草稿头：调度器在每个投机 prefill 块之前排入对应的图像
embedding，并保留其 MRoPE 位置。prefill 后为重试保留的图像片段不再阻止投机 decode。
这仍要求单独请求从位置 0 开始 prefill；下述保留前缀与并发请求限制仍然适用。

2026-09-27 另以 UD-IQ1_S、两张 RTX PRO 4000 Blackwell、`--layer-split 2`、
上下文 1024、常驻共享 Q8_0 MTP 头及 BF16 视觉伴随文件验证。一次预热后，三轮实测的
文本/图像输出均与普通贪心逐 token 一致，MTP 与 ngram 都有实际起草。文本在 64 token
上限停止；图像回答在 204 个可见 token 后以 EOS 完成，数字与颜色描述正确。MTP 的逐轮
配对 decode 工作线程计算时间加速比中位数为文本 1.215 倍、图像 1.292 倍。图像请求计时
不包含同步图像准备与编码；这些短文本复制检查不代表通用质量或完整媒体请求延迟。
另外，普通/MTP HTTP 两种模式各通过 24/24 文本请求与 3/3 图像场景，包含附件顺序与
历史图像。该配置因余量不足拒绝保留缓存，因此后续轮次重新 prefill。本地证据：
`docs/validation/model-matrix-20260927/qwen38/SUMMARY.md`（不提交）。这批历史测试未验证跨节点执行；当前支持范围见[多节点指南](../multi-node-inference.md)。

`--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` 挂上逐 token 的 MTP 块（GGML 后端和 `--backend cuda` 均可；该头必须是单个 GGUF 文件，并在模型加载时挂上）；它只为从位置 0 开始
prefill 的单独请求做投机（与其他序列共享的步，以及延续保留 holder 或共享前缀克隆的轮次，都按普通
方式解码——草稿头有自己的 K/V，无法跨越它从未重放过的位置起草）。在 UD-Q2_K_XL、三 GPU 按层切分
（A40）上以 `--spec-draft 3` 实测：

- **一致性。** 192 token 的代码复制流与普通贪心完全一致，在当前的验证行 kernel 下速度为普通 decode 的
  1.69 倍（2026-09-17 下文改动之前为 1.75-1.96 倍；141/141 草稿被接受，无回滚）；投机 prefill（`SpecForward`）在相同分块下与普通 prefill 逐位一致。2026-09-17 之前，4 行
  verify 的舍入与 1 行 decode 步不同，可能把普通贪心写出的裸 JSON 对象变成 ```` ```json ```` 围栏
  回答；现在验证行使用单 token kernel（见下文）。
- **散文不划算。** 18 个散文请求的接受率为 68-70%（每次 verify 3.0 个 token），但一次 verify 约 45 ms，
  部分接受还要加约 43-46 ms 的回滚（恢复递归状态、重新前向保留的行），而被调速器暂停的步仍然要跑
  带隐藏状态捕获的单行投机前向（约 24-26 ms，普通 decode 为 19 ms）：512 token 散文在 c1 下 decode 为
  44-46 tok/s，普通为 52；8k prompt 之后为 34-38，普通为 40。

在 `ggml_metal` 上配合上面的专家层（48 GiB M5 Pro，UD-Q2_K_XL），MTP 头在代码与列表上有收益：4 行 verify
约 115 ms，普通一步约 50 ms（旧放置下约 197 ms，因为主机要计算 verify 约 3 倍的专家集合），接受率 81%（每次
verify 3.4 个 token）。在 `ggml_metal` 上这些行默认作为一次批量前向来验证（与 NInfer、llama.cpp、vLLM 相同）：
每次 verify 95 ms，代码上比逐行用单 token kernel 验证快 1.3 倍。此时贪心文本会与普通 decode 不同，并取决于
governor 选的 verify 宽度；`TS_Q4E_VERIFY_ROW_KERNELS=1` 恢复精确验证行（投机文本与普通贪心逐位一致，CPU 以及
默认情况下的 CUDA 都保留精确验证行；CUDA 上 `TS_Q4E_VERIFY_ROW_KERNELS=0` 改用批量验证）。
部分接受时不再恢复循环状态并重新前向保留的行（每次约 60 ms）：verify span 保留每一行的 GDN conv/SSM 状态
（上游 `ggml_gated_delta_net` 的逐行快照）与 PLE conv 历史，直接把被接受的行复制进实时状态（NInfer 的 ReplaySSM
思路；状态在主机可见内存中时用 `memcpy`，在 `ggml_cuda` 与 `ggml_vulkan` 上用一个设备端复制图，状态跨设备的
layer split 则仍走旧的回滚；`TS_Q4E_VERIFY_ROW_COMMIT=0` 在所有后端恢复旧的回滚）。使用精确验证行时，投机文本仍与普通贪心完全一致。

批量 verify 还会让窗口读取与 GPU 计算重叠，与 NInfer 的流式窗口同一思路：每行最多两个未命中窗口的专家交给后台
（第二个 I/O 线程组）读取，GPU 同时计算槽位里已有的路由，读到的专家在第二个切点之后计算；这个切点只等待后台读取，
某层没有延迟读取时直接跳过，不额外提交。4 行 verify 的主机时间从约 17–19 ms 降到 3–5 ms，GPU 时间增加约 9 ms
（多出的切点和第二组的乘积），单次 verify 从约 97–99 ms 降到约 94 ms：MTP decode 31.2 / 26.5 / 32.5 →
32.5 / 27.2 / 33.6 tok/s（代码 / 文章 / 列表，贪心文本不变）。单 token decode 与精确验证行仍只用一组。
`TS_MOE_TIER_STATS=1` 会统计后台读取次数。

### 验证行使用单 token kernel

一次 verify 以及对其接受前缀的重放，都会把 2–8 个 token 放进同一张 span 计算图（同样长度的 prefill 也是如此），
而在 CUDA 上 ggml 会按批宽度选择多个 kernel。在这些宽度下，一行的舍入方式与它对应的单 token decode 步并不
相同：F32 投影超过 3 列后离开 `mul_mat_vec_f`，改走 tensor-core / cuBLAS TF32 路径（router logits 偏移
2.6e-3）；BF16 的 QSA indexer 投影从 2 列起走半精度路径（5.9e-3）；路由专家换成多 token MoE kernel
（4.8e-7）；flash attention 换成多 query 启动（2.9e-5）。经过 48 层 MoE 与 QSA 路由，这已不是最后一位的
噪声：在三张 A40 上的 UD-Q2_K_XL 里，3,248 token prompt 之后 teacher-force 前 48 个贪心 token，2、3、4 行
verify 的每一行都与其 decode 步不同，logits 最多相差 2.5，48 行里有 4–6 行改变了贪心 token——并不只发生在
近似平局处。

因此在 CPU 与 CUDA 上，2–8 个 token 的 span 计算图用其单 token 图会运行的 kernel 构建每一行：浮点投影把 token 放在
广播轴上（CUDA 上一次 `mul_mat_vec_f` 启动）。只有实测过的 NVIDIA A40 上的 Q4_K、Q5_K、Q6_K、Q8_0 投影
按至多 4 行一块运行；其他设备和量化类型在广播轴上使用单列归约。Turing 与 GB10 在宽度 1 时的 MMVQ 归约
与 A40 不同，不能全局套用四行分组。路由专家与注意力
逐行展开，每个注意力行读取的 KV 窗口与 mask 行恰好就是它的 decode 步所读的那些。不超过 8 个 token 的图
（包括 decode）还会让两个是否融合取决于内存复用的 ggml-cuda 融合（MoE 加权归约；RMS norm + RoPE）的输入
保持分配，于是这两个融合在任何宽度下都会发生。单 token kernel 不变。CUDA 超过 8 token 的 prefill
会在顺序求和前物化专家加权输出，避免依赖内存分配的 FMA 融合使层切分与张量切分产生不同舍入。
CPU 与 Metal 保留现有的 prefill 图构建方式。

CPU 与 CUDA 将每轮草稿限制为 7 个 token；验证还包含待提交的 anchor，合计最多 8 行。
该硬上限同样约束显式 `--spec-draft` 和自定义 drafter，默认首选窗口仍为 3 个草稿 token。

补充测试现覆盖宽度 1–8 的每一行。macOS ARM CPU 也需要此构建方式：原路径在宽度 2、4 时，虽然保存的 GDN、PLE、
KV 状态相同，logits 仍与逐 token decode 不同。启用 CPU 路径后严格的 fixture 测试通过；测试耗时不视为性能基准。
下方耗时仅来自原 A40 测量，不能作为其他 GPU 架构或广播回退路径的性能结论。测试 hook 构建可在启动时设置
`TS_Q4E_TEST_MMVQ_CHANNELS=1`，在 A40 上验证广播回退路径。

在同一环境下交替重复三次实测：宽度 2、3、4 的每个验证行现在都与其 decode 步逐位相同（48 行中 0 行不同，
没有贪心翻转）。4 行 verify 耗时 32.4–33.1 ms，此前为 29.3–29.6 ms（+11%）；3 行 28.3–29.9 对 26.8–27.2（+8%）；
2 行 24.5–24.8 对 24.1–24.3（+2%）；decode 步（20.6–20.7 ms 对 20.6–21.1）与 3,248 token 的 prefill
（2,335–2,338 ms 对 2,324–2,359）不变。在 192 token 的代码复制流上端到端测量（每种 kernel 各 6 轮，所有输出
都与普通贪心一致），MTP 投机为 83.2 tok/s，此前为 86.5（相对普通 decode 从 1.84 倍变为 1.69 倍）；n-gram 投机为
73.8，此前为 79.5；普通 decode（49.1 对 47.0）与 prefill（830 对 804 tok/s）没有退化：精确性的代价由投机承担。

## 超出内存运行

UD-Q2_K_XL 文件共 78.9 GB，分三个分片：28.8 GB 的 n-gram（PLE）表、46.1 GB 的路由专家，以及约
4 GB 的其余部分。每个 token 只读这张表的 16 行，以及每层 512 个专家中的 10 个，因此 TensorSharp
能在放不下这个文件的机器上运行它，按 token 的需要从 SSD 读取这两部分。

- **n-gram 表从不上传，也从不拷贝。** 在 GGML 后端上，它留在 GGUF 内存映射里，带随机访问提示
  （`madvise(MADV_RANDOM)`），每个 token 的 16 行（每行 90 字节）按需并行收集。这正是 llama.cpp
  `--lazy-mode` 背后的思路，llama.cpp 对超过 4 GiB 的张量默认开启它。这类加载每次都会打印：
  `PLE n-gram table: 28.8 GB read on demand from the GGUF mapping, 16 rows a token (random-access advice).`
  直连 `cuda` 引擎同样从映射中读取这些行。它和 GGML 后端都会在加载后于后台把整张表预热进页缓存，
  但仅限主机能留住它的情况：Linux 上，可用空间取 `MemAvailable` 与进程内存 cgroup 余量中的较小者
  （容器的限额，而不是宿主机的）。大于内存的 Mac 会跳过预热，因为在那里读这张表只会挤掉专家。
  网络文件系统上预热最有用：geesefs 挂载上一行未缓存的读取需要 92–120 ms。
  在 GGML 后端上，同一个后台预热还会读取主机负责的全部路由专家（加速器分层之外的那些，按路由频率从高到低），
  以 cgroup 能缓存的量为限：独立 GPU 上分层之外的每条路由都由主机从映射计算，冷专家会在一次 decode 步骤里
  带来多次存储往返。`TS_HOST_WARM_WAIT=1` 让加载等待预热完成（用于基准测试）。
- **每层的路由专家都在主机接缝处调度，由 GPU 从每层的专家槽位计算。** 在 `ggml_metal` 上
  `--n-cpu-moe` 与 `--cpu-moe` 都未设置时，引擎自行规划放置（与 NInfer 的 Flash Next 专家缓存同一思路，
  落到统一内存上）：
  - *固定槽位* 在整次加载期间保存每层被路由最多的专家，依据 **路由画像** 选出：每个主机接缝统计它路由到的
    专家 id，模型释放时把计数合并进每个模型一个的文件（`~/.cache/tensorsharp/moe-routing/`，
    `TENSORSHARP_MOE_PROFILE_DIR` / `TENSORSHARP_MOE_PROFILE`）。预算按"每字节覆盖最多路由"在各层之间分配，
    默认等于旧方案按整层常驻的 14.4 GB（`TS_MOE_HOT_TIER_GB`）。
  - *窗口槽位*（显式 I/O 预算 `TS_HOST_MOE_IO_CACHE_GB` 的 7/8）按需填充：没有槽位的专家被路由到时，用 `pread`
    直接读进最久未用的槽位（Metal 共享内存，文件以 `F_NOCACHE` 打开）。在 M5 Pro 的 SSD 上，8-16 路并发读取
    可达 13-14 GB/s，而缺页只有 1-3.5 GB/s。
  - 主机只做 id 重映射和读取未命中的专家；GPU 在同一张图里用全 GPU span 自己的 kernel 计算全部 10 条路由。
    没分到槽位的路由（一步宽于窗口时）仍在主机 decode 内核上计算。

  ```
  [moe-offload] qwen4exp (tiered): all 48 layers' routed experts run at a host seam; 48 layers keep their most-routed experts on the accelerator (7659 experts, 14.4 GB of a 14.4 GB budget) covering 67.6% of profiled routes, plus 7466 window slots (14.0 GB) filled on demand.
  ```

  若尚无覆盖该模型的画像，本次加载的所有槽位都是窗口槽位（最近最少使用，没有固定集合），同时由接缝记录画像：
  实测 18.4 / 19.4 / 20.4 tok/s，与有画像的加载相差 3-5%；下一次加载起启用固定槽位。`TS_MOE_HOT_TIER=0` 恢复旧方案（整层常驻末尾若干层）。旧方案浪费内存：常驻的一层要占住全部 512 个
  专家，不论用到与否，还会挤占那些从 SSD 读取专家的层所需的页缓存。
- **Decode** 时，若一步的专家都有槽位，主机不做任何计算。溢出的路由在 TensorSharp 的主机 decode 内核上运行
  （ggml 的 CPU 点积，线程组每层只唤醒一次、算完即休眠；在 Apple 芯片上自旋的线程组会让层间的 GPU 段慢
  1.5-2 倍）。`TS_HOST_MOE_DECODE=0` 让这些路由改走 ggml 图路径。
- **Prefill** 宽于 64 个 token 时，每个分块把各层用到的专家流式送到 GPU，用 `pread` 直接读进 CPU 可见的暂存
  内存（`TS_HOST_MOE_STREAM_PREAD=0` 改回缺页读入映射再拷贝）。64 个 token 以内的分块使用槽位。
- **`--backend mlx` 会被直接拒绝**：MLX 没有稀疏注意力 indexer、hyper-connection、n-gram 表以及
  IQ2_XS/IQ3_XXS 专家的 kernel。请使用 `ggml_metal`。

以下在这台 Mac 上实测（ggml `353b63b`，未修改），对照同一台机器上的 llama.cpp `a868c3e3`。
llama.cpp 默认把模型全部卸载到 GPU，在这里会失败
（`Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`）；它的最佳配置是纯 CPU、
12 线程，n-gram 表由它的 lazy 模式按需读取。

| 真实文本：1,818 token 的 prompt，贪心生成 256 个 token | prefill tok/s | decode tok/s |
| --- | --- | --- |
| TensorSharp `ggml_metal`，自动规划的切分 | 109.4-115.5 | 12.8-12.9 |
| llama.cpp `-ngl 0 -t 12` | 20.4-22.4 | 13.00-13.24 |
| llama.cpp `-ngl 0 -t 12 --no-op-offload` | 28.8-31.6 | 11.94-12.48 |

| llama-bench 的方法：随机 token，pp512 / tg128，同一会话 | prefill tok/s | decode tok/s |
| --- | --- | --- |
| TensorSharp `ggml_metal`，自动规划的切分（15 层专家在 GPU 上） | 147.6 | 21.1 |
| TensorSharp `ggml_metal`，`--n-cpu-moe 32`（16 层在 GPU 上） | 153.4 | 21.2 |
| TensorSharp `ggml_metal`，`--n-cpu-moe 30`（18 层） | 183.9 | 20.6 |
| TensorSharp `ggml_metal`，`--n-cpu-moe 28`（20 层） | 171.9 | 20.3 |
| TensorSharp `ggml_metal`，`--n-cpu-moe 26`（22 层） | 80.7 | 19.0 |
| llama.cpp 纯 CPU，`-t 12 -nopo 1` | 50.07 | 22.55 |

GPU 上放 15 到 16 层时 decode 基本持平（21.1-21.2），超过 16 层后每多一层都更慢；到 22 层时留给主机层的
页缓存太小，prefill 随之崩落。同一天更早、并非并排测得的数据：自动规划的切分 137.3 / 19.2，llama.cpp
48.43 / 22.26（两者相隔一个半小时），`--cpu-moe`（GPU 上不放专家）142.3 / 17.5，18 线程的 `ggml_cpu`
46.1 / 16.6，llama.cpp `-ngl 28 -t 12` 36.64 / 20.06。

在真实文本上，TensorSharp 的 prefill 快 3.5-5 倍、decode 持平；在随机 token 上，prefill 约快 3 倍，
decode 落后 6%（21.1 对 22.55）。
在 Metal 上，decode 受限于 ggml-metal 每次 dispatch 的开销（每个 token 约 4,200 次 dispatch，整个 token 为
单张图时 29.4 ms），外加每个主机接缝约 0.19 ms。真实文本还要为不在页缓存里的专家付出缺页的代价，
因此速度取决于这台 Mac 同时在做什么：其他工作让 5 GB 内存处于压缩状态时，同样的运行 decode 为
10.3-11.3 tok/s。TensorAgent 在 48 GB 的 Mac 上提供这个文件
（[应用内实测](../../TensorAgent/README.md#the-macs-own-models)）。

### 从网络存储加载

大于进程内存一半的检查点过去会完全跳过并行的页缓存预热，加载读取的一切都经由串行缺页到达。在 RTX 5090
VM 上实测（GGUF 位于 geesefs（S3 FUSE）挂载，内存 cgroup 93 GB，`ggml_cuda`，有画像的分层）：加载耗时
**505 s**（F32 张量 43 s，稠密权重上传到 GPU 94 s，23.0 GB 的分层通过缺页填充 363 s，约 63 MB/s），
之后第一次回答又卡在冷的主机专家与 PLE 行上。现在：

- 整个检查点超出预算时，加载一次性读取的张量（路由专家以外的一切）仍并行预热：13 s 读完 3.7 GiB。
- 分层的专家在填充槽位前以 32 路并行大块读取：52 s 读完 31.2 GB（分层本身加上读取合并经过的相邻主机专家）。
- PLE 表与主机负责的专家在后台预热（约 125 s 读完 56 GiB）。
- 专家由主机负责时，kernel 预热只 prefill 64 个 token 而不是 2,048 个：2,048 个 token 的虚拟提示几乎会路由到
  每个专家，把 46 GB 的大部分都流式读一遍（在网络存储上要几分钟，之后第一个请求才能开始）。

模型约 70 s 即可就绪（`TS_HOST_WARM_WAIT=1` 时为 202 s）。所有读取都经由每个分片 32 个句柄的池，在读取任何
张量数据之前一次性打开：geesefs 每次 `open()` 都会丢弃该文件的全部页缓存（第二个描述符打开时，一个 64 MB
的范围从完全驻留变为 0，而映射仍然存在），因此自己打开句柄的读取者会把其他人读过的内容扔掉。出于同样的原因，
在一个进程从这类挂载提供服务时，不要从另一个进程（第二个服务器、`dd`、GGUF 查看工具）打开模型文件：那次
打开会丢掉正在服务的进程已经预热的缓存。

### 专家层，2026-11-06

同一台 Mac、同一文件，ggml `57e2c2ca`（未修改），真实文本：一个 CLI 进程里的四个请求（`--input-jsonl`：
一次预热，然后约 50 token 的代码 prompt、约 1,300 token 的文章 prompt、约 30 token 的列表 prompt，各贪心生成
256 个 token）。同一会话内交错重复三次取中位数；路由画像来自另一次 12 个 prompt 的校准运行。

| | decode tok/s，代码 / 文章 / 列表 | TTFT 秒，代码 / 文章 / 列表 |
| --- | --- | --- |
| 之前（整层常驻） | 13.80 / 13.87 / 15.59 | 3.14 / 13.58 / 2.78 |
| 专家层 + 显式读取 | 19.16 / 19.88 / 20.70 | 0.78 / 6.39 / 0.53 |
| 之前，MTP 头 | 14.67 / 11.98 / 16.66 | 5.31 / 13.61 / 3.27 |
| 专家层，MTP 头，精确验证行（`TS_Q4E_VERIFY_ROW_KERNELS=1`） | 23.47 / 19.01 / 24.60 | 0.77 / 7.39 / 0.57 |
| 专家层，MTP 头（批量 verify，默认） | 30.87 / 21.31 / 28.31 | 0.75 / 7.03 / 0.56 |
| `config/qwen3.8-flash-next.json`（即上一行，使用其采样设置） | 27.00 / 20.62 / 32.72 | 0.80 / 7.96 / 0.53 |

专家层方案用 GPU 的 F32 激活计算专家，而旧方案的 33 个主机层把激活量化为 Q8_K，所以贪心文本可能与之前不同
（此处是文章那个回答，从第 655 个字符起）；当每条路由都有槽位时，专家层与全 GPU span 逐位一致
（`TieredExperts_AnswerAsTheAccelerator_Metal`）。同一文本上的 teacher-forcing 困惑度见 2026-11-06 的报告。

单设备 `ggml_cuda` 也会按当前空闲显存、尚待绑定的浮点权重和缓存、驱动余量及 3 GiB 图工作区
自动规划专家放置。显式 `--n-cpu-moe` / `--cpu-moe` 优先；张量并行和层切分仍使用显式配置。

有覆盖所有层的路由画像时，这一规划使用与 `ggml_metal` 相同的固定槽位，而不是整层常驻：每层把被路由
最多的专家放在显存里（预算即层计划原本给整层的显存），每步其余的路由在主机 decode 内核上计算；独立
显存没有主机可写的窗口槽位。没有这样的画像时保留层计划，同时常驻层的计算图也会记录路由，因此保存的
画像覆盖所有层，之后的加载即可分层。RTX 5090（32 GB，UD-Q2_K_XL，12,245 个专家 / 23.0 GB 放入槽位，
覆盖画像中 85% 的路由，数据全部在内存中）上，普通 decode 实测 51.3 / 50.4 / 45.8 tok/s，层计划为
44.5 / 44.8 / 44.9（代码 / 文章 / 列表）；1,024 个 token 的 teacher-forcing 困惑度为 16.12，层计划为
16.22。起初 MTP 头在那里并不划算：CUDA 用单 token 内核逐行验证（4 行 verify 约 75 ms），部分接受时
需要重新前向，MTP 头还占用 2.8 GB 显存，使槽位降到 20.1 GB，开启后代码和文章分别为 42.2 与 34.2 tok/s。
`TS_MOE_HOT_TIER=0` 保留层计划。

#### CUDA 分层，2026-11-07

同一台 VM 与同一个文件，ggml 未修改，一个 CLI 进程中的四个请求（代码、约 1,300 token 的文章、列表，以及
一个 329 token 的代码编辑回答；其余为 256 个贪心 token）。每次运行都从同一个校准画像的独立副本开始
（`TENSORSHARP_MOE_PROFILE`）：每个进程退出时会把自己的路由合并进画像，共用一个画像的一串运行会把槽位
调到基准提示上（一串运行中，代码的普通 decode 从 42.8 升到 57.5 tok/s）。重复的配置之间相差不超过 1.5 tok/s。

| decode tok/s | 代码 | 文章 | 列表 | 编辑 | 与普通贪心的文本对比 |
| --- | --- | --- | --- | --- | --- |
| 普通 | 52.2 | 52.4 | 47.6 | 43.1 | - |
| 普通，`TS_HOST_MOE_OVERLAP=0` | 49.6 | 49.6 | 45.5 | 41.3 | 一致 |
| MTP，旧回滚（`TS_Q4E_VERIFY_ROW_COMMIT=0`） | 49.6 | 39.5 | 47.8 | 52.5 | 四个都不同 |
| MTP，精确验证行 + 行提交（默认） | 60.9 | 49.6 | 58.1 | 53.1 | 代码与编辑一致 |
| MTP，批量验证（`TS_Q4E_VERIFY_ROW_KERNELS=0`） | 64.3 | 55.0 | 57.9 | 54.8 | 按设计不精确 |
| 普通，自适应槽位（`TS_MOE_TIER_ADAPT=1`） | 57.1 | 61.4 | 58.8 | 53.3 | 取决于之前的输出 |
| MTP + 自适应槽位 | 62.6 | 56.9 | 60.7 | 57.5 | 取决于之前的输出 |

- **接缝重叠**（默认）：分层接缝先异步启动槽位中路由的图片段，主机同时计算其余路由。逐位一致。
- **设备端行提交**：被接受行的 GDN、conv 与 PLE 状态由一个 ggml 复制图写入，不再恢复状态并重新前向保留的行。
  旧回滚在 CUDA 分层加载上同样不精确。
- **自适应槽位**（需显式开启）：Strata 的自适应专家缓存。每层为每个专家维护衰减的路由计数，每 32 次接缝调用
  把最多两个明显更热的主机端专家复制进最冷的槽位。teacher-forcing 困惑度（2,048 个 token，这些文档的八个
  512 token 窗口）中，槽位承接了 889k 条路由而非 685k，困惑度为 22.90，固定槽位为 23.07。哪些路由在主机上运行
  因此取决于之前生成的内容，重复的请求不一定得到相同的贪心文本；这就是它默认关闭的原因。

可选的 CUDA 专家缓存借鉴 Strata 的紧凑量化槽位：只把被路由的专家保留在设备缓冲区，
使用 LRU 淘汰，保留全部选中专家和原有归约顺序。缓存保存原始 GGUF 字节，调用未修改的上游
ggml kernel。进程启动前设置 `TS_HOST_MOE_EXPERT_CACHE_MB`；默认 `0` 继续使用原有主机路径：

```powershell
$env:TS_HOST_MOE_EXPERT_CACHE_MB = '4096'
# 使用 --backend ggml_cuda 和通常的模型选项启动 CLI/server。
```

当自动规划无法放下全部专家、每一层均支持缓存且各层配额能容纳选中专家时，所有专家层都通过主机接缝
使用紧凑槽位；否则继续保留能放下的末尾完整层。
`TS_HOST_MOE_EXPERT_CACHE_LAYERS` 默认 48，按 Qwen3.8 的层数分配预算；小型合成模型需覆盖该值。
缓存支持无偏置、gate/up/down 分别量化的 SiLU 专家以及 1 至 8 行输入。
短 prefill 和目标验证块逐行重放同一标量图，保持 decode 数值和递归状态回滚。
其他形状、后端、不足的预算或不支持的布局继续走原有路径；权重映射失效和模型释放会清空设备槽位。
支持的接缝在 CUDA 缓冲区之间直接复制激活、路由权重和输出；主机 MoE 调试与 GPU 验证仍保留主机暂存接口。
`TS_HOST_MOE_EXPERT_CACHE_OUTPUT_BRIDGE=0` 可恢复输出暂存，
`TS_HOST_MOE_EXPERT_CACHE_BRIDGE=0` 可同时恢复输入与输出暂存。
可选 `TS_HOST_MOE_EXPERT_CACHE_PREFETCH=1` 在上传前并行触碰选中未命中专家的原始字节页，
不锁定或复制整份权重，仍允许操作系统淘汰这些页；冷/热负载测量前默认关闭。

预算计入图分配和保守工作区余量；CUDA 共享池和驱动分配仍需额外空闲显存。
已停用的 CUDA 捕获对象可能保留到上游空闲清理周期，因此该预留量并非进程总显存或卸载后立即释放的总量。
`TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1` 可输出槽位、命中、未命中和预留量。
槽位分配还要求实际空闲显存超过原生安全余量；请求的预算过大时，部分层可能继续使用 CPU 回退。
完整 logits A/B 检查见 `eng/validation/qwen4exp-expert-cache.py`，命令和计时范围见
`eng/validation/Qwen4ExpExpertCacheProbe/README.md`。该功能保持显式开启：合成性能取决于缓存容量，
这些夹具不能证明真实语言质量或与 Strata 的性能持平。

2026-10-02 在 i7-11800H、32 GiB 内存、16 GiB 显存的 RTX 3080 Laptop GPU
和 CUDA 12.6 上验证了真实 UD-IQ1_M 模型。三个分片均通过发布方 SHA-256 检查，
Hugging Face revision 为 `38bb39ee97821de2c9009abb7e93950eec396e66`；推理使用 NVMe SSD 副本。
TensorSharp 的 ggml `353b63b439f27ab2cc19dac97ab1681ba6d2d084` 保持未修改。
Strata `36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca` 使用其固定的 llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`、原始 GGUF 专家、官方专家 profile
和 8 GiB CPU 驻留预算。它将部分稠密投影转换为 BF16，因此不声明跨引擎 logits 逐位一致。

独立语义检查覆盖整数计算、信息提取、Python 函数和 1 至 20 的平方列表。
40 个独立进程均完整生成至 EOS，TensorSharp 与 Strata 的输出 token ID 全部一致。
上下文为 512、F16 KV、关闭思考、标量贪心，不使用 MTP 或后缀草稿。
Strata 的验证器容量设为二，但实际目标窗口均为一行且没有草稿。
TensorSharp 的 8192/9472 MiB 缓存完整最终词表 logits 逐字节一致，
并覆盖每个后续标量层；真实代码案例的自动 CUDA 规划也与显式主机放置一致。

88-token 平方列表以四种引擎/预算各先运行一次，共四轮独立进程。
下表为**中位数（范围）**；TensorSharp 使用 `TS_HOST_MOE_EXPERT_CACHE_MB=9472`
（9.25 GiB 上限）。

| 测量 | TensorSharp | Strata |
|---|---:|---:|
| 引擎报告的 decode tokens/s | 11.09（9.22–14.02） | 10.24（9.37–10.46） |
| 完整进程秒数 | 16.54（14.95–19.31） | 62.15（59.76–66.89） |
| 全设备采样显存峰值，MiB | 14832.5（14831–14842） | 15729（15719–15737） |
| 操作系统工作集峰值，GiB | 19.74（19.66–19.82） | 18.51（18.48–18.53） |

TensorSharp 统计首 token 之后的 87 次 forward，Strata 统计全部 88 次目标运行。
Strata 的 TTFT 包含驻留专家初始化，TensorSharp 不包含模型构建。
完整进程时间包含各引擎加载和输出序列化。未控制操作系统页缓存历史或时钟，
因此这些结果不是冷存储或长期驻留服务的持平证明。所有 prompt 都超过八行，
TensorSharp 使用原有 CPU 专家 prefill，不能把 prompt 耗时差归功于新缓存。
短数学/提取案例的 decode 仍慢于 Strata，大缓存代码案例则更快。
显存采样包含桌面且可能错过瞬时峰值；工作集包含映射页，平方案例中 TensorSharp
主机工作集较高，因此不能声称每一级内存都更少。

另一次 TensorSharp 热代码 A/B（一轮预热、三轮计时）中，直接输入/输出复制为
27.92 tokens/s，仅直接输入为 23.29，完整暂存为 21.97。
所有变体的完整输出 ID 和最终 logits 均一致。预取为 21.43，故继续默认关闭。
该 A/B 不把热 TensorSharp 与新启动 Strata 作比较。CPU 卸载与 CUDA 最终 logits
存在差异（代码案例相对 L2 为 0.09463），虽然 ID 相同，也不声明严格 CPU 数值持平或广泛语言质量。

最终原生二进制（`66e50ad3…`）的 11 个原生测试全部通过且无跳过。
Qwen/MoE 回归通过 CUDA 327 项（跳过 17）和 CPU 319 项（跳过 22）；修改的 Python 工具通过 49 项。
缺少的目标/头部夹具、Metal、QSA 显式开启和不可用的多 GPU 场景不计入通过；
真实 MTP 集成和 TP 类未执行。较广的历史验证脚本仍因 Windows/路径假设、缺失的九月证据
及归档哈希不匹配而失败，未修改模块的失败另行记录。这些检查不能证明真实视觉、长上下文、
困惑度、MTP 或多 GPU 的质量与性能。生成证据留在被忽略的
`docs/validation/qwen38-strata-trained-audit/`、`qwen38-trained-transfer-ab-final/`
及 `strata-qwen38/`；可复用工具和命令位于 `eng/validation/`。

在 CUDA 上，GPU 放不下这个文件时，`--n-cpu-moe` 起同样的作用。在一张 A40（46 GB）上把 12 层的专家
放在主机上，`ggml_cuda` 实测 600 / 30.2 tok/s（随机 token，pp512 / tg128），llama-bench 用
`-ncmoe 12` 为 466.14 / 18.84。

直接 `cuda` 引擎能运行 UD-Q2_K_XL 的 IQ2_XS 与 IQ3_XXS 专家：decode 用从 ggml 点积移植来的逐 token
kernel，仍以捕获的 CUDA 图运行；prefill 则把这两种布局解码进它的 tensor-core 与寄存器暂存
（register-staged）分组 kernel。在分组 kernel 接手之前，它对这个文件的 prefill 走最慢的回退路径，
约 500 tok/s。它也能运行 UD-Q4_K_XL：该文件 48 层路由 down 投影中有 43 层存为 Q5_1（另外 5 层为
Q8_0），gate/up 为 Q4_K/Q5_K。decode 像 ggml 的 `vec_dot_q5_1_q8_1` 一样用 dp4a 把 Q5_1 与 q8_1 激活
做点积（偏移项取自 int8 激活之和），prefill 则把它解码进同样的分组 kernel，全部以 ggml 的
`dequantize_row_q5_1` 为准。此前引擎在加载时拒绝
这个文件，而在 `--layer-split 2` 下两个设备各自的拒绝以未处理的 `AggregateException` 中止了 CLI；
现在它不能解码的专家布局会在任何上传之前被拒绝，只输出一行点名该张量的 `error: model load refused:`。
尚未在直接引擎上测量 UD-Q4_K_XL 的吞吐。没有 nvcc 的构建使用已提交的 PTX，其中已包含 Q5_1
解码器；若加载的 kernel 模块构建自早于该解码器的源码，引擎会拒绝 Q5_1 专家，而不是把它们当作 Q8_0 解码。
该引擎没有主机专家接缝，所以 `--n-cpu-moe` 在它上面只打印警告，专家仍留在 GPU 上。
直接 `cuda` 引擎使用 `--layer-split N` 分配多个 GPU，并拒绝 `--tp N`。
`ggml_cuda` 已支持对 UD-Q2_K_XL 使用 `--tp 2`，保留原始量化权重；支持的格式及设备/布局要求见
[多 GPU](#多-gpu)。

两张 A40、热态，同样的 1,818 token prompt 与 256 个贪心 token。TensorSharp 以
`TensorSharp.Server.Host` 加 `--no-multi-agent --no-skills` 运行，每个进程三个请求，每个请求的首行各不相同，
互不复用前缀；llama-server 关闭 `cache_prompt`，回答两个请求。

| 2x A40，`--layer-split 2`（llama.cpp `-ngl 99`） | prefill tok/s | decode tok/s |
| --- | --- | --- |
| TensorSharp `cuda` | 1,612-1,613 | 56.85-56.93 |
| TensorSharp `ggml_cuda` | 1,174-1,217 | 52.9-53.1 |
| llama.cpp | 752-963 | 59.04-59.86 |

kernel 刚重新构建后，直接引擎的第一个请求在驱动编译 PTX 期间 prefill 只有 335 tok/s；驱动会缓存
编译结果。随机 token（pp512 / tg128）下，直接引擎实测 1,222.0 / 61.0，`ggml_cuda` 为 410.9 / 43.2；
llama-bench 的双 GPU 运行只有 210.74 / 41.99，远低于同一台机器上 llama-server 的数字，因此应以真实
文本那张表为准。按那张表，TensorSharp 的 prefill 快 1.2-2.1 倍，decode 为 llama.cpp 的 88-90%
（`ggml_cuda`）与 95-96%（`cuda`）。

### 内存上限，2026-11-07

在 ISTA-DASLab 的 `Q2_0` 文件上与 Strata 对比时暴露了两个失败。这个文件的专家很小，分层计划会把
分给它的每一个字节都用满。

- **放不下的提示词 span。** span 的计算图随长度增长：4K token 的 span 需要 5.1 GB，在 32 GB 显卡上
  专家层占满之后，旁边 3 GiB 的预留装不下。过去 span 会把自己标记为不支持，而 QSA 没有其他路径，
  之后的每个请求都会失败。现在，若提示词 span 的计算图在任何计算发生之前分配失败，就把它拆成两半
  重新 prefill（递归，最小 64 token），普通提示词和 MTP 头的提示词都适用；能放下的长度会成为之后
  提示词 span 的上限，每次减半都会在日志中说明。verify 窗口、图片提示词、张量并行和层切分仍保持
  单个 span。
- **Metal 内存耗尽。** Metal 无法让一次分配失败后继续运行：内存不足的 command buffer 会让后端
  失效。分层计划原本按 Metal 的工作集（48 GB Mac 上为 40.2 GB）来定，而这只是上限，并不是开着
  编辑器和浏览器时 macOS 真正能给出的内存。现在放在加速器上的专家还必须同时满足：不超过 macOS 在
  不压缩任何内存的前提下能给出的量减去 max(7 GB, 内存的 15%)（Strata 的 arena 规则），并且锁定的
  内存总量不超过内存的 65%；窗口槽位只能用分层之后工作集剩下的部分；保留的 KV 也按同样的方式受限。
  在 48 GB Mac 上跑 `Q2_0` 时，这意味着 23-26 GB 的分层（覆盖 97.6-99% 的已记录路由），之前会让
  Metal 内存耗尽的长提示词场景，现在普通模式和 MTP 模式都能通过。

在同一个文件上并排比较（`eng/qwen38-engine-bench.py`，贪心解码，重复 3 次），除了命中缓存的
后续轮次外，Strata（github.com/Niko1221/Strata `146df3c4`）在各项上都更快：

| | 长回答 decode | 4K 提示词 TTFT | 16K 提示词 TTFT |
| --- | --- | --- | --- |
| RTX 5090，Strata（MTP） | 306 tok/s | 0.92 s | 2.75 s |
| RTX 5090，TensorSharp（MTP） | 74 tok/s | 4.17 s | 27.4 s |
| RTX 5090，TensorSharp `--backend cuda`，普通解码（当天稍后，见下文） | 98 tok/s | 4.24 s | 17.8 s |
| M5 Pro 48 GB，Strata（DFlash2） | 48 tok/s | 6.32 s | 24.4 s |
| M5 Pro 48 GB，TensorSharp（MTP） | 32 tok/s | 16.8 s | 65.3 s |

TensorSharp 在 5090 上的一个 decode token 耗时 11-13 ms：分层 span 中 48 个加速器分段占 8.6-10 ms
（每段一次提交、一次等待；开关 CUDA graph 的差别不到 1 ms），主机专家占 2.3-3.5 ms。Strata 把一个
token 作为一张捕获好的 CUDA graph 运行，在设备上判定命中，未命中的专家由 CPU 线程池就地并行计算。

### 单卡放不下专家时的直接 CUDA 引擎，2026-11-07

`--backend cuda` 把一个 decode 步骤作为一张捕获的 CUDA 计算图运行。它以前要求所有路由专家都在 VRAM 中；
UD-Q2_K_XL（42.9 GiB 专家）和 ISTA-DASLab 的 `Q2_0`（31.6 GiB）都放不进 32 GB 的显卡。现在它按 Strata 运行
该模型的方式规划**专家驻留**：

- 按路由画像跨所有层最热的专家占用 VRAM 槽位（RTX 5090 上 `Q2_0` 覆盖 98.4% 的画像路由，UD-Q2_K_XL 覆盖
  86.4%）。每个堆叠的专家权重带一张每个专家一个地址的表，专家内核按表读取。
- **主机专家**（Strata 的 doorbell）：其余专家一次性复制进锁页主机内存。在最多 8 行的 decode 步骤或验证窗口中，
  每层把输入和选择结果发布到主机映射内存；主机线程用 ggml 的 CPU 内核计算未命中的专家，同时 GPU 计算驻留的专家，
  一个 GPU 内核等待主机的求和结果并把它加上。步骤从不离开它的计算图，也没有专家经过总线传输。未命中专家的表项指向
  VRAM 中一个全零的专家，因此 GPU 那一份对它的贡献为零。有未命中的步骤与全驻留运行在最后几位上不同（主机像 ggml
  的 CPU 内核那样量化激活）。`TS_Q4E_HOST_EXPERTS=0` 改为把每个未命中专家经总线暂存。
- prompt 在步骤内用复制内核把未命中的专家暂存进 VRAM。
- 主机工作线程组使用两个自旋线程（流等待和 doorbell 服务线程）之外剩下的 CPU：在 VM 的 13.6 CPU 配额上，6 个线程
  为 65.5/45.9/58.8 tok/s，10 个为 69.0/48.0/63.5，16 个（超额订阅）为 63.4/45.1/58.0。`--cpu-moe-threads` 可覆盖。
- `TS_MOE_TIER_ADAPT=1`（需显式开启；开启后文本取决于历史）：主机统计每个步骤的选择，每 8 步把每层最多两个更热的
  主机专家与该层最冷的 VRAM 专家互换位置。槽位中的字节换到主机专家的打包位置，内存占用与规划一致。
- 在 CPU 和总线之间拆分每层的未命中（Strata 的 PCIe share）经过测量后已删除：主机线程数在 10 个以上时，它比把所有
  未命中都交给 CPU 更慢（每层 1 个未命中交给 CPU 时为 56.2/41.1/49.0），因为复制位于 GPU 的关键路径上，而 CPU 的工作
  与之重叠。Strata 报告该拆分只在 CPU 线程池很小时才有收益。

引擎现在也能读取 ISTA 的 `Q2_0` 文件：warp-per-row 内核中加入了 Q2_0、Q4_0 和 Q5_0 解码器；为 BF16 的超连接和 PLE
value 矩阵加入了使用 F32 激活的 BF16 内核；为输出行很少的稠密投影加入了 split-K（10240 -> 320 的超连接 down 投影之前
只用 80 个 warp 运行）。没有 split-K、Q2_0 仍按整块 64 值划分时，引擎运行 `Q2_0` 只有 55.9/43.1/56.9 tok/s，低于
ggml_cuda。

RTX 5090（13.6 CPU 配额），`requests.jsonl`（code / prose / list，各 256 token，贪心，无 MTP），固定路由画像，各运行一次：

| 文件、后端 | code | prose | list |
| --- | --- | --- | --- |
| UD-Q2_K_XL，ggml_cuda 分层 | 52 | 52 | 48 |
| UD-Q2_K_XL，cuda，仅暂存 | 52.0 | 38.7 | 45.7 |
| UD-Q2_K_XL，cuda，主机专家 | 70.6 | 47.9 | 64.5 |
| UD-Q2_K_XL，cuda，主机专家 + `TS_MOE_TIER_ADAPT=1` | 71.6 | 50.8 | 77.4 |
| `Q2_0`，ggml_cuda 分层 | 82.6 | 46.4 | 88.4 |
| `Q2_0`，cuda，主机专家 | 94.6 | 65.6 | 97.3 |
| `Q2_0`，cuda，主机专家 + `TS_Q4E_HC_Q8=1`（Strata 的 Q8_0 超连接读取） | 106.0 | 70.4 | 109.1 |

一个 `Q2_0` 步骤的 GPU 时间（`TS_Q4E_PERF=3`，未捕获，开启阶段计时）约为 11 ms（`TS_Q4E_HC_Q8=1` 时为 9.9 ms，其超连接阶段为
2.3 ms）：超连接 3.3 ms，路由专家 2.0，共享专家
1.2，GDN 投影 1.1，注意力输出 0.9，router 0.7，GDN 核心 0.7。它读取的权重按显卡带宽约需 2.3 ms；超连接和共享专家阶段
主要耗在内核边界上，这正是 Strata 融合内核领先的地方。

上面这些 `Q2_0` 数据其实是在**未捕获**状态下跑的。第 10 层共享专家的 down 投影是 IQ4_NL，输入宽度 640；decode 尺寸的稠密内核原先只在
256 的倍数宽度上接受 IQ4_NL，于是第一次捕获就失败，此后引擎每步都逐个发射内核（日志 `decode steps run uncaptured`）。现在对按块解码
的格式（Q4_0、Q5_0、Q8_0、IQ4_NL、MXFP4）只要求整 32 值块。捕获后，普通长生成 decode 从 98 提升到 103 tok/s：这一步受 GPU 限制，
而不是受内核发射限制。

#### `--backend cuda` 上的投机解码，2026-11-07

- **MTP 头。** `--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` 把共享 NextN 块挂到直接引擎上，全部使用主干的内核：
  - token 行嵌入到全部四路流；
  - 每路流拼接 `[rms(e) * enorm, rms(h_s) * hnorm_s]`，由两次跨步拷贝排布；
  - `eh_proj` 按 token x 流的行运行；
  - 接着是该块的注意力（每个序列有自己的 KV）和 MoE；
  - 最后是它自己的输出混合器，再接主干的 LM 头。

  草稿器读取的是输出混合器之前的四路主干流，与 GGML 后端一致。在合成夹具上，连续三步草稿的 logits 与原生 MTP 执行器的相对误差在
  0.5-0.8% 以内，argmax 相同。
- **验证窗口**最多 8 行，作为一个 ubatch 运行，返回每一行的 logits 和流；按（序列、QSA 阶段、行数）捕获为 CUDA graph。被接受的前缀
  通过在其上重放递推来提交，与对同样 token 的普通 decode 逐位一致。
- **一次融合发射**完成半层的超连接 scatter 和下一个混合器的第一遍，每步约少 95 次发射，逐位一致（`TS_Q4E_FUSE_HC=0` 恢复两次发射）。

RTX 5090，`Q2_0` 加 `TS_Q4E_HC_Q8=1`，与上面对比使用同一客户端：

| 配置 | 短对话 TTFT / decode | 长生成 decode | 4K TTFT | 16K TTFT | 缓存后续轮 TTFT |
| --- | --- | --- | --- | --- | --- |
| 普通，未捕获（修复前） | 0.11 s / 65 | 98.1 | 2.67 s | 11.4 s | 0.12 s |
| 普通，已捕获 | 0.10 s / 69 | 103 | 2.70 s | 11.4 s | 0.11 s |
| MTP，窗口 3，已捕获 | 0.13 s / 62 | 110（接受率 87%） | 3.36 s | 14.2 s | 0.13 s |
| Strata，MTP | 0.24 s / 114 | 306 | 0.92 s | 2.75 s | 0.12 s |

MTP 目前收益不大，因为 4 行的验证窗口需要 27-32 ms，约等于三次 decode；一步草稿 0.62 ms，融合的追赶加首步 0.88 ms。MTP 头的
2.9 GiB（Q8_0 专家）占用了专家槽位的显存，常驻路由从 97.6% 降到 95.0%，CPU 要计算的缺失专家约翻倍；而验证窗口的各行缺失的专家
是单行 decode 的四倍，GPU 在每一层都要等 CPU。同样的原因，挂上该头后 prompt 需要暂存更多专家，读取也更慢。

4 行验证窗口的 GPU 时间分布（`TS_Q4E_PERF=3`，未捕获，与 1 行对比）为 18.5-23.6 ms 对 10.9 ms：路由专家 5.8-10.7 ms 对 3.1 ms；
稠密投影和 LM 头随行数增长：注意力投影 1.8 对 0.33 ms，GDN 投影 1.9 对 1.1 ms，LM 头 1.1 对 0.3 ms，因为 decode 尺寸的稠密内核
对每个激活行都重读一遍权重行。把窗口缺失的专家改为经总线暂存、而不是由 CPU 计算（`TS_Q4E_HOST_EXPERT_ROWS=1`），窗口仍是 27-30 ms。
所以下一步是一次读取权重行即可服务多行的稠密内核，以及融合的专家内核，而不是 CPU 一侧。

更大的 prompt ubatch（`TS_Q4E_UBATCH=2048`，默认 512）让每个缺失专家每 2048 行而不是每 512 行暂存一次：4K prompt 从 2.70 s 降到
1.75 s，16K 从 11.4 s 降到 7.5 s（2,290 tok/s）。它的 scratch 从专家槽位中占去 1.2 GiB（常驻路由从 97.6% 降到 96.6%），普通 decode
从 103 降到 98 tok/s。Strata 读取同样的 prompt 用时 0.92 s 和 2.75 s。

#### 移植 Strata 的验证窗口，2026-11-08

Strata 在同一块 RTX 5090、同一个 `Q2_0` 文件上的自测（`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`）：验证窗口平均 2.4-2.75 行
（`--spec 4 --spec-min-p 0.5`），每个窗口 GPU 时间 7.2-7.7 ms，产出 1.9-2.3 个 token。99.8% 的路由专家命中其显存缓存（缓存每个窗口都在
自适应调整），因此每个窗口 CPU 只需计算约 16 个专家。已移植的内容（每项都相对它之前的改动测量）：

- **门铃跳过。** 若一层所选专家都有显存槽位，就不再通知主机：publish 内核在设备内存中记录这一点，wait 立即返回；以前每一层都要等主机回应，
  Nsight 跟踪中占 GPU 时间的 20%。结果逐位一致（`TS_Q4E_DOORBELL_SKIP=0` 让每层都通知）。
- **每个窗口只读一次稠密权重。** 以 warp 逐行计算的格式（Q2_0、Q4_K、Q5_K、IQ4_XS 等）的验证行改为所有行共用一个 block、每个输出行一个 warp：
  4 行窗口的 LM 头从 1.11 ms 降到 0.48 ms，与单行内核的求和逐位一致。
- **共享专家使用独立 stream。** 在 decode 步和验证窗口中，它与路由器、路由专家并行，在合并处汇合，都在捕获的 graph 之内；逐位一致。
- **短专家行。** 640 宽的专家 down 投影改为每行 8 个 lane，而不是一个 warp。
- **按文件中的 BF16 读取路由器**：字节数是 F32 展开的一半，求和相同。
- **更小的草稿专家。** `TS_Q4E_MTP_EXPERTS=q2_0` 在加载时重新量化 MTP 头的专家（Strata 的草稿层就是这样发布的）：2,550 MiB 降到 675 MiB，
  空出的空间留给目标模型的专家；对 GGML 后端同样适用。
- **该模型在所有后端上的逐 token 草稿门限为 0.5**（Strata 的 `--spec-min-p`）：头不确定的草稿要付出一整行验证的代价；`--spec-pmin` 仍可覆盖。
- **为验证窗口启用自适应驻留。** 门铃内核在设备上统计每层的专家选择；每 4 次前向（decode 步、验证窗口、批量步都算）最多做 8 次交换，按收益从大到小，
  把一层中最常被选的主机专家与最少被选的显存专家互换。以前自适应只在单 token 步之后运行，因此在 MTP 下从不运行；实际命中率约 88%，而路由
  profile 预测为 96.8%；自适应使缺失减半。自 2026-11-08 起默认开启（输出会依赖历史；`TS_MOE_TIER_ADAPT=0` 关闭），可用 `TS_MOE_TIER_ADAPT_EVERY` 与
  `TS_MOE_TIER_ADAPT_SWAPS` 调整。

RTX 5090，`Q2_0`，`TS_Q4E_HC_Q8=1`，MTP `--spec-draft 3`，同一客户端（长生成是连续三次约 1,190 token 的回答；自适应的运行会随学习而变快）：

| 配置 | 长生成 decode | 4K TTFT | 16K TTFT | 4 行验证 | 每个 4 行窗口交给 CPU 的专家 |
| --- | --- | --- | --- | --- | --- |
| 本轮之前的 MTP（已捕获） | 109-112 | 3.36 s | 14.2 s | 23.9 ms | |
| 上述除自适应外的全部改动，Q2_0 草稿专家 | 115-125 | 3.02 s | 12.7 s | 21.1 ms | 229 |
| ……再加 `TS_MOE_TIER_ADAPT=1` 和 0.5 门限 | 132-167 | 2.99 s | 12.4 s | 18.3 ms | 123 |
| 同上，32K 上下文（`MAX_CONTEXT=32768`） | 145-180 | 2.53 s | 10.6 s | 15.5 ms | 52 |
| Strata，MTP（131K 上下文，int8 KV，常驻 32K） | 306 | 0.92 s | 2.75 s | 2.5 行约 7.5 ms GPU | 约 16 |

32K 那一行说明 131K 上下文的 F16 注意力缓存要占去 2.9 GiB 专家槽位；Strata 的 KV 用 int8，并且只把最近 32K 个位置放在显存里。32K 下一个
4 行窗口剩下的时间：专家（含 CPU 部分）3-4 ms，GDN 投影 1.9 ms，注意力投影 1.8 ms，超连接 3.1 ms，路由器 1.2 ms。

在 Mac 上（M5 Pro 48 GB，`ggml_metal`，`Q2_0`，MTP，32K）草稿一侧的改动同样适用：

| 配置 | 长生成 decode | 4K decode | 16K decode | 接受率（代码 / 散文 / 列表） |
| --- | --- | --- | --- | --- |
| 门限 0.15，Q8_0 草稿专家（之前） | 25.4-26.8 | 25.3-27.5 | 24.7-26.3 | 88 / 66 / 70% |
| 门限 0.5，Q8_0 草稿专家 | 29.9-31.0 | 30.7-36.2 | 27.5-30.3 | 90 / 72 / 79% |
| 门限 0.5，Q4_0 草稿专家（`TS_Q4E_MTP_EXPERTS=q4_0`） | 29.3-30.4 | 32.0-37.5 | 29.0-34.4 | 90 / 72 / 81% |
| 门限 0.5，Q2_0 草稿专家 | 29.1-30.6 | 29.6-35.5 | 29.5-32.6 | 89 / 68 / 80% |

Metal 的专家分层位于统一内存中，按其预读窗口工作，因此上面的自适应只适用于 CUDA。

#### 直接引擎上的 Q8_0 KV 缓存，2026-11-08

`--backend cuda` 默认使用 `q8_0` KV 缓存，除非 `--kv-cache-dtype` 指定其他类型（`f16` 保持 F16）：12 个注意力层的 K、V 以 int8 编码保存，每 32 个值一个 F16 缩放（ggml 的 Q8_0
算法；编码与缩放分开存放，一个 lane 一次读入自己的 8 个编码），字节数是 F16 缓存的一半，规划器把省下的显存交给专家槽位。`q4_0` 会在
stderr 上说明后改用 F16。与 F16 缓存相比，合成测试模型的 logits 相对变化 3.7e-3 到 4.8e-3；Q8_0 缓存上的验证窗口与同一缓存上的普通
decode 完全一致。

同一会话，RTX 5090，`Q2_0`，`TS_Q4E_HC_Q8=1`，MTP（Q2_0 草稿专家），`TS_MOE_TIER_ADAPT=1`，131K 上下文：

| KV 缓存 | 显存中的专家（profile 路由覆盖） | 长生成 decode | 4K TTFT / decode | 16K TTFT / decode | 4 行验证 | 每窗口 CPU 专家 |
| --- | --- | --- | --- | --- | --- | --- |
| F16 | 15,901（96.9%） | 123-162 | 2.99 s / 87-96 | 12.5 s / 63-78 | 17.9 ms | 113 |
| `q8_0` | 17,085（98.0%） | 135-165 | 2.67 s / 98-104 | 11.1 s / 84-85 | 16.9 ms | 82 |

#### Prompt 分组、拆分注意力与更轻的 MTP 追赶，2026-11-08

在 RTX 5090 上对 4K prompt 做的 `TS_Q4E_PERF=3` 剖析显示了直接引擎的时间去向。每个 512 行 prompt ubatch 的 311 ms GPU
时间里，路由专家占 197 ms，几乎全部用于经总线暂存没有显存槽位的专家，而下一个 ubatch 又会把其中大部分再暂存一次。prompt
结束后，第一次起草要让整个 prompt 走一遍完整的 MTP 层（覆盖整个 prompt 的注意力和每一行的专家：4K prompt 后 59 ms，16K
prompt 共 320 ms，最后一部分计入回答的 decode 时间）。4 行验证窗口中 12 个注意力层在 4K 上下文耗时 2.4 ms，16K 时 4.9 ms：
每个头每行一个 block，逐个依次读取最多 2,051 个单元格。改动如下：

- **按层读取 prompt**（`TS_Q4E_PROMPT_GROUP`，默认 4096 行；单 GPU、启用专家驻留时）。每一层先跑完组内所有 ubatch 再进入下一层，
  并在这些 ubatch 之间保留已暂存的专家（`ts_dsv4_moe_stage_copy_from` 只复制某个 ubatch 新增的专家），所以缺失专家每层每组只经总线
  传输一次。logits 与按 ubatch 顺序逐位一致（`APromptReadLayerByLayer_AnswersBitForBitAsUbatchOrder`）。组的流共 160 MiB，占用
  专家槽位（驻留 16,954 个而不是 17,085 个，profile 路由覆盖 97.9% 而不是 98.0%）。
- **拆分注意力**（`TS_Q4E_ATTN_SPLITS`，默认 16）。decode 规模中超过 256 个单元格的行把它们分给 16 个 block，最后完成的 block
  合并各部分结果（flash decoding），全部在捕获的图内。更短的行整行计算，与之前逐位一致；更长的行只有浮点重结合带来的差异（测试
  上限为相对误差 1e-4），验证行仍然算出与各自 decode 步相同的结果。
- **只写 K/V 的 MTP 追赶**（`TS_Q4E_MTP_KV_ONLY`，默认开启）。追赶的行只写入 MTP 层的 K/V；层的其余部分只在最后一行起草时为它
  运行（Strata 的 `record_front` / `record_rest`）：4K prompt 后从 59 ms 降到 23 ms，16K prompt 共从 320 ms 降到 45 ms。
- **在设备上做贪心验证**（`TS_Q4E_GREEDY_VERIFY`，默认开启）。普通贪心采样器直接从捕获窗口中的 argmax 内核取得每个验证行的首个
  最大值，而不是让每行 1 MB 的 logits 经总线传到主机再扫描。token 完全相同。
- **设备上的草稿链**（`TS_Q4E_MTP_CHAIN`，默认开启；单 GPU、无采样惩罚）。一个窗口的草稿作为一条链运行：挑选内核取得每一步的首个
  最大值 token 及其 top-10 置信度并送入下一步，主机每个窗口只调用一次，也不读取草稿 logits。每个窗口的起草时间从 2.1-2.2 ms 降到
  1.9-2.1 ms；端到端数字仍在运行间噪声范围内（短对话 74-102，长生成 141-175，4K 108-131，16K 154-170 tok/s）。一个草稿步的大部分
  时间是 GPU 工作（覆盖 248K 个 token 的 LM head）。
- **Q2_0 的 prompt 行走张量核**（`TS_Q4E_Q2_0_PROMPT_GEMM`，默认开启）。ISTA 把部分注意力投影、多数共享专家和 PLE key 存为
  `Q2_0`，共享的 prompt 路径不能解码它们，所以这些 prompt 行原先走 warp-per-row 内核，每行都要重读一次权重（4K prompt 中占 0.23 s
  GPU 时间）。现在为本次调用把权重反量化为 F16 副本并使用 cuBLAS GEMM：4K TTFT 从 1.85 s 降到 1.69 s，16K 从 7.4-7.5 s 降到
  6.7-6.9 s，decode 不变。
- **自适应驻留从 prompt 学习**（`TS_MOE_TIER_ADAPT_PROMPT_SWAPS`，默认 64）。在设备上统计 prompt 的专家选择，按 128 行的分量计入，
  并在回答 decode 之前让它最常选中的主机专家最多交换 64 个进入显存。16K prompt 之后，每个窗口交给 CPU 的专家从 156-200 个降到
  64-136 个；4K prompt 之后基本不变。

RTX 5090，ISTA `Q2_0`，`TS_Q4E_HC_Q8=1`，MTP（Q2_0 草稿专家，`--spec-draft 3`），q8_0 KV，自适应驻留，131K 上下文，相同客户端和
场景，每项重复 3 次（TTFT / decode tok/s；每次运行均 15/15 PASS）：

| 运行 | 短对话 | 长生成 | 4K prompt | 16K prompt | 多轮首次 TTFT |
| --- | --- | --- | --- | --- | --- |
| 改动前（2026-11-08 上午） | 0.10 s / 62-80 | 130-167 | 2.65 s / 79-115 | 11.1 s / 74-85 | 1.75-1.87 s |
| prompt 分组、拆分注意力、只写 K/V 的追赶 | 0.10 s / 63-78 | 141-173 | 1.85 s / 89-129 | 7.5 s / 137-141 | 1.24-1.30 s |
| ……再加设备上的贪心验证 | 0.10 s / 69-74 | 142-179 | 1.78-1.86 s / 96-132 | 7.5 s / 132-153 | 1.27-1.30 s |
| ……再加从 prompt 学习的驻留（即现在的默认） | 0.08-0.10 s / 75-96 | 137-175 | 1.77-1.86 s / 112-137 | 7.3-7.5 s / 152-164 | 1.24-1.28 s |
| Strata（MTP `--spec 4`，int8 KV 且 32K 常驻，开启 route tail skip） | 0.24 s / 93-115 | 300-307 | 0.92 s / 376-408 | 2.75 s / 387-421 | 1.08-1.18 s |

剩余差距。长 prompt 之后，4 行窗口仍有 100-270 个专家交给 CPU，每个让窗口多花 40-50 us（有 20 个 CPU 专家时 4 行窗口耗时
12.7 ms，有 170-270 个时 20-22 ms）。Strata 每轮最多交换 96 个专家（这里 `TS_MOE_TIER_ADAPT_SWAPS=96` 让 4K 请求之后的 16K 请求
降到约 50 个，但每轮复制期间 GPU 都要停下等待），并跳过窗口每一行都以第 7 名或更低排名路由到的缺失专家（`STRATA_ROUTE_TAIL_SKIP=7`，
默认开启，会改变输出）。prompt 读取速度仍是 2,150-2,250 tok/s，而 Strata 为 4,360-5,950，且 Strata 的窗口多容纳第五个 token
（`--spec 4`）。

### ggml_metal 上的 DFlash2 草稿模型，2026-11-08

Strata 在 Mac 上用 DFlash2 起草：一种块草稿模型，一次前向最多提出 7 个 token。TensorSharp 现在在 GGML 后端上也能使用同一个草稿模型：
把 [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF) 的官方 GGUF 作为
`--draft-model` 传入即可；Strata 的 Mac 包用的也是同样的原始权重。

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --backend ggml_metal \
  --draft-model Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf
```

- **特征。** 草稿模型训练时读取第 5、15、25、35、45 层完成后的输出，每层在四路超连接流上取平均（每个位置 12,800 个浮点数）。token span
  在自己的计算图中算出这些均值，写入投机前向的特征行，因此验证、普通步和提示词分块都会填充草稿模型的环形缓存。草稿模型借用目标模型的
  token 嵌入和 LM 头。
- **默认值。** 每个窗口最多 6 个草稿，在草稿置信度的乘积低于 0.5 处截断（Strata 用 0.7；在这台 Mac 上 0.5 的代码与散文回答为
  47.3 / 30.8 tok/s，0.7 为 46.4 / 28.0）。`--spec-draft`、`--spec-pmin` 可覆盖。不支持 `--tp` 与分布式阶段。
- **提示词现在会进入草稿模型。** 以前块草稿模型在一次普通 prefill 之后才启用（这一捷径本是给不保存上下文的草稿模型准备的），所以它的
  环形缓存里只有回答自己的 token。现在提示词通过投机上下文 prefill（`ISpeculator.ConditionsOnPrompt`）：短回答 13 -> 18 tok/s，
  4K 提示词的回答 30-39 -> 47-51。所有 DFlash 目标模型（Qwen 3.5/3.8、Muse-Glimmer）都受益。
- **按宽度保存验证图。** 只跑一个 span 的 token 为每种宽度（1..8 行）各保存一张持久图；以前共用一张图，窗口宽度一变就重建。若某张图
  写入的行缓冲区已被更宽的窗口扩容，该图会重建而不是重放。
- **逐行状态原地写入。** 验证窗口的 GatedDeltaNet 输出（含每一行的状态）直接分配在提交时读取的缓冲区里，省去每层每行约 3 MB 的复制
  （Mac 上 4 行窗口省下 7 ms GPU 时间）。
- **提示词从分层槽位取专家。** 提示词宽度的批次会把被卸载层路由到的每个专家放进暂存张量。该层分层中已有的专家现在直接从内存中的槽位
  复制，而不是再从文件读取：4K TTFT 12.7 -> 8.9-10.2 s，16K 62-67 -> 47-48 s。字节相同，结果相同。

Mac M5 Pro 48 GB，ISTA `Q2_0`，`ggml_metal`，32K 上下文，`TS_Q4E_HC_Q8=1`，专家分层，与 Strata 对比时相同的客户端（3 次重复；长生成是
一个代码回答、一个散文回答和一次段落复制）：

| 配置 | 短对话 TTFT / decode | 长生成 decode（代码 / 散文回答） | 4K TTFT / decode | 16K TTFT / decode | 通过 |
| --- | --- | --- | --- | --- | --- |
| TensorSharp MTP，之前 | 0.43 s / 18 | 32.2 | 16.8 s / 38 | 65.3 s / 33 | 15/15 |
| TensorSharp DFlash2 Q4_K_M | 0.32-0.35 s / 15-18 | 32.8-33.4（50.1-52.7 / 32.8-33.5） | 8.9-10.2 s / 35-51 | 46.7-47.9 s / 37-45 | 15/15 |
| Strata DFlash2（`--spec 2`，专家流式读取） | 0.56 s / 19 | 47.8（53.7-55.2 / 47.6-48.0） | 6.32 s / 61 | 24.4 s / 61 | 15/15 |

测量后没有收益、已删除的尝试：把所有专家映射到 Metal（`--n-cpu-moe 0`；36-63 tok/s，但 34 GB 专家加上 4K 提示词的缓冲区让 48 GB 的
Mac 上 Metal 内存不足）；把稠密 IQ4_XS 改写为 IQ4_NL 以使用 ggml-metal 的多行内核；关闭投机调度器（governor）。剩下的差距在验证窗口本身：
4 行约 70 ms，其中 63 ms 在 GPU 上，而单行为 39 ms；Strata 每个约 3.6 行的步骤约 55 ms。

超过 4 行的窗口在 Metal 上作为一次批量前向验证（见“验证行使用单 token kernel”），所以 DFlash 的文本可能与普通贪心解码不同；
`TS_Q4E_VERIFY_ROW_KERNELS=1` 可保持完全一致。

#### TensorSharp 自有的 Metal 专家内核，2026-11-08

Mac 上 token span 的节点剖析（`TS_GGML_NODE_PROFILE=types`，所有专家映射到 GPU，扣除每节点同步）显示路由专家开销最大：ggml-metal 的
MUL_MAT_ID 为每个（行，路由）对单独读取权重，1 行一步 GPU 时间 6.9 ms（89 GB/s），4 行验证窗口 16.2 ms。现在分层的各层用 Strata 的分组专家
内核（MIT，原样移植到 `TensorSharp.GGML.Native/tsm/`）计算其槽位持有的路由：每个不同槽位的行对窗口里路由到它的所有行只读一次。seam
用它本来就要读取的路由构建分组计划，并在 ggml-metal 自己的命令队列上编码这些内核，因此它们按顺序在两个段之间执行，不需要额外的主机往返。
与 MUL_MAT_ID 相差 1e-6，与双精度参考相差 2e-7（`TsmMoeKernelTests`）。`TS_METAL_MOE=0` 保留 MUL_MAT_ID。

M5 Pro，ISTA `Q2_0`，分层：4 行窗口的专家每层 GPU 时间 0.2 ms（1 行 0.08 ms）；4 行验证一次 69.9 -> 59.8 ms，文本不变。同一客户端，
DFlash2 Q4_K_M，15/15：

| 配置 | 短对话 decode | 长生成 decode（代码 / 散文回答） | 4K TTFT / decode | 16K TTFT / decode |
| --- | --- | --- | --- | --- |
| DFlash2，MUL_MAT_ID | 15-18 | 32.8-33.4（50.1-52.7 / 32.8-33.5） | 8.9-10.2 s / 35-51 | 46.7-47.9 s / 37-45 |
| DFlash2，分组内核 | 15.6-19.0 | 34.8-36.1（53.8-57.0 / 34.9-36.2） | 9.1-10.3 s / 45-49 | 46-48 s / 29-45 |
| Strata DFlash2 | 19 | 47.8（53.7-55.2 / 47.6-48.0） | 6.32 s / 61 | 24.4 s / 61 |

这是 TensorSharp 自有 Metal decode 窗口的第一部分（Strata 的设计：一个窗口一个命令缓冲区，权重原地读取，每层主机的处理通过共享事件完成）；
span 的其余部分仍在 ggml-metal 上运行。

#### 共享事件上的 seam 与 GPU 保活，2026-11-08

分层的一次前向原来在 48 个段的每一段结束时提交并等待：主机醒来、读取路由、编码持有的路由和下一段并提交，期间 GPU 空闲。现在 Metal 上的 seam
像 Strata 的主机回合（`run_graph_seg`）一样通过共享事件交接，且不修改 ggml：各段提前两段排入 ggml-metal 自己的队列
（`ggml_backend_graph_compute_async`），每个 seam 是夹在两段之间的一个 TensorSharp 命令缓冲区：预处理输入（hazard tracking 让它排在写出路由的段之后）、
发信号、等待主机放行、按计划读取网格的分组专家内核，以及下一段读取其输出的合并。主机自旋等待信号，读取路由，计算没有槽位的路由，写入计划后放行 GPU。
内核、顺序和缓冲区都与提交并等待的 seam 相同，所以 logits 逐位一致（`TsmSeams_OnASharedEvent_AnswerAsSubmitAndWait_Metal`，覆盖全部路由被持有、
以及一半专家留给主机两种情况）。`TS_METAL_SEAM_EVENTS=0` 保留每个 seam 提交并等待。

一次握手仍是一个往返：M5 Pro 上安静的 GPU 约 190 us 的 GPU 时间。分层前向运行期间，第二个队列上的短自旋内核让 GPU 保持清醒（Strata 的保活）：
往返降到约 83 us，p90 从 880 降到 475 us。`TS_METAL_KEEPALIVE=0` 关闭它。`TS_METAL_SEAM_PROFILE=1` 按 GPU 时钟打印各段时间和 seam 时间
（其内核与等待主机）；`TS_HOST_MOE_TIMING=3` 现在还打印每次前向分层持有的路由、窗口读取和主机路由数。

M5 Pro，ISTA `Q2_0`，分层，每次前向（`TS_HOST_MOE_TIMING=3`）：

| 前向 | 提交并等待 | 事件 seam | + 保活 | Strata 窗口（`STRATA_METAL_PROFILE=1`） |
| --- | --- | --- | --- | --- |
| 1 行 | 37.7-38.9 ms | 37.0-38.3 ms | 32.8-33.9 ms | 28.1 ms |
| 4 行 | 59.0-64.5 ms | 56.7-62.1 ms | 52.8-59.0 ms | 45.7-46.2 ms |

1 行时 GPU 在 ggml 的段里花 21.9 ms，在 seam 里花 8.6-9.7 ms，其中专家内核 3.7 ms；主机回合 3.2-4.3 ms，主要是 9-13 次约 1.3 MB 的窗口读取。
4 行时主机回合 8-14 ms，有 26-47 次窗口读取以及最多 13 条窗口放不下的路由。普通解码 512 token 的回答 17.3-17.6 s，原来 20.0 s。
把一次窗口读取拆成 64 或 128 KB 的请求更慢（请求和唤醒更多），未保留。Strata 默认的 `STRATA_ROUTE_TAIL_SKIP=7` 会跳过窗口里每行都排在第 7 名
或之后的未命中专家；TensorSharp 计算每条路由。

同一客户端，DFlash2 Q4_K_M，15/15（`df2-seams`）：

| 配置 | 短对话 decode | 长生成 decode（代码 / 散文回答） | 4K TTFT / decode | 16K TTFT / decode |
| --- | --- | --- | --- | --- |
| DFlash2，分组内核 | 15.6-19.0 | 34.8-36.1（53.8-57.0 / 34.9-36.2） | 9.1-10.3 s / 45-49 | 46-48 s / 29-45 |
| DFlash2，事件 seam + 保活 | 18.8-19.8 | 37.8-39.5（56.6-59.6 / 37.9-39.6） | 8.9-10.0 s / 42-56 | 41.6-47.3 s / 38-48 |
| Strata DFlash2 | 19 | 47.8（53.7-55.2 / 47.6-48.0） | 6.32 s / 61 | 24.4 s / 61 |

短回答和代码现在与 Strata 持平或更快。散文在接受率相同的情况下仍落后（TensorSharp 440-478 个草稿接受 298-305 个，Strata 469-501 个接受
314-318 个）：TensorSharp 约 3.9 行的验证窗口约 58 ms，Strata 4 行窗口 46 ms；并且散文回答有 38-59 步草稿器不出草稿。

#### 长上下文 QSA 与专家窗口，2026-11-08

16K 提示之后，1 行前向在 ggml 各段的 GPU 时间为 30.0 ms，短上下文时为 21.9 ms。差值全部来自 QSA 层（逐项关闭计时）：每步重新池化每个块的
indexer key（3.8 ms）、按 cell 的选择（1.1 ms），以及在选择掩码下对窗口里每个 cell 做注意力。三项改动，都用上游 ggml 的算子：

- 每个 QSA 层在步之间保留池化 key（`Q4eSeqStateEntry::aux`），一步只池化其 cell 补全的块。文本的块成员由位置决定。回滚让块变为未完成，直到重新池化；
  原始 key 的其他写入者（上传、并发 decode arena）会让下一步重新池化全部块。每块的计算相同，所以 logits 逐位一致
  （`QsaPooledKeys_KeptBetweenSteps_AnswerAsPoolingEveryBlock_Cpu/_Metal`）。`TS_Q4E_QSA_POOL_CACHE=0` 每步池化全部块。
- 选择掩码由 `ggml_top_k_mask` 生成（一个算子替代对整个窗口的填充、散射和加法），decode 步与验证窗口只注意被选中的 cell
  （`ggml_flash_attn_ext_set_n_kv_max`）。`TS_Q4E_QSA_SPARSE_FA=2` 对提示也这样做；M5 Pro 上更慢。
- 统一内存上分层预算的 12% 分给窗口槽位而不是按 profile 排名的固定槽位（`TS_MOE_WINDOW_SHARE`）。窗口跟随当前文本，固定槽位跟随其他提示。

| 前向（M5 Pro，ISTA `Q2_0`，分层） | 之前 | 之后 |
| --- | --- | --- |
| 16K 提示后 1 行，ggml 各段（GPU） | 30.0 ms | 24.7 ms |
| 4 行，短上下文（窗口 0.3 -> 3.1 GB） | 52.8-59.0 ms（窗口读取 26-47 次） | 49.4-51.0 ms（14-19 次） |

同一客户端，DFlash2 Q4_K_M，15/15（`df2-qsa`）：

| 配置 | 短对话 decode | 长生成 decode（代码 / 散文回答） | 4K TTFT / decode | 16K TTFT / decode |
| --- | --- | --- | --- | --- |
| DFlash2，事件 seam + 保活 | 18.8-19.8 | 37.8-39.5（56.6-59.6 / 37.9-39.6） | 8.9-10.0 s / 42-56 | 41.6-47.3 s / 38-48 |
| + QSA 池化 key、稀疏 decode 注意力、窗口比例 | 22.6-24.8 | 42.1-43.4（63.1-69.7 / 42.2-43.5） | 8.9-9.6 s / 50-71 | 42.9-45.4 s / 54-66 |
| Strata DFlash2 | 19 | 47.8（53.7-55.2 / 47.6-48.0） | 6.32 s / 61 | 24.4 s / 61 |

短回答、代码和长上下文解码现在与 Strata 持平或更快；散文约落后 10%。剩下的是读取提示：一个 4096 token 的块在短上下文时计算 8.2 s（16K 时 9.9 s），
其中约 1 s 是在 GPU 计算之前暂存每层的全部专家（`TS_HOST_MOE_TIMING=1`）。

#### 读取提示：提前暂存专家、显式读取 PLE 行，2026-11-08

`TS_Q4E_PHASE=1` 拆分一个提示块的耗时。GPU 之前有两项主机开销：

- 每个卸载层在其 MoE 图运行前暂存所路由的专家（分层复制与文件读取），每层约 20 ms。提示几乎会路由到每个专家，所以现在在 GPU 运行前一个段时，
  在线程上把下一个卸载层的专家整体暂存，其 MoE 图直接绑定（`TS_HOST_MOE_STAGE_PREFETCH`）。关键路径上的暂存每 30 层从约 600 降到 24 ms。
  只用一个缓冲区，在上一层计算完成后复用：两个缓冲区在 48 GB Mac 上的 16K 提示让 Metal 内存不足。
- PLE n-gram 行（每 token 从 29 GB 表中取 16 行）原来通过映射缺页读取：一个 4096 token 块的 65K 行需要 1.1-3.1 s。现在提示规模的 gather
  在原生 I/O 线程组上每行一次 `pread`，每块 0.52-0.54 s，与 Strata 报告的速度相同（`TS_PLE_EXPLICIT_READ`）。

两者暂存的字节相同，所以 logits 逐位一致（`PromptExpertPrefetch_AnswersAsStagingEachLayer_Metal`、
`PleRows_ReadExplicitly_AnswerAsTheMapping_Metal`）。先按块选择的 QSA（先选约 515 个最佳块，再选其 cell）试过后放弃：没有可测的收益，并且会改变
批量 arena 一致性测试所固定的截断处的并列选择。

| 配置（同一客户端，DFlash2 Q4_K_M，3 次） | 4K TTFT | 16K TTFT |
| --- | --- | --- |
| QSA 池化 key（`df2-qsa`） | 8.9-9.6 s | 42.9-45.4 s |
| + 提前暂存专家、显式读取 PLE（`df2-pf1`） | 8.46-8.81 s | 38.0-38.3 s |
| Strata DFlash2 | 6.32 s | 24.4 s |

剩下的是块的计算，每 4096 token 约 7.5 s：路由专家（约 1.5 s）、ggml 的 gated delta net（约 0.9 s）、hyper-connection 与 GDN 块逐 token 的
逐元素与布局操作（约 3 s：[T, 10240] 转置、conv 输入拼接、q/k/v 拷贝、q/k 头重复），以及稠密投影（约 1.6 s）。Strata 用融合的提示内核完成这些
（`gdn_prompt.metal`：conv + 归一化 + 门控、按列并行的状态扫描、门控归一化）。

## 多 GPU

`ggml_cuda` 的 `--tp N` 切分每个路由专家及共享专家：gate/up 按中间通道切分，汇集激活后，
down 按输出行切分，再汇集输出供 hyper-connection 写回。每个 rank 保留全部专家 ID。
down 点积保持原始完整宽度，避免求和顺序的微小差异被后续激活量化放大。
注意力、GDN、QSA、PLE 使用复制的权重及各 rank 独立状态，
输出头仅在 rank 0 执行。图像与工具调用沿用同一目标图；投机回滚会恢复每个 rank
的 GDN/PLE 状态。TP 下仍不支持共享前缀检查点。`ggml_cuda` 下可使用
`--tp-node-id` 和 `--tp-peers` 扩展到多个节点，FFN 行切分依据全局 rank 数量。
分布式按层切分则传递完整的 F32 超连接残差，GDN、卷积、PLE 与 QSA 状态留在所属层。
分布式模式当前仅支持串行文本请求，不支持媒体、投机解码及保留序列快照。
两次汇集使用 TensorSharp CUDA FP32 collective；回退路径以零复制切片避开
上游 CUDA 自动 BF16 阈值，设备 collective 不可用时使用 FP32 主机归约。
两次通信增加数据量以保持数值精度，无需修改 ggml。

FFN 中间宽度和输出宽度必须能被并行度整除；投影按完整输出行切片，保留完整输入量化块。
量化 prefill 保留原始 MMQ tile 和归约几何，gate/up 切片保留重叠的 128 行边界 tile，再裁剪输出。
在中间宽度 640 的检查点上，TP2 每个 rank 为逻辑 320 行保存 384 行，TP4 为逻辑 160 行保存 256 行。
不支持的配置会在批量加载权重前依据 GGUF 元数据拒绝。当前模型路径要求 CUDA MMQ stream-K，
FFN 类型限于 Q5_1、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ1_S、IQ3_S、IQ4_XS、IQ4_NL 和 Q8_0。
UD-Q4_K_XL 的 Q5_1 down 权重使用 TensorSharp 自有的精确输出行切分内核。
这包含 UD-Q2_K_XL 的 IQ2_XS/IQ3_XXS 路由 gate/up 权重、Q5_K/Q6_K 共享权重和 IQ4_NL/Q8_0 down 权重。
完整输出行数及 down 输出切片须为 128 的倍数。IQ1_M 仍被拒绝，因为上游 ggml 没有该格式的 MMQ tile kernel。
其他 FFN 类型（包括 F32/F16/BF16）、设备或无法保持归约顺序的布局会明确拒绝。
下述历史物理 GPU 验证使用 NVIDIA A40，量化格式以各次运行的说明为准。
专家切片目前另占总路由专家字节数及重叠行的
主机缓冲区，并保留至模型释放。加载时不再预读整个稀疏 PLE 表，所需行按需读取。

`--layer-split N` 仍表示按完整层连续分配至多个 GPU，适用于 `ggml_cuda`、
`ggml_vulkan` 和直接 `cuda` 引擎。不得同时使用 `--tp` 与 `--layer-split`。
`eng/tests/qwen4exp-tensor-parallel.py` 验证两层 prefill、重放、QSA、多轴 RoPE、
全部 logits 和多 rank 状态回滚。量化模式
（`--quantized-ffn --tokens 1,2,3,4,5,6,7,8 --rollback-width 8`）在 CUDA TP2 与 TP4 上均通过全部 102 项检查，
包含验证宽度 8 的循环状态、QSA 和 PLE 快照恢复，hidden 与 logits 误差均为零。
CPU TP4 loopback 另行通过全部 42 项 F32 检查，仅用于正确性验证。
合成 F32 CUDA TP4 的宽度 17 重放超过原有误差门限（hidden 最大误差 3.49e-5）；
公共模型入口明确拒绝 F32 FFN，此场景不计为通过。
`eng/tests/qwen4exp-tp-quantized-ffn.py --hidden 2560` 另行验证真实 640 通道宽度、
IQ3_S/IQ4_NL 与 IQ4_XS/Q8_0 gate/down、Q8_0 共享专家及宽度 1 至 8、17、31、128；
CUDA TP2 在 `--require-bitwise` 下逐位一致。
`eng/ForcedLogitProbe` 在固定相同 token 历史下比较完整模型 logits，避免早期贪心
分歧掩盖后续 decode 的数值误差。

### UD-Q2_K_XL 的 TP2 验证，2026-10-05

提供的检查点现可在两张 NVIDIA A40 上使用 `--backend ggml_cuda --tp 2`，
保留原始量化字节及上游 MMQ 算术。部署的原生库 SHA-256 为
`a7d30f8ac0648372b8fd63881b9a2c915032b75ee2e51299265d61349d793301`；
上游 ggml 保持未修改，revision 为
`ffa4e8b80930029a35991f94e7c8a93cd67730ab`。
该 VM 的 CUDA peer access 行为探测失败，F32 FFN 汇集使用 NCCL 共享内存传输。

- 托管 TP 测试全部 39 项通过，无失败或跳过。
- 原生 strip 检查通过 410 项 CPU 比较和 820 项 CUDA 比较，覆盖全部 14 种支持格式，
  每张 A40 各 410 项。CUDA gate/up 输入宽度为 2560；token 宽度包含
  1、4、8、9、17、31 和 129。
  其中 524 项使用自有 MMQ strip 路径，296 项使用小形状下选择的上游 CUDA 路径。
- 26 项完整合成 FFN 比较均逐位一致，hidden 宽度 2560、FFN 宽度 640、16 个专家且选择 10 个。
  覆盖 IQ2_XS/IQ4_NL 路由 gate/down 与 Q5_K/Q8_0 共享权重组合，以及
  IQ3_XXS/IQ4_NL 与 Q6_K/Q8_0；宽度为 1 至 9、17、31、128 和 129。
- 5 个完整模型固定历史案例，每例 24 行，共 29,798,400 个 F32 logits，
  与原始 layer-split-2 二进制逐字节一致。5 个贪心回答通过算术、提取、Python 行为、
  有序平方数和思考检查；回答也与候选构建的 layer-split-2 逐字节一致。
- 精确使用 `--interactive --think --max-tokens 20000`，配合本模型及
  `--backend ggml_cuda --tp 2`，完成两轮并以 EOS 结束，回答分别为 `703`、`720`。
  实际生成 163、76 个 token，报告速率为 46.1、45.4 token/s；第二轮复用 256 个提示 token 中的 232 个。

启用 CUDA 图的数值检查通过。针对 Q5_K、使用 `GGML_CUDA_DISABLE_GRAPHS=1` 的
memcheck 报告零错误；启用图的 sanitizer 尝试报告已处理的 CUDA 图更新 API 状态 910，
不计为干净的图捕获 sanitizer 验证。这两轮短交互不验证完整的 20,000-token 生成；
本轮未评估长上下文、视觉、困惑度、TP4 或其他设备。生成证据保存在被忽略的
`docs/validation/qwen38-q2-tp/`；可复用回答检查工具为
`eng/validation/validate-qwen4exp-tp-answers.py`，请求夹具为
`InferenceWeb.Tests/Fixtures/Qwen4Exp/tp-quality-requests.jsonl`。

四次新 CLI 启动按 layer2、TP2、TP2、layer2 的顺序比较，使用 F16 KV、上下文 4096、
两个 CPU 线程及固定输入 pp512/tg128。每次均正常预热内核，再计时四轮。
吞吐单元格依次为**首轮 / 第 2–4 轮中位数 / 全部四轮范围**，单位 token/s；加载不含预热。

| 模式 / 启动序号 | 加载（秒） | 预热（秒） | pp512：首轮 / 中位数 / 范围 | tg128：首轮 / 中位数 / 范围 |
| --- | ---: | ---: | --- | --- |
| layer2 / 1 | 20.49 | 138.68 | 942.0 / 998.6 / 942.0–1010.6 | 53.6 / 53.9 / 53.6–54.1 |
| TP2 / 2 | 46.56 | 2.47 | 806.8 / 882.2 / 806.8–884.6 | 49.4 / 51.5 / 49.4–52.2 |
| TP2 / 3 | 48.42 | 2.43 | 797.2 / 824.8 / 797.2–841.8 | 48.5 / 50.3 / 48.5–51.4 |
| layer2 / 4 | 19.51 | 138.08 | 942.2 / 986.2 / 942.2–998.1 | 52.8 / 53.4 / 52.8–53.6 |

合并每种模式的六个后续轮次，中位数为 TP2 **848.0 pp / 51.4 tg**，
layer2 **997.4 pp / 53.65 tg** token/s；该负载下 TP2 的 prefill 慢 15.0%，decode 慢 4.2%。
在此 VM 上追求本轮实测最高吞吐时，使用 `--layer-split 2`。
TP2 需要额外计算重叠边界行、复制注意力，并在每层 FFN 做两次 F32 汇集；本轮未单独隔离各项开销。
两种模式在不同启动阶段上传权重，评估启动耗时时需同时看加载和预热。

四次不计时的完整贪心序列（prefill 加 128 个 decode token）均一致，运行二进制哈希及
干净上游身份在前后保持不变。计时使用重复的 17-token 输入周期，可能受益于 PLE 行和专家局部性；
未清除 OS 缓存，GPU 时钟未锁定。结果不含自然对话格式化、HTTP、调度或采样，
不代表真实文本、长上下文或冷存储吞吐。完整日志、计时和身份快照保存在被忽略的
`docs/validation/qwen38-q2-tp/benchmark-20261006T010554Z-21317/` 目录。

### 先前的 UD-IQ4_XS 验证

UD-IQ4_XS 检查点在 NVIDIA A40 上，TP2、TP4 各有 120 行完整词表 logits 与稳定舍入后的普通 layer2 执行逐字节一致：
三个文本 prompt、一个单 token 合成 prompt、一个 128 token 合成 prefill，每例固定历史运行 24 步。
TP2 两种路径均使用 native `da25f156`；TP4 使用 native `1ba6d7a4`，与保存的 `da25f156` 普通执行参照比较。
下述最终 HTTP 与投机检查使用 native `473ee64d`。
所用 ggml 为未修改的 `353b63b439f27ab2cc19dac97ab1681ba6d2d084`。
CUDA prefill 舍入稳定化可能改变旧二进制的 logits 或低 margin 贪心选择；旧参照向量单独保留，不宣称与旧版逐位兼容。
最终 HTTP 对比中，与旧二进制的首个 logits 向量相对 L2 误差为 0.0416；16 个文本结果有 14 个完全相同，
另两个仅有标点差异。该历史数值对比不通过严格一致性门限。

最终 CUDA TP2 HTTP 测试与同版本 layer2 执行在全部 16 个文本 prompt、4 个工具调用往返和 4 个图像回答轮次上一致，
首个真实 prefill 的 248,320 个 logits 逐字节相同。学习型 MTP 和 n-gram 投机在文本与图像测试中
均保持全部 96 个普通贪心 token 一致。压力配置为 `TS_SPEC_DRAFT=7 TS_SPEC_PMIN=0`；
MTP 在两个场景中均实际达到验证宽度 8，并覆盖拒绝回滚。所有进程正常退出，运行前后检查确认二进制未改变。

另一次六进程启动基准使用 2× NVIDIA A40、UD-IQ4_XS、上下文 4096、F16 KV、128 token 内核预热及两个 CPU 线程。
下表按执行顺序保留每种模式的两次启动。每个进程对固定输入的 pp512/tg128 计时五轮，
另以不计时的完整 128 token 贪心序列检查正确性。吞吐单元格依次为
**首轮 / 第 2–5 轮中位数 / 全部五轮范围**，单位 token/s；加载时间不含内核预热。

| 模式 / 启动序号 | 加载（秒） | 预热（秒） | pp512：首轮 / 中位数 / 范围 | tg128：首轮 / 中位数 / 范围 |
|---|---:|---:|---|---|
| 旧版 layer2 / 1 | 64.85 | 20.67 | 259.2 / 406.60 / 259.2–439.7 | 24.9 / 29.45 / 24.9–36.6 |
| 当前 layer2 / 1 | 48.98 | 21.51 | 262.7 / 427.75 / 262.7–446.5 | 30.3 / 37.90 / 29.2–38.0 |
| 当前 TP2 / 1 | 85.68 | 10.78 | 229.0 / 345.75 / 229.0–358.9 | 22.0 / 24.10 / 18.2–25.9 |
| 旧版 layer2 / 2 | 68.74 | 22.14 | 255.9 / 419.25 / 234.0–454.1 | 29.2 / 30.40 / 29.2–35.7 |
| 当前 TP2 / 2 | 80.22 | 12.56 | 230.3 / 349.15 / 213.9–356.7 | 18.6 / 27.30 / 14.2–34.3 |
| 当前 layer2 / 2 | 25.66 | 18.13 | 258.3 / 414.65 / 234.9–422.1 | 30.3 / 33.85 / 30.2–37.8 |

六个进程均正常退出，并通过运行时文件身份检查。当前 layer2 与 TP2 的两次启动均生成完全相同的完整贪心序列。
汇总稳态吞吐为 layer2 **421.20 / 35.875**，TP2 **347.45 / 25.70** token/s：
此机器上 TP 的 **prefill 慢 17.5%，decode 慢 28.4%**。Attention 与循环状态在各 rank 复制，
每层两次精确 F32 FFN 集合通信增加了这些 PCIe GPU 的通信开销（`NCCL_P2P_DISABLE=1`）。
本次实测按层切分更快。

旧二进制的合成贪心序列从 decode 下标 50（从零计数）起与全部当前运行产生分歧，
严格兼容性对比仍然失败；当前与旧版的吞吐比值未通过 token 一致性资格检查。
这些加载是未清除页缓存的混合/热缓存测量，GPU 时钟仅记录、未锁定。
明显的计时和加载波动不支持冷存储或普遍加速的结论。

另有一次相同二进制和设置的 TP2 诊断，仅改为
`GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_BF16_THRESHOLD=0`。
完整贪心序列及运行时检查通过，但性能取舍不一致：pp512
**216.3 / 309.15 / 202.7–321.4**，tg128 **16.5 / 30.30 / 16.5–40.1**
（首轮 / 稳态中位数 / 全部五轮范围），加载 100.69 秒、预热 10.05 秒。
这一次启动的 prefill 更慢，不足以支持更改默认 NCCL 传输。

下列历史性能数据仅对应按层切分。

实测：2× A100-80GB，Qwen3.8-Flash-Next-UD-Q2_K_XL（73.4 GiB）：

- 1 卡与 2 卡运行的贪心输出**逐字节一致**（SHA-256 相同）。
- 显存 24.2 GB + 26.2 GB——大约每张卡各放半个模型，而不是一张卡放下全部。
- 吞吐不变：两种情况下 prefill 都在 ~1520–1550 t/s，decode 都在 ~56 t/s。
  作为参照，同一台机器上的 llama.cpp：1 张 GPU pp1536 1094 / tg128 61.2；
  2 张 GPU `-sm layer` 1200 / 61.5——也就是说 llama.cpp 从第二张卡上同样只拿到
  约 10% 的 prefill 提升、decode 基本为 0。

启动时会打印实际走的是哪种模式，以及每张 GPU 分到的层数 / 字节数。
`TS_Q4E_LAYER_SPLIT=20,28` 可以用显式的每卡层数覆盖自动均衡（精神上等同于
llama.cpp 的 `--tensor-split`），并且在无法满足给定值时直接抛异常，而不是悄悄忽略
——这很有用，因为自动均衡只按权重计价，看不见视觉塔，而视觉塔加载得更晚、会落在
GPU 0 上。

## 基准矩阵

[`benchmark_config_glm53_qwen38.json`](../../benchmarks/engine_comparison/benchmark_config_glm53_qwen38.json)
以 `qwen38-flash-next` 的名字把本模型注册到固定的 Hugging Face revision 上，并挂上
它的 `mmproj-BF16.gguf`，好让 `image` 场景能跑。其中两条事实值得在这里重复。

一是已发布的 Q8_0 分片里**完全没有** `nextn` / `mtp` 张量，因此 `mtp_supported`
为 false，`--mtp on` 的格子会带着理由被跳过，而不是悄悄按普通解码跑掉。

二是**本模型只能跑在会传 `--layer-split N` 的那一列上**。原因就在上一节：切分度来自 `--layer-split`，
所以在不传 `--layer-split` 的后端列上，TensorSharp 只会建单设备上下文，175.3 GiB 会全部压到
一张卡上。因此配置里给了它 `min_tp`（4，仅按权重算出的下限——8 才是这台 8×A40 机器
应当使用的度数），在不传 `--layer-split` 的那一列上，这些格子会被记为
`needs --tp 4 (does not fit 1 GPU(s))` 的跳过，而不是留给它去 OOM。这里的 `--tp` 是
基准工具保留的 GPU 数量选择参数；所选后端列向 TensorSharp 传入的是 `--layer-split`。跑法：

```
python run_matrix.py --config benchmark_config_glm53_qwen38.json \
    --models qwen38-flash-next --backends ggml_cuda_split
```

那一列会让 llama.cpp 用 `--split-mode layer` 切在同样这些 GPU 上，于是参照列两边是
同一种整层放置方式。这里的历史结果只验证层切分，不代表上文新增张量并行的性能。
