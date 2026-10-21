# Qwen 3.8 Flash Next（`qwen4exp`）

[← 返回模型索引](README_zh-cn.md) | [English](qwen38-flash-next.md)

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
`docs/validation/model-matrix-20260927/qwen38/SUMMARY.md`（不提交）。该架构现支持下文所述的本地张量并行；跨节点执行仍不支持。

`--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` 挂上逐 token 的 MTP 块（仅限 GGML 后端；该头必须是单个 GGUF 文件，并在模型加载时挂上）；它只为从位置 0 开始
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
  直连 `cuda` 引擎同样从映射中读取这些行，并在加载后把整张表预热进页缓存。
- **最前面若干层的路由专家在主机上运行，同样直接读这份映射。** 在 GGML 的 GPU 后端上由
  `--n-cpu-moe N` / `--cpu-moe` 选择这些层（在 `ggml_metal` 与 `ggml_cuda` 上实测）。两者都未设置时，
  `ggml_metal` 与 `ggml_cuda` 会自行规划切分；`ggml_cuda` 按每张 GPU 规划，按层切分时也是如此（见
  [`ggml_cuda` 上的专家放置](#ggml_cuda-上的专家放置)）。在 `ggml_metal` 上，引擎在既放得进 Metal
  工作集、又给主机层留够页缓存所需内存的前提下，尽量把整层专家留在 GPU 上，其余卸载到主机。在 48 GB
  的 M5 Pro 上（51.5 GB 内存、40.2 GB 的 Metal 工作集）：

  ```
  [moe-offload] qwen4exp (planned): routed experts of 33 of 48 layers run on the host from the GGUF mapping (31.7 GB read on demand); the accelerator holds 15 layers' (14.4 GB). Metal working set 40.2 GB, RAM 51.5 GB; --n-cpu-moe N overrides.
  ```

  常驻的层数超过一定程度后，再多常驻也不会更快：常驻的一层要占住全部 512 个专家，不论用到与否，
  还会挤占那些从 SSD 读取专家的层所需的页缓存。
- **Decode** 在 TensorSharp 自己的内核上运行每个主机层的 10 个专家：用的是 ggml 的 CPU 点积，
  线程组每层只唤醒一次，该层算完即休眠。在 Apple 芯片上，自旋的线程组让层间的 GPU 段慢了
  1.5-2 倍，所以线程组改为休眠。`TS_HOST_MOE_DECODE=0` 恢复 ggml 图路径。
- **Prefill** 达到 128 个 token 及以上时（`TS_HOST_MOE_DEVICE_MIN_BATCH`），每个分块把各主机层用到的
  专家流式送到 GPU，并先用 16 个线程把这些专家的页面缺页读入：在 M5 Pro 上，这一步让 1,818 token 的
  prefill 从 37.2-38.0 秒缩短到 15.7-16.6 秒，decode 不受影响。
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

`ggml_cuda` 也会按每张 GPU 的空闲显存规划专家放置，单卡与按层切分（`--layer-split N`）都适用。
显式 `--n-cpu-moe` / `--cpu-moe` 会覆盖该规划；张量并行运行不卸载专家。规划方式、启动输出以及
哪些情况会拒绝加载，见 [`ggml_cuda` 上的专家放置](#ggml_cuda-上的专家放置)。

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

在 CUDA 上，GPU 放不下这个文件时，同样的卸载起同样的作用：`ggml_cuda` 会自行规划，`--n-cpu-moe N`
可将其固定。在一张 A40（46 GB）上把 12 层的专家放在主机上，`ggml_cuda` 实测 600 / 30.2 tok/s（随机
token，pp512 / tg128），llama-bench 用 `-ncmoe 12` 为 466.14 / 18.84。

直接 `cuda` 引擎能运行 UD-Q2_K_XL 的 IQ2_XS 与 IQ3_XXS 专家：decode 用从 ggml 点积移植来的逐 token
kernel，仍以捕获的 CUDA 图运行；prefill 则把这两种布局解码进它的 tensor-core 与寄存器暂存
（register-staged）分组 kernel。在分组 kernel 接手之前，它对这个文件的 prefill 走最慢的回退路径，
约 500 tok/s。该引擎没有主机专家接缝，所以 `--n-cpu-moe` 在它上面只打印警告，专家仍留在 GPU 上。
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

### `ggml_cuda` 上的专家放置

未指定 `--n-cpu-moe` 与 `--cpu-moe` 时，`ggml_cuda` 会自行决定哪些层的路由专家在主机上运行，单卡与
按层切分（`--layer-split N`）都适用；`--tp N` 下不卸载任何专家。按层切分过去把每个路由专家都按常驻
GPU 计价，也没有任何机制按 GPU 卸载专家，因此两张 20 GB 的 RTX 3080 各被要求放下 27-38 GB，加载在
kernel 预热时失败。

- **单卡。** 规划以量化稠密权重上传后的空闲显存为起点，扣除尚待绑定的浮点权重与缓存、驱动余量
  （`TS_VRAM_HEADROOM_MB`；默认取 512 MiB 与显卡容量 1/16 中的较大者），以及按 span 宽度和主机路由
  层数计算的 prefill span 工作区。过去固定预留 3 GiB，而 4,096 token 的 span 加 40 个主机路由层就会
  超出，把 16 GB 的显卡推入 WDDM 换页。
- **按层切分。** 在上传任何权重之前，引擎测量每张 GPU 的空闲显存（同样扣除余量），并同时选定连续的
  层段与主机路由层，让尽可能多的层把专家留在 GPU 上。在每张 GPU 的层段内，靠前的层把专家路由到主机
  （从 GGUF 映射按需读取），靠后的层把专家留在 GPU 上，与 llama.cpp `--n-cpu-moe` 的顺序相同。稠密
  权重上传之后，会再次测量每张 GPU 并重新拟合它的卸载。启动时会打印规划，例如：

  ```
  Layer split across 2 GPUs (sized to free VRAM): gpu0=layers 0-23 (10 with experts on the GPU, 14 on the host), ...
  [moe-offload] qwen4exp (planned per GPU): routed experts of 28 of 48 layers run on the host from the GGUF mapping (...); gpu0 holds the experts of 10 of layers 0-23 (...), ...
  ```

  按层切分增加的是容量而不是速度：每个 token 都要依次经过各张 GPU；专家放在主机上时，decode 速度
  取决于主机内存为这些主机路由层提供的带宽。

覆盖方式，以及哪些情况会拒绝加载（退出码 2）：

- `--n-cpu-moe N` / `--cpu-moe`（`TS_N_CPU_MOE` / `TS_CPU_MOE`）固定主机层集合，即前 N 层；
  `--n-cpu-moe 0` 表示一层都不卸载。按层切分时，层段会围绕这一集合排布，主机路由层的专家不再计入
  其所在 GPU，放不下的放置会被拒绝，消息例如 "Re-run with --n-cpu-moe N, or omit the flag to let
  TensorSharp place the experts"。
- `TS_Q4E_LAYER_SPLIT=a,b` 固定层段（每张 GPU 的层数），每张 GPU 仍会卸载自己靠前的层；若某张 GPU
  即使把全部专家放到主机也放不下它的层段，则拒绝加载。
- 若即使全部专家都在主机上，切分仍放不下，加载会被拒绝，并列出每张 GPU 所需与空闲的显存。可以降低
  `MAX_CONTEXT` 或 `TS_Q4E_PREFILL_CHUNK`、释放显存，或增加 GPU。

**Prefill span。** `TS_Q4E_PREFILL_CHUNK`（token 数，至少 128）是 qwen4exp 在不使用 `--tp` 的
`ggml_cuda` 上运行的最宽 prefill span；更长的提示词分块按连续的多个 span 运行。所有路由专家都留在
GPU 上时默认 4096，只要有一层路由到主机就默认 2048。每个主机路由层在每个 span 都要把自己的整套专家
流式送到 GPU 一次：span 越窄，所需工作区越小，能常驻的层越多（decode 更快）；span 越宽，流式传输的
开销分摊得越开（长提示词 prefill 更快）。span 所读取的 KV 超过 16,384 行后，span 还会自动变窄，使长
上下文不超出预留的工作区。图像与投机解码的前向同样按 span 切分：每个 span 取用自己那几行的图像嵌入与
位置表，并写入自己那几行的草稿模型隐藏状态与 logits。取值不是不小于 128 的整数时，加载会报错终止。有专家被卸载时，`[moe-offload]` 那一行会写明当前使用的宽度。span 边界就是 prefill 分块边界，
因此[连续批处理](#连续批处理)末尾关于 prefill 分块形状的数值限制同样适用。

**上下文。** 放置完成后，若各 GPU 已没有空间让 KV 缓存继续增长，加载时会限制上下文长度：
`[moe-offload] qwen4exp: context capped at N tokens (was M): GPU d has no room to grow the KV cache further after this placement. ...`
设置 `MAX_CONTEXT` 会预先分配整个窗口，规划也会为它预留空间。

**视觉投影器。** 通过 `--mmproj` 指定的投影器（以及 CLI 在给出 `--image` 或 `--video`
时在模型旁边找到的配套投影器）会计入 GPU 0 的规划：它的视觉塔以 F32 权重形式驻留，另需约 512 MiB
工作区。因此多模态运行会相应多卸载一些专家，而不是之后让 GPU 0 显存耗尽。

**预热。** 预热会构建规划所允许的最宽 span，因此预留不足会在第一个请求之前暴露，而不是在请求中途失败。
在自动放置下，若预热时显存不足（其他进程占用了显存，或驱动所需超出规划），该 GPU 会把更多层的专家
移到主机（它仍持有的层的四分之一，至少两层）并重新预热，最多四次：
`[moe-offload] qwen4exp: GPU d ran out of memory during warmup; routing the experts of layers a-b to the host as well ...`。
若这样也无济于事（显式指定了 `--n-cpu-moe` / `--cpu-moe`，或全部专家已在主机上），CLI 与服务端都会以
一行信息拒绝加载（退出码 2），而不是带着堆栈失败，或启动一个无法应答的模型。这一行会写明当前放置以及
相应的调整办法：调大 `TS_VRAM_HEADROOM_MB`（在每张 GPU 上多留空闲，从而把更多专家移到主机）、调大
`--n-cpu-moe`，或调低 `TS_Q4E_PREFILL_CHUNK`、`MAX_CONTEXT`。

**并发请求。** 在放得很紧的切分上，每多一个并发请求，就要在每张 GPU 上为它准备独立的 KV 与递归状态
（16K 上下文时合计约 0.5 GiB，分布在各张 GPU 上）；单独一个请求会复用预热时放置的状态。若要支持多个
并发会话，请降低 `MAX_CONTEXT` 或卸载更多层。

**接缝缓冲区。** 对 64 token 及以上的 span，主机路由层的 MoE 交接缓冲区会在层与层之间复用，而不是
在整个前向期间一直占用。在 16 GB 的 RTX 3080 Laptop、40 个主机路由层上实测：span 图缓冲区（UD-IQ1_M，
`MAX_CONTEXT=8192`，用 `TS_GGML_LOG_VRAM=1` 读取）在 2,048 token 时从 2,164 MB 降到 572 MB，在
4,096 token 时从 4,705 MB 降到 1,561 MB；生成文本逐字节一致；一次 6,544 token 的 prefill 因显卡不再
换页而从 116 秒降到 55 秒。这只是单张显卡上的结果，不是吞吐对比。

要在一张卡上验证按层切分及其按 GPU 的卸载，可以使用上游 ggml-cuda 的 `GGML_CUDA_DEVICES=2`：它模拟
两张各报告一半显存的 GPU。这只是模拟，不是性能测量。

## 多 GPU

`ggml_cuda` 的 `--tp N` 切分每个路由专家及共享专家：gate/up 按中间通道切分，汇集激活后，
down 按输出行切分，再汇集输出供 hyper-connection 写回。每个 rank 保留全部专家 ID。
down 点积保持原始完整宽度，避免求和顺序的微小差异被后续激活量化放大。
注意力、GDN、QSA、PLE 使用复制的权重及各 rank 独立状态，
输出头仅在 rank 0 执行。图像与工具调用沿用同一目标图；投机回滚会恢复每个 rank
的 GDN/PLE 状态。TP 下仍不支持共享前缀检查点，也不支持跨节点执行。
两次汇集使用 TensorSharp CUDA FP32 collective；回退路径以零复制切片避开
上游 CUDA 自动 BF16 阈值，设备 collective 不可用时使用 FP32 主机归约。
两次通信增加数据量以保持数值精度，无需修改 ggml。

FFN 中间宽度和输出宽度必须能被并行度整除；投影按完整输出行切片，保留完整输入量化块。
量化 prefill 保留原始 MMQ tile 和归约几何，gate/up 切片保留重叠的 128 行边界 tile，再裁剪输出。
在中间宽度 640 的检查点上，TP2 每个 rank 为逻辑 320 行保存 384 行，TP4 为逻辑 160 行保存 256 行。
不支持的配置会在批量加载权重前依据 GGUF 元数据拒绝。当前模型路径要求 CUDA MMQ stream-K，
FFN 类型限于 Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ1_S、IQ3_S、IQ4_XS、IQ4_NL 和 Q8_0。
这包含 UD-Q2_K_XL 的 IQ2_XS/IQ3_XXS 路由 gate/up 权重、Q5_K/Q6_K 共享权重和 IQ4_NL/Q8_0 down 权重。
完整输出行数及 down 输出切片须为 128 的倍数。IQ1_M 仍被拒绝，因为上游 ggml 没有该格式的 MMQ tile kernel。
其他 FFN 类型（包括 F32/F16/BF16）、设备或无法保持归约顺序的布局会明确拒绝。
下述历史物理 GPU 验证使用 NVIDIA A40，量化格式以各次运行的说明为准。
专家切片目前另占总路由专家字节数及重叠行的
主机缓冲区，并保留至模型释放。加载时不再预读整个稀疏 PLE 表，所需行按需读取。

**容量。** `--tp N` 下所有路由专家都留在 GPU 上并切成 N 份，同时每张 GPU 还各自持有注意力、循环与 PLE
权重、输出头以及全部缓存的副本；这种模式下不能卸载专家。无法这样放下的检查点会在加载时、把专家切分到主机
缓冲区之前被拒绝，并列出每张 GPU 的需求与空闲显存（退出码 2）。97 GB 的 IQ4_XS 检查点（三个分片）约含 61 GiB 路由
专家，因此在 2 张 20 GB GPU 上每张需要 30 GiB 以上：这种情况下请使用 `--layer-split 2`，它会把放得下的
部分留在 GPU 上，其余专家放在系统内存中运行。在这样的主机上，带专家卸载的张量并行本来也不会更快：在两张
GPU 上复制注意力与缓存会比按层切分留给专家的空间更少，而 decode 速度无论哪种方式都取决于驻留主机的专家。

`--layer-split N` 仍表示按完整层连续分配至多个 GPU，适用于 `ggml_cuda`、
`ggml_vulkan` 和直接 `cuda` 引擎。在 `ggml_cuda` 上，层段与路由专家卸载会按每张 GPU 的空闲显存一并
确定（见 [`ggml_cuda` 上的专家放置](#ggml_cuda-上的专家放置)）；在 `ggml_vulkan` 上，层段按权重字节数
均衡。不得同时使用 `--tp` 与 `--layer-split`。
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

启动时会打印实际走的是哪种模式，以及每张 GPU 的划分：在 `ggml_cuda` 上是每张 GPU 的层段、其中多少层
把专家留在 GPU 上，以及规划用量与空闲显存的对比；在 `ggml_vulkan` 上是每张 GPU 分到的层数 / 字节数。
`TS_Q4E_LAYER_SPLIT=20,28` 可以用显式的每卡层数覆盖自动划分（精神上等同于
llama.cpp 的 `--tensor-split`），并且在无法满足给定值时直接抛异常，而不是悄悄忽略。
在 `ggml_cuda` 上，每张 GPU 仍会卸载自己靠前那些层的专家；某张 GPU 即使把全部专家放到主机也放不下
它的层段时会拒绝加载；规划也已把 `--mmproj` 指定的投影器计入 GPU 0。在 `ggml_vulkan` 上，自动均衡
只按权重计价，看不见视觉塔，而视觉塔加载得更晚、会落在 GPU 0 上，因此要靠这个覆盖值为它留出空间。

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
