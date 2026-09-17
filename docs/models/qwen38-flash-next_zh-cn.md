# Qwen 3.8 Flash Next（`qwen4exp`）

[← 返回模型索引](README_zh-cn.md) | [English](qwen38-flash-next.md)

Qwen3.8-Flash-Next 是一个混合型 MoE：GatedDeltaNet 递归层与全注意力层交错
（其中一部分全注意力层挂在 Qwen Sparse Attention 的 indexer 后面），再加上一个
PLE n-gram 嵌入块、×4 hyper-connection 流以及 512 专家的 MoE。GGUF 架构 id 是
`qwen4exp`。权重：
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
（每个量化档一个子目录、均为多分片；`--model` 指向 `-00001-of-` 那一片；把
`mmproj-BF16.gguf` 放在模型旁边即可启用图像输入）。

## TensorSharp 如何运行它

在 GGML 后端上，整个 token（几乎）只跑一张图——嵌入、PLE（在图内）、全部 48 层、
最后的 mixer 以及 LM head——并配一个按形状索引的已捕获图缓存
（`TS_Q4E_TOKEN_GRAPH=0` 回退到逐层融合 kernel，后者再逐算子回退）。视觉沿用
Qwen3.5-VL 塔，位置用 (T,H,W) IMRoPE；支持多图与多轮图像会话，并在轮次之间复用
KV（GDN 递归无法回退，因此只有当新 prompt **恰好扩展**已缓存前缀时才复用；见[保留前缀复用](#保留前缀复用)）。

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

## 连续批处理

并发请求通过**逐序列状态持有者**（per-sequence state holders）来服务：每个在飞请求
各自拥有自己的注意力 KV 与 QSA indexer 缓存、GDN 卷积与 delta-net 状态、PLE 卷积
历史与 n-gram 窗口，以及固定下来的 kernel 描述符。原生 kernel 用持有者的 host 种子
指针作为设备驻留递归状态的键，用描述符地址作为已缓存图的键，所以切换请求只是一次
引用交换——不需要状态下载 / 上传，也不需要重建图——每个序列都在自己那张已捕获的
单图融合 decode 上解码。引擎按步在各序列间轮询（`SupportsPerSequenceFusedForward`）；
融合的 N 路批量 decode 属于后续优化。

**并发时的贪心输出可能与单独运行不同，原因在于 prefill 的分块形状。** 调度器对单独一个请求用
一个大块 prefill（`TS_SCHED_SOLO_PREFILL_CHUNK` 与 `TS_SCHED_MAX_BATCHED_TOKENS` 中较小者），对并发
请求则按步预算分份，而本模型的 logits 依赖分块大小。在 UD-Q2_K_XL、三 GPU 按层切分上用
`benchmarks/ChunkParityProbe` 对一个 19,121 token 的 prompt 实测：重复同样的 4096 分块逐位一致
（max |Δlogit| 为 0）；1024 与 512 token 的分块让 logits 最多偏移 1.3，并在近似平局处翻转贪心解码——
最早在第 9 个输出 token，top-2 差值为 0.002（标题的第一个词）。保持分块形状相同就消除了这一效应：
一个 2,928 token 的 prompt，每个请求都一次 prefill 完（`TS_SCHED_PREFILL_CHUNK=4096`、
`TS_SCHED_MAX_BATCHED_TOKENS=16384`），4 路并发 × 3 轮的输出与单独运行逐字节一致（512 token，12/12），
所以逐序列 holder 之间没有状态泄漏，轮询 decode 也不依赖并发度。CUDA 上不承诺 prefill 形状的宽度不变性，
所以 fixture 的分块与整段关卡给差异设上界，而不是要求逐位一致；见 [保留前缀复用](#保留前缀复用)。

## 保留前缀复用

`Qwen4ExpModel.RetainedCache.cs` 为 `qwen4exp` 提供与 Qwen 3.5、DeepSeek V4 路径相同的保留 holder 复用：

- 结束的会话的整个逐序列 holder 会被**保留**，并为恰好扩展它的下一轮重新设键。什么都不移动：以该
  holder 为键的原生状态条目、它的已捕获图以及草稿头的私有 K/V 都留在原处。
- 所有聊天共享的 prompt 结尾处的状态会被**检查点**为以主机为准的深拷贝（注意力 K/V、QSA 原始 key
  与位置、GDN/PLE 递归状态、私有 MTP 状态），并**克隆**进每个新聊天。缺少权威原生状态时，克隆会
  拒绝执行，而不是拷贝陈旧的主机种子。
- 复用**仅限精确前缀**（`IExactFusedCacheReuse`）：新 prompt 没有逐 token 复现到最后一个的 holder
  不是它的延续，任何部分匹配都会重新 prefill。
- 保留的会话与检查点共用一个预算 `TS_Q4E_RETAINED_CACHE_MB`（默认 4096，并受实测内存余量限制；
  `0` 或无法解析的值会拒绝所有保留），先驱逐最早保留的会话。`TS_Q4E_RETAINED_CACHE=0` 关闭该功能。
- 它需要完整的 GGML token-span 路径（每一份逐序列状态都驻留在设备上并以 holder 为键），以及原生
  条目可以精确拷贝的 GDN 状态布局。保留在按层切分下可用；检查点在按层切分下被接受，在张量并行下
  被拒绝。

证据（合成 fixture，不代表训练模型的验收或性能）：
[`eng/validation/qwen38_mtp_followup/retained-cache-20260916`](../../eng/validation/qwen38_mtp_followup/retained-cache-20260916/README.md)
——`Qwen4ExpRetainedCacheTests` / `Qwen4ExpRetainedCachePolicyTests` 覆盖保留 A/B/A、检查点克隆、投机重绑定、
预算驱逐、缺失状态拒绝以及 QSA 首次/重置增长，并在 CUDA 上覆盖真实双 GPU 按层切分的检查点生命周期。
所有关卡在 CPU 上都逐位一致。在单卡 CUDA 上，一次 4 token 的目标验证与 4 次单 token 前向逐位相同
（`TeacherForcedTargetVerify_…`），以 2–4 为块提交的 32 个 teacher-forced token 在每一行上都与标量解码
逐位相同（`RepeatedTargetBlocks_…`）——见 [验证行使用单 token kernel](#验证行使用单-token-kernel)。
16 token 的 prefill 再接 4 个 token，在 CUDA 上与一次 20 token 的 prefill 并不逐位相同，因为 prefill 的
kernel 按批宽度选择：`SharedPrefixChunking_…` 在 CUDA 上把差异上界设为 1e-2（实测 logits 相差 1.7e-4 到
4.4e-4；它当初要抓的陈旧种子缺陷让 logits 偏移了 0.3155），并且只允许在 top-2 差值不超过实测差异两倍的
近似平局处改变贪心结果
（[`verify-row-kernels-20260917`](../../eng/validation/qwen38_mtp_followup/verify-row-kernels-20260917/README.md)）。

## 共享 MTP 头的投机解码

`--draft-model mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` 挂上逐 token 的 MTP 块；它只为从位置 0 开始
prefill 的单独请求做投机（与其他序列共享的步，以及延续保留 holder 或共享前缀克隆的轮次，都按普通
方式解码——草稿头有自己的 K/V，无法跨越它从未重放过的位置起草）。在 UD-Q2_K_XL、三 GPU 按层切分
（A40）上以 `--spec-draft 3` 实测：

- **一致性。** 192 token 的代码复制流与普通贪心完全一致，速度 1.75-1.96 倍（141/141 草稿被接受，
  无回滚）；投机 prefill（`SpecForward`）在相同分块下与普通 prefill 逐位一致。2026-09-17 之前，4 行
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
保持分配，于是这两个融合在任何宽度下都会发生。单 token 与 prefill 的 kernel 不变；Metal 保留现有的图构建方式。

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
证据以及逐断言诊断：
[`verify-row-kernels-20260917`](../../eng/validation/qwen38_mtp_followup/verify-row-kernels-20260917/README.md)。

## 多 GPU

`qwen4exp` 上的 `--tp N` 跑的是**按层切分**：每张 GPU 持有一段连续的完整层。它不是
张量并行——`qwen4exp` 不切分任何权重——而且这也正是 llama.cpp 为该架构提供的
（唯一）多 GPU 模式（`-sm row` 直接拒绝加载）。它是**容量**特性，不是速度特性：
单卡装不下时靠它把模型装下。

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

二是**本模型只能跑在会传 `--tp N` 的那一列上**。原因就在上一节：切分度来自 `--tp`，
所以在不传 `--tp` 的后端列上，TensorSharp 只会建单设备上下文，175.3 GiB 会全部压到
一张卡上。因此配置里给了它 `min_tp`（4，仅按权重算出的下限——8 才是这台 8×A40 机器
应当使用的度数），在不传 `--tp` 的那一列上，这些格子会被记为
`needs --tp 4 (does not fit 1 GPU(s))` 的跳过，而不是留给它去 OOM。跑法：

```
python run_matrix.py --config benchmark_config_glm53_qwen38.json \
    --models qwen38-flash-next --backends ggml_cuda_tp
```

那一列会让 llama.cpp 用 `--split-mode layer` 切在同样这些 GPU 上，于是参照列两边是
同一种放置方式——对 `qwen4exp` 而言，这也是两个引擎各自唯一的多卡模式。
