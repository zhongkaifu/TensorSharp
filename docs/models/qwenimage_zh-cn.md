# Qwen-Image-Edit

[← 返回模型索引](README_zh-cn.md)

## 状态快照

| 字段 | 状态 |
|---|---|
| GGUF 架构键 | `qwen_image`（MMDiT 扩散 Transformer） |
| 源类 | [`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs)（+ 内部 [`QwenImagePipeline`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)） |
| 任务 | 图像编辑——提示词 + 输入图像 → 编辑后的图像 |
| 模态 | 图像输入 + 文本输入 → 图像输出 |
| 思考 / 工具 | 不适用 |
| 生成方式 | FlowMatch-Euler 扩散去噪（true-CFG），**非**自回归 token 解码 |
| CLI 支持 | `TensorSharp.Cli` 检测到 `QwenImageModel` 后进入图像编辑模式（`--image`、`--prompt`、`--output`、`--cfg`、`--diffusion-steps`、`--diffusion-seed`） |
| 服务器支持 | Web UI 图像编辑流程：`POST /api/image-edit` 与 `POST /api/image-edit/stream`（SSE，含实时去噪预览） |
| 连续批处理 | 无——图像编辑串行执行（扩散网络非线程安全），并发请求逐个处理 |

## 1. 来源与意图

Qwen-Image-Edit 是一个指令驱动的图像编辑器。与自回归 LLM 不同，所加载的
`qwen_image` GGUF **仅**是 MMDiT（多模态扩散 Transformer）；图像编辑还需要两个
DiT GGUF 本身不含的网络：

- **Qwen-Image VAE**——图像 ↔ 16 通道潜变量（空间 8× 下采样），以及
- **Qwen2.5-VL-7B 文本编码器**——提示词 → 3584 维条件，外加可选的 `mmproj`
  视觉塔以实现图像接地（image-grounded）条件。

`ModelBase.Create()` 将 `general.architecture = qwen_image` 路由到
`QwenImageModel`。伴随 GGUF 从 DiT GGUF 所在目录解析，或通过
`TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` 环境变量指定
（VAE 可为原始 `.safetensors`，直接加载并将 BF16→F32 上转）。`QwenImageModel`
不是 `IModelArchitecture` 文本生成器：`Forward()` 抛异常，编辑通过
[`EditImage()`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) 驱动。

## 2. 流水线（去噪循环）

[`QwenImagePipeline.Edit`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)
编排完整的编辑过程：

```text
输入图像
  -> 预处理（保持纵横比；尺寸为 16 的倍数；面积按 VRAM 预算钳制，最高至原生 ~1 MP）
  -> VAE 编码 -> 归一化（diffusers latents 均值/标准差）-> pack 为 [refSeq, 64]
  -> 文本条件：Qwen2.5-VL 编码(提示词 [+ 视觉接地的图像 token])
       （CfgScale > 1 时再编码负向提示词，用于 true-CFG）
  -> 释放文本 + 视觉编码器（为 DiT 回收 VRAM）
  -> 噪声潜变量（带种子高斯）pack 为 [genSeq, 64]
  -> 拼接 [生成 | 参考] token（modulateIndex 标记参考半区）
  -> FlowMatch-Euler 调度器（timestep == sigma），First-Block-Cache 复位
  -> 每步：
       DiT 速度预测（cond）[+ neg -> true-CFG 合并 + 逐行重归一化]
       scheduler.Step(latents, v, step)
       通过 OnStep 回调可选地输出解码的 RGB 预览（降采样）
  -> unpack -> 反归一化 -> VAE 解码 -> 输出图像
```

true-CFG 逐 token 行合并有条件与无条件速度（`comb = neg + scale·(cond − neg)`），
并将每行重归一化回有条件范数。当 `CfgScale <= 1` 时跳过负向分支（每步只跑一次
DiT 前向，而非两次）。

## 3. MMDiT 架构常量

`qwen_image` GGUF 不携带超参数；它们固定在
[`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) 中：

| 常量 | 值 |
|---|---|
| 隐藏维度 | 3072 |
| 层数（双流块） | 60 |
| 注意力头数 | 24（头维度 128） |
| DiT 输入通道 | 64（16 潜通道 × 2×2 patch） |
| 文本条件维度 | 3584（Qwen2.5-VL 隐藏维度） |
| VAE 潜通道 | 16 |
| VAE 缩放因子 | 8 |
| RoPE 轴维度 | {16, 56, 56}（和 = 头维度 128） |
| Eps | 1e-6 |

每个 MMDiT 块以多模态 RoPE 联合关注图像与文本 token；timestep/文本调制通过逐
token 的 `modulateIndex`（0 = 生成半区，1 = 参考半区）施加。参考潜变量被拼接进
token 序列，使编辑接地于原图。

## 4. TensorSharp 实现

| 组件 | 文件 |
|---|---|
| 模型入口 / 伴随文件解析 | [`QwenImageModel.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) |
| 去噪编排 | [`QwenImagePipeline.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs) |
| 生成参数 | [`QwenImageParams.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageParams.cs) |
| MMDiT（图/缓存/CFG-batch/native/整模型） | `QwenImageDiT*.cs` |
| FlowMatch-Euler 调度器 | [`QwenImageScheduler.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageScheduler.cs) |
| VAE（+ 参考数学） | [`QwenImageVae.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVae.cs) |
| Qwen2.5-VL 文本编码器 | [`QwenImageTextEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageTextEncoder.cs) |
| 视觉塔（接地） | [`QwenImageVisionEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVisionEncoder.cs) |
| 图像解码 / 编码（PNG） | [`ImageIO.cs`](../../TensorSharp.Models/Models/QwenImage/ImageIO.cs) |
| 融合 native 内核 | [`ggml_ops_qwen_image.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen_image.cpp) |

## 5. 加速状态

- **CUDA 图捕获的整 DiT 前向**（`TSGgml_QwenImageForward`）：权重一次性常驻，整个
  60 块前向被捕获进单个可重放的图并配专用 `gallocr`。这把去噪从受启动开销限制
  （~40% GPU）变为受计算限制（~100% GPU）——单次前向成本下降约 2.9×，在被测
  CUDA 机器上 8 步去噪从 ~153 s 降到 ~63 s。用 `TS_QWEN_DIT_WHOLE_CAPTURE=0`
  关闭。
- **默认 flash 注意力**（ggml-cuda 路径；`TS_QWEN_DIT_FLASH=0` 强制显式分数路径）：
  注意力内存为 O(n)，因此在 VAE 解码的 im2col 暂存成为瓶颈前可容纳更高分辨率。
- **CFG-batching**：当合并的 token 块能装入 VRAM 时，两个引导分支在一次摊销启动
  的融合前向中运行（`TS_QWEN_DIT_CFG_BATCH_MAXTOK`）；更大的图像回退为每步两次
  前向。
- **Lightning 蒸馏 LoRA（4/8 步编辑）**——`--qwen-image-lora <lora.safetensors>`
  （CLI + 服务器）/ `TS_QWEN_IMAGE_LORA` 以**运行时旁路**在每个目标投影旁应用
  DiT LoRA：`y = W_quant·x + b + (alpha/rank)·up·(down·x)`（F32 计算），量化基础
  权重保持不变。（合并进权重——stable-diffusion.cpp 的反量化→加→重量化路径——
  只对 F16/Q8_0 存储成立：Lightning 增量约 1e-4 RMS，远低于 Q2_K 量化步长，
  合并会被重量化噪声吞掉。）LoRA 因子与权重一样常驻显存、可被 CUDA 图捕获，
  每次前向额外开销几个百分点 + ~1.6 GB 显存。配合 lightx2v 的
  [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
  检查点，采样默认值自动切换（从文件名解析）：其训练步数（4 或 8）、cfg 1.0
  （每步单次前向）、以及蒸馏所用的固定 timestep shift 3——把默认的 60 次 DiT
  前向减少到 4–8 次。显式 `steps`/`cfg` 仍然优先；`TS_QWEN_IMAGE_LORA_SCALE`
  调节 LoRA 乘数（默认 1.0）。旁路实现在整模型前向（CUDA 默认路径）；回退到
  逐块/托管路径时会打印警告（该路径不应用 LoRA）。
- **整步去噪缓存（EasyCache）**——默认缓存模式，移植自 stable-diffusion.cpp 的
  `easycache.hpp`。它用测得的输入潜变量变化（乘以经验跟踪的输入→输出变换率）
  预测每步的输出变化；当累计预测保持在阈值以下时，**跳过两个 CFG 分支的整个
  DiT 前向**，各分支的速度重建为 `input + cached(output − input)`。最初 ~15% 和
  最后 ~5% 的步总是计算。默认阈值下通常跳过 40–55% 的步，结果感知等价。
  开关：`TS_QWEN_DIT_CACHE_MODE`（`easycache`/`fbc`/`both`/`off`）、
  `TS_QWEN_DIT_EASYCACHE_THRESHOLD`（默认 0.2——越低越接近无缓存）、
  `TS_QWEN_DIT_EASYCACHE_START`/`_END`（窗口比例，0.15/0.95）、
  `TS_QWEN_DIT_CACHE_DEBUG=1`（每步决策日志）。
- 跨去噪步的 **First-Block-Cache**（每次生成复位）；此前的默认，现用
  `TS_QWEN_DIT_CACHE_MODE=fbc` 选择。它每分支总是计算 block 0，在低变化步跳过
  blocks 1..59——节省严格少于整步缓存，但其决策基于真实的 block-0 残差而非预测。
- **融合条件编码器主干**（`TSGgml_QwenTeTrunk`，默认开启）：Qwen2.5-VL 文本编码器
  LLM（28 层，GQA，因果）与其视觉塔（32 层，MHA，窗口/全注意力以加性掩码实现）
  各自把整个层栈作为**一个**设备图运行——逐算子路径每层要付约 10 次设备⇄主机
  往返（M-RoPE、bias 加法、SiLU、块掩码 softmax 都是主机循环）。RoPE cos/sin 表
  在主机预计算；权重按 GGUF mmap 指针常驻绑定。默认编辑下：视觉 4.7 s → **1.6 s**、
  LLM 预填充 2.8 s → **0.8 s**（文本条件阶段 11.2 s → 2.5 s）。已对 numpy oracle
  验证（`te-verify` cosine 0.9989、`vis-verify` 0.9994——均不低于逐算子路径自身）。
  后端无法运行时回退到逐算子路径；`TS_QWEN_TE_FUSED=0` 禁用两者。
- **VRAM 预算**：条件生成后释放文本 + 视觉编码器，使 DiT（及其注意力暂存）独占
  工作集；除非固定 `Width`/`Height`，否则目标输出面积自动钳制以适配设备 VRAM，
  并在最终 VAE 解码前归还持久复用 `gallocr`。
- **融合整 VAE 图**（`TSGgml_QwenVaeRun`，默认开启）：整个 VAE 编码/解码作为**一个**
  设备常驻 ggml 图运行——C# 侧按已验证的 `VaeReferenceMath` 拓扑发出扁平算子列表，
  特征全程留在 GPU 上，权重从稳定缓冲常驻绑定（只上传一次），每个卷积在其临时
  F16 im2col 适合预算时用 im2col+GEMM（tensor core；`TS_QWEN_VAE_FUSED_IM2COL_BUDGET`，
  默认 2 GiB；gallocr 复用暂存，峰值只有一个卷积的 im2col），超出预算则用
  `ggml_conv_2d_direct`。取代了逐卷积路径的数 GB PCIe 往返 + CPU SiLU/norm 循环：
  928×688 下编码 19.5 s → **0.95 s**、解码 22.8 s → **1.35 s**，与 diffusers oracle
  位级等价（PSNR 99 dB，与旧路径相同）。后端无法运行时回退到逐卷积路径；
  `TS_QWEN_VAE_FUSED=0` 禁用。
- **分带切块的 VAE 卷积**（`VaeReferenceMath.TryGpuConv2dMaybeTiled`）：`ggml_conv_2d`
  会物化一个 `~IC·KH·KW·OH·OW·2` 字节的 F16 im2col，高分辨率下达数 GB，曾溢出到
  共享显存（VAE 慢约 3×）甚至 OOM——这正是把输出限制到 ~0.55 MP 的真正瓶颈。现按
  输出水平条带切分（预算 `TS_QWEN_VAE_CONV_TILE_BYTES`，默认 1 GiB），每条带在手动
  补零的输入切片上重跑同一卷积，结果**逐位一致**（无拼缝——只有临时 im2col 被限界），
  使面积钳制可达模型**原生 ~1 MP**，显著改善面部 / 细节。

重要开关：

| 变量 | 效果 |
|---|---|
| `TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` | 覆盖解析到的伴随 GGUF（CLI 暴露为 `--qwen-image-vae` / `--qwen-image-vl` / `--qwen-image-mmproj`） |
| `TS_QWEN_IMAGE_NO_VISION=1` | 跳过视觉接地（更快的纯文本无接地条件） |
| `TS_QWEN_IMAGE_LORA` | 以运行时 F32 旁路应用的 DiT LoRA safetensors（CLI/服务器：`--qwen-image-lora`）；Lightning 检查点还会切换采样默认值（其步数、cfg 1.0、固定 shift 3） |
| `TS_QWEN_IMAGE_LORA_SCALE` | LoRA 乘数（默认 1.0；0 = 结构上启用但零效果） |
| `TS_QWEN_IMAGE_MAX_AREA` | 覆盖 Metal 默认的目标面积钳制 |
| `TS_QWEN_DIT_WHOLE_CAPTURE=0` | 禁用 CUDA 图捕获的整 DiT 前向 |
| `TS_QWEN_DIT_FLASH=0` | 强制显式分数注意力路径（更紧的二次 VRAM 预算） |
| `TS_QWEN_DIT_CFG_BATCH_MAXTOK` | 保持 CFG-batching 启用的 token 预算 |
| `TS_QWEN_DIT_CACHE_MODE` | 去噪缓存：`easycache`（默认，整步跳过）、`fbc`（First-Block-Cache）、`both`、`off` |
| `TS_QWEN_DIT_EASYCACHE_THRESHOLD` | EasyCache 累计变化阈值（默认 0.2；越低越接近无缓存，越高跳过越多） |
| `TS_QWEN_DIT_CACHE=0` | 旧版总开关——禁用所有去噪缓存 |
| `TS_QIMG_DEBUG=1` | 每步速度 / 潜变量统计 |

## 6. 生成参数（`QwenImageParams`）

| 属性 | 默认 | 含义 |
|---|---:|---|
| `Steps` | 0（自动） | FlowMatch-Euler 去噪步数。自动 = 30，或已加载 Lightning LoRA 的训练步数（4/8） |
| `CfgScale` | 0（自动） | true-CFG 引导尺度；`<= 1` 关闭负向分支。自动 = 2.5（Qwen-Image-Edit-2511 推荐值——4.0 会过度引导：面部扭曲、颜色过饱和），加载 Lightning LoRA 时为 1.0。需要更强风格化可调到 3.5–4，但会牺牲面部保真度 |
| `NegativePrompt` | `" "` | CFG 分支的负向提示词（仅 `CfgScale > 1` 时使用） |
| `Seed` | 0 | 确定性初始噪声种子 |
| `TargetArea` | 1024×1024 | 输出面积（像素，纵横比随输入；尺寸对齐到 /16） |
| `Width` / `Height` | 0 | 显式输出尺寸（绕过 VRAM 面积钳制） |
| `OnStep` | null | 每步回调 `(step, totalSteps, preview)`，用于实时 UI 反馈 |
| `PreviewCount` | 0 | 整个循环中输出的解码 RGB 预览数量 |

## 7. 服务器行为

当 Web UI 托管一个 `qwen_image` DiT GGUF 时，上传 + 编辑端点接管：

- `POST /api/image-edit` 运行单次编辑并返回输出图像。
- `POST /api/image-edit/stream` 流式发送 SSE 帧：进度滴答加上节流的解码预览
  （`{ imageEdit: true, step, total, image, width, height }`），随后在 `done`
  之前发送最终的全分辨率图像。
- 编辑在共享锁后串行执行（扩散网络非线程安全），因此并发请求逐个运行。
- Ollama / OpenAI 聊天适配器是自回归的，不暴露图像编辑。

## 8. 待办

- 并发编辑被串行化；跨请求的批量 / 流水线编辑尚未实现。
- 视觉接地条件准确但开销大；最实用的全质量编辑在带整 DiT 捕获的 CUDA 路径上。
