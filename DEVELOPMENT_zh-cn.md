# 开发
[English](DEVELOPMENT.md) | [中文](DEVELOPMENT_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分：如何构建 TensorSharp、仓库结构、包分层、内部架构与测试工具。

## 前置要求

### 安装 .NET 10 SDK

TensorSharp 的所有项目都面向 `net10.0`。从源码构建必须安装完整的 **.NET 10 SDK**；仅安装 Runtime 不够。安装 SDK 时也会安装 CLI 与服务器运行所需的 .NET 和 ASP.NET Core Runtime。

| 平台 | 推荐安装方式 |
|---|---|
| **Windows** | 打开 PowerShell 并运行 `winget install Microsoft.DotNet.SDK.10`。也可以按 Microsoft 的 [Windows 安装指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/windows)，选择适合当前架构的 .NET 10 **SDK** 安装程序。 |
| **macOS** | 下载并运行 [.NET 10 SDK 安装程序](https://dotnet.microsoft.com/zh-cn/download/dotnet/10.0)。Apple 芯片选择 **Arm64**，Intel Mac 选择 **x64**；另见 Microsoft 的 [macOS 安装指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/macos)。 |
| **Linux** | 使用 Microsoft 的 [Linux 安装指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux)选择发行版和版本、配置相应软件源，再安装 .NET 10 SDK 包（通常名为 `dotnet-sdk-10.0`）。各发行版的软件源与架构支持不同，请以链接中的发行版专用步骤为准。 |

安装后打开新终端，确认列表中包含 `10.0.x` SDK：

```bash
dotnet --list-sdks
```

安装程序、包管理器、手动安装和非管理员安装方式见 Microsoft 的 [.NET 跨平台安装概览](https://learn.microsoft.com/zh-cn/dotnet/core/install/)。

### 其他构建前置

- **`git` 与网络访问：** GGML/CUDA 原生构建会在首次构建时从 [github.com/ggml-org/ggml](https://github.com/ggml-org/ggml) 克隆 ggml 源码到 `ExternalProjects/ggml/`（参见 `eng/fetch-ggml.sh` / `eng/fetch-ggml.ps1`）。克隆默认跟踪 ggml 的默认分支（`master`）；可用 `TENSORSHARP_GGML_GIT_REF` 指定其他引用，或在克隆完成后设置 `TENSORSHARP_GGML_NO_UPDATE=1` 跳过网络更新（用于离线重建）
- **macOS（Metal 后端）：** 用于构建原生 GGML 库的 CMake 3.20+ 与 Xcode 命令行工具——GGML 以源码形式内嵌 Metal kernel 并在运行时编译，因此构建期不需要 Metal 编译器。若需使用 MLX 后端，还需通过 `bash TensorSharp.Backends.MLX/build-native-macos.sh` 从 `TensorSharp.Backends.MLX/Native/` 构建 `libmlxc`；该构建**会**编译 Metal shader，因此需要**完整的 Xcode 以及 Metal 工具链**，仅安装命令行工具是不够的。首次构建时 `eng/ensure-metal-toolchain.sh` 会自动完成这一准备工作，详见[构建原生 MLX 库](#构建原生-mlx-库仅-macos)
- **Windows（GGML CPU / CUDA 后端）：** CMake 3.20+ 与 Visual Studio 2022 或 2026 C++ 构建工具；若使用 `ggml_cuda` 或 `cuda`，还需要 NVIDIA 驱动和带 cuBLAS 的 CUDA Toolkit 12.x 或其他兼容版本。装有多个工具包时，可用 `CUDACXX`（或 `-DCMAKE_CUDA_COMPILER=`）指定用哪个 `nvcc` 构建；`build-windows.ps1` 会优先采用它而不是 `PATH`/`CUDA_PATH`，把选择打印在 `Configuring ...` 一行中，并丢弃使用其他编译器配置过的构建树（CMake 只在首次配置时缓存 CUDA 编译器，此后便会忽略 `CUDACXX`）。注意 CMake 仅在非 Visual Studio 生成器下读取 `CUDACXX`；使用 `Visual Studio NN` 生成器时请改用 `-T cuda=<版本或路径>`。Visual Studio 2026 的 MSVC 14.5x 工具集比当前 CUDA 工具包官方支持的宿主编译器更新，构建会自动向 `nvcc` 传递 `-allow-unsupported-compiler`；请同时安装“适用于 Windows 的 C++ CMake 工具”组件，以便构建使用 Ninja 生成器（Visual Studio 生成器还额外需要为对应 VS 版本提供 MSBuild 集成的 CUDA 工具包）。**cuDNN 会自动准备**：首次 CUDA 构建时由 `eng/fetch-cudnn.ps1` 下载到 `ExternalProjects/cudnn/`（下载约 1.8 GB，磁盘占用约 1.1 GB，只解出 `include/` 与 `bin/`），使 Wan / Qwen-Image VAE 的卷积可以走 cuDNN 而不是 ggml 的 im2col+GEMM 下降路径。已安装的版本（`TS_CUDNN_DIR`、`CUDNN_DIR`、`CUDA_PATH`）优先于下载，`TENSORSHARP_CUDNN=OFF` 可完全跳过；构建不与它链接——运行时用 `LoadLibrary` 解析，因此有没有它，构建与产物都照常工作
- **Linux（GGML CPU / CUDA 后端）：** CMake 3.20+；若使用 `ggml_cuda` 或 `cuda`，还需要 NVIDIA 驱动和带 cuBLAS 的 CUDA Toolkit 12.x 或其他兼容版本。**cuDNN 会自动准备**：首次 CUDA 构建时，`eng/fetch-cudnn.sh` 会从 NVIDIA 公开的 redist 渠道把固定版本下载到 `ExternalProjects/cudnn/`（无需账号，也无需点选许可），Wan / Qwen-Image VAE 的卷积随后走 cuDNN 而不是 ggml 的 im2col+GEMM 下降路径。系统上已装的 cuDNN（`libcudnn9-dev-cuda-12`、`CUDA_HOME`、`TS_CUDNN_DIR`）优先于下载。它严格可选：获取失败不会让构建失败，编译期只需要头文件，库本身在运行时用 `dlopen` 解析，因此带 cuDNN 构建出的二进制在没有 cuDNN 的机器上照样能跑。`TENSORSHARP_CUDNN=OFF` 完全跳过；configure 阶段会打印实际生效的是哪一条
- **Windows（GGML Vulkan 后端）：** 这里同样需要 CMake 3.20+ 与 Visual Studio 2022 或 2026 C++ 构建工具——Vulkan 工具链的准备和后端的编译都由 CMake 完成，缺少它会在 `eng/fetch-vulkan-toolchain.ps1` 阶段就失败。Visual Studio 的“适用于 Windows 的 C++ CMake 工具”组件同时自带 `cmake.exe` 与 `ninja.exe`，两者都不在 `PATH` 上时 `build-windows.ps1` 会回退到该副本；否则请从 [cmake.org/download](https://cmake.org/download/) 安装 CMake。原生构建**仅支持 x64**——`build-windows.ps1` 会自行导入 `vcvars64` 环境，包括覆盖已经激活的 *x86* 环境（普通的“Developer PowerShell for VS”默认是 x86，而 ggml 的 Vulkan 后端无法以 32 位编译）。机器有 Vulkan 运行时（每个较新的 GPU 驱动都带的 `System32\vulkan-1.dll`）时自动启用。已安装 [LunarG Vulkan SDK](https://vulkan.lunarg.com/) 时直接使用；未安装时构建会通过 `eng/fetch-vulkan-toolchain.ps1` 自动把便携工具链（Vulkan-Headers、由系统 loader 生成的 vulkan-1 导入库、glslc、SPIRV-Headers）准备到 `ExternalProjects/vulkan-toolchain/`。用 `build-windows.ps1 --no-vulkan` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF` 退出。运行时需要支持 Vulkan 1.3 的 GPU 驱动
- **Linux（GGML Vulkan 后端）：** 已安装 Vulkan loader（`libvulkan.so.1`）时自动启用。存在发行版开发包时直接使用（`apt install libvulkan-dev glslc spirv-headers`）；否则构建会通过 `eng/fetch-vulkan-toolchain.sh` 把缺失的部分（Vulkan-Headers、shaderc CI 预编译的 glslc、SPIRV-Headers）自动下载到 `ExternalProjects/vulkan-toolchain/`。用 `build-linux.sh --no-vulkan` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF` 退出
- GGUF 模型文件（例如来自 [Hugging Face](https://huggingface.co)）

## 构建

### 构建整个解决方案

```bash
dotnet build TensorSharp.slnx
```

解决方案构建默认使用 `Any CPU` 平台（见 `Directory.Solution.props`），因此在 Visual Studio 开发者命令提示符中也能正常工作——这类提示符会向环境导出 `Platform=x64`，否则会把构建引导到不存在的 `Release|x64` 解决方案配置。显式传入的 `-p:Platform=...` 仍然优先。

### 构建单独应用

```bash
# 控制台应用
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj

# Web 应用
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj
```

### 构建原生 GGML 库

如果原生库不存在，首次执行 `dotnet build` 时会自动构建。也可以手动构建：

```bash
cd TensorSharp.GGML.Native
```

macOS：

```bash
bash build-macos.sh
```

Linux（仅 CPU）：

```bash
bash build-linux.sh
```

Linux（启用 GGML_CUDA）：

```bash
bash build-linux.sh --cuda
```

Windows（仅 CPU）：

```powershell
.\build-windows.ps1 --no-cuda
```

Windows（启用 GGML_CUDA）：

```powershell
.\build-windows.ps1 --cuda
```

在 Windows 和 Linux 上，原生构建脚本会自动检测可见 NVIDIA GPU 的 compute capability，并把一个精简的 `CMAKE_CUDA_ARCHITECTURES` 列表传给 ggml-cuda（例如在 RTX 3080 上为 `86-real`），从而显著降低 CUDA 构建时间。原生构建默认还会并行运行，并根据内存容量限制并行任务数（`nvcc` 单个编译单元峰值约 3 GB），避免拖慢普通开发机器。

在 Windows 上，`build-windows.ps1` 优先使用 **Ninja** 生成器，其次是 `Visual Studio NN` 生成器，最后才交由 CMake 自行选择。这对构建时间影响很大：Ninja 会把所有编译单元放进同一个依赖图并行编译，而 Visual Studio 生成器只在 CMake 项目之间并行，导致 ggml-cuda 的约 190 个 `nvcc` 编译任务逐个串行执行。脚本会在 `PATH` 或 Visual Studio 安装目录中查找 `ninja.exe`（"适用于 Windows 的 C++ CMake 工具"组件自带一份），并自行导入 MSVC 的 `vcvars64` 环境，因此不再需要从"x64 Native Tools"命令提示符启动。它还会在**已经激活的 x86 环境之上**导入 `vcvars64`：普通的"Developer PowerShell for VS"与"Developer Command Prompt"快捷方式默认使用 x86 工具集，而 32 位构建会在 ggml 的 Vulkan 后端深处失败，且报错完全不会提到"32 位"。已经激活的 *x64* 环境则保持不动，因此显式固定的工具集（`vcvarsall.bat x64 -vcvars_ver=...`）不会被覆盖。`cmake.exe` 的解析方式与 `ninja.exe` 相同——先找 `PATH`，再找 VS 的"适用于 Windows 的 C++ CMake 工具"副本——缺少 CMake 时会提前明确报错，而不是抛出"无法将“cmake”项识别为 cmdlet"。生成器与实际并行度会打印在 `Configuring TensorSharp.GGML.Native (...)` 一行中；如果脚本警告可能回退到串行的 `NMake Makefiles` 生成器，请安装上述 VS 组件或把 `ninja.exe` 加入 `PATH`。

Visual Studio 的定位由 `eng/vs-locate.ps1` 完成，它能识别被 VS 安装程序标记为"不完整"的安装（对这类安装 `vswhere -latest` 会静默地报告*找不到*任何安装，这正是 CMake 之前回退到串行生成器的原因）。可用 `TENSORSHARP_VS_INSTALL_DIR` 覆盖检测到的安装目录，或通过 `CMAKE_GENERATOR` 环境变量以及向脚本传入 `-G` 来强制指定生成器。

如需覆盖自动检测到的架构列表或默认的并行度，可使用以下任一方式：

```bash
TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real' bash build-linux.sh --cuda
bash build-linux.sh --cuda --cuda-arch='86-real;89-real'
TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2 bash build-linux.sh --cuda
```

```powershell
$env:TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real'; .\build-windows.ps1 --cuda
.\build-windows.ps1 --cuda --cuda-arch='86-real;89-real'
$env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2; .\build-windows.ps1 --cuda
```

也可以在 `dotnet build` 时通过环境变量请求 CUDA 版本的原生库：

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

```powershell
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA='ON'; dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

在 macOS 上会生成带 Metal GPU 支持的 `libGgmlOps.dylib`。在 Windows 和 Linux 上，原生脚本会保留已有的 CUDA 构建，并在检测到 CUDA 工具链时自动启用 GGML_CUDA；也可以通过 `build-windows.ps1 --cuda`、`build-linux.sh --cuda` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` 显式启用。GGML Vulkan 后端在机器有 Vulkan 运行时时同样自动启用，并在首次使用时下载其构建工具链；`--vulkan` / `--no-vulkan` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON/OFF` 可显式指定，显式选择会在后续重建中保持（构建自动准备的 Vulkan 工具链见[前置要求](#前置要求)）。构建产物会自动复制到应用输出目录。

Direct `cuda` 后端由托管 C# 代码和 PTX 内核组成。执行 `dotnet build` 时，`TensorSharp.Backends.Cuda` 会在检测到 `nvcc` 后把 `native/kernels/*.cu` 编译到中间目录（`obj/cuda_ptx/ptx/`），各输出目录的 `cuda_kernels/` 使用的就是这份本地编译的 PTX——构建不会修改 git 跟踪的 `native/ptx/` 文件。如果缺少 `nvcc`，则改用 `native/ptx/` 中提交的 PTX 基线；若该基线也无法加载，PTX 覆盖的算子会使用 CPU 回退。cuBLAS GEMM 仍要求运行时能够找到 CUDA 运行库。

修改 `.cu` 内核后，请显式刷新提交的 PTX 基线并提交差异——没有 `nvcc` 的机器运行的是提交的 PTX，未刷新的内核改动会让这些机器悄悄运行过期内核：

```powershell
dotnet build TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj -p:TensorSharpUpdateCommittedPtx=true
```

#### Apple Silicon 上的 Metal 4 tensor API

在 M5 及更新的 GPU 上，ggml 可以让矩阵乘走 Metal 4 tensor API，在 M5 Pro 上实测 **prefill 吞吐提升 2.6 倍**（807 → 2107 tok/s，Gemma 4 E4B Q8_0，2048 token prefill）。decode 不受影响，这符合预期——单序列 decode 受内存带宽限制。

ggml 只有在运行时成功编译一个包含 `<metal_tensor>` 的探测 kernel 后才会启用它，而编译该探测 kernel 使用的是*默认*的 Metal Shading Language 版本。这个默认值由 Metal 依据**主可执行文件**中记录的 SDK 推导，而不是依据 `libGgmlOps.dylib`。微软预先构建并分发的 .NET apphost（以及 `dotnet` 启动器）是针对 macOS 15.5 SDK 编译的，因此 .NET 进程的默认值是 MSL 3.2，在该版本下 `<metal_tensor>` 不声明任何内容，于是 ggml 会在完全支持该特性的硬件上禁用 tensor API：

```
ggml_metal_device_init: - the tensor API is not supported in this environment - disabling
ggml_metal_device_init: has tensor            = false
```

`tsg_metal_msl_default.m` 在我们自己的库中修正这个默认值，而这也是该修复唯一能放置的位置：`ExternalProjects/ggml` 不纳入 git 跟踪，且 `eng/fetch-ggml.sh` 每次构建都会将其硬重置到上游，因此写在那里的改动会被下一次构建抹掉。仅当 GPU 声明支持 Metal 4 family、且继承到的默认值更旧时，它才会把进程级默认值提升到 MSL 4.0——也就是原生链接的宿主本来就会得到的值。自行设置 `languageVersion` 的代码保持其自身选择，因此 MLX 后端不受影响（MLX 总是显式设置该值）。生效时它会输出一行日志，`has tensor` 随之变为 `true`。

| 环境变量 | 作用 |
|---|---|
| `TENSORSHARP_METAL_MSL_DEFAULT=off` | 保留宿主过时的默认值（恢复 `has tensor = false`） |
| `TENSORSHARP_METAL_MSL_DEFAULT=<主版本>.<次版本>` | 强制指定默认 MSL 版本，例如 `4.0` |

ggml 自身的开关依然在此之上生效：`GGML_METAL_TENSOR_DISABLE=1` 关闭 tensor API，`GGML_METAL_TENSOR_ENABLE=1` 则绕过 ggml 将其限制在 M5/M6/A19/A20 设备的白名单。

##### Wan 视频与 tensor API

ggml 的 tensor-API `mul_mm` 在 M5 上会偶发性地误读 Wan VAE 卷积 GEMM 的操作数列——仅计算图首次执行、与缓冲区布局相关（即历史上"32×32 latent 解码为全黑帧而 33×33 正常"的问题），任何运行时开关都无法规避；而同样的 GEMM 在隔离环境下逐位正确，LLM/DiT 类计算图也从未出错。这是上游 ggml-metal/驱动层缺陷，且 `has_tensor` 在设备初始化时固定，无法按算子选择 kernel。

tensor API 开启时 VAE 依然**正确**：在支持 tensor API 的设备上，VAE 卷积改走 `ggml_conv_2d_direct`（`ggml_ops_wan.cpp` 的 `wan_vae_gemm_budget`），解码结果与 CPU 后端仅差 F16 舍入，但更慢——这是**固定的**每视频开销，而 tensor API 的 DiT 加速随步数和模型规模**线性放大**。M5 Pro 480×480×9帧、6 步实测：

| | DiT 每步（tensor 开/关） | VAE 编+解码（直接/GEMM） | 盈亏平衡 |
|---|---|---|---|
| A14B I2V Q4_K_M | **17.1s** / 30.2s（1.77×） | 135s / 19s | 约 9 步 → 40 步默认配方下开启 tensor API **快约 33%**（约 13.7 vs 20.5 分钟） |
| TI2V-5B Q8_0 | **1.6s** / 2.9s（1.8×） | 179s / 13s | 约 128 步 → 永不划算 |

因此默认按 DiT 规模选择（`WanVideoArchitecture.ApplyNativeTunables`）：A14B/14B 级模型（`patch_embedding` 输出维度 ≥ 5120）**启用**，更小的模型**禁用**。待上游修复 tensor-API `mul_mm` 后，可全面启用并移除直接卷积隔离方案。

| 环境变量 | 作用 |
|---|---|
| `TS_WAN_METAL_TENSOR_API=1` / `=0` | 为 Wan 进程强制开/关 tensor API，覆盖按模型规模的默认值 |
| `TS_WAN_VAE_GEMM_MAX_MB=<n>` | 强制走 im2col+GEMM VAE 路径并设定 `n` MB 的 im2col 预算（0 强制直接卷积）——双向覆盖自动选择 |

#### MiniMax-H3 与 FP16 flash-attention 分子

与上一节不同，这条不限于 Apple：只要 flash-attention kernel 把 softmax 分子保存在 FP16 里就会遇到，CUDA 属于这种情况（ggml-metal 的 F16/F32 kernel 以 F32 累加输出）。

MiniMax-H3 对**一条**无 mask 的打包序列做双向注意力——文本、条件帧、目标音频与目标视频都在其中——因此 key 数量**就是**整段片段：22 帧为 2364 个打包 token，107 帧为 8646 个。vendored ggml 中 CUDA 的 flash-attention kernel 在支持快速 FP16 的 GPU 上用 FP16 寄存器累加 `sum_j exp(s_j - max) * V_j`（`ggml-cuda/fattn-mma-f16.cuh` 中的 `T_C_VKQ = tile<…, half2>`）。在 CUDA 上用 `ggml_prec_set_acc(out, GGML_PREC_F32)`（取代已弃用的 `ggml_flash_attn_ext_set_prec`）请求 F32 累加器不起作用——`ggml-cuda/` 里没有任何代码读取 `GGML_OP_FLASH_ATTN_EXT` 的这一精度请求。ggml-metal 同样不读取它，但其 F16/F32 kernel 本来就以 F32 累加输出（`ggml-metal/kernels/fa.metal` 中 `o_t = float`）；ggml-vulkan 则会读取它并改用 F32 累加器（三者均对照 vendored ggml `353b63b` 核实）。CUDA kernel 留给累加器的余量只有 3 个 bit（`FATTN_KQ_MAX_OFFSET` 把运行最大值抬高 log(8)，使每个 softmax 权重上限为 1/8），因此 N 个 key 的一行会累加到 N/8 × |V|，一旦 N × |V| > 8 × 65504 就溢出为 Inf。只有 H3 会撞上这个上限：checkpoint 里有 `q_norm` 和 `k_norm`，唯独没有 `v_norm`，value 这一路的幅值不受约束。640×384 实测：73 帧正常，107 帧则在**第一个**去噪步就出现 Inf，返回的视频每个像素全黑、每个音频采样被钳位——视频与音频共用同一主干，溢出会连同声音一起拖垮。

修复在 `h3_attend`（`ggml_ops_minimax_h3.cpp`）：按 key 数量取"能把它压到 `kH3FlashKeyBudget` 以内的最小 2 的幂"预先缩放 V，输出时再还原。注意力对 V 是线性的，所以这个修正是**精确**的；又因为倍数是 2 的幂，经过 F16 转换也只是指数位平移而非舍入，因此足够短的序列（包括测试套件里所有 oracle 用例）保持逐位一致。`h3_mm` 对两个激活无界的量化 matmul（送入 `o_proj` 的注意力输出、送入 `down_proj` 的 SwiGLU 隐状态）做同样处理，那里的上限则是 q8_1 每个 block 的 FP16 求和。

| 环境变量 | 作用 |
|---|---|
| `TS_H3_TRACE=1` | 打印每个去噪步的 latent 与 velocity 幅值——样本发散时，latent 的 absmax 会比真正变成无穷早若干步露出端倪 |
| `TS_H3_NO_FLASH=1` | 强制走显式 softmax 路径，用于区分是 flash-attention kernel 的问题还是建模的问题 |

采样器也不再无条件相信结果：velocity 出现非有限值时直接让请求失败并指明是第几步（`MiniMaxH3Pipeline.cs` 的 `RequireFinite`），而不是写出一个长度、帧率、音轨时长都正常但通体全黑的文件——这种失败本来是无声的，因为 RGB 钳位会把 NaN 像素固定成 0，WAV 写出会把 NaN 采样钳到 -1。

#### 后端没有 kernel 的 flash-attention 形状

`ggml_backend_graph_compute` 从不询问 `ggml_backend_supports_op`。因此 ggml-cuda 没有 kernel 的 `GGML_OP_FLASH_ATTN_EXT` 节点会一路走到 `ggml_cuda_flash_attn_ext`，以 `ggml-cuda/fattn.cu:730: fatal error`（`BEST_FATTN_KERNEL_NONE`）终止整个进程。在 ggml 456172ec（`eng/Dockerfile.gb10` 固定的未修改修订版；默认构建跟踪 `master`，见[前置要求](#前置要求)）中，`ggml_cuda_get_best_fattn_kernel` 在以下情况返回 none：

- K 的 head 大小不是 40、64、72、80、96、112、128 或 256（V 的 head 大小相同），也不是 192（V 为 128）、320（V 为 256）、512 或 576（V 为 512）；
- head 大小为 192、320、512 或 576，但 grouped-query 路径不成立：它要求 query/KV head 之比至少为 2（192 时为 8 的倍数，320 时为 32 的倍数）、有 mask、没有 ALiBi、KV 长度是 256（`FATTN_KQ_STRIDE`）的倍数，并且未量化的 Q/K/V/mask 的每个 `nb[1..3]` 都能被 16 整除；
- K 或 V 不是 F32、F16、BF16、Q4_0、Q4_1、Q5_0、Q5_1 或 Q8_0，或者 mask 的 `ne[2] != 1`。

`ggml_backend_supports_op` 回答的正是这个判定（它调用同一个函数），Metal 与 Vulkan 也以同样的方式报告各自的限制。实际中触发过两次：`HunyuanDenseServingTests` 的合成 `mistral3` 模型 head 大小为 16，其批处理 prefill 分块经过了 `TSGgml_PagedAttentionForward`；以及从 `TS_KV_INITIAL_TOKENS=8` 起步的 Gemma 4 全局缓存增长到 16 行，于是 512 维的全局层读取 16 行的窗口（`flash_attn_kv_length` 会补齐到 256，但从不超过缓存长度）。任何长度不是 256 倍数的 512/576 维缓存都有同样的截断，例如 `MAX_CONTEXT` 设为 4000 时，上下文超过 3840 个 token 之后。

`TensorSharp.GGML.Native` 直接在某个后端上计算的图，都通过 `tsg_flash_attn_ext_guarded`（`ggml_ops_flash_attn_guard.h`；针对当前后端用 `tsg::flash_attn_ext_guarded`）构建 flash attention。它先构建 flash 节点，只要后端有对应 kernel 就直接返回，因此受支持的形状得到与之前相同的图。否则它返回以显式算子写成的同一个注意力——F32 `mul_mat`、带 mask/ALiBi/sinks/logit softcap 的 `soft_max_ext`、`mul_mat`——输出布局与 flash 节点一致，query 行分块处理以保证打分矩阵不超过 256 MiB，并且每个调用点打印一次警告：

```
[TensorSharp] warning: CUDA0 has no flash-attention kernel for paged attention (K head 16, V head 16, KV rows 64, query rows 8, heads 4/2, K/V f32/f32, mask yes); running this attention as explicit F32 mul_mat + soft_max instead (same math, slower, more memory). Reported once per call site.
```

`TSGgml_FlashAttnFallbackCount`（`GgmlBasicOps.FlashAttnFallbackCount()`）统计走显式路径的建图次数。覆盖的调用点：分页注意力（两个变体）与分页 KV 池；通用 transformer decode 入口；Qwen 3 decode 与 prefill；Qwen 3.5/3.6 的层 decode、整模型 decode、批处理 decode、verify 与层 prefill；Qwen 3.8 Flash Next（`qwen4exp`）；Gemma 4 dense 与 MoE 的 decode 和批处理 decode；GPT-OSS 的 decode、prefill、批处理 decode 与层 prefill；Muse-Glimmer 及其 DFlash drafter；GLM 5.x 的 decode 与 forward（询问该层所在设备的后端）；以及 CPU/Metal 上的视觉注意力。原本就询问后端的调用点保留各自的处理：Gemma 4 与 Qwen 3.5 verify、GPT-OSS 与 Qwen 3.5 slot arena 向托管调用方返回错误（现在检查每个 tile 与每一层，而不只是第一个），视觉、diffusion、Wan、Qwen-Image、MiniMax-H3 与 embedding 图照旧回退。DeepSeek V4/V4.1 的注意力不变：其滑动窗口环与压缩行缓存在构造上就补齐到 256 行，V4.1 运行 TensorSharp 自己的 F32 注意力。

`GgmlOpsFlashAttnGuardTest`（ctest `flash-attn-unsupported-shape-fallback`）在 CPU 后端上把显式路径与双精度参考实现对比（mask、GQA、带步长的 F16 窗口、sinks、softcap、ALiBi、query 分块），并在第一个 GPU 设备上检查 head 16 以及过短或未对齐的 512 维窗口会以相同结果回退、受支持的形状仍走 kernel。在 CUDA 上，若 ggml 升级后 ggml-cuda 对这些形状的 kernel 可用性发生变化，该测试也会失败。`GgmlOpsFlashAttnGuardTest --unguarded` 在 GPU 上计算裸的 head-16 节点，复现该 abort。`FlashAttnUnsupportedShapeTests`（`Requires=Cuda`）以 head 大小 16 调用 `TSGgml_PagedAttentionForward`，并与托管参考实现对比。

### 构建原生 MLX 库（仅 macOS）

MLX 后端依赖 `libmlxc`（[MLX](https://github.com/ml-explore/mlx) 的 C 绑定）。仓库在 `TensorSharp.Backends.MLX/Native/MLX_C_VERSION` 中固定了已知可用的 `mlx-c` tag，并提供一个辅助脚本来获取和构建：

```bash
bash TensorSharp.Backends.MLX/build-native-macos.sh
```

脚本会把生成的库（`libmlxc.dylib`、`libmlx.dylib` 以及任何后端依赖）写入 `TensorSharp.Backends.MLX/Native/dist/`，构建过程会将它们连同 `mlx.metallib` 一起复制到输出目录。该 metallib 包含 MLX 预编译的 Metal kernel，体积较大（约 150 MB）但**不可省略**：MLX 通过对自身代码调用 `dladdr` 来定位它，因此它必须与 `libmlx.dylib` **位于同一目录**。它唯一的兜底路径是编译期写死的、指向构建目录的路径，所以缺少该文件的部署可以正常加载，却会在第一次 GPU 运算时抛出 `Failed to load the default metallib`。如需自行打包，请务必把它与这些 dylib 放在一起。运行时后端会优先在应用目录下查找；也可以使用 `TENSORSHARP_MLX_LIBRARY=<libmlxc.dylib 路径>` 或 `TENSORSHARP_MLX_LIBRARY_DIR=<包含 libmlxc 的目录>` 指定自定义安装位置。如果找不到对应库，后端会报告不可用，启动时 `--backend mlx` 会被拒绝。

#### Metal 工具链（自动准备）

MLX 在构建期需要编译 Metal shader，因此 `xcrun metal` 必须可用。有两个常见原因会导致它不可用，`build-native-macos.sh` 会在 configure 之前调用 `eng/ensure-metal-toolchain.sh` 来自动修复：

1. **当前激活的 developer 目录是命令行工具。** `/Library/Developer/CommandLineTools` 完全不包含 Metal 编译器，因此当 `xcode-select -p` 指向它时会报 `xcrun: error: unable to find utility "metal", not a developer tool or in PATH`。脚本会定位已安装的 `Xcode.app`，并将 `DEVELOPER_DIR` 指向它来构建。脚本**不会**执行 `sudo xcode-select -s`——该覆盖仅对本次构建生效；如需全局生效请自行执行该命令。
2. **Xcode 16 及更高版本不再内置 Metal 编译器。** 它是一个约 700 MB 的独立可下载组件；缺少它时 `metal` 虽然存在但无法运行（`cannot execute tool 'metal' due to missing Metal Toolchain`）。脚本会通过 `xcodebuild -downloadComponent MetalToolchain` 下载，该操作不需要 `sudo`，安装到系统资产库后可被所有项目共享，并且在 Xcode 升级后依然有效。

这两个问题在 MLX 中都会表现为不易理解的 `error Metal compiler header resolution failed for .../reduce_utils.h`。

Xcode 本身无法无人值守地自动下载（App Store 与 developer.apple.com 都要求登录 Apple ID），因此当系统中没有 `Xcode.app` 时，脚本会中止并给出安装指引。相关开关：

| 变量 | 作用 |
| --- | --- |
| `TENSORSHARP_XCODE_DEVELOPER_DIR` | 指定使用的 `<Xcode.app>/Contents/Developer`，跳过自动探测（适用于多版本并存或 beta 版 Xcode） |
| `TENSORSHARP_MLX_SKIP_METAL_SETUP` | `1`/`true`——完全跳过工具链检查，适用于已在别处准备好环境的机器 |
| `TENSORSHARP_MLX_NATIVE_SKIP` | `true`——完全跳过 MLX 原生构建，以便在没有 Metal 工具链的情况下构建 TensorSharp 的其余部分 |

切换 developer 目录会使 CMake 缓存失效（旧的 SDK 已固化在 `CMakeCache.txt` 中），因此脚本会丢弃过期的构建树并重新 configure。已获取的 `_deps/*-src` 源码目录会被保留，因此代价只是重新 configure，而不需要重新克隆 MLX。

#### macOS SDK 27

macOS 27 SDK 默认按 Metal 4.1 编译 Metal shader，其地址空间规则会拒绝所固定的 MLX kernel。因此当 `xcrun --sdk macosx --show-sdk-version` 报告的 SDK 为 27.0 或更新时，`TensorSharp.Backends.MLX/Native/CMakeLists.txt` 会设置 `CMAKE_OSX_DEPLOYMENT_TARGET=26.2`，从而选用 Metal 4.0 并保留 MLX 的 NAX kernel。显式设置的 `CMAKE_OSX_DEPLOYMENT_TARGET` 或 `MACOSX_DEPLOYMENT_TARGET` 优先，更旧的 SDK 保持其默认值。

### GB10 / DGX Spark 构建容器（实验性）

`eng/Dockerfile.gb10` 在**原生 Linux ARM64 Docker 构建机**上构建本 checkout 的主解决方案与原生 GGML/CUDA 代码。它固定了 CUDA 13.0.2 组件、.NET SDK 10.0.401 以及未经修改的 GGML 修订版 `456172ec733a135778adcd32d00e576a58232e45`。精度相关 kernel 仍位于 TensorSharp 自有代码中；构建会检查获取到的 ggml 工作树未被修改。SDK 与干净的 Ubuntu 镜像按 digest 固定；NVIDIA 签名软件源提供显式指定版本的编译器、运行时与 cuBLAS 包，从而避免把完整 CUDA devel 镜像的 profiler 与无用数学库拉到小型托管 runner 上。未使用的 cuBLAS 静态库在安装层中删除；动态链接的 CUDA 库及其许可声明保持不变。Ubuntu 构建依赖从 Ubuntu 软件源安装。这是构建/开发镜像，不是部署镜像；下文的 `runtime` 与 `artifacts` target 负责生成并验证部署布局。

在仓库根目录、可用的 ARM64 构建机上运行：

```bash
docker build --platform linux/arm64 -f eng/Dockerfile.gb10 \
  --build-arg SOURCE_REVISION="$(git describe --always --dirty)" \
  --build-arg GGML_BUILD_JOBS=2 \
  -t tensorsharp-gb10-build .
```

镜像包含本地源码改动，而不是重新克隆上游 TensorSharp。该 Dockerfile 专用的 ignore 文件会排除 Git 元数据、常见的本地凭据、模型下载与生成的输出，且不影响其他 Docker 构建。使用远程构建机前请检查构建上下文——该 ignore 文件不是密钥扫描器。`SOURCE_REVISION` 是调用方提供的来源信息，`git describe --dirty` 检测不到未跟踪文件。远程构建时，`source` target 会导出同样过滤后的 checkout，既不编译也不要求 ARM64；请传输该归档而不是未过滤的工作目录，并在构建解出的源码时沿用原 checkout 的 `SOURCE_REVISION`：

```bash
docker build -f eng/Dockerfile.gb10 --target source \
  --output type=tar,dest=../tensorsharp-gb10-source.tar .
```

GB10 的 compute capability 为 12.1：GGML 目标为 `121a-real`，Direct CUDA 后端编译 `compute_121` PTX。构建会拒绝 13.x 以外的 CUDA 版本，也会拒绝 x64/QEMU 构建机；编译不需要发现或访问 GPU。`GGML_NATIVE=OFF` 防止构建机的 CPU 指令混入产物；NCCL、Vulkan 与可选的 cuDNN 加速均被禁用。原生编译默认两个任务，可用 `GGML_BUILD_JOBS` 调整。生成的开发镜像在 `/src` 下包含 checkout 与构建产物，默认命令为 Bash；它不下载模型、不运行推理、不发布发行归档，也不安装或修改宿主的 NVIDIA 驱动。在共享的 Spark 上构建或测试前，请为 vLLM 等竞争负载安排停机——即使不用 GPU 的构建也会占用 CPU 与共享内存。

聚焦的容器契约检查可在任何装有 Python、Bash 与本地 Docker daemon 的开发机上运行，不需要 CUDA、模型下载或 GPU：

```bash
python eng/tests/gb10-container.py
```

#### GB10 归档与干净运行时

`artifacts` target 为 `linux-arm64` 发布自包含的 CLI 与 Server.Host 应用，打包 CUDA 13 运行时/cuBLAS 库与 ARM64 原生媒体库，并在纯净的 Ubuntu 24.04 中验证解压后的归档。该验证不使用 SDK、CUDA Toolkit、源码 checkout、构建缓存或 `LD_LIBRARY_PATH`；GGML 通过 `$ORIGIN` 定位其伴随库。缺少必需资产会让构建失败；NVIDIA 驱动从不打包。发布路径跳过完整的开发/测试解决方案构建，依次处理两个应用包（含干净运行时验证），且不在发布构建缓存中保留重复的解压应用；可选的 `runtime` target 仍保留两个应用供交互使用。

```bash
docker build --platform linux/arm64 -f eng/Dockerfile.gb10 \
  --build-arg SOURCE_REVISION="$(git describe --always --dirty)" \
  --target artifacts --output type=local,dest=artifacts/gb10 .
```

版本默认取自 `Directory.Build.props`；候选版本可用 `--build-arg RELEASE_VERSION=2.8.6-gb10.1`。输出为 `tensorsharp-cli-<version>-linux-arm64-cuda13-GB10.tar.gz`、`tensorsharp-server-<version>-linux-arm64-cuda13-GB10.tar.gz`、`SHA256SUMS` 以及一份无头验证记录。归档启动检查会加载真实打包的 cuBLAS 与 OpenCV 绑定，并在调用各应用的 `--help` 之前通过 ImageMagick 编码一个像素——它检查的是依赖加载，而不是模型推理。

解压任一归档后运行 `./TensorSharp.Cli` 或 `./TensorSharp.Server.Host`，无需安装 .NET 或 CUDA Toolkit。在 Ubuntu 24.04 上需安装 `ca-certificates libgomp1 libgssapi-krb5-2 libicu74 libssl3t64 libstdc++6 zlib1g`；GPU 执行还需要兼容的 NVIDIA 驱动（构建机使用 CUDA 13.0.2）。请保留打包的许可声明，并遵守媒体依赖的再分发/源码义务。

若要把干净环境保留为镜像，把导出参数换成 `--target runtime -t tensorsharp-gb10-runtime`；解压后的应用位于 `/opt/tensorsharp/cli` 与 `/opt/tensorsharp/server`，默认命令为 Bash。后续做 GPU 检查时加 `--gpus all`；构建镜像本身从不需要 GPU。无头构建允许宿主缺少 `libcuda.so.1`，并会报告（但不失败）.NET 可选 LTTng 跟踪提供程序所需的 `liblttng-ust.so.0` 不可用，这与 [CoreCLR 的可选加载行为](https://github.com/dotnet/runtime/blob/v10.0.12/src/coreclr/pal/src/misc/tracepointprovider.cpp)一致；其他原生依赖都是必需的。安排好 Spark 的可用时间后，可运行要求驱动的复查：

```bash
docker run --rm --gpus all --network none \
  -v "$PWD/artifacts/gb10:/archives:ro" tensorsharp-gb10-runtime \
  bash /validation/verify-gb10-release.sh /archives /tmp/gb10-check --require-driver
```

默认的 Docker target 仍是开发镜像；现有的 x64/macOS 发布工作流及其 CUDA 版本不变。

托管的 GB10 构建使用全新的 `docker-container` BuildKit worker，而不是把开发镜像加载进 runner 的 Docker 镜像存储。上游整合之前的完整冷启动归档构建在**硬性 10 GiB 构建存储上限**下验证通过：文件系统峰值占用 **8,970,973,184 字节**，导出占用 **1,313,840,034 字节**，在四个 CPU 上耗时 **691.7 秒**。CI 为 Docker 预留 10 GiB、为导出预留 2 GiB（二者共用文件系统时共 12 GiB），空间不足时明确失败；它不会删除无关工具，也不会回退到自托管 Spark。worker 内存上限为 8 GiB；托管 job 的实际执行仍需在工作流进入 GitHub 后另行检查。如需在 fork 或分支上做 CI 预演，可手动触发 **Release Binaries** 并填写 `version` 与 `gb10_only=true`：这只运行 GB10 构建/验证/上传 job，不创建 GitHub Release，也不运行现有的平台矩阵。新建的 fork 需先在仓库的 Actions 标签页启用工作流；手动触发前，GitHub 需要先在 fork 的默认分支上登记该工作流，之后才能选择其他分支做测试运行。

#### 已验证的 GB10 配置与限制

以下是**2026-09-17**、上游重新整合之前核验的历史基线：NVIDIA GB10（compute capability 12.1）、Ubuntu 24.04.5 ARM64、NVIDIA 驱动 **580.178.04**、CUDA **13.0.2**、.NET SDK **10.0.401**。干净运行时不含 SDK 或 CUDA Toolkit。两个解压后的 apphost 都通过了原生依赖检查，并在 `ggml_cuda` 上生成了文本，包括服务端的 OpenAI 兼容聊天端点。

那次正确性验证运行了 **3,908 个 CPU 分组测试**、**129 个托管 CUDA 测试**与 **3 个原生 CUDA 测试**，全部通过且无跳过。原生检查覆盖显式 F32 matmul 精度、稀疏 flash attention 与激活量化。CPU 分组使用单独构建的纯 CPU GGML 库；其中现有的 shell 测试需要带 pip/venv 的 Python 与 Node.js，且有一个 pip 安装回归测试会访问软件包索引，因此该分组不是离线测试套件。

冒烟模型为 [`ISTA-DASLab/Qwen3.5-4B-GGUF-GSQ`](https://huggingface.co/ISTA-DASLab/Qwen3.5-4B-GGUF-GSQ) 的 `Qwen3.5-4B-Q2_K_XL.gguf`，修订版 `3fd6825d7b0f014adb03d3981074a43753c0b8be`，SHA-256 为 `b7d9ec51fb4d726d31e6bfd47062fb74574a974b93951da6a435b5e35185cb5a`。GSQ 沿用标准 GGUF K-Quant 编码，没有新增模型专用加载器。

五次隔离的 CLI 运行使用内置 prompt 与 `--max-tokens 16 --temperature 0 --seed 123 --warmup-runs 1`，限制为四个 CPU 与 16 GiB 容器内存。每次运行在四 token 预热后重置 KV 缓存，测量 19 个输入 / 16 个输出 token。prefill 中位数为 **440.4 tokens/s**（438.5–451.7），decode 中位数为 **76.9 tokens/s**（76.3–77.5）。容器 `memory.peak` 最大为 **737,058,816 字节**（约 703 MiB）；该 cgroup 计数**不是** GPU/统一内存的总用量，运行时另外报告了 1,836 MB 驻留设备的量化权重，二者不可相加。这些短暂的热运行只是可复现性基线，不是长上下文基准，也不构成 CPU 对 GPU 的加速结论。

这些检查只覆盖单设备 GB10 上的 Qwen 文本冒烟与上面列出的 kernel/媒体契约，并非全部模型家族。GB10 上的 CUDA 12、其他 ARM64 NVIDIA 产品、多 GPU/多 Spark 运行、Vulkan/MLX、可选的 cuDNN 加速以及完整的图像/视频生成都不在验证范围内。GB10 上可能出现通用的集成 GPU 性能警告，它并不是本机与 CPU 的实测对比。做硬件检查前请明确预留 Spark；常规 CI 不得占用正在运行的 vLLM 实例的资源。上游重新整合改变了推理与精度实现，因此这些历史硬件计数与耗时不能为重新整合后的代码背书；托管 CI 会重新检查当前的 CPU 与归档路径，硬件资格验证需重做后才能沿用这些测量结果。


## 项目结构

```text
TensorSharp/
├── TensorSharp.Core/            # 核心张量库（Tensor、Ops、内存、设备抽象，含 CPU SIMD/托管量化内核）
├── TensorSharp.Runtime/         # GGUF、分词器、模板、采样、协议解析
│   ├── Paged/                   # 分页 KV 缓存原语（BlockPool、BlockTable、KvBlock、BlockHashIndex、PagedKvStorage、PagedKvBatchOps、ManagedPagedAttention）
│   ├── Scheduling/              # 连续批处理引擎（InferenceEngine、BatchExecutor、ContinuousBatchScheduler、SequenceState、SchedulerConfig/Output、InferenceRequestHandle）；PrefixCache/ 是默认启用的基数树前缀缓存（PrefixTree、PrefixCacheCoordinator、IPrefixCacheModel 契约），PrefixCheckpointFileStore 负责持久化前缀检查点
│   ├── Speculative/             # 投机解码：起草/验证/回滚核心（SpeculativeExecution）、ISpeculator 各算法（DraftHeadSpeculator、BlockDraftSpeculator、NGramSpeculator）与 SpeculatorRegistry、模型侧契约（ISpecTrunk、SpeculativeModelContracts）、共用的参数解析（SpeculativeCliFlags、SpeculationOptions）以及运行期成本裁判
│   ├── PagedKvCacheManager.cs   # 独立的单会话分页 KV 管理器（块分配、前缀复用、RAM / SSD / Redis 分层）；只有 CLI 的 --paged-bench 会用到它
│   ├── PagedKvBlockStore.cs     # 带可选 SSD 溢出的 RAM/磁盘分级分页块存储
│   ├── SsdKvBlockTier.cs        # 分页块的 SSD 冷层
│   ├── TurboQuantKvCodec.cs     # 实现 IKvBlockCodec 的量化 KV 块编解码器（2-bit / Q4 / Q8）
│   ├── PrefillChunking.cs       # SWA / 超长 prompt 使用的分块 prefill 辅助
│   ├── KvBlockHash.cs           # 内容寻址的块哈希，用于跨请求前缀复用
│   └── Logging/                 # JSON-line 文件日志器 + 每轮遥测
├── TensorSharp.AgentHost/       # 构建在运行时之上的智能体层：Agent Skills、代码执行与子智能体委派
│   ├── Agents/                  # 子智能体：请求级限额（MultiAgentOptions）、spawn_agent / wait_agent / send_input / close_agent / list_agents 五个工具声明（MultiAgentTools）、请求自有的智能体树及其角色、只读工具门控与串行化的宿主工具（MultiAgentSession）、协调提示词（MultiAgentPrompt）、宿主在 prefill 之前据以设置检查点的子智能体提示词画像（MultiAgentPromptProfile），以及 Web UI 展示的进度快照（MultiAgentProgress）；详见 docs/multi_agent.md
│   ├── Skills/                  # Agent Skills：SKILL.md frontmatter 解析（YamlFrontmatter、SkillManifest）、未配置技能根目录时使用的仓库技能根——从工作目录向上直到 Git 根目录的每个已存在的 .agents/skills（由近及远），排在 <程序目录>/skills 之前（SkillDiscovery）、发现 / 安装 / 查找（SkillRegistry、SkillArchive）、目录边界约束（SkillPathGuard）、提示词规划（SkillPrompt）、内置的 skills_list / skills_read / skills_run 工具与进程内披露循环（SkillTools、SkillAgentLoop、SkillScriptRunner）、共用的参数解析（SkillHostOptions）以及对外客户端（SkillsChatClient），以及各平台沙箱（SkillSandbox、SkillSandboxWindows）及其违规监视器、会话级工作区（SessionWorkspace）
│   └── CodeExec/                # 文件工具（read_file、write_file）、shell 与 apply_patch：执行引擎（ShellRunner）、提供给模型的四个工具声明——read_file、apply_patch、write_file（仅用于新建文件）与 shell——以及小模型所需的宽松参数读取（ShellTools）、command 参数的解读——把一行命令拆成各个简单命令、判定其中哪些是软件包安装（这决定了这一行到底能不能拿到套接字）、并在 shell 看到之前拦截 apply_patch heredoc（ShellCommand）、会话级的工作目录与导出环境变量——因为没有常驻的 shell 进程，它们通过文件持久化（ShellSession）、shell 的发现与方言（ShellProgram）、补丁信封解析（CodePatch）及其匹配引擎——对参考实现 V4A applier 的逐行移植（V4ADiff）、把一次失败改写成下一条该敲的命令（CodeDiagnostics）、当一次运行是因为模型猜错了库的 API 而失败时，直接从已安装的包里读出真实 API（ApiProbe）、检查命令写入或补丁改动后的文件是否仍能解析（SyntaxCheck）、把宿主的绝对路径从命令的全部输出中改写掉（OutputPaths）、字符串替换编辑器及其容错阶梯——已不再向模型提供，仅为让旧客户端的编辑调用仍能分派而保留（FileEdit）、所有工具与拒绝信息展示文件时共用的带行号清单渲染器（NumberedListing）、注入系统提示词的八条编辑规则（CodePrompt）、发现「为改两行而重打整个文件」的行为（RewriteWatch）、为 import 失败的技能脚本由宿主发起的安装（PackageInstaller）、宿主侧的执行条款（CodeExecOptions）、受限启动——它也可以只启动进程而不等它结束，后台任务即由此实现（ConfinedProcess）、解释器发现（CodeEnvironment）、产物捕获（CodeArtifactStore）、安装期的软件源代理（EgressProxy）、结果记录（CodeExecResult），以及技能层所见的 ICodeRunner 接缝（CodeRunnerAdapter）
├── TensorSharp.Models/          # 模型架构实现与多模态编码/注入
│   ├── Models/<Family>/         # 每个架构一个目录（DeepSeek4、DiffusionGemma、Gemma4、GlmDsa、GptOss、HunyuanDense、MiniMaxH3、Mistral3、MuseGlimmer、Nemotron、Qwen3、Qwen35、Qwen4Exp、QwenImage、WanVideo）
│   │   ├── <Family>Architecture.cs         # 在 Architecture/BuiltInArchitectures.cs 中登记的插件描述符
│   │   ├── <Family>Model.cs                # 旧的单序列 ModelBase 实现
│   │   └── <Family>Model.BatchedForward.cs # IBatchedPagedModel.ForwardBatch —— 批处理/分页路径（Mistral3、Gemma4、GptOss、HunyuanDense、Qwen3、Qwen35、Nemotron）
│   ├── Models/DeepSeek4/        # DeepSeek V4 Flash 与 V4.1 Flash（DeepSeek41Architecture / DeepSeek41Model，含其视觉伴随模块）：使用整模型执行器而非逐算子前向
│   │   ├── DeepSeek4Model.cs               # GGUF 元数据、分词器、聊天模板、执行器选择
│   │   ├── DeepSeek4CudaExecutor.cs        # 对接 Direct CUDA 整模型引擎
│   │   ├── DeepSeek4CpuExecutor*.cs        # 100% 纯 C# 整模型执行器（零原生依赖）
│   │   ├── DeepSeek4Model.Dspark.cs        # DSpark 块级草稿器（draft / 置信度 / Markov 头）
│   │   └── DeepSeek4Model.PerSeqCache.cs   # 让该模型可被服务端托管的原生 per-sequence slot
│   ├── Models/GlmDsa/           # GLM 5.x：原生执行器驱动、MLA + DSA indexer 逐算子参考实现、序列 slot、NextN/MTP 草稿头
│   ├── Models/MuseGlimmer/      # Muse-Glimmer：融合整模型前向、视觉编码器、张量并行变体、DFlash 块级草稿器
│   ├── Models/MiniMaxH3/        # MiniMax-H3 视频 + 联合 32 kHz 立体声音频：打包序列 DiT、Qwen3-VL 文本编码器与视觉塔、视频与音频 VAE、flow-match 调度器、pipeline
│   ├── Models/WanVideo/         # Wan 2.1/2.2，仅视频：DiT、UMT5-XXL 文本编码器、因果 3D VAE、UniPC 调度器，以及不依赖 ggml 的 WanDirect* `cuda`/`cpu` 路径
│   ├── Models/Video/            # 两个视频家族共同实现的接缝：IVideoGenerationModel、VideoGenerationParams/Progress、GeneratedVideoAudio、WAV 写出
│   ├── Paged/                   # 张量侧的分页注意力辅助（TensorPagedAttention）
│   ├── KvBlockTransfer.cs       # 跨序列的 KV 块 extract/inject 辅助
│   ├── SpeculativeDecoder.cs    # 独立的单序列投机生成循环（CLI、测试与离线调用方），包裹共享的 SpeculativeExecution 核心；与算法无关（NextN/MTP 头、DSpark/DFlash 块级草稿器、n-gram）
│   ├── SpeculativeDraftHeadLoader.cs # 把独立的 --draft-model GGUF（Gemma 4 gemma4-assistant、DFlash/DFlash2、Qwen 3.8 Flash Next 共享 MTP 头）挂到主干上；在模型构造期间加载的块级草稿器（DSpark）视为已挂载
│   └── ModelMultimodalInjector.cs # 视觉 / 音频 / 视频嵌入注入
├── TensorSharp.Backends.GGML/   # GGML 后端绑定（通过原生库支持 Metal/CUDA/Vulkan/CPU）
├── TensorSharp.Backends.Cuda/   # Direct CUDA 后端（CUDA Driver API、cuBLAS、PTX 内核）
│   └── Dsv4/                    # DeepSeek V4 Direct CUDA 整模型引擎（不依赖 ggml）：GGUF→显存流式加载器、按设备权重竞技场、层切分、DSpark 草稿器
├── TensorSharp.Backends.MLX/    # Apple Silicon MLX 后端（mlx-c / Metal），原生桥接由 `build-native-macos.sh` 编译
├── TensorSharp.GGML.Native/     # 到 ggml 的原生 C++ 桥接（构建 libGgmlOps，拆分为多个专注源文件）
│   ├── ggml_ops_core.cpp                  # 元素级、归约、基础 shape 操作
│   ├── ggml_ops_elementwise.cpp           # 元素级 / 激活融合
│   ├── ggml_ops_matmul.cpp                # GEMM / 量化 matmul
│   ├── ggml_ops_{attention,matmul,q8}_precision.* # TensorSharp 自有的显式 F32 注意力、matmul 与 Q8_0 matmul 内核（ggml 本身保持原样）
│   ├── ggml_ops_fused.cpp                 # 跨域融合的每层内核
│   ├── ggml_ops_norm_attn.cpp             # Norm + 注意力融合
│   ├── ggml_ops_flash_attn_guard.cpp      # Flash attention；后端对该形状没有 kernel 时改为显式注意力并警告一次
│   ├── ggml_ops_transformer.cpp           # 通用融合 Transformer 层/整模型 decode 与 flash-attn decode
│   ├── ggml_ops_transformer_common.h      # 共享的 Transformer 辅助函数与 C# 层描述符结构体
│   ├── ggml_ops_transformer_prefill.cpp   # 融合层 prefill（Gemma 4、GPT-OSS、Qwen 3.5）
│   ├── ggml_ops_qwen3_decode.cpp / _prefill.cpp # Qwen 3（`qwen3` 描述符）整模型 decode 与 prompt prefill 计算图
│   ├── ggml_ops_qwen35_decode.cpp         # Qwen 3.5/3.6 融合 decode（单层、整模型、批量）
│   ├── ggml_ops_qwen35_verify.cpp         # Qwen 3.5/3.6 融合多 token verify
│   ├── ggml_ops_qwen35_gdn_tp.cpp         # Qwen 3.5/3.6 按 rank 的打包 GatedDeltaNet 内核（张量并行）
│   ├── ggml_ops_qwen35_recurrent_prefill.cpp # Qwen 3.5/3.6 递归层 prefill
│   ├── ggml_ops_qwen4exp*.cpp             # Qwen 3.8 Flash Next（`qwen4exp`）：融合的块级与 token 区间计算图及按序列的状态快照、按 token 批量 decode 的槽位稳定 arena，以及共享 MTP 头
│   ├── ggml_ops_gptoss_decode.cpp         # GPT OSS 整模型 decode 计算图（每 token 一次调度，共享 KV 窗口）
│   ├── ggml_ops_gptoss_prefill.cpp        # GPT OSS 整模型 prefill：N 个 token 走完全部注意力 + MoE 层，连同折叠的最终 norm 与 LM head 合为一张图
│   ├── ggml_ops_deepseek4.cpp             # DeepSeek V4 原生整模型执行器（层切分、压缩 KV 缓存、计算图缓存）
│   ├── ggml_ops_deepseek41_tp.cpp / _vision.cpp # DeepSeek V4.1 路由 MoE 张量并行（按 rank 切分专家权重条带）及其视觉伴随编码器
│   ├── ggml_ops_glm_dsa.cpp               # GLM 5.x 原生整模型执行器（MLA + DSA indexer、张量并行、序列 slot、NextN/MTP verify + 草稿图）
│   ├── ggml_ops_muse_glimmer.cpp          # Muse-Glimmer 整模型前向：decode 用持久图（供 ggml-cuda 捕获 CUDA 图）、prefill 用瞬时图，另含张量并行图
│   ├── ggml_ops_muse_glimmer_vision.cpp   # Muse-Glimmer ViT 块的设备端实现（最大尺寸图像有 16,224 个 patch，逐算子派发会让每步都经宿主同步）
│   ├── ggml_ops_dflash.cpp                # Muse-Glimmer DFlash 块级草稿器，每个投机步融合为一张图（草稿块 + 借用主干的 LM head）
│   ├── ggml_ops_dsv4_fused.cu / _cpu.cpp  # DeepSeek V4 在 ggml-cuda 流上的融合自定义算子（及其 CPU 版本）
│   ├── ggml_ops_gemma4_decode.cpp         # Gemma 4 稠密整模型 decode（CUDA graph 持久化）
│   ├── ggml_ops_gemma4_batched.cpp        # Gemma 4 稠密 + MoE 按 token 批量 decode
│   ├── ggml_ops_gemma4_verify.cpp         # Gemma 4 稠密 verify + MTP 草稿步
│   ├── ggml_ops_gemma4_moe.cpp            # Gemma 4 MoE 层/整模型 decode 与 verify
│   ├── ggml_ops_moe.cpp                   # 专家混合前向 / 融合路由
│   ├── ggml_ops_gated_delta_net.cpp       # Qwen 3.5/3.6 GatedDeltaNet 内核（按序列 + 批处理）
│   ├── ggml_ops_mamba2.cpp                # Nemotron Mamba2 内核（按序列 + 批处理 SIMD）
│   ├── ggml_ops_bonsai.cpp                # Bonsai2 PRISM 带符号 Hadamard（FWHT）变换，包裹其 matmul 与按行取数
│   ├── ggml_ops_paged_attention.cpp       # 分页注意力原生内核（驱动 ggml_flash_attn_ext + sinks 变体）
│   ├── ggml_ops_tensor_parallel.cpp       # 多 rank TP 组、分段融合计算图执行、集合通信
│   ├── ggml_ops_tp_probe.cu               # 选择 TP 传输方式的 peer-copy / NCCL AllReduce 预检
│   ├── ggml_ops_diffusion.cpp             # DiffusionGemma 融合 decode-layer / 整模型 / lm-head 内核
│   ├── ggml_ops_qwen_image.cpp            # Qwen-Image 共享内核：融合的 Qwen3-VL 文本编码器主干、整 VAE 计算图、VAE 注意力与 2D 卷积入口
│   ├── ggml_ops_qwen_image21.cpp          # Qwen-Image-2.1 扩散 Transformer 整图（在多次预测间保留并重放）
│   ├── ggml_ops_minimax_h3.cpp            # MiniMax-H3 整网络计算图：音视频打包的 DiT、Qwen3-VL 文本编码器与视觉塔、视频 / 音频 VAE 的编码与解码（七个入口，权重直接从调用方 mmap 常驻绑定）
│   ├── ggml_ops_wan.cpp                   # Wan 2.1/2.2 整图入口：UMT5-XXL 文本编码器、每步 DiT 速度预测（按 shape 持久化以便捕获 CUDA 图）、因果 3D 视频 VAE 编码与解码
│   ├── ggml_ops_embeddings.cpp            # BERT / XLM-RoBERTa 句向量编码器计算图（GGUF `bert`）
│   ├── ggml_ops_training.cpp              # 仅训练用内核（运行时不使用）
│   └── tests/                              # 原生单元 + 烟雾测试
├── TensorSharp.Chat/            # 与宿主无关的聊天流水线，由 Server、CLI 与 iOS 应用共用（不依赖 ASP.NET Core 与 Distributed）
│   ├── ModelService.cs          # 模型加载/释放、InferenceEngineHost 与生成流水线之上的门面（TensorParallelGroupFactory、SchedulerConfigOverride、UnloadModel）
│   ├── ModelLifecycleService.cs # 模型加载/释放与后端选择；张量并行组由宿主传入
│   ├── InferenceEngineHost.cs   # 单模型 InferenceEngine 单例（连续批处理入口）
│   ├── ChatGenerationPipeline.cs # Prompt 渲染，将请求提交到 InferenceEngine，流式返回 token，处理 stop
│   ├── DiffusionBatchScheduler.cs # DiffusionGemma 连续批处理：所有进行中的聊天请求一起去噪，在 block 边界加入与退出
│   ├── InferenceTelemetry.cs    # Prompt/eval 计时、TTFT、tokens/sec、有界输入摘要与输出日志
│   ├── ChatHistoryPreparer.cs   # 历史归一化、raw token 拼接、多模态顺序辅助
│   ├── ChatSession.cs / SessionManager.cs # 单会话历史跟踪与线程安全的会话注册
│   ├── ConversationTranscriptStore.cs # 本服务生成的 assistant 轮次的原始输出 token，使同一对话的下一个请求渲染出 KV 缓存中实际保存的内容
│   ├── InferenceQueue.cs        # 已弃用的空操作队列垫片，仍回答队列状态（并发由引擎本身处理）
│   ├── ModelService.Agents.cs   # 流水线之上的子智能体委派：每个子智能体一个全新的 ChatSession，父智能体渲染好的子智能体画像作为公共前缀检查点提供
│   ├── WebUiChatService.cs      # 去掉传输层的 Web UI 请求/流式契约（会话、模型、上传、图像生成与编辑、视频、聊天帧）
│   ├── WebUiChatPolicy.cs       # Web UI 聊天请求合法性校验（模型经 /api/models/load 选择，不能按消息切换）
│   ├── TextUploadHelper.cs      # 无损文本上传归一化辅助（上传时不设字符或 token 预算）
│   ├── SkillsService.cs         # 与传输无关的 /api/skills 与 /v1/skills 管理接口
│   ├── BackendCatalog.cs        # 后端词汇与规范名称；可用性探测留在 Server
│   ├── Jev/                     # Jev（POST /v1/systemone）：基于 DiffusionGemma 的单步结构化读取，可带图像输入
│   ├── Skills/                  # SkillRequestPlan、SkillChatLoop、RequestWorkspaceLease：流水线之上的进程内披露循环，也负责提供子智能体工具
│   ├── Hosting/                 # ServerHostingOptions、采样默认值、上传存储/内容策略、托管模型守卫、启动加载器
│   ├── RequestParsers/          # JSON 请求解析（聊天消息、采样、工具函数、技能、视频参数、上传引用）
│   ├── ResponseSerializers/     # WebUiSseEvents（Web UI 帧形状）+ 共用 JSON 选项
│   └── ProtocolAdapters/        # ChatStreamCollector + FinishReasonMapper（流水线词汇 -> 协议词汇）
├── TensorSharp.Server/          # Web 聊天 + API 服务库（ASP.NET Core）；可运行的应用是 TensorSharp.Server.Host
│   ├── BackendCatalogProbes.cs  # TensorSharp.Chat 的 BackendCatalog 背后的 CUDA / MLX / GGML 可用性探测
│   ├── OpenAIResponseFormatParser.cs  # OpenAI response_format（json_object / json_schema）解析
│   ├── Hosting/                 # 启动期相关：选项装配（ServerOptionsBuilder，也负责解析子智能体限额）、wwwroot 解析、/uploads 静态文件与清理、embedding 托管、多节点张量并行工厂、paged-KV / 连续批处理 CLI 翻译
│   ├── ResponseSerializers/     # 各协议响应形状构造（Ollama / OpenAI）
│   ├── Responses/               # 存储的 /v1/responses 记录（内存，或经 --redis-url 存到 Redis）
│   ├── StreamingWriters/        # SSE 与 NDJSON 线协议辅助
│   ├── ProtocolAdapters/        # 各协议的请求处理器（OllamaAdapter、OpenAIChatAdapter、OpenAIResponsesAdapter、JevAdapter、EmbeddingAdapter；WebUiAdapter 与 SkillsAdapter 是 TensorSharp.Chat 之上的薄 HTTP 外壳）
│   ├── Endpoints/               # ASP.NET Core 路由映射（每协议一个扩展方法）
│   └── Logging/                 # 请求日志中间件 + 低噪声路径支持
├── TensorSharp.Server.Host/     # 可运行的 Web 应用
│   ├── Program.cs               # 精简启动：配置文件展开、DI 注册、中间件、端点映射、paged-KV + 连续批处理 CLI 翻译
│   ├── Hosting/                 # --help 文本（ServerUsage）、启动横幅、日志设置
│   ├── wwwroot/index.html       # 聊天界面
│   ├── testdata/                # 集成测试套件（bash + Python）
│   ├── Dockers/                 # Hugging Face Space 的 Dockerfile（CPU 与 GPU）及其 README
│   └── API_EXAMPLES.md          # 详细 API 文档
├── TensorSharp.Cli/             # CLI 应用（单次生成、交互式 REPL、JSONL 批处理、基准）
├── TensorSharp.TestMatrix/      # 测试 / 基准矩阵运行器、默认提示、环境变量扫描与主机基线
├── TensorAgent/                 # iPhone / iPad 应用：自带适合手机的聊天页面，与 Server 的 Web UI 绑定同一套 WebUiChatService / SkillsService API，完全在设备上运行
│   ├── src/TensorAgent.Core/    # 与平台无关：模型目录与存储、可续传下载、已保存的对话、设置、回环服务器及其路由表，以及把这些组装起来的 AgentAppHost（放在这里而不是 iOS 头工程中，以便测试能启动它、通过 HTTP 驱动它并将其拆除）
│   │   ├── Catalog/ Downloads/ Sessions/ Settings/ # ModelCatalog + ModelStore、ModelDownloadManager + ResumableDownloader、ChatTurnManager + ConversationStore、AppSettings
│   │   ├── Hosting/             # AgentAppHost、LoopbackServer + WebUiRoutes、EngineMemoryPolicy、SpeculationPolicy、TensorAgentSkillRouter、ProcessMemory
│   │   ├── Sharing/             # 接收分享扩展交来的内容
│   │   ├── Interop/             # 原生库解析器
│   │   ├── Shell/               # 进程内 POSIX shell——管道、重定向、heredoc、glob、函数，以及编程模型常用的 coreutils（含 awk）——外加让智能体宿主在不支持 Process.Start 的平台上运行代码的 IShellBackend，以及宿主运行在 macOS、Linux 或 Windows 时借助操作系统沙箱运行原生进程的 DesktopShellBackend
│   │   ├── Sandbox/             # ExecutionPolicy 与 ConfinedPaths：shell、Python 与 JavaScript 共同遵守的一套规则，因为没有操作系统沙箱可依赖
│   │   ├── Python/              # 通过 P/Invoke 嵌入的 CPython 3.13、其 audit-hook 沙箱，以及纯 wheel 安装器
│   │   ├── JavaScript/          # 基于 C API 的 JavaScriptCore，提供 Node 风格的 console/process/require/fs/timers
│   │   └── WebUi/               # tensoragent.js：回环服务器追加到应用自有页面上的脚本
│   ├── src/TensorAgent.Maui/    # net10.0-ios 头工程：手机页面（wwwroot/index.html）、WebView + 附件 + 听写、模型 / 对话 / 设置页面、后台与分享收件箱处理，以及本设备上各类文件的存放位置
│   ├── src/TensorAgent.Sharing/ # 应用与其扩展共用的分享信封契约
│   ├── src/TensorAgent.ShareExtension/ # iOS 分享扩展“Ask TensorAgent”
│   ├── tests/TensorAgent.Tests/ # 应用宿主测试（不在 PR CI 中运行）
│   ├── skills/                  # 12 个技能（也是桌面宿主的技能根目录）；verdicts.json 记录 verify-skills.py 对每个技能的静态检查结论（对照打包的解释器解析 Python import，检查 shell 脚本是否调用 npm/npx/pnpm/yarn/parcel/vite）。10 个通过并打包进 iOS 应用；playwright 与 web-artifacts-builder 未通过，由 TensorAgent.Maui.csproj 排除
│   └── scripts/                 # 模拟器的构建 / 运行 / 验证（build-sim.sh、run-sim.sh、verify-sim.sh）、真机（build-device.sh、deploy-device.sh、bench-spec-device.sh）、verify-background.sh（模拟器或真机）、verify-share-rule.sh、prepare-python.sh、build-lxml-ios.sh、verify-skills.py
├── InferenceWeb.Tests/          # xUnit 单元测试，覆盖算子、KV 缓存、分页调度器、批处理模型正确性以及 Web/服务辅助逻辑
├── AdvUtils/                    # 工具库（日志）
├── docs/                        # 开发者参考文档
│   ├── models/                  # 按模型架构卡片（每个模型一份 .md，中英双语）
│   ├── PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md  # 分页 KV 缓存、前缀共享、调度器、按模型批处理状态
│   ├── speculative_decoding.md  # 起草-验证设计：ISpeculativeTarget / ISpeculator / IDraftHead 三层，以及 draft-head、block 与 ngram 三种算法（英文）
│   ├── agent_skills.md          # Agent Skills：SKILL.md 格式、渐进式披露与其预算、进程内工具循环、路径 / ZIP / 脚本执行的安全模型，以及 HTTP 与 C# 两套接口（英文）
│   ├── multi_agent.md           # 子智能体委派：工具、角色与只读子智能体、限额、与父智能体共享前缀、宿主集成（英文）
│   ├── playwright_agent.md      # Playwright 浏览器技能：所需参数、macOS 沙箱配置、账号交接、已验证的宿主（英文）
│   └── env_var_feature_matrix.md  # TestMatrix 使用的运行时开关 × 模型/后端/功能覆盖矩阵
├── benchmarks/                  # 可重现的基准脚本
└── ExternalProjects/            # ggml/ 在构建时从 github.com/ggml-org/ggml 克隆（不纳入版本控制）
```

## 项目 / NuGet 包分层

仓库按包边界拆成独立层，使用者可以只引用真正需要的部分。

**状态核验于 2026-09-25。** 发布集合共 **13** 个包，即下表的每一行。`eng/verify-packages.ps1` 是这份清单的权威来源，并且它是发布流水线的门禁：在两个可打包项目之间新增 `ProjectReference` 而不同步更新该脚本，会直接让发布失败。

[NuGet.org](https://www.nuget.org/profiles/TensorSharp) 上的现状：13 个包 ID 均已发布 **2026.9.1**，于 2026-09-17 由标签 `v2026.09.01` 发布。**3.4.0**（2026-09-12，标签 `v3.4.0.0`）是首个包含 `TensorSharp.Runtime.Logging`、`TensorSharp.AgentHost`、`TensorSharp.Chat`、`TensorSharp.Server.Host` 与 `TensorSharp.Distributed` 的版本。更早的 **3.1.2**（2026-07-21）只有 8 个包 ID——`TensorSharp.Tensors`、`TensorSharp.Runtime`、`TensorSharp.Models`、`TensorSharp.Backends.GGML`、`TensorSharp.Backends.Cuda`、`TensorSharp.Backends.MLX`、`TensorSharp.Server` 与 `TensorSharp.Cli`——且其中的 `TensorSharp.Server` 早于日志层与聊天层拆分，与这里描述的分层并不一致。每个包版本都是其标签时的快照；`v2026.09.01` 之后的源码改动要等下一个版本标签才会发布到 NuGet.org。

| 项目 | NuGet 包 | 对外 namespace | 职责 |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Tensors` | `TensorSharp` | Tensor 原语、Ops、分配器、存储与设备抽象 |
| `TensorSharp.Runtime.Logging` | `TensorSharp.Runtime.Logging` | `TensorSharp.Runtime.Logging` | 各宿主共用的日志抽象与输出目标；单独拆出后，引擎各层不必再背上宿主的日志栈 |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF 解析、分词器、Prompt 渲染、采样、输出协议解析、分页 KV 缓存、连续批处理调度器 |
| `TensorSharp.AgentHost` | `TensorSharp.AgentHost` | `TensorSharp.AgentHost` | Agent Skills、代码执行（`read_file` + `write_file` + `shell` + `apply_patch`，含操作系统级沙箱、Web/CLI 会话级与 HTTP 请求级工作区，以及由宿主判定的软件包安装），以及子智能体委派的基础组件（`MultiAgentSession`、`MultiAgentTools`、`MultiAgentOptions`）——构建在 `TensorSharp.Runtime` 之上 |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`、各模型架构、多模态编码器、批处理 / 分页前向、模型侧执行辅助 |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML 执行后端与原生互操作 |
| `TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` | `TensorSharp.Cuda` | Direct CUDA 分配器、存储、cuBLAS GEMM、PTX 内核和量化 CUDA 算子 |
| `TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` | `TensorSharp.MLX` | Apple Silicon MLX 后端（mlx-c / Metal），含量化 / 融合 / 编译内核与 MoE 专家 offload |
| `TensorSharp.Distributed` | `TensorSharp.Distributed` | `TensorSharp.Distributed` | 用于多节点张量并行的点对点 TCP 协调层 |
| `TensorSharp.Chat` | `TensorSharp.Chat` | `TensorSharp.Chat`（新类型）；迁入的流水线保留 `TensorSharp.Server.*` 命名空间 | 与宿主无关的聊天流水线：`ModelService`、会话、生成、技能循环与 Web UI 请求/流式契约（`WebUiChatService`、`SkillsService`）——不依赖 ASP.NET Core 与 `TensorSharp.Distributed`；由 Server、CLI 与 iOS 应用共用 |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core 服务、OpenAI/Ollama 适配层、TensorSharp.Chat 之上的 HTTP 传输层与 Web UI |
| `TensorSharp.Server.Host` | `TensorSharp.Server.Host` | `TensorSharp.Server.Host` | **可运行**的 Web 应用：`Program.cs`、宿主装配、`wwwroot/` 与命令行。`TensorSharp.Server` 是它依赖的库——单独构建或运行 `TensorSharp.Server` 不会产生任何可执行文件 |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | 控制台宿主、调试工具与 JSONL 批处理 |

这样的拆分让引擎使用者不必带上 Web 依赖，也能把 API 层改动和核心运行时隔离开，并让后续 benchmark / eval harness 更容易独立发布。

`TensorSharp.Chat` 是流水线与各宿主之间的接缝。原先位于 `TensorSharp.Server` 内、却不涉及 ASP.NET Core 的部分——`ModelService`、`ModelLifecycleService`、`InferenceEngineHost`、`ChatGenerationPipeline`、会话、技能循环、请求解析器与 Web UI 帧构造器——都原样迁到了这里（命名空间不变）；Web UI 的处理器则变成 `WebUiChatService` / `SkillsService`：输入输出都是 JSON 负载对象，拒绝以 `WebUiRequestRejectedException(StatusCode, Payload)` 抛出，流式输出是由帧组成的 `IAsyncEnumerable<object>`。Server 的 `WebUiAdapter` 与 `SkillsAdapter` 如今每个路由只有几行（读取 `HttpContext`、调用服务、写出状态码 + JSON 或 `data:` 事件），因此进程内的回环服务器或原生视图模型可以驱动完全相同的契约。该库刻意不引用两样东西：`TensorSharp.Distributed`（它引用 CUDA 后端，而 iOS 构建无法链接后者——Server 通过 `ModelService.TensorParallelGroupFactory` 把多节点张量并行传进来），以及 CUDA/MLX 后端探测（`BackendCatalogProbes` 留在 Server）。没有 shell 的宿主通过 `ModelService.SchedulerConfigOverride` 设定 KV 池大小，并通过 `ModelService.UnloadModel()` 在不释放服务的前提下卸载模型。上述任何一个依赖方向被反转，`ChatLayeringTests` 都会失败。

> **注意：** 核心层发布的包名是 **`TensorSharp.Tensors`**，不是 `TensorSharp.Core`。NuGet.org 上的 `TensorSharp.Core` 包 ID 属于另一个已废弃的无关项目（所有版本均已 unlist，源码仓库已删除），推送到该 ID 会返回 403。差异仅限于 NuGet 包名——项目名、程序集名与 `TensorSharp` namespace 均保持不变，因此 `using` 语句不受影响，只有 `dotnet add package` 命令行不同。

发布前可验证包元数据与 README 依赖边界：

```powershell
pwsh ./eng/verify-packages.ps1
```

该验证会对上表公开包运行 `dotnet pack`，并在 `AdvUtils` 等内部依赖泄漏到 `.nuspec`，或 TensorSharp 包依赖了上表之外的分层时失败。

### 发布包版本（维护者）

[`Publish NuGet`](.github/workflows/publish-nuget.yml) 工作流在推送版本标签时打包上表中的公开项目，并推送到 NuGet.org 与 GitHub Packages。这里描述的是发布流程，而不是当前包的可用情况：

```bash
git tag vX.Y.Z.W      # 标签决定包版本 X.Y.Z.W
git push origin vX.Y.Z.W
```

- 标签（去掉开头的 `v`）会覆盖每个包的 `TensorSharpVersion`，因此所有包都以同一个协调一致的版本发布，无需先修改 `Directory.Build.props`。
- 打包只包含托管代码——原生 GGML/CUDA/MLX 库不会嵌入包内——因此工作流在普通 runner 上以 `eng/verify-packages.ps1 -SkipNativeBuild` 运行（它同时设置 `TensorSharpSkipGgmlNative=true` / `TensorSharpSkipMlxNative=true`）。
- 发布到 NuGet.org 使用 [Trusted Publishing](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing)（OIDC），无需管理 API key 机密。`NuGet/login@v1` 把 job 的 GitHub OIDC 令牌换成一个有效期一小时的 key。nuget.org 上的策略固定了仓库所有者、仓库、工作流**文件名**（`publish-nuget.yml`）以及 `production` environment，因此重命名该工作流文件，或从 job 中去掉 `environment: production`，都会让发布失败，直到策略同步更新。
- 包是逐个推送而不是按通配符批量推送：被其他 NuGet.org 账号占用的包 ID 会返回 403，而 `--skip-duplicate` **不会**吸收这种错误，因此运行结果会准确列出哪些包被拒，而不会在第一个失败处停下。
- 包附带 Source Link 与配套的 `.snupkg` 符号包。`ContinuousIntegrationBuild` 只在 GitHub Actions 下设置，因此本地打包保留正常的源码路径。
- 如需预演而不发布，可手动触发该工作流（`workflow_dispatch`），填写 `version` 输入并勾选 `dry_run`——它会打包、校验，并把 `.nupkg` 文件作为构建产物上传，但不推送。

### 平台二进制发行状态

**状态核验于 2026-09-25**：最新发行版是 [v2026.09.01](https://github.com/zhongkaifu/TensorSharp/releases/tag/v2026.09.01)（2026-09-17），此前依次为 [v3.4.0.0](https://github.com/zhongkaifu/TensorSharp/releases/tag/v3.4.0.0)（2026-09-12）与 [v3.3.0.0](https://github.com/zhongkaifu/TensorSharp/releases/tag/v3.3.0.0)（2026-08-30）。每个发行版都发布服务端与 **TensorSharp.Cli** 的十个自包含应用归档，内含 .NET 10 运行时与原生库：下方五种平台/后端后缀各有一份 Server 归档和一份 CLI 归档。请到 [Releases 页面](https://github.com/zhongkaifu/TensorSharp/releases)检查更新版本，不要假定这份带日期的状态会永远代表最新发行。

已发布的归档矩阵（v3.3.0.0 至 v2026.09.01）如下：

| 归档后缀 | 内置的原生后端 | 格式 |
|---|---|---|
| `win-x64-cpu` | GGML CPU | `.zip` |
| `win-x64-cuda` | GGML CUDA + 纯 C# CUDA（PTX）+ CUDA 12.x 运行时 | `.zip` |
| `linux-x64-cpu` | GGML CPU | `.tar.gz` |
| `linux-x64-cuda` | GGML CUDA + 纯 C# CUDA（PTX）+ CUDA 12.x 运行时 | `.tar.gz` |
| `osx-arm64` | GGML Metal + MLX | `.tar.gz` |

另有实验性的 GB10 目标，通过 [GB10 Docker 流水线](#gb10--dgx-spark-构建容器实验性)生成 `linux-arm64-cuda13-GB10` 的 CLI 与 Server.Host 归档。目前还没有任何已发布的发行版包含它（v3.3.0.0 至 v2026.09.01 只有上面五种后缀）；请查看具体发行版的资产，不要假定其中包含它。

- 当前工作流以 `TensorSharp.Server.Host`（可运行的服务端）作为服务端应用发布，归档仍沿用 `tensorsharp-server-…` 的名称。v3.4.0.0 及之后的版本发布的是 `TensorSharp.Server.Host`；v3.3.0.0 早于 Server / Server.Host 拆分，因此其服务端归档包含的是 `TensorSharp.Server`。
- 推送 `v*` 标签会触发归档与 NuGet 工作流；只有所需 job 全部成功后才会发布产物。
- `-cuda` 归档已内置 CUDA 运行时库（`cudart` / `cublas` / `cublasLt`），但运行时仍需 NVIDIA GPU 与兼容驱动；`-cpu` 归档可在任意机器运行。macOS 归档需 Apple Silicon。
- 如需预演，可手动触发该工作流（`workflow_dispatch`）并填写 `version` 输入——它会构建全部平台并创建**草稿** Release。可用 `cuda_arch` 输入覆盖 CUDA 构建的目标 GPU 架构。


## 架构说明

TensorSharp 采用分层系统结构：

1. **TensorSharp.Core** 提供核心 `Tensor` 类型、存储抽象和可扩展的操作注册表（`Ops`）。CPU 实现使用 `System.Numerics.Vectors` 进行 SIMD 加速。

2. **TensorSharp.Runtime** 负责运行时契约与通用服务：GGUF 解析、分词（SentencePiece / BPE）、聊天模板渲染、可配置 token 采样、输出解析、分页 KV 缓存（`Runtime/Paged/*`）、连续批处理调度器 / 引擎（`Runtime/Scheduling/*`）、`IKvBlockCodec` 接口及其 `TurboQuantKvCodec` 2-bit / Q4 / Q8 实现，以及 `IModelArchitecture`、`IBatchedPagedModel`、`IPromptRenderer`、`IOutputProtocolParser`、`IMultimodalInjector`、`IBackendExecutionPlan` 等抽象。它刻意不包含智能体层：技能、代码执行与子智能体委派位于 **TensorSharp.AgentHost**，该项目引用运行时、且运行时绝不反向引用，因此只需提供 OpenAI / Ollama 聊天补全的宿主可以只依赖运行时，不携带技能注册表、沙箱与代码执行器。（若该方向被反转，`AgentHostLayeringTests` 会失败。）

3. **TensorSharp.Models** 实现 `ModelBase` 以及已登记的全部 16 个架构描述符（`Architecture/BuiltInArchitectures.cs`）和多模态辅助组件——13 个文本家族（DeepSeek V4 Flash、DeepSeek V4.1 Flash、GLM 5.x、Gemma 4、DiffusionGemma、Qwen 3 / Qwen 2 / Qwen 2.5-VL（仅文本）、Qwen 3.5/3.6 系列、Qwen 3.8 Flash Next、GPT OSS、Nemotron-H、Mistral 3、Hunyuan Dense、Muse-Glimmer）与 3 个媒体输出家族（Qwen-Image-2.1、MiniMax-H3、Wan 2.1/2.2）。自回归架构提供旧的单序列前向，多数架构还提供面向连续批处理的 `IBatchedPagedModel.ForwardBatch` 实现（`<Family>Model.BatchedForward.cs`）。DiffusionGemma 刻意不同：它不支持 `Forward()`，生成必须通过 `DiffusionGemmaSampler` 在固定长度 canvas 上迭代去噪。Qwen-Image-2.1（`QwenImageModel`）同样非自回归：`Forward()` 抛异常，图像生成与编辑通过 `GenerateImage()` / `EditImage()` 进行，由它们编排扩散 Transformer、专用 2.1 VAE 与 Qwen3-VL-8B 文本编码器。LoRA 插件（`--lora` / `--lora-scale` / `--lora-config`，由 `TensorSharp.Runtime/LoraSpec.cs` 为两个宿主统一解析）以不合并的方式作用于扩散 Transformer（`QwenImage21LoraSet`）；`config/lora/` 提供十二个预设。视频家族更进一步：`MiniMaxH3Model` 与 `WanVideoModel` 的 `ForwardCore()` 都直接抛异常，生成统一走 `GenerateVideo(prompt, VideoGenerationParams)`，其背后是 `Models/Video/` 里共享的 `IVideoGenerationModel` 接缝——CLI 与服务端因此只用一条路径驱动两者（以及日后新增的模型），而不必逐个判断具体模型类型。MiniMax-H3 在同一个打包 latent 里**同时**去噪视频与 32 kHz 立体声音频，共有七张原生整网络计算图（DiT、Qwen3-VL 文本编码器、视觉塔、视频与音频 VAE 的编码与解码）；Wan 2.1/2.2 则是仅视频的家族，其 DiT、UMT5-XXL 编码器与因果 3D VAE 同样以整图方式运行。模型通过 `ModelBase.Create()` 加载，并依据 GGUF 元数据自动识别架构——不带架构元数据的文件例外，它们依据张量识别：MiniMax-H3 公开发布的 GGUF 完全不带元数据（`LooksLikeMiniMaxH3`，经由 `MiniMaxH3Architecture.DetectFromTensors` 接入），不带元数据的 Qwen-Image-2.1 GGUF（例如 Unsloth 的 Q8_0）也是如此（`QwenImageArchitecture.DetectFromTensors`）。以通用 `llama` 标签发布的文件则由描述符的 `RecognizeRelabelledFile` 逐个文件认领，目前只有 Mistral 3 实现了它（例如标为 `llama` 的 Mistral Small 3.1）。

4. **TensorSharp.Backends.GGML** 通过原生 C++ 桥接库（`libGgmlOps` / `GgmlOps.dll`）注册同名操作的加速实现，并链接 [ggml](https://github.com/ggml-org/ggml)。在 macOS 上可提供 Metal GPU 计算，在 Windows/Linux 上可启用面向 NVIDIA GPU 的 GGML CUDA。除原生量化 matmul（Q4_K_M、Q8_0 等，无需反量化到 FP32）外，还提供分页注意力（`TSGgml_PagedAttentionForward`，含 / 不含注意力 sinks 两种版本）以及架构特定的批处理内核（Mamba2、GatedDeltaNet）。

5. **TensorSharp.Backends.Cuda** 是 Direct CUDA 路径。它使用 CUDA Driver API 管理设备、上下文与存储，用 cuBLAS 执行 Float32 GEMM，用 PTX 内核覆盖热点标量与 Transformer 辅助算子，并对尚未实现的原生内核使用 CPU 回退。

6. **TensorSharp.Backends.MLX** 是 Apple Silicon 上的 MLX 路径。它封装 [mlx-c](https://github.com/ml-explore/mlx-c)（`libmlxc`），提供分配器、存储、异步 worker 派发、量化 / 融合 / 编译内核、MoE 专家 offload，以及对未实现算子的 CPU 回退层。

7. **TensorSharp.Server** 是 HTTP / 应用层，提供兼容 Ollama 与 OpenAI 的 REST API、浏览器聊天 UI、上传处理、中间件与 SSE 写出。它所服务的流水线位于 **TensorSharp.Chat**——`ModelService`、持有自回归模型连续批处理引擎的 `InferenceEngineHost`、为所有协议（Web UI、OpenAI、Ollama）的 DiffusionGemma 聊天轮次去噪的 `DiffusionBatchScheduler`、会话、技能循环以及 Web UI 请求/流式契约（`WebUiChatService`、`SkillsService`）——因此 CLI 与 iOS 应用无需 ASP.NET Core 即可驱动同一套代码；旧的队列状态接口保留作为向后兼容。

8. **TensorSharp.Cli** 是控制台 / 应用层，用于本地 prompt 运行、多模态实验、prompt 检查、JSONL 批处理、交互式 REPL 与内置的 prefill / decode 基准。

### 新增模型、模态或对话格式

一个架构需要声明的全部信息都集中在三张表里：新增一个模型家族只需要改动它自己的
目录，外加每张表一行——加载器、调度规划器、CLI 与服务端都不需要动。

**1. 架构插件。** 在模型旁边写 `Models/<Family>/<Family>Architecture.cs`：

```csharp
internal static class MyFamilyArchitecture
{
    public static ModelArchitectureDescriptor Descriptor { get; } = new()
    {
        Id = "myfamily",
        DisplayName = "My Family",
        Aliases = new[] { "myfamily", "myfamily_moe" },   // general.architecture 取值
        Factory = c => new MyFamilyModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
        // 以下均可选，都有默认值：
        //   MultiGpu / MultiGpuLimitation   如何使用多卡，以及为什么不能张量并行
        //   ProjectorFileHints              自动发现的 mmproj 伴随文件名
        //   DetectFromTensors               针对不带架构元数据的 GGUF
        //   RecognizeRelabelledFile         认领被转换器标成通用架构的文件
        //                                   （需同时提供 RelabelledFileDescription，用于拒绝信息）
        //   ApplyNativeTunables             加载前需要设置的进程级 ggml 开关
    };
}
```

然后在 `Architecture/BuiltInArchitectures.cs` 里加一行。`ModelBase.Create` 通过
`ModelArchitectureRegistry` 解析，不再有 switch 需要扩展；而且
`ModelArchitectureDescriptor.Validate()` 会拒绝“声明了降级的多卡模式却不说明原因”
的描述符。

**2. 模态是能力接口，不是类型判断。** 能看图的模型实现 `IVisionCapableModel`
（加载视觉塔、接收一段 embedding）与 `IMultimodalPromptExpander`（展开自己的占位
符）；音频再加 `IAudioCapableModel` / `IAudioEncoderLoader`；按轴旋转位置编码再加
`IMRoPEPositionSink`。`ModelMultimodalInjector` 拥有全部通用逻辑——按请求分桶、
span 记账、前缀裁剪、截断、切片——并且不再出现任何模型类型名。CLI、交互式 REPL 与
服务端都驱动同一个 injector，因此一次接好的模态在所有入口都能用。

**3. 对话格式是一个 `ChatProtocol`。** 提示词框架、是否绕过 GGUF 自带的 Jinja
模板、媒体占位符 token、输出解析器、该解析器是否必须运行、结构化输出语法从何处开始
生效、KV cache 的生成后缀，以及视频抽帧上限——这些统一为 `ChatProtocolRegistry`
里的**一条**记录。它们过去分散为 `ChatTemplate`、`OutputParser`、
`KVCachePromptRenderer` 与服务端里大约二十多处按架构名的比较，漏掉任何一处都会静默
出错（未解析的回复会把推理标签当答案流式吐给客户端；缺失的媒体占位符会让图片被丢弃；
缺失的生成后缀会让多轮前缀复用率归零，而回答本身看起来仍然正确）。

运行期路由保持不变，仍由能力接口驱动：`ExecutionCapabilities.FromModel` 每步读取一次
`IBatchedPagedModel`、`ISpeculativeTarget` 等接口。上述三张表都不在逐 token 的热路径上。

### 性能优化

下表是跨架构汇总；[`docs/models/`](docs/models/README_zh-cn.md) 里每个模型卡片会在上下文中走一遍同样的内核，包含具体派发的 GGML 图与触发融合路径的条件。

- **融合 GPU decode**（Gemma 4）：在 Metal 上将所有 Transformer 层合并为单次 GGML 计算图调度，将每个 token 的 CPU-GPU 往返从数百次降低到一次。相较逐算子调度约提升 2.6 倍。
- **融合 GPU prefill**（Gemma 4）：对于密集（非 MoE、非 KV 共享、无 PLE/多模态）层，`Gemma4LayerPrefill` 将整个 Transformer 块（RMSNorm + QKV + QK-norm + RoPE + 注意力 + 输出投影 + post-attn norm + GeGLU FFN + post-FFN norm + 残差 + 层缩放因子）合并为 prefill 期间每层一次的 GGML 计算图调度，将融合方法从单 token decode 扩展到多 token prefill。
- **分块 prefill**（Gemma 4）：长提示被拆分为有界的分块（2 倍滑动窗口，最大 2048 tokens），以避免 SWA 层上 O(n²) 的注意力分数张量。分块在纯文本（无多模态嵌入）时自动应用，确保每个分块在 SWA 窗口预算内。
- **融合 Qwen 3.5/3.6-family attention 层 decode**：单次 GGML 计算图为每个 FullAttention 层完成 RMSNorm + 融合 QKV + Q/gate 反交错 + 每头 QK norm + RoPE + KV 缓存追加 + flash attention + sigmoid 门控混合 + 输出投影 + 残差加法。替换了原本每层 ~2 次独立 GGML 调用与 ~6 个小型 CPU/GPU 同步点。当缓存序列长度超过 4096 token 时启用（可通过 `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖）。
- **融合 prefill 注意力**（Qwen 3.5/3.6-family）：`FusedPrefillAttention` 将 Q*K^T、因果掩码、softmax 和 *V 合并为 prefill 期间每个注意力层一次的 GGML 计算图调度，消除了每个注意力层约 5 次独立的 C# 到 GGML 往返。同时支持初始 prefill 和带有已有 KV 缓存条目的续接。
- **整模型 Metal prefill 与 decode**（Qwen 3.5/3.6-family）：受支持的 dense 单设备模型会在一张 GGML 计算图内执行全部 attention 与 GatedDeltaNet 层、最终 RMSNorm 与 LM head。prefill 使用融合的多 token verify 图；decode 保留按序列的计算图，直接读取量化的 token embedding，把 Metal KV 拷贝视图限制在 64 token 的注意力桶内，并让计算图提交与 logits 回读重叠。
- **原地 Metal GatedDeltaNet 状态**（Qwen 3.5/3.6-family）：单 token decode 让每个递归层融合 GDN 的输出与其状态输入共用同一块内存，在 64 层的 Qwen 3.6-27B 上每 token 省去 48 次状态拷贝调度与约 302 MB 的状态读写流量。设置 `TS_QWEN35_METAL_GDN_INPLACE_STATE=0` 可保留独立拷贝路径用于诊断。
- **融合输出投影 + FFN**（Qwen 3.5/3.6-family）：对于 FullAttention 和 GatedDeltaNet 中的 dense FFN 层，`FusedOutProjFFN` 将输出投影、残差加法、post-attention RMSNorm 以及完整的 SwiGLU FFN（gate_up matmul + SiLU + down matmul + 残差加法）合并为单次 GGML 计算图调度，将每层 2 次 GPU 往返减少为 1 次。
- **融合输出投影 + 归一化 + 路由器**（Qwen 3.5/3.6-family MoE）：`FusedOutProjNormRouter` 将 GatedDeltaNet 输出投影、残差加法、post-attention RMSNorm 和 MoE 路由器投影合并为一次调度。预计算的路由器 logits 随后由批量 MoE 内核直接消费，消除了每个 MoE 层的独立路由器调度。
- **融合视觉编码器**（Qwen 3.5/3.6-family）：`FusedVisionAttention` 将 LayerNorm + QKV + 偏置 + 2D RoPE + 缩放点积注意力 + 输出投影 + 偏置 + 残差合并为一次 GGML 计算图调度（~8 个算子 → 1）。`FusedVisionMLP` 将 LayerNorm + up + 偏置 + GELU + down + 偏置 + 残差合并为一次调度（7 个算子 → 1）。两者结合将每个编码器块的 GPU 往返从约 15 次减少到 2 次。
- **融合权重投影**：同类型的 Q/K/V 投影融合为单次 QKV matmul；混合类型的 importance-matrix / UD 量化投影保持独立，以免产生数 GB 的 FP32 展开。gate 与 up 投影融合为单次 gate_up matmul。
- **原生量化计算**：量化权重（Q4_K_M、Q6_K、Q8_0、IQ2_XXS、MXFP4、NVFP4 等）直接参与 matmul，无需展开为 FP32，节省内存与带宽。批量 `AddmmQuantBatch` 内核可在一次调度内完成对同一量化权重块的多个子矩阵 matmul。
- **Direct CUDA 内核**：`cuda` 后端加速 fill/copy、unary ops、融合激活、RMSNorm、softmax、index select、因果掩码、RoPE/RoPEEx、cuBLAS GEMM，以及受支持的量化 matmul/get-rows；未覆盖算子会安全回退。
- **批量 GPU MoE**：`MoEExpertsSwiGLUResidual`（Qwen 3.5/3.6-family）和 `MoEExpertsForward`（Nemotron-H）将每个 MoE 层中所有被选中的专家——以及 Qwen 3.5/3.6-family 中可选的 shared expert 与残差加法——合并为一次 GGML 计算图调度。
- **整模型融合 decode 计算图**（Gemma 4 dense + MoE、Qwen 3.5/3.6、GPT OSS）：一个 decode token 的全部计算——每一层、MoE 路由与专家、最终 norm 与 LM head——作为**一次** GGML 计算图提交，而不是每层一次。在 CUDA/Vulkan 上该图只构建一次、张量地址保持稳定后反复重放（KV 写入用 `ggml_set_rows`、行号作为 I64 输入；注意力窗口按 stride 补齐、掩码作为 F16 输入），这正是 ggml-cuda 能把它捕获成 CUDA 图的前提。GPT OSS decode 在 A40 上从 24 → 154 tok/s，且随上下文长度基本持平（16K 时 133 tok/s，而逐层路径已跌到 2.3）。补齐的注意力窗口必须清零而不能留作未初始化——残留显存按 F16 解读会产生能穿过 `-inf` 掩码的 NaN。按模型的关闭开关：`TS_GPTOSS_MODEL_DECODE=0`、`TS_GEMMA4_FD_PERSIST=0`、`TS_QWEN35_FD_PERSIST=0`。
- **GLM 5.x 整模型执行器**：`glm-dsa` 是同样的形态。原生 ggml 执行器（`ggml_ops_glm_dsa.cpp`）自行加载分片 GGUF（GLM-5.2 为六个分片，GLM-5.3 的 UD-Q2_K_XL 为七个），并持有 MLA 缓存（每层每 token 一行 576 宽，逐 head 的 K/V 解压被折进 query 和输出）以及 DSA lightning indexer 缓存——78 层里只有 21 层会刷新它。它既可以按层切分到各张可见 GPU，也可以在 `--tp N` 下让每一层跑在每个 rank 上：注意力 head 按列/行并行，路由专家则在**每个专家内部**按行切开，因为 `ggml_mul_mat_id` 要求同一个 token 选中的专家 id 互不相同。并发靠原生序列 slot 而不是分页 KV，在其之上默认启用批量融合 decode（`TS_BATCHED_FUSED_DECODE=0` 可关闭）。`TensorSharp.Models/Models/GlmDsa/` 里是原生路径用来对照的托管逐算子参考实现。末尾的 NextN 块（`blk.78`）在 `--spec` 下驱动 [MTP 投机解码](docs/models/glm_zh-cn.md#nextn--mtp-投机解码)：主干图多输出一个 `h_nextn`，第二张图运行草稿块；该块只在请求了投机时才加载，因为它是一整层额外的解码层，会与 KV 缓存争用显存。
- **DeepSeek V4 整模型执行器**：`deepseek4` 完全绕开通用的逐算子前向。原生 ggml 执行器（`ggml_ops_deepseek4.cpp`）自行加载分片 GGUF，把权重按层切分到所有可见 GPU，在设备上持有全部 DSV4 KV 状态（原始 SWA 环、CSA/HCA 压缩 K 缓存、lightning indexer 缓存、压缩器状态环），并把每个 prefill/decode ubatch 作为一张 `ggml_backend_sched` 计算图执行，配合按形状签名的图缓存，使稳态 decode 直接重放已捕获的 CUDA 图。decode 注意力通过融合的 index-gather 算子取出紧凑的 `[ring | top-512]` K，而不是扫描整个上下文。Direct CUDA 引擎（`TensorSharp.Backends.Cuda/Dsv4/`）在不依赖 ggml 的前提下实现同一模型，把量化权重从分片直接流式写入按设备的显存竞技场。二者都构建在共享的 `Tensor` / `IAllocator` / `Ops` 之上；只有真正 DSV4 特有的计算才留在 DeepSeek V4 的文件里。
- **DSpark 块级投机解码**（DeepSeek V4）：独立的草稿 GGUF（`--draft-model`）每步提议一整块 token，主干用一次批量前向验证整块。在 ggml 上草稿器就是计算图里额外的三层，其 key ring 由主干图自己提交，因此投机不产生任何主机往返。4×A40 实测 decode 提速 1.3–1.4×（多轮对话最高 2.0×），贪心输出与非投机基线逐字节一致。
- **基于 GEMM 的视觉 patch embedding**（Qwen 3.5/3.6-family）：将 patch embedding 重构为并行 im2col + 矩阵乘法，把单线程标量五重嵌套循环替换为可在 GPU 上加速的 matmul。
- **并行化 Q/gate 反交错**（Qwen 3.5/3.6-family）：FullAttention prefill 中的 Q + sigmoid-gate 反交错按 token 并行化，长 prompt 时可随 CPU 核心数线性扩展。
- **优化后的纯 C# CPU 路径**：托管 GEMM 快速路径和连续 Float32 内核加速了 decode、softmax、RMSNorm、RoPE、融合激活等热点路径，同时在 CPU 加载时保持量化 GGUF 权重压缩状态。
- **环形 KV 缓存**：滑动窗口注意力层使用固定大小环形缓冲区，使内存占用不随序列长度增长。
- **KV 缓存前缀复用**：多轮对话会复用各轮之间最长的匹配 token 前缀。对 SWA 模型，截断会自动按滑动窗口大小回退，使后缀部分可以重建 SWA 上下文。
- **分页 KV 缓存 & 基数树前缀缓存**：连续批处理引擎把 KV 切分成固定大小的块，并默认通过基数树在并发 / 历史请求间复用 prompt 前缀（`TS_PREFIX_CACHE_MODE=tree`）；`TS_PREFIX_CACHE_MODE=legacy` 选择旧的“对写满的块做内容哈希后共享”方式，`--no-prefix-cache` / `TS_SCHED_PREFIX_CACHE=0` 在两种模式下都关闭复用。DiffusionGemma 与媒体模型不参与。尚未实现 `IBatchedPagedModel` 的模型仍会走同一引擎内隔离的按序列 KV-swap 回退路径。
- **原生分页注意力内核**：`TSGgml_PagedAttentionForward`（及面向 GPT OSS 的 `WithSinks` 变体）在 C++ 中按序列从分页缓冲区聚合 K/V，按序列构建小型 GGML 图，并派发 `ggml_flash_attn_ext`——也就是旧的单序列路径所使用的同一融合 GPU flash 注意力内核（Metal/CUDA/Vulkan）。在 Ministral-3-14B 长上下文（4×~800 tokens）上比旧的按序列 GGML 路径**快 ~21%**。
- **批处理 / 分页前向**：Mistral 3、Gemma 4、GPT OSS、Qwen 3、Hunyuan Dense、Qwen 3.5/3.6（含 GatedDeltaNet 递归状态池）、Nemotron-H（含 Mamba2 递归状态池 + 原生批处理 Mamba2 内核）把 N 个序列打包到一次 `ForwardBatch` 调用中，每层执行一次批处理线性投影 matmul，通过 `slotMapping` 写入分页 K/V，并通过原生内核做按序列注意力。Gemma 4 批处理路径在 batch=8 短 prompt 下达到 **1.5×** 旧吞吐，在 4×800-token prompt 下达到 **1.6×**；Nemotron-H Mamba2 批处理在 Apple M4 Pro 上 batch=3 时达到 **3.95×**。详见 [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。
- **MTP / NextN 投机解码**：单序列可运行多 token 预测草稿头（Qwen 3.6 与 Qwen 3.8-27B 内嵌 NextN 块；GLM 5.2 与 GLM-5.3 内嵌 NextN 块——只在不传 `--tp` 时可用，因为二者都不带自己的 head，draft 块借用按列切分的主干 LM head；Gemma 4 独立 `gemma4-assistant` 草稿 GGUF；Qwen 3.8 Flash Next 通过 `--draft-model` 加载共享 MTP 头 GGUF，仅限 GGML 后端）。草稿头最多提议 `--spec-draft` 个 token，主干用一次批量前向验证，二者均由该请求自己的采样器驱动，因此在不改变输出的前提下加速 decode。在 ggml 后端上，融合的单图多 token 验证与草稿步内核（`NativeGemma4ModelVerify` / `TryFusedMoEModelVerify` / `NativeGemma4DraftStep`，以及 Qwen 3.6 的 NextN 图）摊销了验证开销；Gemma 4 路径还增加了 gallocr 验证 scratch 以及部分接受时避免重跑已保留前缀的稠密快速回滚。纯 C# `cuda` 后端运行完全驻留 GPU 的逐算子验证 / 草稿（donor 缓存注意力、GQA decode 内核、GPU RoPE），使验证层循环零宿主端同步停顿。默认关闭；用 `--spec` 启用，或用 `--draft-model` 指定独立的草稿器 GGUF（Gemma 4 assistant、Qwen 3.8 Flash Next 的 MTP 头）——给出文件本身即可启用投机。
- **DiffusionGemma prompt-KV 缓存与融合去噪**：GPU 后端会在每个 block 中只对 `[prompt | canvas]` 的 prompt 部分预填充一次 K/V，并在去噪多步中复用；GGML 后端默认使用融合整模型 diffusion decode 与融合 lm-head tail。所有协议的并发 DiffusionGemma 聊天请求都经 `DiffusionBatchScheduler` 在 block 边界批处理；Web UI 另外流式显示去噪预览。
- **内核预热**：CLI 和 Server 在启动时运行一次微型前向传播，以预编译 GPU 内核（Metal pipeline state、CUDA JIT）并预热内存池，避免首次推理请求的冷启动延迟。
- **Prefill 缓存**（Gemma 4、Qwen 3.5/3.6-family）：逐 forward 传播的 SWA 掩码缓存（Gemma 4）、跨全局层的 NeoX RoPE cos/sin 查找表缓存（Gemma 4）、以及跨层的 RoPE 位置张量缓存（Gemma 4、Qwen 3.5/3.6-family），消除了 prefill 期间的冗余重复计算。
- **原地 QK RMSNorm**（Qwen 3.5/3.6-family）：逐头 QK 归一化通过 `View` 原地执行，避免了每层每个 Q/K 的一次张量分配与拷贝。

### 内存优化

- **零拷贝文件映射量化权重**（Direct CUDA、GGML CUDA、GGML Metal、GGML CPU）：GGUF 模型文件以内存映射方式打开，量化张量通过 host 指针缓冲区直接绑定到原生算子。这样省去了之前每张张量从磁盘复制到新分配原生堆缓冲区的过程——这一过程在 Apple Silicon 上会让大型量化模型的常驻内存几乎翻倍。例如，`Qwen3.5-35B-A3B-IQ2_XXS`（约 10 GB GGUF）在 Metal 后端的实际工作内存峰值从约 17 GB 降至约 7 GB。映射文件由操作系统的页缓存管理，必要时可换出，且在 Apple Silicon（统一内存）上不会带来推理性能损失。
- **最佳匹配内存池**：GGML 主机分配器使用 best-fit 而非 first-fit 在已池化块中检索可重用空间，避免把大块草稿内存交给小型中间张量请求，从而把工作集严格控制在合理范围内。
- **有界池保留量**：集成 GPU / CPU 内存池现在将单个保留块上限设为 64 MB，整池上限设为 32 块。结合 mmap 后的权重，可在快速复用短生命中间张量的同时限制峰值常驻内存。
- **高内存效率模型加载**：大张量直接流式加载到原生内存，避免中间托管分配。F32 权重与 norm 仍按需加载；量化权重在受支持的后端上通过 mmap 方式绑定。
- **分页 KV 块池；RAM / SSD 分层仅在独立管理器中**：服务端请求路径的活跃块由每个引擎的 `BlockPool` 统一管理，按 LRU 淘汰缓存的前缀块。分层的 `PagedKvBlockStore`（`TS_KV_CACHE_MAX_RAM_MB`、`TS_KV_CACHE_SSD_DIR`、`TS_KV_CACHE_MAX_SSD_MB`）属于独立的 `PagedKvCacheManager`，只有 CLI 的 `--paged-bench` 会用到它。服务端接受并记录 `--paged-kv*` 参数（以及 `--redis-url` 中 KV 的那一半），但它们不改变请求的服务方式。
- **KV 块编解码器**：在上述独立管理器中，`TurboQuantKvCodec`（2-bit 仿射、Q4 或 Q8）可通过 `--paged-kv-quant-bits` / `TS_KV_PAGED_QUANT_BITS` 压缩分页块，以精度换取更小的每块带宽与内存占用——大致减半（Q8）、减为四分之一（Q4）或约十分之一（2-bit，fp32 块）。2-bit 档位使用每组仿射 min+scale（即 llama.cpp Q2_K 背后的 block-min 思路），让四个码值覆盖该组的实际取值范围；它面向超长上下文的远端前缀复用，此时注意力权重远大于量化噪声。带递归状态的模型会自动回退到 passthrough。


## 测试

### 单元测试（xUnit）

`InferenceWeb.Tests` 覆盖无需启动服务的进程内行为：托管量化算子、可用 CUDA 设备上的 Direct CUDA 后端内核、可用 MLX 时的 MLX 后端内核、分页 KV 缓存调度（`ContinuousBatchSchedulerTests`、`PagedKvCacheTests`、`PagedKvCacheCodecTests`）、批处理执行器正确性（`BatchedExecutorTests`）、按模型批处理前向与旧路径的一致性（`Qwen35BatchedCorrectnessTests`、`Mistral3BatchedForwardTests`、`Gemma4BatchedForwardTests`、`GptOssBatchedCorrectnessTests`、`NemotronBatchedCorrectnessTests`）、MTP / NextN 投机解码正确性与可选端到端探针（`SpeculativeExecutionTests`、`Qwen36SpeculativeTests`、`Gemma4SpeculativeTests`）、DiffusionGemma 去噪 / prompt-KV / 批处理生成探针（`DiffusionGemmaTests`）、按模型批处理性能微基准（`*BatchedPerfBench.cs`）、`TurboQuantKvCodec` 编解码往返、prefill 分块、KV 缓存策略、KV 缓存 Prompt 渲染与多轮集成、聊天会话与 SessionManager 隔离、ModelService 历史跟踪、请求日志中间件与文件日志 Provider、图像预处理、媒体辅助逻辑、结构化输出校验、文本上传辅助、ModelService 上传日志、Web UI 聊天策略、模型上下文长度解析、可用后端发现，服务器 CLI 选项构造（`ServerOptionsBuilderTests`），Agent Skills —— `SKILL.md` frontmatter 解析及其各类告警情形（`SkillManifestParserTests`），与技能注册表的发现、优先级、ZIP 安装防护和路径边界约束（`SkillRegistryTests`），以及子智能体委派 —— 工具 schema、限额、宿主接线与进度快照（`MultiAgentTests`、`MultiAgentHostTests`、`MultiAgentToolSchemaTests`、`MultiAgentProgressTests`、`MultiAgentClientTests`）。

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

iOS 应用的宿主有自己的测试项目 `TensorAgent/tests/TensorAgent.Tests`（见 [TensorAgent/README.md](TensorAgent/README.md)）；PR CI 不运行它，不过 `InferenceWeb.Tests` 会检查该应用的项目文件、plist、entitlement 与原生导出清单（`TensorAgentMauiProjectTests`）。

#### 测试分组（Test lanes）

测试按两个维度打标。基础分类说明见 `InferenceWeb.Tests/TestAssemblyConfig.cs`；`Video` 与 `NativeTestHooks` 来自门控特性，部分测试套件还有自己的 `Category` 取值。`Category=Bench` 用普通 `[Trait]` 标记含时延/吞吐断言的基准测试类；前缀缓存相关的套件则标为 `Category=PrefixCacheUnit`、`PrefixCacheModel` 或 `PrefixCacheProperty`（下文的基数树性质测试）。`Requires=Cuda|Mlx|Models|Video|NativeTestHooks|GgmlCpu|GgmlMetal|GgmlCuda|GgmlVulkan` 标记测试对环境的依赖（`Models` = 需要测试模型目录下的真实 GGUF 权重；`Video` = 需要能编码视频的 OpenCV 构建；`NativeTestHooks` = 需要带测试钩子构建的原生库，用于 GLM 快照边界的 theory 测试）。依赖环境的测试使用 `InferenceWeb.Tests/GatedFacts.cs` 中的门控特性编写 —— `[CudaFact]`/`[CudaTheory]`、`[MlxFact]`/`[MlxTheory]`、`[VideoFact]`/`[VideoTheory]`、`[ModelFact("ENV_VAR", "gguf-substring")]`/`[ModelTheory(...)]` —— 它们会自动附加对应的 `Requires` trait，并在前提条件缺失时显式跳过。构造固定 GGML 后端但不需要权重的测试使用 `[GgmlFact(BackendType.GgmlCpu)]`/`[GgmlTheory(...)]`（trait 为 `Requires=GgmlCpu` 等）：原生桥接每个进程只允许一个 GGML 后端，因此除非 `TS_TEST_GGML_BACKEND` 固定为该后端，否则会显式跳过。在测试中直接调用 `CudaBackend.IsAvailable()`/`MlxBackend.IsAvailable()` 会产生编译错误（`BannedSymbols.txt`）：请改用门控特性。未打标的 `[Fact]`/`[Theory]` 是自包含的正确性测试，可在任何环境运行。不带过滤器的 `dotnet test` 运行全部测试；用 `--filter` 选择分组：

```bash
# 内循环（边改边测）：与环境无关的正确性测试，任何机器上数秒内跑完。
# PR CI 也运行这一分组（.github/workflows/pr-unit-tests.yml）。
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --filter "Category!=Bench&Requires!=Cuda&Requires!=Mlx&Requires!=Models"

# 完整正确性（推送前，在有 GPU 和模型文件的机器上）：除基准外的全部测试。
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --filter "Category!=Bench"

# 仅基准测试：其断言对时序敏感，应在空闲机器上有意运行。
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --filter "Category=Bench"
```

门控测试在前提条件缺失时会报告为**已跳过**（被跳过的 `[Theory]` 只计一次，不按数据行展开），因此在没有相应硬件/权重的机器上，绿色结果会显示为"N 通过，M 跳过"，而不是静默通过从未执行的测试。少数前提条件复杂的测试类（多个环境变量、按方法选择模型）仍在测试体内做门控，并保留显式的 `[Trait("Requires", ...)]` 标注。

PR CI（`.github/workflows/pr-unit-tests.yml`）在 GitHub 托管的 x64（`ubuntu-latest`）与 ARM64（`ubuntu-24.04-arm`）runner 上运行第一个分组，触发于 pull request 以及推送到 `main`。它强制关闭 GPU 原生构建（`TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=OFF`、`TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`、`TENSORSHARP_CUDNN=OFF`），固定 `TS_TEST_GGML_BACKEND=cpu`，并先运行不需要 GPU 的 GB10 容器契约检查 `eng/tests/gb10-container.py`。它不下载模型，也不需要 GPU。

基数树前缀缓存的树级性质测试（`InferenceWeb.Tests/PrefixCache/TreeTraceHarnessTests.cs`，trait 为 `Category=PrefixCacheProperty`）默认运行 1,000 个带种子的操作序列，属于可移植测试分组。三个仅供测试使用的环境变量控制它：`PREFIX_CACHE_TREE_SEEDS=20000` 运行完整的 20,000 个种子，`PREFIX_CACHE_TREE_SEED_START=<n>` 设置种子起点，`PREFIX_CACHE_SEED=<n>` 重放单个失败的种子。树操作的时延与分配门槛由 [`benchmarks/RadixTreeBench`](benchmarks/RadixTreeBench/README.md) 检查，它不需要加载模型。

### 服务端集成测试

服务端的集成测试位于 `TensorSharp.Server.Host/testdata/`。测试覆盖所有三种 API 风格（Web UI SSE、Ollama、OpenAI）、多轮对话、思维链模式、工具调用、结构化输出、队列状态兼容、并发请求和中断支持。架构特定能力（思维链、工具调用）会自动检测，当前模型不支持时会自动跳过。

```bash
# 先启动服务端（dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ...），然后运行：
python3 TensorSharp.Server.Host/testdata/test_multiturn.py
# 或
bash TensorSharp.Server.Host/testdata/test_multiturn.sh
```

完整测试矩阵见 [TensorSharp.Server.Host/testdata/README_zh-cn.md](TensorSharp.Server.Host/testdata/README_zh-cn.md)。

### 原生回归测试（CTest）

`TensorSharp.GGML.Native/tests/` 下的 C++ 测试只有在 `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`（默认 `OFF`）时才会构建：向 `build-linux.sh` 或 `build-windows.ps1` 传入 `--tests`（两者也都读取该环境变量）。`build-macos.sh` 只构建库本身，因此在 macOS 上请自行用 `-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON` 配置 CMake。用 `ctest --test-dir <构建目录> -R <名称> --output-on-failure` 运行。登记了跳过返回码的测试（包括 Qwen-Image-2.1 这一组）在设备不可用时报告为已跳过，而不是通过。

示例：`flash-attn-unsupported-shape-fallback`（见上文），以及 Qwen-Image-2.1 这一组——`qwen-image21-whole-graph-cpu` 写出一个合成的显式注意力参考，`qwen-image21-whole-graph-{metal,vulkan,cuda}` 与之对比（包括前缀 KV 缓存与两个 rank 的张量并行切分：CPU 与 Metal 上用回环组；CUDA 与 Vulkan 需要两块 GPU，单 GPU 机器会在通过的测试中打印 `SKIP tensor parallel`，这部分不计为覆盖），另有 `qwen-image21-vae-shortcuts-{cpu,cuda}` 与 `qwen-image21-vae-f32-convolution-metal[-no-mps]`。每个 GPU 变体只有在构建启用了对应后端时才会登记。详见 [TensorSharp.GGML.Native/tests/qwen_image21_tests.md](TensorSharp.GGML.Native/tests/qwen_image21_tests.md)；这些是数值回归测试，不检查图像质量或速度。

### 推理矩阵运行器

`TensorSharp.TestMatrix` 是更大的 CLI 驱动覆盖工具，用于长时间模型 / 后端验证。它会发现 GGUF 文件，过滤不可用后端与不受支持的提示类型，运行 baseline 与环境变量 sweep，用每个 cell 一个 JSON 的形式保存结果，生成汇总 Markdown 报告，并可按需与每类主机的基线做回归对比。

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

当前运行器契约见 [TensorSharp.TestMatrix/README_zh-cn.md](TensorSharp.TestMatrix/README_zh-cn.md) 与 [docs/env_var_feature_matrix_zh-cn.md](docs/env_var_feature_matrix_zh-cn.md)。

### 验证工具与验证记录

原生开发与验证工作遵守仓库的两条规则（[AGENTS.md](AGENTS.md)）：

- **ggml 保持原样。** 没有任何 ggml 补丁，也不改写获取到的文件；ggml 不提供的行为在 TensorSharp 自有代码（`TensorSharp.GGML.Native`）中实现。原生改动要针对未经修改的上游 checkout 验证，并记录 ggml 修订版、实际运行的测试与基准的局限；被跳过或不可用的模型/设备场景不计为通过。
- **生成的验证记录不进 Git。** 日志、报告与快照放在被忽略的 `docs/validation/` 或 `artifacts/` 目录中，绝不强制添加。可复用的工具放在 `eng/` 下（`eng/tests/`、`eng/validation/`），测试所需的 fixture 放在对应的测试项目中（例如 `InferenceWeb.Tests/Fixtures/`）。

Qwen-Image-2.1 的工具体现了这种分工（它们生成的记录应放在 `docs/validation/` 或 `artifacts/`）：

| 工具 | 检查内容 |
|---|---|
| `eng/tests/qwen-image21-dit.py` | 原生扩散 Transformer 的独立 NumPy 参考实现，包含 LoRA 适配器；`--backend cpu\|metal\|cuda\|vulkan`，不需要模型权重 |
| `eng/tests/qwen-image21-vae-attention.py` | VAE 空间注意力的 NumPy 参考实现 |
| `eng/validation/qwen-image21-bench.py` | 在相同输入上依次运行 TensorSharp 与 stable-diffusion.cpp，报告墙钟时间、引擎各阶段、每步耗时、峰值 RSS 与输出哈希 |
| `eng/validation/qwen-image21-ab.py` | 以轮换顺序对比两个构建或两组设置；PNG 的 SHA-256 相同即证明改动逐位一致 |
| `eng/validation/qwen-image21-http.py` | 通过 HTTP 驱动真实的 `TensorSharp.Server.Host`（传输、状态复用、预览、alpha、错误处理）；不是质量或吞吐基准 |
| [`eng/QwenImage21CompanionProbe`](eng/QwenImage21CompanionProbe/README.md) | 对比托管算子链与融合原生主干产生的 Qwen3-VL-8B 条件向量 |
