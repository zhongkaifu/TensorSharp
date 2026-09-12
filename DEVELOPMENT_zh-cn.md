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

与上一节不同，这条不限于 Apple：只要 flash-attention kernel 把 softmax 分子保存在 FP16 里就会遇到，而这里构建的每个后端都是如此。

MiniMax-H3 对**一条**无 mask 的打包序列做双向注意力——文本、条件帧、目标音频与目标视频都在其中——因此 key 数量**就是**整段片段：22 帧为 2364 个打包 token，107 帧为 8646 个。vendored ggml 的每个 flash-attention kernel 都用 FP16 寄存器累加 `sum_j exp(s_j - max) * V_j`，而 `ggml_flash_attn_ext_set_prec(GGML_PREC_F32)` 是空操作——`ggml-cuda/` 与 `ggml-metal/` 里没有任何代码读取 `GGML_OP_FLASH_ATTN_EXT` 的 `op_params`。kernel 留给累加器的余量只有 3 个 bit（`FATTN_KQ_MAX_OFFSET` 把运行最大值抬高 log(8)，使每个 softmax 权重上限为 1/8），因此 N 个 key 的一行会累加到 N/8 × |V|，一旦 N × |V| > 8 × 65504 就溢出为 Inf。只有 H3 会撞上这个上限：checkpoint 里有 `q_norm` 和 `k_norm`，唯独没有 `v_norm`，value 这一路的幅值不受约束。640×384 实测：73 帧正常，107 帧则在**第一个**去噪步就出现 Inf，返回的视频每个像素全黑、每个音频采样被钳位——视频与音频共用同一主干，溢出会连同声音一起拖垮。

修复在 `h3_attend`（`ggml_ops_minimax_h3.cpp`）：按 key 数量取"能把它压到 `kH3FlashKeyBudget` 以内的最小 2 的幂"预先缩放 V，输出时再还原。注意力对 V 是线性的，所以这个修正是**精确**的；又因为倍数是 2 的幂，经过 F16 转换也只是指数位平移而非舍入，因此足够短的序列（包括测试套件里所有 oracle 用例）保持逐位一致。`h3_mm` 对两个激活无界的量化 matmul（送入 `o_proj` 的注意力输出、送入 `down_proj` 的 SwiGLU 隐状态）做同样处理，那里的上限则是 q8_1 每个 block 的 FP16 求和。

| 环境变量 | 作用 |
|---|---|
| `TS_H3_TRACE=1` | 打印每个去噪步的 latent 与 velocity 幅值——样本发散时，latent 的 absmax 会比真正变成无穷早若干步露出端倪 |
| `TS_H3_NO_FLASH=1` | 强制走显式 softmax 路径，用于区分是 flash-attention kernel 的问题还是建模的问题 |

采样器也不再无条件相信结果：velocity 出现非有限值时直接让请求失败并指明是第几步（`MiniMaxH3Pipeline.cs` 的 `RequireFinite`），而不是写出一个长度、帧率、音轨时长都正常但通体全黑的文件——这种失败本来是无声的，因为 RGB 钳位会把 NaN 像素固定成 0，WAV 写出会把 NaN 采样钳到 -1。

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


## 项目结构

```text
TensorSharp/
├── TensorSharp.Core/            # 核心张量库（Tensor、Ops、内存、设备抽象，含 CPU SIMD/托管量化内核）
├── TensorSharp.Runtime/         # GGUF、分词器、模板、采样、协议解析
│   ├── Paged/                   # 分页 KV 缓存原语（BlockPool、BlockTable、KvBlock、BlockHashIndex、PagedKvStorage、PagedKvBatchOps、ManagedPagedAttention）
│   ├── Scheduling/              # 连续批处理引擎（InferenceEngine、BatchExecutor、ContinuousBatchScheduler、SequenceState、SchedulerConfig/Output、InferenceRequestHandle）
│   ├── Speculative/             # 投机解码：起草/验证/回滚核心（SpeculativeExecution）、ISpeculator 各算法（DraftHeadSpeculator、BlockDraftSpeculator、NGramSpeculator）与 SpeculatorRegistry、模型侧契约（ISpecTrunk、SpeculativeModelContracts）、共用的参数解析（SpeculativeCliFlags、SpeculationOptions）以及运行期成本裁判
│   ├── PagedKvCacheManager.cs   # 单会话分页 KV 管理（块分配、前缀复用）
│   ├── PagedKvBlockStore.cs     # 带可选 SSD 溢出的 RAM/磁盘分级分页块存储
│   ├── SsdKvBlockTier.cs        # 分页块的 SSD 冷层
│   ├── TurboQuantKvCodec.cs     # 实现 IKvBlockCodec 的量化 KV 块编解码器（2-bit / Q4 / Q8）
│   ├── PrefillChunking.cs       # SWA / 超长 prompt 使用的分块 prefill 辅助
│   ├── KvBlockHash.cs           # 内容寻址的块哈希，用于跨请求前缀复用
│   └── Logging/                 # JSON-line 文件日志器 + 每轮遥测
├── TensorSharp.AgentHost/       # 构建在运行时之上的智能体层：Agent Skills 与代码执行
│   ├── Skills/                  # Agent Skills：SKILL.md frontmatter 解析（YamlFrontmatter、SkillManifest）、发现 / 安装 / 查找（SkillRegistry、SkillArchive）、目录边界约束（SkillPathGuard）、提示词规划（SkillPrompt）、内置的 skills_list / skills_read / skills_run 工具与进程内披露循环（SkillTools、SkillAgentLoop、SkillScriptRunner）、共用的参数解析（SkillHostOptions）以及对外客户端（SkillsChatClient），以及各平台沙箱（SkillSandbox、SkillSandboxWindows）及其违规监视器、会话级工作区（SessionWorkspace）
│   └── CodeExec/                # 文件工具（read_file、edit_file、write_file）、shell 与 apply_patch：执行引擎（ShellRunner）、两个工具声明以及小模型所需的宽松参数读取（ShellTools）、command 参数的解读——把一行命令拆成各个简单命令、判定其中哪些是软件包安装（这决定了这一行到底能不能拿到套接字）、并在 shell 看到之前拦截 apply_patch heredoc（ShellCommand）、会话级的工作目录与导出环境变量——因为没有常驻的 shell 进程，它们通过文件持久化（ShellSession）、shell 的发现与方言（ShellProgram）、补丁信封解析（CodePatch）及其匹配引擎——对参考实现 V4A applier 的逐行移植（V4ADiff）、把一次失败改写成下一条该敲的命令（CodeDiagnostics）、当一次运行是因为模型猜错了库的 API 而失败时，直接从已安装的包里读出真实 API（ApiProbe）、检查命令写入或补丁改动后的文件是否仍能解析（SyntaxCheck）、把宿主的绝对路径从命令的全部输出中改写掉（OutputPaths）、发现「为改两行而重打整个文件」的行为（RewriteWatch）、为 import 失败的技能脚本由宿主发起的安装（PackageInstaller）、宿主侧的执行条款（CodeExecOptions）、受限启动——它也可以只启动进程而不等它结束，后台任务即由此实现（ConfinedProcess）、解释器发现（CodeEnvironment）、产物捕获（CodeArtifactStore）、安装期的软件源代理（EgressProxy）、结果记录（CodeExecResult），以及技能层所见的 ICodeRunner 接缝（CodeRunnerAdapter）
├── TensorSharp.Models/          # 模型架构实现与多模态编码/注入
│   ├── Models/<Family>/         # 每个架构一个目录（DeepSeek4、DiffusionGemma、Gemma4、GlmDsa、GptOss、MiniMaxH3、Mistral3、MuseGlimmer、Nemotron、Qwen35、Qwen4Exp、QwenImage、WanVideo）
│   │   ├── <Family>Model.cs                # 旧的单序列 ModelBase 实现
│   │   └── <Family>Model.BatchedForward.cs # IBatchedPagedModel.ForwardBatch —— 批处理/分页路径（Mistral3、Gemma4、GptOss、Qwen35、Nemotron）
│   ├── Models/DeepSeek4/        # DeepSeek V4 Flash：使用整模型执行器而非逐算子前向
│   │   ├── DeepSeek4Model.cs               # GGUF 元数据、分词器、聊天模板、执行器选择
│   │   ├── DeepSeek4CudaExecutor.cs        # 对接 Direct CUDA 整模型引擎
│   │   ├── DeepSeek4CpuExecutor*.cs        # 100% 纯 C# 整模型执行器（零原生依赖）
│   │   ├── DeepSeek4Model.Dspark.cs        # DSpark 块级草稿器（draft / 置信度 / Markov 头）
│   │   └── DeepSeek4Model.PerSeqCache.cs   # 让该模型可被服务端托管的原生 per-sequence slot
│   ├── Models/GlmDsa/           # GLM 5.x：原生执行器驱动、MLA + DSA indexer 逐算子参考实现、序列 slot
│   ├── Models/MuseGlimmer/      # Muse-Glimmer：融合整模型前向、视觉编码器、张量并行变体、DFlash 块级草稿器
│   ├── Models/MiniMaxH3/        # MiniMax-H3 视频 + 联合 32 kHz 立体声音频：打包序列 DiT、Qwen3-VL 文本编码器与视觉塔、视频与音频 VAE、flow-match 调度器、pipeline
│   ├── Models/WanVideo/         # Wan 2.1/2.2，仅视频：DiT、UMT5-XXL 文本编码器、因果 3D VAE、UniPC 调度器，以及不依赖 ggml 的 WanDirect* `cuda`/`cpu` 路径
│   ├── Models/Video/            # 两个视频家族共同实现的接缝：IVideoGenerationModel、VideoGenerationParams/Progress、GeneratedVideoAudio、WAV 写出
│   ├── Paged/                   # 张量侧的分页注意力辅助（TensorPagedAttention）
│   ├── KvBlockTransfer.cs       # 跨序列的 KV 块 extract/inject 辅助
│   ├── SpeculativeDecoder.cs    # Qwen 3.6、GLM 5.2 与 Gemma 4 共用的模型侧起草-验证-回滚驱动
│   ├── SpeculativeDraftHeadLoader.cs # 加载独立的草稿器 GGUF（Gemma 4 gemma4-assistant、DSpark、DFlash）并绑定到主干
│   └── ModelMultimodalInjector.cs # 视觉 / 音频 / 视频嵌入注入
├── TensorSharp.Backends.GGML/   # GGML 后端绑定（通过原生库支持 Metal/CUDA/Vulkan/CPU）
├── TensorSharp.Backends.Cuda/   # Direct CUDA 后端（CUDA Driver API、cuBLAS、PTX 内核）
│   └── Dsv4/                    # DeepSeek V4 Direct CUDA 整模型引擎（不依赖 ggml）：GGUF→显存流式加载器、按设备权重竞技场、层切分、DSpark 草稿器
├── TensorSharp.Backends.MLX/    # Apple Silicon MLX 后端（mlx-c / Metal），原生桥接由 `build-native-macos.sh` 编译
├── TensorSharp.GGML.Native/     # 到 ggml 的原生 C++ 桥接（构建 libGgmlOps，拆分为多个专注源文件）
│   ├── ggml_ops_core.cpp                  # 元素级、归约、基础 shape 操作
│   ├── ggml_ops_elementwise.cpp           # 元素级 / 激活融合
│   ├── ggml_ops_matmul.cpp                # GEMM / 量化 matmul
│   ├── ggml_ops_fused.cpp                 # 跨域融合的每层内核
│   ├── ggml_ops_norm_attn.cpp             # Norm + 注意力融合
│   ├── ggml_ops_transformer.cpp           # 通用融合 Transformer 层/整模型 decode 与 flash-attn decode
│   ├── ggml_ops_transformer_common.h      # 共享的 Transformer 辅助函数与 C# 层描述符结构体
│   ├── ggml_ops_transformer_prefill.cpp   # 融合层 prefill（Gemma 4、GPT-OSS、Qwen 3.5）
│   ├── ggml_ops_qwen35_decode.cpp         # Qwen 3.5/3.6 融合 decode（单层、整模型、批量）
│   ├── ggml_ops_qwen35_verify.cpp         # Qwen 3.5/3.6 融合多 token verify
│   ├── ggml_ops_qwen35_gdn_tp.cpp         # Qwen 3.5/3.6 按 rank 的打包 GatedDeltaNet 内核（张量并行）
│   ├── ggml_ops_qwen35_recurrent_prefill.cpp # Qwen 3.5/3.6 递归层 prefill
│   ├── ggml_ops_gptoss_decode.cpp         # GPT OSS 整模型 decode 计算图（每 token 一次调度，共享 KV 窗口）
│   ├── ggml_ops_gptoss_prefill.cpp        # GPT OSS 整模型 prefill：N 个 token 走完全部注意力 + MoE 层，连同折叠的最终 norm 与 LM head 合为一张图
│   ├── ggml_ops_deepseek4.cpp             # DeepSeek V4 原生整模型执行器（层切分、压缩 KV 缓存、计算图缓存）
│   ├── ggml_ops_glm_dsa.cpp               # GLM 5.x 原生整模型执行器（MLA + DSA indexer、张量并行、序列 slot）
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
│   ├── ggml_ops_paged_attention.cpp       # 分页注意力原生内核（驱动 ggml_flash_attn_ext + sinks 变体）
│   ├── ggml_ops_tensor_parallel.cpp       # 多 rank TP 组、分段融合计算图执行、集合通信
│   ├── ggml_ops_tp_probe.cu               # 选择 TP 传输方式的 peer-copy / NCCL AllReduce 预检
│   ├── ggml_ops_diffusion.cpp             # DiffusionGemma 融合 decode-layer / 整模型 / lm-head 内核
│   ├── ggml_ops_qwen_image.cpp            # Qwen-Image-Edit MMDiT 整模型前向（CUDA 图捕获）+ CFG-batched 内核
│   ├── ggml_ops_minimax_h3.cpp            # MiniMax-H3 整网络计算图：音视频打包的 DiT、Qwen3-VL 文本编码器与视觉塔、视频 / 音频 VAE 的编码与解码（七个入口，权重直接从调用方 mmap 常驻绑定）
│   ├── ggml_ops_wan.cpp                   # Wan 2.1/2.2 整图入口：UMT5-XXL 文本编码器、每步 DiT 速度预测（按 shape 持久化以便捕获 CUDA 图）、因果 3D 视频 VAE 编码与解码
│   ├── ggml_ops_training.cpp              # 仅训练用内核（运行时不使用）
│   └── tests/                              # 原生单元 + 烟雾测试
├── TensorSharp.Server/          # Web 聊天 + API 服务（ASP.NET Core）
│   ├── Program.cs               # 精简启动：DI 注册、中间件、端点映射、paged-KV + 连续批处理 CLI 翻译
│   ├── ModelService.cs          # 保持服务端推理公共 API 稳定的门面，持有 InferenceEngineHost
│   ├── ModelLifecycleService.cs # 模型加载/释放与后端选择（CPU / CUDA / MLX / GGML CPU/Metal/CUDA/Vulkan）
│   ├── InferenceEngineHost.cs   # DI 注册的单模型 InferenceEngine 单例（连续批处理入口）
│   ├── ChatGenerationPipeline.cs # Prompt 渲染，将请求提交到 InferenceEngine，流式返回 token，处理 stop
│   ├── InferenceTelemetry.cs    # Prompt/eval 计时、TTFT、tokens/sec、有界输入摘要与输出日志
│   ├── ChatHistoryPreparer.cs   # 历史归一化、raw token 拼接、多模态顺序辅助
│   ├── ChatSession.cs           # 单会话历史跟踪与 assistant raw token
│   ├── SessionManager.cs        # 线程安全的会话注册（默认会话 + 每个 UI Tab 的会话）
│   ├── InferenceQueue.cs        # 向后兼容的队列状态接口（并发由引擎本身处理）
│   ├── BackendCatalog.cs        # 可用计算后端的发现（CPU / CUDA / MLX / GGML*）
│   ├── TextUploadHelper.cs      # 无损文本上传归一化辅助
│   ├── WebUiChatPolicy.cs       # Web UI 聊天请求合法性校验
│   ├── OpenAIResponseFormatParser.cs  # OpenAI response_format（json_object / json_schema）解析
│   ├── Hosting/                 # 启动期相关：选项装配（ServerOptionsBuilder）、后端选择、日志、wwwroot 解析、paged-KV / 连续批处理 CLI 翻译
│   ├── RequestParsers/          # JSON 请求解析（采样配置、聊天消息、工具函数）
│   ├── ResponseSerializers/     # 各协议响应形状构造（Ollama / OpenAI / Web UI）
│   ├── StreamingWriters/        # SSE 与 NDJSON 线协议辅助
│   ├── ProtocolAdapters/        # 各协议的请求处理器（WebUiAdapter、OllamaAdapter、OpenAIChatAdapter）
│   ├── Endpoints/               # ASP.NET Core 路由映射（每协议一个扩展方法）
│   ├── Logging/                 # 请求日志中间件 + 低噪声路径支持
│   ├── wwwroot/index.html       # 聊天界面
│   ├── testdata/                # 集成测试套件（bash + Python）
│   └── API_EXAMPLES.md          # 详细 API 文档
├── TensorSharp.Cli/             # CLI 应用（单次生成、交互式 REPL、JSONL 批处理、基准）
├── TensorSharp.TestMatrix/      # 测试 / 基准矩阵运行器、默认提示、环境变量扫描与主机基线
├── InferenceWeb.Tests/          # xUnit 单元测试，覆盖算子、KV 缓存、分页调度器、批处理模型正确性以及 Web/服务辅助逻辑
├── AdvUtils/                    # 工具库（日志）
├── docs/                        # 开发者参考文档
│   ├── models/                  # 按模型架构卡片（每个模型一份 .md，中英双语）
│   ├── PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md  # 分页 KV 缓存、前缀共享、调度器、按模型批处理状态
│   ├── speculative_decoding.md  # 起草-验证设计：ISpeculativeTarget / ISpeculator / IDraftHead 三层，以及 draft-head、block 与 ngram 三种算法
│   ├── agent_skills.md          # Agent Skills：SKILL.md 格式、渐进式披露与其预算、进程内工具循环、路径 / ZIP / 脚本执行的安全模型，以及 HTTP 与 C# 两套接口
│   └── env_var_feature_matrix.md  # TestMatrix 使用的运行时开关 × 模型/后端/功能覆盖矩阵
├── benchmarks/                  # 可重现的基准脚本
└── ExternalProjects/            # ggml/ 在构建时从 github.com/ggml-org/ggml 克隆（不纳入版本控制）
```

## 项目 / NuGet 包分层

仓库按包边界拆成独立层，使用者可以只引用真正需要的部分。

**状态核验于 2026-09-08。** 发布集合共 **13** 个包，即下表的每一行。`eng/verify-packages.ps1` 是这份清单的权威来源，并且它是发布流水线的门禁：在两个可打包项目之间新增 `ProjectReference` 而不同步更新该脚本，会直接让发布失败。

当前真正发布到 [NuGet.org](https://www.nuget.org/profiles/TensorSharp) 的只是其中一个子集：**8** 个包 ID，版本 **3.1.2**，发布于 2026-07-21——`TensorSharp.Tensors`、`TensorSharp.Runtime`、`TensorSharp.Models`、`TensorSharp.Backends.GGML`、`TensorSharp.Backends.Cuda`、`TensorSharp.Backends.MLX`、`TensorSharp.Server` 与 `TensorSharp.Cli`。它们落后于当前源码与 v3.3.0.0 应用发行版；尤其是已发布的 `TensorSharp.Server` 早于日志层与聊天层拆分，与这里描述的分层并不一致。

其余 5 个——`TensorSharp.Runtime.Logging`、`TensorSharp.AgentHost`、`TensorSharp.Chat`、`TensorSharp.Server.Host` 与 `TensorSharp.Distributed`——已可打包并通过校验，但尚未推送，将随下一个版本标签一同发布。在此之前，使用这些层仍需从源码 checkout 添加项目引用。

| 项目 | NuGet 包 | 对外 namespace | 职责 |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Tensors` | `TensorSharp` | Tensor 原语、Ops、分配器、存储与设备抽象 |
| `TensorSharp.Runtime.Logging` | `TensorSharp.Runtime.Logging` | `TensorSharp.Runtime.Logging` | 各宿主共用的日志抽象与输出目标；单独拆出后，引擎各层不必再背上宿主的日志栈 |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF 解析、分词器、Prompt 渲染、采样、输出协议解析、分页 KV 缓存、连续批处理调度器 |
| `TensorSharp.AgentHost` | `TensorSharp.AgentHost` | `TensorSharp.AgentHost` | Agent Skills 与代码执行（`read_file` + `edit_file` + `write_file` + `shell` + `apply_patch`），含操作系统级沙箱、Web/CLI 会话级与 HTTP 请求级工作区，以及由宿主判定的软件包安装——构建在 `TensorSharp.Runtime` 之上 |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`、各模型架构、多模态编码器、批处理 / 分页前向、模型侧执行辅助 |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML 执行后端与原生互操作 |
| `TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` | `TensorSharp.Cuda` | Direct CUDA 分配器、存储、cuBLAS GEMM、PTX 内核和量化 CUDA 算子 |
| `TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` | `TensorSharp.MLX` | Apple Silicon MLX 后端（mlx-c / Metal），含量化 / 融合 / 编译内核与 MoE 专家 offload |
| `TensorSharp.Distributed` | `TensorSharp.Distributed` | `TensorSharp.Distributed` | 用于多节点张量并行的点对点 TCP 协调层 |
| `TensorSharp.Chat` | `TensorSharp.Chat` | `TensorSharp.Chat`（新类型）；迁入的流水线保留 `TensorSharp.Server.*` 命名空间 | 与宿主无关的聊天流水线：`ModelService`、会话、生成、技能循环与 Web UI 请求/流式契约（`WebUiChatService`、`SkillsService`）——不依赖 ASP.NET Core 与 `TensorSharp.Distributed`；由 Server、CLI 与 iOS 应用共用 |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core 服务、OpenAI/Ollama 适配层、推理引擎宿主与 Web UI |
| `TensorSharp.Server.Host` | `TensorSharp.Server.Host` | `TensorSharp.Server.Host` | **可运行**的 Web 应用：`Program.cs`、宿主装配、`wwwroot/` 与命令行。`TensorSharp.Server` 是它依赖的库——单独构建或运行 `TensorSharp.Server` 不会产生任何可执行文件 |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | 控制台宿主、调试工具与 JSONL 批处理 |

这样的拆分让引擎使用者不必带上 Web 依赖，也能把 API 层改动和核心运行时隔离开，并让后续 benchmark / eval harness 更容易独立发布。

> **注意：** 核心层发布的包名是 **`TensorSharp.Tensors`**，不是 `TensorSharp.Core`。NuGet.org 上的 `TensorSharp.Core` 包 ID 属于另一个已废弃的无关项目（所有版本均已 unlist，源码仓库已删除），推送到该 ID 会返回 403。差异仅限于 NuGet 包名——项目名、程序集名与 `TensorSharp` namespace 均保持不变，因此 `using` 语句不受影响，只有 `dotnet add package` 命令行不同。

发布前可验证包元数据与 README 依赖边界：

```powershell
pwsh ./eng/verify-packages.ps1
```

该验证会对上表公开包运行 `dotnet pack`，并在 `AdvUtils` 等内部依赖泄漏到 `.nuspec`，或 TensorSharp 包依赖了上表之外的分层时失败。

### 平台二进制发行状态

**状态核验于 2026-09-01：**[v3.3.0.0](https://github.com/zhongkaifu/TensorSharp/releases/tag/v3.3.0.0) 已发布 **TensorSharp.Server** 与 **TensorSharp.Cli** 的十个自包含应用归档，内含 .NET 10 运行时与原生库。下方五种平台/后端后缀各有一份 Server 归档和一份 CLI 归档。请到 [Releases 页面](https://github.com/zhongkaifu/TensorSharp/releases)检查更新版本，不要假定这份带日期的状态会永远代表最新发行。

v3.3.0.0 已发布的归档矩阵如下：

| 归档后缀 | 内置的原生后端 | 格式 |
|---|---|---|
| `win-x64-cpu` | GGML CPU | `.zip` |
| `win-x64-cuda` | GGML CUDA + 纯 C# CUDA（PTX）+ CUDA 12.x 运行时 | `.zip` |
| `linux-x64-cpu` | GGML CPU | `.tar.gz` |
| `linux-x64-cuda` | GGML CUDA + 纯 C# CUDA（PTX）+ CUDA 12.x 运行时 | `.tar.gz` |
| `osx-arm64` | GGML Metal + MLX | `.tar.gz` |

- 推送 `v*` 标签会触发归档与 NuGet 工作流；只有所需 job 全部成功后才会发布产物。
- `-cuda` 归档已内置 CUDA 运行时库（`cudart` / `cublas` / `cublasLt`），但运行时仍需 NVIDIA GPU 与兼容驱动；`-cpu` 归档可在任意机器运行。macOS 归档需 Apple Silicon。
- 如需预演，可手动触发该工作流（`workflow_dispatch`）并填写 `version` 输入——它会构建全部平台并创建**草稿** Release。可用 `cuda_arch` 输入覆盖 CUDA 构建的目标 GPU 架构。


## 架构说明

TensorSharp 采用分层系统结构：

1. **TensorSharp.Core** 提供核心 `Tensor` 类型、存储抽象和可扩展的操作注册表（`Ops`）。CPU 实现使用 `System.Numerics.Vectors` 进行 SIMD 加速。

2. **TensorSharp.Runtime** 负责运行时契约与通用服务：GGUF 解析、分词（SentencePiece / BPE）、聊天模板渲染、可配置 token 采样、输出解析、分页 KV 缓存（`Runtime/Paged/*`）、连续批处理调度器 / 引擎（`Runtime/Scheduling/*`）、`IKvBlockCodec` 接口及其 `TurboQuantKvCodec` 2-bit / Q4 / Q8 实现，以及 `IModelArchitecture`、`IBatchedPagedModel`、`IPromptRenderer`、`IOutputProtocolParser`、`IMultimodalInjector`、`IKVCachePolicy`、`IBackendExecutionPlan` 等抽象。它刻意不包含智能体层：技能与代码执行位于 **TensorSharp.AgentHost**，该项目引用运行时、且运行时绝不反向引用，因此只需提供 OpenAI / Ollama 聊天补全的宿主可以只依赖运行时，不携带技能注册表、沙箱与代码执行器。（若该方向被反转，`AgentHostLayeringTests` 会失败。）

3. **TensorSharp.Models** 实现 `ModelBase` 以及全部 13 个具体模型架构和多模态辅助组件——10 个文本家族（DeepSeek V4 Flash、GLM 5.x、Gemma 4、DiffusionGemma、Qwen 3.5/3.6 系列、Qwen 3.8 Flash Next、GPT OSS、Nemotron-H、Mistral 3、Muse-Glimmer）与 3 个媒体输出家族（Qwen-Image-Edit、MiniMax-H3、Wan 2.1/2.2）。自回归架构提供旧的单序列前向，多数架构还提供面向连续批处理的 `IBatchedPagedModel.ForwardBatch` 实现（`<Family>Model.BatchedForward.cs`）。DiffusionGemma 刻意不同：它不支持 `Forward()`，生成必须通过 `DiffusionGemmaSampler` 在固定长度 canvas 上迭代去噪。Qwen-Image-Edit（`QwenImageModel`）同样非自回归：`Forward()` 抛异常，图像编辑通过 `EditImage()` 进行，由它编排 MMDiT 扩散 Transformer、Qwen-Image VAE 与 Qwen2.5-VL 文本编码器。视频家族更进一步：`MiniMaxH3Model` 与 `WanVideoModel` 的 `ForwardCore()` 都直接抛异常，生成统一走 `GenerateVideo(prompt, VideoGenerationParams)`，其背后是 `Models/Video/` 里共享的 `IVideoGenerationModel` 接缝——CLI 与服务端因此只用一条路径驱动两者（以及日后新增的模型），而不必逐个判断具体模型类型。MiniMax-H3 在同一个打包 latent 里**同时**去噪视频与 32 kHz 立体声音频，共有七张原生整网络计算图（DiT、Qwen3-VL 文本编码器、视觉塔、视频与音频 VAE 的编码与解码）；Wan 2.1/2.2 则是仅视频的家族，其 DiT、UMT5-XXL 编码器与因果 3D VAE 同样以整图方式运行。模型通过 `ModelBase.Create()` 加载，并依据 GGUF 元数据自动识别架构——MiniMax-H3 例外：其公开发布的 GGUF 完全不带元数据，只能依据张量识别（`LooksLikeMiniMaxH3`，经由 `MiniMaxH3Architecture.DetectFromTensors` 接入）。

4. **TensorSharp.Backends.GGML** 通过原生 C++ 桥接库（`libGgmlOps` / `GgmlOps.dll`）注册同名操作的加速实现，并链接 [ggml](https://github.com/ggml-org/ggml)。在 macOS 上可提供 Metal GPU 计算，在 Windows/Linux 上可启用面向 NVIDIA GPU 的 GGML CUDA。除原生量化 matmul（Q4_K_M、Q8_0 等，无需反量化到 FP32）外，还提供分页注意力（`TSGgml_PagedAttentionForward`，含 / 不含注意力 sinks 两种版本）以及架构特定的批处理内核（Mamba2、GatedDeltaNet）。

5. **TensorSharp.Backends.Cuda** 是 Direct CUDA 路径。它使用 CUDA Driver API 管理设备、上下文与存储，用 cuBLAS 执行 Float32 GEMM，用 PTX 内核覆盖热点标量与 Transformer 辅助算子，并对尚未实现的原生内核使用 CPU 回退。

6. **TensorSharp.Backends.MLX** 是 Apple Silicon 上的 MLX 路径。它封装 [mlx-c](https://github.com/ml-explore/mlx-c)（`libmlxc`），提供分配器、存储、异步 worker 派发、量化 / 融合 / 编译内核、MoE 专家 offload，以及对未实现算子的 CPU 回退层。

7. **TensorSharp.Server** 是 HTTP / 应用层，提供兼容 Ollama 与 OpenAI 的 REST API、浏览器聊天 UI、上传处理；其中 `InferenceEngineHost` 持有自回归模型的连续批处理引擎，`DiffusionBatchScheduler` 处理 DiffusionGemma 的 Web UI 轮次，旧的队列状态接口保留作为向后兼容。

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
- **GLM 5.x 整模型执行器**：`glm-dsa` 是同样的形态。原生 ggml 执行器（`ggml_ops_glm_dsa.cpp`）自行加载分片 GGUF（GLM-5.2 为六个分片，GLM-5.3 的 UD-Q2_K_XL 为七个），并持有 MLA 缓存（每层每 token 一行 576 宽，逐 head 的 K/V 解压被折进 query 和输出）以及 DSA lightning indexer 缓存——78 层里只有 21 层会刷新它。它既可以按层切分到各张可见 GPU，也可以在 `--tp N` 下让每一层跑在每个 rank 上：注意力 head 按列/行并行，路由专家则在**每个专家内部**按行切开，因为 `ggml_mul_mat_id` 要求同一个 token 选中的专家 id 互不相同。并发靠原生序列 slot 而不是分页 KV，在其之上默认启用批量融合 decode（`TS_BATCHED_FUSED_DECODE=0` 可关闭）。`TensorSharp.Models/Models/GlmDsa/` 里是原生路径用来对照的托管逐算子参考实现。
- **DeepSeek V4 整模型执行器**：`deepseek4` 完全绕开通用的逐算子前向。原生 ggml 执行器（`ggml_ops_deepseek4.cpp`）自行加载分片 GGUF，把权重按层切分到所有可见 GPU，在设备上持有全部 DSV4 KV 状态（原始 SWA 环、CSA/HCA 压缩 K 缓存、lightning indexer 缓存、压缩器状态环），并把每个 prefill/decode ubatch 作为一张 `ggml_backend_sched` 计算图执行，配合按形状签名的图缓存，使稳态 decode 直接重放已捕获的 CUDA 图。decode 注意力通过融合的 index-gather 算子取出紧凑的 `[ring | top-512]` K，而不是扫描整个上下文。Direct CUDA 引擎（`TensorSharp.Backends.Cuda/Dsv4/`）在不依赖 ggml 的前提下实现同一模型，把量化权重从分片直接流式写入按设备的显存竞技场。二者都构建在共享的 `Tensor` / `IAllocator` / `Ops` 之上；只有真正 DSV4 特有的计算才留在 DeepSeek V4 的文件里。
- **DSpark 块级投机解码**（DeepSeek V4）：独立的草稿 GGUF（`--draft-model`）每步提议一整块 token，主干用一次批量前向验证整块。在 ggml 上草稿器就是计算图里额外的三层，其 key ring 由主干图自己提交，因此投机不产生任何主机往返。4×A40 实测 decode 提速 1.3–1.4×（多轮对话最高 2.0×），贪心输出与非投机基线逐字节一致。
- **基于 GEMM 的视觉 patch embedding**（Qwen 3.5/3.6-family）：将 patch embedding 重构为并行 im2col + 矩阵乘法，把单线程标量五重嵌套循环替换为可在 GPU 上加速的 matmul。
- **并行化 Q/gate 反交错**（Qwen 3.5/3.6-family）：FullAttention prefill 中的 Q + sigmoid-gate 反交错按 token 并行化，长 prompt 时可随 CPU 核心数线性扩展。
- **优化后的纯 C# CPU 路径**：托管 GEMM 快速路径和连续 Float32 内核加速了 decode、softmax、RMSNorm、RoPE、融合激活等热点路径，同时在 CPU 加载时保持量化 GGUF 权重压缩状态。
- **环形 KV 缓存**：滑动窗口注意力层使用固定大小环形缓冲区，使内存占用不随序列长度增长。
- **KV 缓存前缀复用**：多轮对话会复用各轮之间最长的匹配 token 前缀。对 SWA 模型，截断会自动按滑动窗口大小回退，使后缀部分可以重建 SWA 上下文。
- **分页 KV 缓存 & 块哈希前缀共享**：连续批处理引擎把 KV 切分成固定大小的块，对每个写满的块做内容哈希，并在并发 / 历史请求间共享。尚未实现 `IBatchedPagedModel` 的模型仍会走同一引擎内隔离的按序列 KV-swap 回退路径。
- **原生分页注意力内核**：`TSGgml_PagedAttentionForward`（及面向 GPT OSS 的 `WithSinks` 变体）在 C++ 中按序列从分页缓冲区聚合 K/V，按序列构建小型 GGML 图，并派发 `ggml_flash_attn_ext`——也就是旧的单序列路径所使用的同一融合 GPU flash 注意力内核（Metal/CUDA/Vulkan）。在 Ministral-3-14B 长上下文（4×~800 tokens）上比旧的按序列 GGML 路径**快 ~21%**。
- **批处理 / 分页前向**：Mistral 3、Gemma 4、GPT OSS、Qwen 3.5/3.6（含 GatedDeltaNet 递归状态池）、Nemotron-H（含 Mamba2 递归状态池 + 原生批处理 Mamba2 内核）把 N 个序列打包到一次 `ForwardBatch` 调用中，每层执行一次批处理线性投影 matmul，通过 `slotMapping` 写入分页 K/V，并通过原生内核做按序列注意力。Gemma 4 批处理路径在 batch=8 短 prompt 下达到 **1.5×** 旧吞吐，在 4×800-token prompt 下达到 **1.6×**；Nemotron-H Mamba2 批处理在 Apple M4 Pro 上 batch=3 时达到 **3.95×**。详见 [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。
- **MTP / NextN 投机解码**：单序列可运行多 token 预测草稿头（Qwen 3.6、GLM 5.2 与 GLM-5.3 内嵌 NextN 块——GLM-5.3 只在不传 `--tp` 时可用，因为它的 draft 块借用按列切分的主干 LM head；Gemma 4 独立 `gemma4-assistant` 草稿 GGUF）。草稿头最多提议 `--spec-draft` 个 token，主干用一次批量前向验证，二者均由该请求自己的采样器驱动，因此在不改变输出的前提下加速 decode。在 ggml 后端上，融合的单图多 token 验证与草稿步内核（`NativeGemma4ModelVerify` / `TryFusedMoEModelVerify` / `NativeGemma4DraftStep`，以及 Qwen 3.6 的 NextN 图）摊销了验证开销；Gemma 4 路径还增加了 gallocr 验证 scratch 以及部分接受时避免重跑已保留前缀的稠密快速回滚。纯 C# `cuda` 后端运行完全驻留 GPU 的逐算子验证 / 草稿（donor 缓存注意力、GQA decode 内核、GPU RoPE），使验证层循环零宿主端同步停顿。默认关闭；`--spec`（Gemma 4 的 assistant GGUF 用 `--draft-model` 加载，给出文件本身即可启用投机）。
- **DiffusionGemma prompt-KV 缓存与融合去噪**：GPU 后端会在每个 block 中只对 `[prompt | canvas]` 的 prompt 部分预填充一次 K/V，并在去噪多步中复用；GGML 后端默认使用融合整模型 diffusion decode 与融合 lm-head tail。Web UI 通过 `DiffusionBatchScheduler` 在 block 边界批处理并发 diffusion 请求。
- **内核预热**：CLI 和 Server 在启动时运行一次微型前向传播，以预编译 GPU 内核（Metal pipeline state、CUDA JIT）并预热内存池，避免首次推理请求的冷启动延迟。
- **Prefill 缓存**（Gemma 4、Qwen 3.5/3.6-family）：逐 forward 传播的 SWA 掩码缓存（Gemma 4）、跨全局层的 NeoX RoPE cos/sin 查找表缓存（Gemma 4）、以及跨层的 RoPE 位置张量缓存（Gemma 4、Qwen 3.5/3.6-family），消除了 prefill 期间的冗余重复计算。
- **原地 QK RMSNorm**（Qwen 3.5/3.6-family）：逐头 QK 归一化通过 `View` 原地执行，避免了每层每个 Q/K 的一次张量分配与拷贝。

### 内存优化

- **零拷贝文件映射量化权重**（Direct CUDA、GGML CUDA、GGML Metal、GGML CPU）：GGUF 模型文件以内存映射方式打开，量化张量通过 host 指针缓冲区直接绑定到原生算子。这样省去了之前每张张量从磁盘复制到新分配原生堆缓冲区的过程——这一过程在 Apple Silicon 上会让大型量化模型的常驻内存几乎翻倍。例如，`Qwen3.5-35B-A3B-IQ2_XXS`（约 10 GB GGUF）在 Metal 后端的实际工作内存峰值从约 17 GB 降至约 7 GB。映射文件由操作系统的页缓存管理，必要时可换出，且在 Apple Silicon（统一内存）上不会带来推理性能损失。
- **最佳匹配内存池**：GGML 主机分配器使用 best-fit 而非 first-fit 在已池化块中检索可重用空间，避免把大块草稿内存交给小型中间张量请求，从而把工作集严格控制在合理范围内。
- **有界池保留量**：集成 GPU / CPU 内存池现在将单个保留块上限设为 64 MB，整池上限设为 32 块。结合 mmap 后的权重，可在快速复用短生命中间张量的同时限制峰值常驻内存。
- **高内存效率模型加载**：大张量直接流式加载到原生内存，避免中间托管分配。F32 权重与 norm 仍按需加载；量化权重在受支持的后端上通过 mmap 方式绑定。
- **可选 SSD 溢出的分页 KV 块池**：`PagedKvBlockStore` 保留了 RAM / SSD 分层块存储能力（`TS_KV_CACHE_MAX_RAM_MB`、`TS_KV_CACHE_SSD_DIR`、`TS_KV_CACHE_MAX_SSD_MB`），主要服务独立分页 KV 组件与后续扩展；服务端请求路径的活跃块由每个引擎的 `BlockPool` 统一管理。
- **KV 块编解码器**：`TurboQuantKvCodec`（2-bit 仿射、Q4 或 Q8）可压缩分页块，以精度换取更小的每块带宽与内存占用——大致减半（Q8）、减为四分之一（Q4）或约十分之一（2-bit，fp32 块）。2-bit 档位使用每组仿射 min+scale（即 llama.cpp Q2_K 背后的 block-min 思路），让四个码值覆盖该组的实际取值范围；它面向超长上下文的远端前缀复用，此时注意力权重远大于量化噪声。带递归状态的模型会自动回退到 passthrough。


## 测试

### 单元测试（xUnit）

`InferenceWeb.Tests` 覆盖无需启动服务的进程内行为：托管量化算子、可用 CUDA 设备上的 Direct CUDA 后端内核、可用 MLX 时的 MLX 后端内核、分页 KV 缓存调度（`ContinuousBatchSchedulerTests`、`PagedKvCacheTests`、`PagedKvCacheCodecTests`）、批处理执行器正确性（`BatchedExecutorTests`）、按模型批处理前向与旧路径的一致性（`Qwen35BatchedCorrectnessTests`、`Mistral3BatchedForwardTests`、`Gemma4BatchedForwardTests`、`GptOssBatchedCorrectnessTests`、`NemotronBatchedCorrectnessTests`）、MTP / NextN 投机解码正确性与可选端到端探针（`SpeculativeExecutionTests`、`Qwen36SpeculativeTests`、`Gemma4SpeculativeTests`）、DiffusionGemma 去噪 / prompt-KV / 批处理生成探针（`DiffusionGemmaTests`）、按模型批处理性能微基准（`*BatchedPerfBench.cs`）、`TurboQuantKvCodec` 编解码往返、prefill 分块、KV 缓存策略、KV 缓存 Prompt 渲染与多轮集成、聊天会话与 SessionManager 隔离、ModelService 历史跟踪、请求日志中间件与文件日志 Provider、图像预处理、媒体辅助逻辑、结构化输出校验、文本上传辅助、ModelService 上传日志、Web UI 聊天策略、模型上下文长度解析、可用后端发现，服务器 CLI 选项构造（`ServerOptionsBuilderTests`），以及 Agent Skills —— `SKILL.md` frontmatter 解析及其各类告警情形（`SkillManifestParserTests`），与技能注册表的发现、优先级、ZIP 安装防护和路径边界约束（`SkillRegistryTests`）。

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

#### 测试分组（Test lanes）

测试按两个维度打标（声明见 `InferenceWeb.Tests/TestAssemblyConfig.cs`）：`Category=Bench` 用普通 `[Trait]` 标记含时延/吞吐断言的基准测试类；`Requires=Cuda|Mlx|Models` 标记测试对环境的依赖（`Models` = 需要测试模型目录下的真实 GGUF 权重）。依赖环境的测试使用 `InferenceWeb.Tests/GatedFacts.cs` 中的门控特性编写 —— `[CudaFact]`/`[CudaTheory]`、`[MlxFact]`/`[MlxTheory]`、`[ModelFact("ENV_VAR", "gguf-substring")]`/`[ModelTheory(...)]` —— 它们会自动附加对应的 `Requires` trait，并在前提条件缺失时显式跳过。在测试中直接调用 `CudaBackend.IsAvailable()`/`MlxBackend.IsAvailable()` 会产生编译错误（`BannedSymbols.txt`）：请改用门控特性。未打标的 `[Fact]`/`[Theory]` 是自包含的正确性测试，可在任何环境运行。不带过滤器的 `dotnet test` 运行全部测试；用 `--filter` 选择分组：

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

### 服务端集成测试

TensorSharp.Server 的集成测试位于 `TensorSharp.Server/testdata/`。测试覆盖所有三种 API 风格（Web UI SSE、Ollama、OpenAI）、多轮对话、思维链模式、工具调用、结构化输出、队列状态兼容、并发请求和中断支持。架构特定能力（思维链、工具调用）会自动检测，当前模型不支持时会自动跳过。

```bash
# 先启动 TensorSharp.Server，然后运行：
python3 TensorSharp.Server/testdata/test_multiturn.py
# 或
bash TensorSharp.Server/testdata/test_multiturn.sh
```

完整测试矩阵见 [TensorSharp.Server.Host/testdata/README.md](TensorSharp.Server.Host/testdata/README.md)。

### 推理矩阵运行器

`TensorSharp.TestMatrix` 是更大的 CLI 驱动覆盖工具，用于长时间模型 / 后端验证。它会发现 GGUF 文件，过滤不可用后端与不受支持的提示类型，运行 baseline 与环境变量 sweep，用每个 cell 一个 JSON 的形式保存结果，生成汇总 Markdown 报告，并可按需与每类主机的基线做回归对比。

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

当前运行器契约见 [TensorSharp.TestMatrix/README_zh-cn.md](TensorSharp.TestMatrix/README_zh-cn.md) 与 [docs/env_var_feature_matrix_zh-cn.md](docs/env_var_feature_matrix_zh-cn.md)。
