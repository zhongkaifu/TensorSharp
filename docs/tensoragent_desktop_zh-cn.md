# TensorAgent 桌面版用户指南

[English](tensoragent_desktop.md) | [中文](tensoragent_desktop_zh-cn.md)

TensorAgent 在你的电脑上运行 AI 模型。本指南介绍 Mac 和 Windows 应用的下载、安装、首次聊天、附件与可选代码工具。使用桌面安装包无需构建 TensorSharp，也无需安装 .NET SDK。模型权重需要在应用内另外下载。

**[下载 TensorAgent 桌面版：最新发布](https://github.com/zhongkaifu/TensorSharp/releases/latest)**
· [所有发布](https://github.com/zhongkaifu/TensorSharp/releases)
· [源码构建与实际验证范围](../TensorAgent/README.md)

在发布页面展开 **Assets**，选择以 `tensoragent-desktop-` 开头的文件。桌面安装包由更新后的 **Release Binaries** 工作流发布；更新之前的发布可能只有 CLI / Server 归档。如果最新发布没有 Desktop 资源，该版本暂时没有桌面下载：可按[源码构建说明](../TensorAgent/README.md#on-the-desktop-macos-and-windows)构建，或等待下次包含桌面安装包的发布。CLI 和 Server 归档是另外的应用。

知识库、回答调整、LTX 2.5 和原生技能发现功能对应 `v2026.10.10` 标签之后的当前源码。在包含这些改动的发布出现前，需从源码构建；已有桌面安装包不代表其中已包含这些功能。

## 1. 选择下载文件

下表中的 `<version>` 是去掉开头 `v` 的发布标签。例如，`v2026.10.03` 对应文件名中的 `2026.10.03`；此处仅举例，不表示该发布已经存在。

| 你的电脑 | 首选文件 | 其他格式 |
|---|---|---|
| Apple Silicon Mac（M1 或更新），macOS 14 或更新 | `tensoragent-desktop-<version>-osx-arm64.dmg` | `.pkg` 安装程序或 `.zip` 应用包 |
| Windows x64，CPU / AMD / Intel 显卡 | `tensoragent-desktop-<version>-win-x64-cpu.msi` | `.zip` 便携文件夹 |
| Windows x64，兼容的 NVIDIA GPU 与驱动 | `tensoragent-desktop-<version>-win-x64-cuda.msi` | `.zip` 便携文件夹 |

Windows CPU 包使用 CPU 推理，不包含 Vulkan 后端。CUDA 包包含 CUDA 运行库，但 NVIDIA 驱动需另外安装；没有兼容 NVIDIA GPU 时选择 CPU 包。Mac 包使用 Metal，并可回退至 CPU。此工作流不生成 Intel Mac 或 Windows ARM64 桌面安装包。

Windows 应用声明的最低版本为 Windows 10 1809（build 17763）。请使用仍获得更新的 Windows 版本，例如受支持的 Windows 11 安装；缺少浏览器运行库时，安装 [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)。应用包含 .NET、Windows App SDK 与 Visual C++ 运行库，但 WebView2 是独立的前置依赖。详见 Microsoft 的 [WebView2 要求](https://learn.microsoft.com/en-us/windows/apps/develop/ui/controls/webview2)。

**内存与磁盘：** 当前内置模型最低需要 **12 GB 物理系统内存**（Mac 的统一内存）；8 GB 电脑没有符合条件的内置模型。部分模型需要 16、24、32 或 48 GB，模型页面会显示具体限制。GPU 显存与磁盘空间是另外的要求。请为应用、模型卡片显示的下载量及生成文件保留足够空间。较小的聊天模型连同配套文件也约需下载 5–6 GB，图像和视频模型则大得多。先用符合条件的小型聊天模型完成首次运行。

每个平台还发布 `SHA256SUMS-tensoragent-desktop-<version>-<variant>.txt`，其中 `<variant>` 为 `osx-arm64`、`win-x64-cpu` 或 `win-x64-cuda`。下载对应校验文件，把下例的 `VERSION` 换成你下载的版本，在下载文件所在目录运行，并将结果与校验文件中的哈希值比较：

```bash
# macOS 终端
shasum -a 256 "tensoragent-desktop-VERSION-osx-arm64.dmg"
```

```powershell
# Windows PowerShell
Get-FileHash ".\tensoragent-desktop-VERSION-win-x64-cpu.msi" -Algorithm SHA256
```

当前 Mac 包使用临时签名（ad hoc），Windows 安装程序未签名；它们不是经 Apple 公证或经 Windows 发布者认证的安装包。哈希匹配用于核对下载与发布校验值，不能代替可信发布者的签名。

## 2. 安装并打开

### macOS

1. 下载 **DMG** 并打开，将 **TensorAgent.app** 拖到 **Applications（应用程序）**。
2. 推出磁盘映像，从“应用程序”或 Spotlight 打开 **TensorAgent**。
3. 如果 macOS 因开发者无法验证而阻止打开，先确认文件来自项目的发布页面。尝试打开后，前往 **系统设置 → 隐私与安全性 → 仍要打开**，按 macOS 提示确认。详见 Apple 的[打开可信应用说明](https://support.apple.com/en-us/102445)。

也可选择 **PKG**：双击后按 Installer 指引安装到 `/Applications/TensorAgent.app`，可能需要管理员授权。选择 **ZIP** 时，先解压再把 **TensorAgent.app** 移到“应用程序”。三种格式包含同一应用，任选一种即可；安装完成后打开已安装的应用，不要一直从挂载的 DMG 运行。

### Windows

1. 按上表下载 **CPU** 或 **CUDA MSI**，打开并完成当前用户的安装。
2. 从“开始”菜单打开 **TensorAgent**。安装位置为 `%LOCALAPPDATA%\Programs\TensorAgent`。
3. 如果 Windows 显示未知发布者提示，继续前核对来源和校验值。组织管理的电脑如阻止未签名应用，请遵守组织的安装策略。
4. 如果聊天窗口空白或报 WebView2 错误，安装 [WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)，然后重新打开 TensorAgent。离线电脑可下载 **x64** 独立安装程序。

选择 **ZIP** 时，使用“全部解压缩”，把完整文件夹放到可写位置，再打开 **TensorAgent.Maui.exe**。DLL、运行库、`webui` 和 `skills` 必须随完整文件夹保留；只复制 EXE 或直接在 ZIP 内打开无法正常运行。便携版与安装版使用相同的用户数据文件夹。

## 3. 配置模型并发送第一条消息

1. 如需更换界面语言，打开 **☰ → 设置 → 应用语言**。模型会用你发送消息的语言回复。
2. 系统盘空间不足时，先在 **设置 → 存储 → 模型下载和缓存文件夹** 填入可写的绝对路径，选择 **保存文件夹**。更改路径不会移动旧文件，详见[存储与备份](#存储与备份)。
3. 在空聊天页面选择模型，或打开 **☰ → 模型**。阅读输入能力、内存要求、许可与下载大小。
4. 为符合条件的聊天模型（如 **Gemma 4 E2B** 或 **Gemma 4 E4B (IQ4_XS)**）选择 **下载**。希望模型理解图片时，保持 **设置 → 下载 → 包含可选文件** 开启。模型下载需要网络，与代码工具的网络开关分开。
5. 等待下载和校验完成，选择 **使用**。等待加载完成、模型名出现在输入框上方。首次加载与首次回复可能因缓存准备而耗时较长。
6. 输入 `请用三个简短要点介绍你能帮我做什么。`，按 **Enter** 或发送按钮。**Shift+Enter** 换行，**停止** 中断回复，**新聊天** 开启独立对话。

下载中断会保留部分文件，回到“模型”选择 **继续** 或 **重试**。应用会在使用前校验模型文件；使用内置目录时无需手动下载 GGUF。

模型正在推理或回答时，仍可输入并发送新指令，例如 `请重点介绍安装步骤。`。输入框会显示 **调整回答**；发送后中断当前回复，并结合对话、已生成的部分回答和新指令继续。**停止** 仍可单独中止回复。连续发送的指令会等待前一轮生成和工具清理完成，再启动下一轮。

回答流式输出时，只有停留在底部聊天才会自动跟随。向上滚动阅读前面的内容时，画面会停在原处；点击输入框上方的 **↓** 按钮，或手动滚回底部，即可重新跟随回答。使用能说话的模型（LFM3-Omni）时，**朗读回复**会边生成边播放。

## 4. 使用附件、技能与生成文件

**询问文件或图片。** 通过 **+ → 文件** 或 **照片** 选择内容，等待上传完成，再发送具体问题。理解图片需要当前模型支持图像输入，并已安装视觉文件。卡片显示“文本已就绪 · 视觉文件未下载”时，选择 **添加视觉** 或 **启用视觉**，下载完成后再次选择 **使用**。音频和视频输入也取决于模型及平台，请查看卡片。

**查找以前的工作。** 通过 **☰ → 聊天** 打开保存的对话。完成的回复与产物链接自动保存。打开生成文件的链接即可查看或另存到应用之外。需要长期保留的文档、图片和视频应另外保存；产物文件放在应用缓存中，可能被清理。

**语音输入。** Mac 可长按输入框切换至按住说话，并在提示时授予麦克风与语音识别权限。支持时 Apple 听写请求设备端识别，否则 Apple 的识别服务可能在设备外处理音频。Windows 可在输入框使用系统语音键入（**Windows+H**）。

**使用技能。** 打开 **☰ → 技能**，文档或代码任务保持技能总开关开启。查看技能后可选择“在此聊天中使用”；关闭总开关即可进行不含技能列表的普通聊天。例如 `创建一个简短的 Word 行李清单文档，并提供文件` 需要代码执行权限与下节介绍的工具。

桌面共享技能目录现有 17 个包，新增 `find-skills`、`grill-me`、
`grill-with-docs`、`tdd` 和 `handoff`。较长的文档访谈、TDD 与交接流程
使用 Gemma 4 E4B、16K 上下文和 24 轮工具调用完成验证；不能据此推断
小模型或较低预算也能完成。实际覆盖与失败记录见[技能审核](agent_skills.md#audited-skillssh-bundle)（英文）。

**发现并安装技能。** 开启 **设置 → 沙盒 → 允许访问网络**，保持技能开启，
直接提出任务即可。没有已安装技能可用时，模型可以搜索 skills.sh，把找到的技能
固定到一个提交安装并在同一轮中使用，无需 Node.js、npm 或 Git，也无需你另行要求
安装（关闭网络开关即可停用）。对话会显示 `Installed <包> from <网址> @ <提交>`，
技能面板会显示每个中心技能的来源。搜索会先列出适合的已安装技能（网络查询即内置的
`research`）；若已安装技能已占用某个中心技能的名称，除非你按包名明确要求，否则不会
再装一份同名技能。开启“运行代码”后，内置 research 技能的网页搜索与网页读取会以
`web_search` 和 `web_fetch` 工具提供给模型，在沙盒中运行该技能的脚本。显式不选任何技能则关闭发现工具。
安装不执行任何代码：脚本经 `skills_run` 在“运行代码”权限下执行，并会先安装
脚本声明的 PEP 723 依赖；流程所需的凭据与其他程序仍需自行准备。
技能面板仍支持 ZIP 文件与 ZIP/列表 URL。验证记录保留了模型输出失败，
安装成功不代表任务完成，详见[发现与安装说明](agent_skills.md#discover-and-install-community-skills)（英文）。

**生成或编辑图片。** 内存足够时，下载并 **使用 Qwen-Image 2.1**。在输入框描述图片，或附上照片并说明要修改的内容；需要蒙版编辑时，通过图像编辑器选择区域。完成后可比较原图并再次编辑。像“调亮一点”这样的后续消息会修改上方的图片，新的描述则生成新图片；每张图片都会注明是如何生成的，最新一张还可改用另一种理解重新生成。模型名菜单中的 **LoRA 插件** 为可选项，需要单独下载。**Qwen-Image 2.1 Turbo** 用 8 步代替 40 步完成同样的工作，用时约为五分之一；它与 Qwen-Image 2.1 共用文本编码器、VAE 和视觉文件，同时使用两者也只需下载一次；它只提供在其上验证过的风格插件。

**生成视频。** 下载并 **使用 MiniMax-H3**，描述短视频，或附上作为起始画面的照片。在聊天中附加视频，可从该视频的精确末帧继续生成；在 Video Studio 中也可选择视频末帧作为下一段的起始画面。参考照片和录音可使用 MiniMax-H3 参考模式，切换模式时无需重新加载模型。**LTX 2.5 Distilled** 支持文本/图像生成带 48 kHz 立体声音频的视频，以及按时间位置和强度指定的图像关键帧。该条目为实验功能，需要 48 GB 系统内存档位。五个公开基础文件共 43.08 GB；可选且受限的 Ingredients IC-LoRA 增加 1.31 GB，需先获得 Hugging Face 授权。默认生成 17 帧预览，附上图像引导后可在输入框选择更长的片段及关键帧位置/强度。身份参考使用该 IC-LoRA 检查点，内置条目中的关键帧引导的是视频时间线。配置和实测限制见 [LTX 2.5 指南](models/ltx-2.5_zh-cn.md)。这些任务需要较高的桌面内存档位。Mac 应用已有媒体生成实测，Windows 图像 / 音频 / 视频生成尚未验证。详见[模型与验证说明](../TensorAgent/README.md#what-has-not-been-verified)。

## 5. 建立并使用知识库

1. 打开 **☰ → 知识库**，输入集合名称并选择 **创建**。
2. 选择 **上传文件** 或 **上传文件夹**；嵌套文件保留相对路径。**添加聊天** 可导入选中的已保存对话、可用附件和生成文件；导入的是快照，对话变化后需要再次导入以更新。
3. 检查抽取状态和警告，再搜索集合核对片段。`Ready`、`Partial` 和 `MetadataOnly` 表示实际建立了什么索引；保留原文件并不表示全部内容都已理解。
4. 选择 **在此聊天中使用**。已选集合显示在输入框上方；上传本身不会选择集合。新聊天不选择任何集合，重新打开已保存聊天会恢复其选择。
5. 提问时会自动检索相关片段。支持工具的模型还可继续搜索、读取周边内容；开启代码工具时可打开完整原文件进行计算。回答中的来源链接和检索来源列表可打开引用预览及原文件。

文本、代码、CSV/TSV、JSON、HTML、字幕与部分 Office 内容无需聊天模型即可导入。PDF 保留文本层，并对扫描或含图片页面执行本地 OCR：支持的 Mac 使用 Apple Vision，Windows 自动准备 Poppler/Tesseract。首次便携 OCR 配置需要联网，之后识别在本地完成。图像与视频采样帧需要已加载的兼容视觉模型；音频可使用已加载的音频塔或已配置的 `TS_JEV_TRANSCRIPTION_URL` 转录伴随服务。启用相应能力后，对已有仅元数据来源选择 **重新索引**，即可从保留的原文件提取内容。

默认使用本地 BM25 检索，可通过可选 GGUF 嵌入模型增加语义检索。抽取与生成的回答可能遗漏或误读细节，重要引用应核对原文。子代理继承同一集合选择，不能自行扩大范围。格式、预览、配额、OCR、嵌入配置与 API 示例见[知识库指南](knowledge-system_zh-cn.md)。

## 6. 配置可选代码工具

本地聊天与推理不需要 Python 或 Node。桌面技能使用电脑上安装的程序；桌面安装包**不包含 Python、Node、npm 或浏览器**。按任务需求从官方 [Python](https://www.python.org/downloads/) 与 [Node.js](https://nodejs.org/en/download) 页面安装，把程序加入 `PATH` 后重启 TensorAgent。Mac 应用读取登录 shell 的 `PATH`，Windows 使用进程环境。Python 文档技能还可能需要 `python-docx`、`python-pptx` 或 `openpyxl` 等包。技能说明和缺失工具报错会提示具体依赖；安装应用并不意味着所有内置技能的工具都已就绪。

打开 **☰ → 设置 → 沙盒**：

| 设置 | 作用 |
|---|---|
| **运行代码** | 允许 shell 命令与技能脚本，默认开启。普通聊天不需要执行程序时可关闭。 |
| **不使用沙盒运行**（Windows） | Windows 程序执行还需要此项，默认关闭。程序可访问你的 Windows 账户能够访问的文件和网络资源，仅在接受此权限时开启。 |
| **允许访问网络** | 原生技能发现/下载与代码工具的网络访问，默认关闭。研究或安装依赖时按需开启；它不控制模型下载。Windows 无沙盒进程没有强制的文件或网络边界。 |
| **子代理** | 让同一模型的辅助对话处理可独立执行的任务，默认开启；关闭可减少并行工作和内存使用。 |

macOS 使用 TensorSharp 的 Seatbelt 配置限制子进程，只允许写入聊天工作区及允许的临时目录。Windows 应用没有等效隔离。Windows 的 **不使用沙盒运行** 关闭时，仍可聊天、使用支持的附件并进行本地推理。“设置”的 **当前：** 行显示检测到的执行环境与 Python/Node 工具情况。网络和执行设置影响后续命令，子代理设置影响下一条消息。

尝试文档任务时，先安装依赖，选择支持工具调用的文本模型，开启相关技能和权限，再请求一个小文件。输入框上方会显示活动状态，完成后打开文件链接。浏览器任务另见 [Playwright 指南](playwright_agent.md)（英文）。

## 存储与备份

安装程序与个人数据位于不同位置：

| 内容 | macOS | Windows |
|---|---|---|
| 设置、保存的聊天、已安装技能 | `~/Library/Application Support/TensorAgent` | `%LOCALAPPDATA%\TensorAgent\Data` |
| 知识库原文件与索引 | `~/Library/Application Support/TensorAgent/knowledge` | `%LOCALAPPDATA%\TensorAgent\Data\knowledge` |
| 默认模型、LoRA、上传、工作区、产物与日志 | `~/Library/Caches/TensorAgent` | `%LOCALAPPDATA%\TensorAgent\Cache` |
| 警告 / 错误日志 | `~/Library/Caches/TensorAgent/logs/errors.log` | `%LOCALAPPDATA%\TensorAgent\Cache\logs\errors.log` |

**设置 → 存储** 显示有效模型路径和数量。迁移模型时，等下载 / 导入完成后关闭应用，把原目录内的 `<catalog-id>` 模型子文件夹复制到新目录，再打开应用并保存新目录的完整路径。应用不会自动移动模型；**使用默认文件夹** 可恢复默认路径。LoRA 和其他缓存仍留在原缓存位置。

备份时先关闭 TensorAgent，再复制数据文件夹及需保留的生成文件。如果希望已保存聊天中的上传文件和产物链接继续有效，还需备份缓存。自定义模型文件夹需要另外复制，以免再次下载权重。**模型 → 删除** 移除模型；**设置 → 删除所有聊天** 移除聊天及关联工作。已导入知识库的原文件保留在独立数据目录中，需在 **知识库** 中删除对应来源。删除无法撤销。

## 更新或卸载

应用内没有自动更新器。下载下个发布的对应安装包，并在更新前关闭 TensorAgent。Mac 可替换“应用程序”中的应用或运行新 PKG。Windows 运行新版 MSI，CPU 和 CUDA 安装程序更新同一应用。ZIP 版本应完整解压到新目录，避免混用版本文件。每次只运行一个实例。

替换程序不会删除独立的数据与模型文件夹。更新前请备份；新的模型目录可能退役旧条目并回收其下载文件。兼容性变化以发布说明为准。

Mac 卸载时把 **TensorAgent.app** 移到废纸篓；Windows 使用 **设置 → 应用 → 安装的应用 → TensorAgent → 卸载**（Windows 10 为“应用和功能”）；ZIP 版删除其程序文件夹即可。个人数据和缓存会保留。若需一并删除，先关闭应用并另存需要的内容，再删除上表中的 TensorAgent 目录及自定义模型目录。

## 故障排查

| 问题 | 可以尝试 |
|---|---|
| Assets 中没有 `tensoragent-desktop-` 文件 | 该发布没有桌面安装包。检查其他发布或成功的打包运行，或在包含安装包的发布出现前从源码构建。 |
| Mac 应用打不开 | 确认 Apple Silicon 和 macOS 14+，核对校验值，按上文 Apple 的可信应用流程打开。源码构建的系统下限可能不同。 |
| Windows 聊天窗口空白 | 安装 / 更新 WebView2 Evergreen Runtime，重新打开，并确认完整安装或解压所有文件。 |
| CUDA 包无法启动或加载模型 | 检查 NVIDIA 驱动及 GPU 兼容性，试用 CPU 包以区分 GPU 与应用问题；不要混用两个版本的文件。 |
| 模型显示 **过大** | 物理内存低于模型档位，增加磁盘空间或分页文件不会改变目录资格。改选符合条件的模型。 |
| 下载停止或校验失败 | 检查网络、可写目录和剩余空间，选择 **继续 / 重试**；持续失败时从“模型”删除该模型后重新下载。 |
| 无法理解附件图片 | 确认支持图像输入，下载视觉文件并重新选择 **使用**；等全部上传完成再发送。 |
| 命令或文档技能无法运行 | 检查 **运行代码**、Windows **不使用沙盒运行**、Python/Node/依赖包和 **当前：** 行；安装工具后重启。 |
| 研究或依赖安装无法联网 | 检查 **允许访问网络** 与系统 / 组织限制；模型下载的联网权限独立。 |
| 回复慢或内存不足 | 关闭其他大型应用，改选小模型，尝试新聊天并关闭子代理。CPU 推理和首次预热可能较慢。KV 缓存精度需重新加载模型才生效，并非所有模型都支持。 |

无法解决时，提交 [GitHub issue](https://github.com/zhongkaifu/TensorSharp/issues)，附上发布版本、安装包变体、系统、内存 / GPU、所选模型、原始错误及 `errors.log` 中的相关片段。分享前检查日志，可能包含提示、文件名或工具输出。没有模型 / 设备与已知媒体验证缺口不计为验证通过；[覆盖范围说明](../TensorAgent/README.md#what-has-not-been-verified)记录实际执行的验证。
