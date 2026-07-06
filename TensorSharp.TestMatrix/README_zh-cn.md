# TensorSharp.TestMatrix

[English](README.md) | [中文](README_zh-cn.md)

TensorSharp 推理测试与基准矩阵运行器。它通过把 `TensorSharp.Cli` 作为子进程
启动，扫描 **(model × backend × feature × env-var)** 的组合，从结构化日志中
解析每次运行的指标，为每个 cell 保存一个 JSON 文件，并生成汇总 Markdown 报告。

配套文档 [`docs/env_var_feature_matrix_zh-cn.md`](../docs/env_var_feature_matrix_zh-cn.md)
是环境变量会影响哪些功能的权威参考。新增或删除开关时请同步更新两边。

## 覆盖内容

| 维度 | 取值 |
|---|---|
| **模型** | 默认从 `/Users/ZhongkaiFu/work/model` 自动发现 GGUF（可配置）；也支持显式配置覆盖。DiffusionGemma 目前还不会被自动分类为一等矩阵模型家族；实验时请使用显式配置项。 |
| **后端** | `cpu`、`ggml_cpu`、`ggml_metal`、`ggml_cuda`、`ggml_vulkan`、`cuda`、`mlx`（按主机能力过滤） |
| **功能 / prompt 类型** | 自回归 CLI 功能：合成 prefill（512、2048）、合成 decode（128）、短文本、长文本、上传文本、多轮聊天、函数 / 工具调用、思维链模式、图像、音频、视频。目前没有专门的 diffusion feature。 |
| **环境变量 sweep** | Baseline cell 加上 [`Defaults/matrix-config.json`](Defaults/matrix-config.json) 中 `default_env_vars` 选择的高影响开关。已注册全集在 `EnvVarMatrix.All`；DiffusionGemma 的 `DIFFUSION_*` 变量当前在矩阵外，默认不会清理或 sweep。详见[矩阵文档](../docs/env_var_feature_matrix_zh-cn.md)。 |

## 构建

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release   # 必需：子进程驱动目标
```

运行器需要已构建的 `TensorSharp.Cli`。默认会在自身输出目录的相对路径
`../TensorSharp.Cli/bin/TensorSharp.Cli` 附近自动发现；也可以用
`--cli-executable <path>` 或配置里的 `cli_executable` 覆盖。

## 运行

默认运行（主机支持的全部默认 backend / feature / env-var sweep）：

```bash
dotnet run --project TensorSharp.TestMatrix -c Release
```

每个适用 cell 都会先运行一个不强制环境变量的 baseline。Sweep case 会在清理继承的
`TS_*` 等相关变量后，只传入一个 env var/value pair。

交互式开发常用的子集：

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- \
  --backends ggml_metal,mlx \
  --features short_text,long_text,image \
  --env-vars KV_CACHE_DTYPE,TS_QWEN35_BATCHED \
  --model-dir /Users/ZhongkaiFu/work/model \
  --results-dir results/dev \
  --report report-dev.md
```

Dry run（只打印将要执行的 case id）：

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

只运行 baseline cell，关闭环境变量 sweep：

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- --env-vars none
```

断点续跑，跳过磁盘上已有的 cell：

```bash
dotnet run --project TensorSharp.TestMatrix -c Release -- --skip-existing
```

## 配置

JSON 配置文件控制模型发现、默认值与每模型覆盖。默认配置位于
[`Defaults/matrix-config.json`](Defaults/matrix-config.json)，构建时会复制到程序集旁边。
命令行覆盖项优先于配置文件。

每模型覆盖示例：

```json
{
    "models": [
        {
            "id": "gemma4-e4b-q8",
            "family": "gemma4",
            "display_name": "Gemma 4 E4B Q8_0",
            "gguf": "gemma-4-E4B-it-Q8_0.gguf",
            "mmproj": "gemma-4-mmproj-F16.gguf",
            "supports_image": true,
            "supports_audio": true,
            "supports_video": true,
            "supports_tools": true,
            "supports_thinking": true,
            "enabled": true
        }
    ]
}
```

相对路径会基于 `model_dir` 解析。自动发现的模型如果与配置项共享同一个 `id`，
配置项会替换自动发现结果。

DiffusionGemma GGUF 在 `ModelDiscovery` 支持 `diffusion-gemma` /
`diffusion_gemma` 架构识别之前，应通过显式 `models[]` 条目加入。当前 feature
目录仍偏向 append/decode 式自回归生成；如需把 TestMatrix 输出作为正式 diffusion
回归信号，请先新增专门的 diffusion feature。

## 媒体文件（图像 / 音频 / 视频）

`Inputs` 目录只随仓库提供 prompt，不包含媒体文件。请把样例媒体放到：

- `TensorSharp.TestMatrix/Inputs/media/apple.png`
- `TensorSharp.TestMatrix/Inputs/media/sample.mp3`
- `TensorSharp.TestMatrix/Inputs/media/sample.mp4`

也可以在配置中设置 `media_dir` 指向共享资源目录。媒体文件缺失时，对应 case 会给出
明确错误；不会静默跳过。

## 输出

- `results/<case-id>.json`：每个 cell 一个文件，包含完整指标、stdout 尾部与正确性结论
- `report.md`：聚合 Markdown 报告，包含 baseline 表、环境变量 sweep 表、敏感性摘要、跳过组合、正确性失败与运行时失败

CI 上传的是报告文件；JSON 文件是后续与历史运行做 delta 分析的事实来源。

## “通过”代表什么

一个 cell 只有同时满足以下条件，才会标记为 `ok = true`：

1. `TensorSharp.Cli` 退出码为 0（无崩溃、无超时）。
2. 从结构化日志中解析到 `prefill_tps`、`decode_tps` 或 `model_load_ms` 中的至少一个，证明模型已加载并执行。
3. **正确性检查**：如果该 feature 配置了 `ExpectedContains`，assistant 输出中必须包含所有要求子串（大小写不敏感）。失败会单独归入“Correctness failures”，与运行时失败区分。

默认期望子串定义在 [`Matrix/FeatureCatalog.cs`](Matrix/FeatureCatalog.cs)：

| 功能 | 输出中必须包含 |
|---|---|
| `short_text`（“why is the sky blue”） | `blue` |
| `long_text`（分页 KV 报告摘要） | `paged` |
| `uploaded_text`（服务日志分析） | `08:01:12` |
| `multi_turn`（三轮姓名 / 颜色记忆） | `alex`, `teal` |
| `tools`（东京天气） | `get_current_weather`, `tokyo` |
| `thinking`（双火车应用题） | `10:38` |
| `image`（默认 apple.png） | `apple` |
| `pp512`, `tg128`, `pp2048` | 无，合成基准不检查文本 |
| `audio`, `video` | 默认无，取决于运行环境提供的样例媒体 |

该检查刻意较弱：单个关键词是正确答案的必要条件而非充分条件。它用于捕获灾难性
回归，例如模型加载后生成乱码、多轮 KV 复用失效、工具名没有输出，而不是做模型质量评测。
如需更强验证，可覆盖某个 feature 的 `ExpectedContains`，或加入与黄金输出对比的回归模式。

如果更换默认媒体文件（图像、音频、视频），请同步更新对应 feature 的
`ExpectedContains`。

## 回归检测（baselines）

矩阵的价值在于发现回归。运行器会从
[`Baselines/baseline-<host>.json`](Baselines/README_zh-cn.md) 加载每类主机的基线，
并与当前运行结果比较：

- **新的运行时失败**：过去通过，现在崩溃 / 超时 / 无指标
- **新的正确性失败**：过去通过，现在缺少 `ExpectedContains` 子串
- **吞吐回归**：decode TPS（或 prefill-only 场景的 prefill TPS）下降超过 `--regression-threshold-pct`（默认 10%）
- **改进**：过去失败，现在通过（信息性）
- **未跟踪**：该 cell 尚无 baseline 条目（新 cell）

运行时 / 正确性 / 吞吐回归是阻断项：设置 `--fail-on-regression` 时运行器会非零退出，
使 PR 失败。改进和未跟踪 cell 只作为信息展示。

```bash
# 与提交的 baseline 对比，并在阻断回归时失败
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll --fail-on-regression

# 合理变更后重新认可 baseline（diff 需自行提交）
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll --update-baseline

# 与指定 baseline 文件对比（例如和旧快照 A/B）
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll \
  --baseline ./oldbaseline.json --fail-on-regression
```

文件格式、主机分类和更新流程见 [`Baselines/README_zh-cn.md`](Baselines/README_zh-cn.md)。

## 扩展矩阵

| 要新增 | 修改位置 |
|---|---|
| 新后端 | [`Matrix/Backend.cs`](Matrix/Backend.cs)：新增 `BackendInfo` 并注册到 `BackendCatalog.All` |
| 新 prompt 类型 / feature | [`Matrix/FeatureCatalog.cs`](Matrix/FeatureCatalog.cs) + [`Runners/CliRunner.cs`](Runners/CliRunner.cs) 的 `BuildArgs` switch；把新 prompt 放到 `Inputs/prompts/` |
| 新环境变量 sweep | [`Matrix/EnvVarMatrix.cs`](Matrix/EnvVarMatrix.cs)：新增带 `AppliesTo` 谓词的 `EnvVarSpec`；如需默认运行则加入 [`Defaults/matrix-config.json`](Defaults/matrix-config.json)；同步更新 [`docs/env_var_feature_matrix_zh-cn.md`](../docs/env_var_feature_matrix_zh-cn.md) 及英文版 |
| 自动发现漏掉的模型 | 在 `matrix-config.json` 的 `models[]` 中加入 `ModelConfig` |
| DiffusionGemma 覆盖 | 增加家族识别或显式配置、在 `FeatureCatalog.cs` 中新增 diffusion feature、为 diffusion 生成补充 CLI 参数映射，并在需要 sweep 时注册 `DIFFUSION_*` env-var |

## CI

仓库的 GitHub Actions workflow [`test-matrix.yml`](../.github/workflows/test-matrix.yml)
使用标记为 `tensorsharp-cuda` 与 `tensorsharp-mlx` 的自托管 runner。

为降低 CI 时延，**自动触发（每个 PR、每次推送到 main）只运行快速的「smoke」
组合**，而不是完整矩阵：

| | Smoke（自动：PR / push） | 完整（手动 / 本地） |
|---|---|---|
| 特性 | `pp512`、`tg128`、`short_text` | 全部（12 个） |
| 后端 | 仅生产快后端（`ggml_cuda,cuda` / `ggml_metal,mlx`） | 全部（含纯 C# 的 `cpu`） |
| 环境变量 sweep | 无（仅 baseline cell） | 全部 `default_env_vars` |
| 回归门禁 | 是（`--fail-on-regression`） | 是 |

Smoke 组合仍能抓住每次提交最关心的问题——吞吐回归（`pp512`/`tg128`）或模型能
加载但输出乱码（`short_text`）——同时去掉了占主要耗时的环境变量 sweep。

**运行完整矩阵：** 它不再自动运行。可在 Actions 页通过 **`workflow_dispatch`**
按需触发（把 `backends` / `features` / `env_vars` 输入留空即跑完整矩阵，或填入
自定义子集），或在本地运行任意子集——见上文[运行](#运行)。PR 评论仍会包含上传的
`report.md` artifact 链接。runner 准备要求（模型目录、NVIDIA 驱动等）见 workflow 文件。
