# 在 Pi coding agent 中使用 TensorSharp

[Pi](https://pi.dev)（`@earendil-works/pi-coding-agent`）是一个终端编程智能体：它用自带的工具读文件、
执行 shell 命令、修改代码，可以接入任何 OpenAI 兼容的端点。本示例把它接到本机的 TensorSharp 服务端上，
服务端运行 Qwen3.8-27B 在 Apple Silicon 上最快的配置：

- UD-IQ3_XXS GGUF，后端 `ggml_metal`；
- DFlash2 块草稿模型，每步 7 个草稿，不设置信度门限；
- 贪心解码；
- 64K 上下文，q8_0 KV 缓存。

示例提供两种用法：`pi` 命令和一个调用 Pi SDK 的 Node.js 小程序。两种用法都会让 Pi 修好一个示例项目中失败的测试。

[English](README.md)

| 路径 | 内容 |
|---|---|
| [`tensorsharp-pi.json`](tensorsharp-pi.json) | 面向 Pi 的服务端配置叠加层，叠加在预设 [`config/qwen3.8-27b-metal-dflash2.json`](../../config/qwen3.8-27b-metal-dflash2.json) 之上 |
| [`scripts/start-server.sh`](scripts/start-server.sh) | 以预设加叠加层启动 `TensorSharp.Server.Host` |
| [`pi-agent/models.json`](pi-agent/models.json) | Pi 的 provider 配置：端点、模型 id、上下文窗口、思考开关 |
| [`pi-agent/settings.json`](pi-agent/settings.json) | 把该模型设为 Pi 的默认模型 |
| [`scripts/pi.sh`](scripts/pi.sh) | 用独立的配置目录（`~/.pi/tensorsharp-agent`）运行 `pi`，不改动你的 `~/.pi/agent` |
| [`sdk/run-task.mjs`](sdk/run-task.mjs) | 以库的方式使用 Pi：执行一个编程任务，流式输出文本与工具调用 |
| [`sample-project/`](sample-project/) | 一个 Node.js 小模块：3 个 bug，5 个测试，其中 3 个失败 |
| [`scripts/e2e.sh`](scripts/e2e.sh) | 端到端检查：CLI 与 SDK 各自修复一份全新的示例项目副本 |

## 环境要求

| | 版本或大小 |
|---|---|
| Mac | Apple Silicon，统一内存 32 GB 或以上。权重 10.9 GB，草稿模型 1.1 GB，其余是 KV 缓存和 Metal 缓冲区。验证机器为 M5 Pro 48 GB，macOS 27.2 |
| .NET | .NET 10 SDK、CMake 3.20+、Xcode 命令行工具，用于从源码构建 TensorSharp（见[入门指南](../../docs/getting_started.md#install-and-first-run)） |
| Node.js | 22.19 或更高（Pi 的要求），验证版本 26.8.1 |
| Pi | `@earendil-works/pi-coding-agent` 1.5.1（旧包 `@mariozechner/pi-coding-agent` 已弃用） |
| 磁盘 | 两个模型文件共 12 GB，首次启动时自动下载 |

请从源码构建。Release 压缩包只包含其标签对应的功能；本示例依赖服务端一项新修复：工具循环的缓存能够延续
（见[缓存如何跟上 Pi](#缓存如何跟上-pi)）。

## 1. 构建 TensorSharp

在仓库根目录执行：

```bash
dotnet build TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true
```

这一步会构建带 Metal 的 native ggml 库，并生成 `TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll`，
`start-server.sh` 运行的就是它。该文件不存在时，脚本改用会先构建的
`dotnet run --project TensorSharp.Server.Host -c Release ...`。

## 2. 启动服务端

```bash
examples/pi-coding-agent/scripts/start-server.sh
```

脚本实际执行的是：

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --config config/qwen3.8-27b-metal-dflash2.json \
  --config examples/pi-coding-agent/tensorsharp-pi.json
```

首次启动时，预设会从固定的 Hugging Face 版本下载以下两个文件，并校验 sha256：

- `Qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf`（来自 unsloth）；
- `Qwen/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（来自 z-lab）。

文件保存在 `$TENSORSHARP_MODELS` 下，默认 `<repo>/models`。已有这些文件时，把该变量指向它们的上级目录：

```bash
TENSORSHARP_MODELS=~/work/models examples/pi-coding-agent/scripts/start-server.sh
```

脚本名后面的参数会原样传给服务端，优先级高于两个配置文件，例如 `--port 5001`。

`/v1/models` 列出模型时，服务端即已就绪：

```bash
curl -s http://127.0.0.1:5000/v1/models
# {"object":"list","data":[{"id":"Qwen3.8-27B-UD-IQ3_XXS","object":"model","owned_by":"local","loras":[]}]}
```

### 叠加层改了什么

预设提供模型、草稿模型、速度设置（7 个草稿，`spec-pmin` 为 0）、贪心采样和 `no-skills`。后加载的 `--config`
文件会覆盖前面文件里设过的值，服务端会逐项在日志中说明。叠加层设置了以下各项：

| 设置 | 值 | 原因 |
|---|---|---|
| `host`、`port` | `127.0.0.1`、`5000` | 只有本机能访问。服务端本身没有认证，请勿对外暴露 |
| `max-tokens` | `16384` | 配置文件里的 `max-tokens` 是固定上限，也会限制每个请求。预设的 1,024 会让思考模型在写代码中途被截断。Pi 按模型的 `maxTokens` 申请，这里是 16,384 |
| `mmproj` | `"none"` | Pi 只发文本。跳过 0.93 GB 的视觉投影器及其下载 |
| `kv-cache-dtype` | `q8_0` | KV 显存减半，解码速度不变 |
| `environment.MAX_CONTEXT` | `65536` | Pi 的系统提示词与工具定义在第一条消息之前就占约 4K token，文件内容和测试输出也增长很快。64K 的 q8_0 KV 在 32 GB 的 Mac 上能与权重共存 |

## 3. 安装并配置 Pi

```bash
npm install -g @earendil-works/pi-coding-agent@1.5.1
pi --version
```

Pi 从配置目录下的 `models.json` 读取 provider。本服务端对应的配置见
[`pi-agent/models.json`](pi-agent/models.json)：

```json
{
  "$schema": "https://pi.dev/schemas/models.schema.json",
  "providers": {
    "tensorsharp": {
      "name": "TensorSharp (local)",
      "baseUrl": "http://127.0.0.1:5000/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "compat": {
        "supportsDeveloperRole": true,
        "supportsReasoningEffort": false,
        "supportsStore": false,
        "maxTokensField": "max_completion_tokens",
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "Qwen3.8-27B-UD-IQ3_XXS",
          "name": "Qwen3.8 27B (TensorSharp, DFlash2)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 16384,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

- `id` 必须与 `GET /v1/models` 列出的名字完全一致，即 GGUF 文件名去掉 `.gguf`。写错会得到 HTTP 404。
- `contextWindow` 与服务端的 `MAX_CONTEXT` 一致。上下文距离它不足 `compaction.reserveTokens`（默认 16,384）时，
  Pi 开始压缩旧轮次，所以本配置下约在 48K token 时压缩。
- `maxTokens` 与叠加层的 `max-tokens` 一致。
- `"thinkingFormat": "qwen-chat-template"` 通过 `chat_template_kwargs.enable_thinking` 开关 Qwen 的思考。
  Qwen3.8 只有这一个开关：Pi 的 `off` 级别关闭思考，其他级别都打开思考。`supportsReasoningEffort: false`
  让 Pi 不再额外发送 `reasoning_effort`。
- `supportsDeveloperRole: true`：Pi 以 `developer` 消息发送指令，TensorSharp 会把它渲染为系统消息。
- `apiKey` 可以是任意字符串，服务端不检查。

[`pi-agent/settings.json`](pi-agent/settings.json) 把该模型设为默认，并打开思考：

```json
{
  "defaultProvider": "tensorsharp",
  "defaultModel": "Qwen3.8-27B-UD-IQ3_XXS",
  "defaultThinkingLevel": "medium",
  "enableInstallTelemetry": false
}
```

这两个文件有两种安装方式：

- **独立配置目录（脚本采用的方式）。** `scripts/pi.sh` 设置 `PI_CODING_AGENT_DIR=~/.pi/tensorsharp-agent`，
  首次运行时把两个文件复制过去。之后不再覆盖，你在那里做的修改会保留。你平时的 `~/.pi/agent` 不受影响。
- **你平时的 Pi 配置。** 把 `tensorsharp` provider 合并进 `~/.pi/agent/models.json`，再用
  `pi --model tensorsharp/Qwen3.8-27B-UD-IQ3_XXS` 或 Pi 的模型选择器选中它。

`pi.sh` 还设置了 `PI_OFFLINE=1`、`PI_SKIP_VERSION_CHECK=1`、`PI_TELEMETRY=0`。这样 Pi 自身不发起任何网络请求：
不刷新模型目录、不检查版本、不上报遥测。

## 4. 使用 Pi

交互式使用，在要处理的项目目录中：

```bash
cd ~/src/my-project
~/work/TensorSharp/examples/pi-coding-agent/scripts/pi.sh
```

不进入交互界面、只执行一个任务，Pi 会输出最终回答。在脚本中运行时要把 stdin 设为空，
否则 `pi -p` 会一直等待输入：

```bash
cd ~/src/my-project
~/work/TensorSharp/examples/pi-coding-agent/scripts/pi.sh -p "Explain what src/stats.js does" < /dev/null
```

其他常用选项：

```bash
pi.sh --thinking off -p "..." < /dev/null              # 不思考：更快，适合简单修改
pi.sh --mode json --no-session "..." < /dev/null       # 每行一个 JSON 事件，不保存会话
pi.sh --continue                                       # 继续本目录最近的会话
```

## 5. 在代码中使用 Pi（SDK）

[`sdk/run-task.mjs`](sdk/run-task.mjs) 以库的方式运行一个任务，使用同一份 `models.json`。核心代码：

```js
import { createAgentSession, ModelRuntime, SessionManager } from "@earendil-works/pi-coding-agent";

const modelRuntime = await ModelRuntime.create({
  modelsPath: path.join(agentDir, "models.json"),
  authPath: path.join(agentDir, "auth.json"),
});
const model = modelRuntime.getModel("tensorsharp", "Qwen3.8-27B-UD-IQ3_XXS");

const { session } = await createAgentSession({
  cwd: projectDir,                           // Pi 的 read/bash/edit/write 工具在此目录下工作
  agentDir,
  modelRuntime,
  model,
  thinkingLevel: "medium",
  sessionManager: SessionManager.inMemory(), // 不写会话文件
});

session.subscribe((event) => {
  if (event.type === "tool_execution_start") console.log(`[tool] ${event.toolName}`, event.args);
  if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta")
    process.stdout.write(event.assistantMessageEvent.delta);
});

await session.prompt("Run the test suite with `npm test`. Fix the bugs in src/stats.js ...");
console.log(session.getLastAssistantText());
session.dispose();
```

运行前先安装 SDK。智能体会修改文件，所以请对示例项目的副本运行：

```bash
cd examples/pi-coding-agent/sdk
npm install
cp -R ../sample-project /tmp/stats-demo
PI_CODING_AGENT_DIR=~/.pi/tensorsharp-agent PI_OFFLINE=1 node run-task.mjs /tmp/stats-demo
```

`PI_CODING_AGENT_DIR` 目录中必须有 `models.json`。`scripts/pi.sh` 首次运行时会创建 `~/.pi/tensorsharp-agent`；
没运行过的话，可以手动创建：

```bash
mkdir -p ~/.pi/tensorsharp-agent && cp ../pi-agent/*.json ~/.pi/tensorsharp-agent/
```

`TENSORSHARP_URL` 可以覆盖 provider 的 `baseUrl`。程序会输出智能体的文本、每次工具调用、最终回答，
以及 token 统计（包括服务端从前缀缓存中复用了多少提示词 token）。

## 6. 端到端检查

[`scripts/e2e.sh`](scripts/e2e.sh) 把同一个任务交给 Pi 两次：一次通过 `pi` 命令，一次通过 SDK，
每次都使用 `sample-project/` 的全新副本。任务是：运行 `npm test`，在不改测试的前提下修好 `src/stats.js`，
再运行一次测试。示例项目里有三个 bug：`median` 把数字按字符串排序；`median` 在元素个数为偶数时没有取中间两个数的平均；
`percentile` 下标差一。

之后脚本自己检查每份副本。只有测试文件与原文件逐字节相同，并且 `node --test` 报告 5 个测试全部通过，才算 PASS。

```bash
examples/pi-coding-agent/scripts/start-server.sh &   # 或者用 START_SERVER=1 让 e2e.sh 自行启动
examples/pi-coding-agent/scripts/e2e.sh
```

```text
==> server: {"object":"list","data":[{"id":"Qwen3.8-27B-UD-IQ3_XXS",...}]}
==> pi (CLI) on .../examples/pi-coding-agent/.work/cli-project
    88 s; tool calls: 10
PASS [cli]: # tests 5 # pass 5 # fail 0
==> pi (SDK) on .../examples/pi-coding-agent/.work/sdk-project
...
--- 8 model requests in 83.4 s: 6204 prompt tokens (41962 served from the server's prefix cache), 2117 generated ---
PASS [sdk]: # tests 5 # pass 5 # fail 0
```

两次都通过时脚本才返回 0。副本、Pi 的 JSON 事件、测试输出和日志都写到 git 忽略的 `.work/`，可以用 `WORK_DIR`
改到别处。其他设置：`TENSORSHARP_URL`（默认 `http://127.0.0.1:5000/v1`）和 `PI_BIN`（不在 `PATH` 上的 `pi`）。

### 实测结果

2026-11-09 在 M5 Pro 48 GB、macOS 27.2 上测得。软件版本：TensorSharp `b722bad1`（ggml `cd6d2b7`，未修改的上游代码）、
Pi 1.5.1、Node.js 26.8.1。每种方式各运行一次。模型每次输出不同，所以轮数和工具调用次数也会变化。

| 运行方式 | 结果 | 总耗时 | 模型请求数 | 由缓存提供的提示词 token |
|---|---|---|---|---|
| `pi` CLI（`--mode json`） | PASS，5/5 测试通过，测试文件未改动 | 88 s | 10（10 次工具调用） | 首个请求之后每个请求 86-99% |
| SDK（`run-task.mjs`） | PASS，5/5 测试通过，测试文件未改动 | 83.4 s | 8 | 48,166 中的 41,962（87%） |

服务端日志中的逐请求数据：

- **会话的第一个请求：** 3.9-4.1K token 的提示词（Pi 的系统提示词、工具定义和任务）需要冷预填充，首 token 时间 13-14 s。
- **之后的每一轮：** 提示词 4.2K-9.1K token，首 token 时间 0.45-3.5 s。
- **解码：** 20-43 tok/s。写代码时 DFlash2 草稿大多被接受；草稿被拒较多时（例如普通文字和简短回答），
  投机解码调速器会退回普通解码，约 21 tok/s。

这只是一个示例项目上的结果，不是编程能力的基准测试，也只覆盖上述 Apple Silicon 配置。

## 缓存如何跟上 Pi

Pi 自己执行工具。每次工具调用之后，它都会重新发送整段对话：系统提示词、带推理内容的历史轮次、工具调用和工具结果。
服务端复用上一轮的 KV 状态，只预填充新增的 token。前提是请求中的历史轮次与服务端当初生成的内容完全一致。

Pi 回传每轮推理时保留了流式输出中的原样内容，包括 `</think>` 之前的换行。服务端以前拿它和自己去掉首尾空白的副本比较，
两者永远不相等。结果每一轮都只复用了共享的系统前缀，约 3.8K token，到第六轮时首 token 时间已增长到 5.3 s。
现在服务端比较时忽略这些首尾空白，每一轮都能接着上一轮继续（提交 `b722bad1`）。以下两种情况仍会重新预填充对话：

- 客户端修改了之前的推理或消息；
- 在另一个目录中开始新会话，因为 Pi 的系统提示词里写有工作目录。

自带 `tools` 的请求不会得到服务端的任何工具（技能、代码执行、子智能体），模型看到的正好是 Pi 的四个工具。
参见 [Agent clients that run their own tools](../../TensorSharp.Server.Host/API_EXAMPLES.md#agent-clients-that-run-their-own-tools-pi-codex-opencode)（英文）。

## 其他硬件

本示例只在 Apple Silicon 上用上述预设验证过。在 NVIDIA GPU 上，Qwen3.8-27B 最快的预设是
[`config/qwen3.8-27b-ninfer.json`](../../config/qwen3.8-27b-ninfer.json)：直接 CUDA 引擎运行 NVFP4 NInfer 文件，
带它自己的 DFlash2 权重。用 `TENSORSHARP_PRESET` 选择它：

```bash
TENSORSHARP_PRESET=config/qwen3.8-27b-ninfer.json examples/pi-coding-agent/scripts/start-server.sh
```

该预设在自己的 `environment` 中把 `MAX_CONTEXT` 设为 40960。配置文件之间后者优先，所以叠加层的 65536 会覆盖它；
shell 中导出的 `MAX_CONTEXT` 又优先于两者。例如 `MAX_CONTEXT=40960 TENSORSHARP_PRESET=... scripts/start-server.sh`
保留预设的长度。请把 `models.json` 的 `contextWindow` 设为服务端实际使用的长度。

这一组合尚未与 Pi 一起运行过，包括叠加层的 q8_0 KV 缓存在直接 CUDA 引擎上的效果。

## 故障排查

| 现象 | 原因与解决 |
|---|---|
| 脚本中的 `pi -p` 一直不返回 | Pi 在等待 stdin。加上 `< /dev/null` |
| `404` / `model not found` | `models.json` 中的 `id` 与 `GET /v1/models` 不一致。应为 `Qwen3.8-27B-UD-IQ3_XXS` |
| 回答在 1,024 token 左右截断 | 叠加层没有生效。它必须放在预设之后：`--config <preset> --config tensorsharp-pi.json` |
| `context_length_exceeded` | 对话超出了服务端的上下文窗口。Pi 收到该错误会自动压缩并重试。如果调低了 `MAX_CONTEXT`，请同时调低 `contextWindow`，并在 Pi 的 `settings.json` 中设置 `compaction.reserveTokens`（例如 32K 窗口用 8192） |
| 第一个回答很慢 | 第一个请求要预填充 Pi 约 4K token 的系统提示词（13-14 s），之后的轮次会复用它。刚启动的服务端还要先加载 12 GB 权重 |
| `Address already in use` | 5000 端口已有其他服务。给 `start-server.sh` 加 `--port 5001`，并修改 `models.json` 的 `baseUrl`（`e2e.sh` 与 SDK 用 `TENSORSHARP_URL`） |
| 32 GB 的 Mac 内存紧张 | 缩短上下文：`MAX_CONTEXT=32768 scripts/start-server.sh`（shell 中导出的变量优先于配置文件）。同时修改 `models.json` 的 `contextWindow`，并按上文调低 `compaction.reserveTokens` |
