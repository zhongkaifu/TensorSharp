# 子智能体

[English](sub_agents.md) | [中文](sub_agents_zh-cn.md)

子智能体是模型自己为某一项任务启动的另一个模型副本。它拥有与启动它的智能体相同的工具，
在同一个工作目录中工作，但拥有**自己的上下文**：它看到的是自己的任务，而不是整段对话。
它与父智能体并行工作，最后交回一个最终答案。

TensorSharp 采用 OpenAI Codex 所用的那套工具接口来实现这一功能，并把每个子智能体作为
普通序列运行在**同一个已加载模型**上。连续批处理引擎会把父智能体与它的所有子智能体放在
一起解码；每个子智能体的提示词都以父智能体一字不差的指令与工具块开头，因此引擎会复用这段
已缓存的前缀，而不必重新预填充。

子智能体**默认关闭**。用 `--sub-agents` 开启。

## 快速开始

```bash
TensorSharp.Server --model gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal \
    --code-exec --sub-agents
```

然后明确要求委派——工具说明会告诉模型：只有在用户或它正在遵循的某个技能要求时，才启动
子智能体：

```bash
curl -s http://localhost:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-E4B-it-Q8_0",
  "messages": [{"role": "user", "content":
    "Use three sub-agents working in parallel, one per task: compute the sum of 1..1000, the number of primes below 10000, and the 50th Fibonacci number, each with a python3 program. Wait for all three and reply with the three numbers."}]
}'
```

每个聊天入口都会得到这些工具：Web UI、`/v1/chat/completions`、`/v1/responses` 以及
Ollama 的 `/api/chat`。它们只在本来就带有工具——Agent Skills 或 `--code-exec`——的请求上
提供，因为一个没有工具的子智能体做不了任何父智能体自己做不了的事。

## 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--sub-agents` | 关闭 | 提供五个子智能体工具。环境变量：`TS_SUB_AGENTS`（非 `0` 的任意值）。 |
| `--sub-agents-max-threads <n>` | `4` | 一轮中同时打开的子智能体数，1–16。再启动一个时，会先关闭答案已交付、且完成时间最早的那个智能体；只有所有已打开的智能体都仍在工作时才会失败。环境变量：`TS_SUB_AGENTS_MAX_THREADS`。 |
| `--sub-agents-max-depth <n>` | `1` | 子智能体可以嵌套多深，1–4。为 `1` 时，模型可以启动子智能体，但子智能体不能再启动自己的子智能体。环境变量：`TS_SUB_AGENTS_MAX_DEPTH`。 |

`--sub-agents` 还会把引擎的保留状态上限（`TS_RETAINED_FUSED_CACHE_MAX`）从 4 提高到
`2 × (max-threads + 1)`，除非你自己设置了它；启动时会说明这一点。原因见
[KV 缓存](#kv-缓存)。

同一个开关也以 `SkillsChatClientOptions.SubAgents` 的形式提供给 C# 客户端——该客户端在本地
针对远程端点运行工具循环。

## 工具

| 工具 | 参数 | 作用 |
|---|---|---|
| `spawn_agent` | `message`（必填）、`fork_context` | 以 `message` 为任务启动一个子智能体，并立即返回它的 id（`agent_1`、`agent_2`……）。传入 `fork_context: true` 时，该智能体从对话的一份副本开始，而不是从全新的上下文开始。 |
| `send_input` | `target`、`message` | 追加一条后续指令。已完成的智能体会带着先前的上下文重新开始工作；正在工作的智能体会在下一步之前读到这条消息。 |
| `wait_agent` | `targets`（id，以逗号分隔，或 `all`）、`timeout_ms` | 等到任一被等待的智能体完成，然后返回所有已完成、且尚未报告过的被等待智能体的最终答案。默认 300 秒，限制在 10 秒 – 1 小时之间。 |
| `close_agent` | `target` | 停止一个智能体及其启动的所有智能体；返回它关闭前的状态。 |
| `list_agents` | — | 调用者自己的智能体：id、状态、任务。 |

一个智能体只能指挥**它自己**启动的智能体。

### 父智能体收到什么

已完成智能体的答案**恰好送达一次**：要么作为 `wait_agent` 的结果，要么——如果父智能体
从不等待——附加在它收到的下一个工具结果之后，包在 `<subagent_notification>` 里。每个答案
都附带宿主自己记录的、该智能体实际做了什么，与智能体自己的说法并列：

```
agent_2 completed in 7.4 s (3 rounds; tools it ran: write_file, shell). Its final answer:
The sum of the integers from 1 to 1000 is 500500. File: sum.py
```

没有运行任何工具的智能体也会如实报告（`(1 round; it called no tools)`），因为小模型的答案
只是一种说法，而父智能体无法读取该智能体的对话记录来核对它。

### 一轮的结束

Codex 给模型的指引是“在交还控制权之前先等待子智能体”（"wait for sub-agents before
yielding"）。本地小模型会忘记这一点。因此，当模型作答时，如果它启动的智能体仍在工作，
或者有它尚未看到的答案，宿主会等待它们、把结果交给模型，并再给模型一轮来使用这些结果。
没有这些结果时写出的那份答案不会显示。如果已经没有剩余轮次，仍在工作的智能体会被停止，
答案也会说明这一点。子智能体永远不会活过这一轮：请求结束时仍在运行的，一律停止。

## 运行方式

- **每轮一个运行时。** `SubAgentRuntime`（`TensorSharp.AgentHost/Agents/`）负责这一轮
  启动的每一个智能体——id、树结构、上限、状态、结果交付——并在这一轮结束时释放。
- **每个智能体都是一个工具循环。** 子智能体以与父智能体相同的工具上下文运行
  `SkillAgentLoop`。宿主只提供“一组消息如何变成一次生成”：在服务端，每一轮都和父智能体
  的轮次一样经过同一个 `ChatGenerationPipeline`，作为一条独立序列。
- **在同一个模型上并行。** 引擎跨序列批量解码，因此各智能体的轮次彼此同时运行，也与父
  智能体的轮次同时运行。
- **共享文件，各自的 shell。** 各智能体共享父智能体的工作区——一个智能体写的文件，其他
  智能体都能看到——但每个都有自己的**通道**（`SessionWorkspace.ForAgent`）：自己的 shell
  工作目录、导出的环境变量和读取记录，因此一个智能体的 `cd` 不会挪动另一个智能体的下一条
  命令。代码工具仍然一次一个地获取工作区的执行锁。
- **流式输出。** `wait_agent` 运行期间，Web UI 会在 “Waiting for sub-agents…” 下方显示
  每个智能体正在做什么（`agent_1: round 2: shell`）。

## KV 缓存

一个子智能体的开销由两件事决定。

**共享前缀。** 全新子智能体的提示词依次是：原样复制的父智能体开头的系统消息、原封不动的
父智能体工具列表，以及作为第一条用户消息的任务。引擎会为公共前缀——系统消息加工具块——
建立检查点，每个子智能体都从这个检查点恢复，而不必再次预填充这些指令。这也是为什么即使
对已到达深度上限的智能体，子智能体工具仍然保持声明（改为拒绝这次调用）：少了一个工具的
工具列表会在最初几百个 token 内就与父智能体的分叉。

**分叉。** 分叉出的子智能体（`fork_context: true`）运行在父智能体的缓存作用域内，以父智能体
截至其 spawn 调用为止的对话作为提示词，因此它从父智能体自己的缓存状态恢复，只需预填充
这次 spawn 调用和它的任务。

**保留状态。** 引擎会保留有限数量的已结束对话的 KV 以供复用（`TS_RETAINED_FUSED_CACHE_MAX`，
默认 4，一个全局 LRU）。每个子智能体轮次都会向这个池发布状态，因此在默认值下，一个正在
等待四个智能体的父智能体会在它们的第一轮期间就丢掉自己已缓存的对话，并在 `wait_agent`
之后重新预填充。所以 `--sub-agents` 会把上限提高到每个并发对话两份状态。

## 可观测性

服务端对每个事件记录一行日志，全部使用 `SkillToolInvoked` 事件 id：

```
agents.spawn id=agent_1 parent=root depth=1 fork=False evicted=- task=...
agents.round id=agent_1 fork=False promptTokens=3381 kvReused=3178 evalTokens=31 ttftMs=44 tokensPerSec=44.7 finishReason=eos
agents.finish id=agent_1 status=completed turns=1 rounds=3 toolCalls=2 ms=5440 hitRoundLimit=False answerChars=141
agents.deliver id=agent_1 via=wait_agent
agents.collect-before-answer round=3 ms=2210 delivered=True
agents.turn spawned=3 completed=3 failed=0 closed=0 stoppedAtTurnEnd=0
```

`eng/validation/sub-agents-e2e.py` 会驱动一个正在运行的服务端跑完一组场景，并按请求读回
这些日志行。

## TensorSharp 在哪些地方沿用 Codex，在哪些地方不沿用

| | Codex | TensorSharp | 原因 |
|---|---|---|---|
| 工具 | `spawn_agent`、`send_input`、`wait_agent`、`close_agent`、`resume_agent`；v2 增加 `list_agents` | 相同，但去掉 `resume_agent`，加上 `list_agents` | 按轮存在的运行时没有什么可恢复的 |
| Id | UUID（v1）、模型自选路径（v2） | `agent_1`、`agent_2`…… | 小模型能可靠地抄写短 id |
| 启动策略 | 仅当用户或指令要求时 | 相同，写在 `spawn_agent` 的说明里 | |
| 深度 | 1；到达上限时隐藏工具 | 1；工具仍声明，启动请求以 Codex 的原话拒绝 | 完全相同的工具块才能让 KV 前缀保持共享 |
| 打开的智能体 | 6（v1），到达上限即报错；v2 会逐出空闲的已完成智能体 | 4，先逐出答案已交付的已完成智能体，然后才报错 | 在不依赖 `close_agent` 的前提下限定 GPU 争用 |
| 结果 | v1：`wait_agent` 返回结果，*同时*还有一条通知重复它们；v2：只用邮箱 | 恰好一次，经由 `wait_agent` 或下一个工具结果里的通知 | 8k 上下文只负担得起一次答案 |
| `wait_agent` 超时 | 默认 30 秒，10 秒 – 1 小时 | 默认 300 秒，限制范围相同 | 反正会提前返回；本地智能体更慢 |
| 分叉 | 从复制的历史中去掉工具调用 | 保留它们 | 复制的前缀只有与父智能体的 KV 一致才值得复制 |
| 一轮的结束 | 以指令要求模型等待 | 由宿主收集尚未交付的智能体 | 小模型会忘记 |

## 限制与注意事项

- 只有当每个子任务的工作量足够大、而结果很短时，委派才划算。如果父智能体必须逐字复述
  其智能体的输出——四个智能体写的四段文字，被粘贴进答案——父智能体最后一次生成就与自己
  完成这项工作一样长，委派反而更慢。
- 不支持跨序列批量解码的模型家族只能轮流地、一次推进一步地运行各智能体；它们仍然获得
  上下文隔离，但得不到加速。
- TensorAgent 应用不启用子智能体：为了适应手机内存，它只保留一个对话的状态，而子智能体
  会把它逐出。
