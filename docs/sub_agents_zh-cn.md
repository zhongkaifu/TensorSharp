# 子智能体

[English](sub_agents.md) | [中文](sub_agents_zh-cn.md)

子智能体是模型自己为某一项任务启动的另一个模型副本。它拥有与启动它的智能体相同的工具，
在同一个工作目录中工作，但拥有**自己的上下文**：它看到的是自己的任务，而不是整段对话。
它与父智能体并行工作，最后交回一个最终答案。

TensorSharp 采用 OpenAI Codex 所用的那套工具接口来实现这一功能，并把每个子智能体作为
普通序列运行在**同一个已加载模型**上。连续批处理引擎会把父智能体与它的所有子智能体放在
一起解码；每个子智能体的提示词都以父智能体一字不差的指令与工具块开头，因此引擎会复用这段
已缓存的前缀，而不必重新预填充。

子智能体**默认关闭**。在服务端或 CLI 上用 `--sub-agents` 开启。

## 快速开始

```bash
TensorSharp.Server --model gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal \
    --code-exec --sub-agents
```

CLI 接受同样的参数（`TensorSharp.Cli --model ... --code-exec --sub-agents`），并把子智能体
的活动以 `[agent] ...` 行输出到 stderr。

然后明确要求委派——工具说明会告诉模型：只有在用户或它正在遵循的某个技能要求时，才启动
子智能体：

```bash
curl -s http://localhost:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-E4B-it-Q8_0",
  "messages": [{"role": "user", "content":
    "Use three sub-agents working in parallel, one per task: compute the sum of 1..1000, the number of primes below 10000, and the 50th Fibonacci number, each with a python3 program. Wait for all three and reply with the three numbers."}]
}'
```

每个聊天入口都会得到这些工具：Web UI、`/v1/chat/completions`、`/v1/responses`、
Ollama 的 `/api/chat`，以及 CLI 的两种模式（单次与交互式）。它们只在本来就带有工具——Agent Skills 或 `--code-exec`——的请求上
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
在服务端，没有这些结果时写出的那份答案不会显示（交互式 CLI 已经把它打印出来，收集结果后的
答案会接在后面）。如果已经没有剩余轮次，仍在工作的智能体会被停止；来得太晚、没能被使用的
智能体答案会在回复之后显示——回复会说明哪一种是哪一种，因此不会有结果无声消失。子智能体
永远不会活过这一轮：请求结束时仍在运行的，一律停止。

### 子智能体的边界

子智能体的一轮比父智能体的受到更紧的约束，因为父智能体的整轮都在等它：

- **轮次：** 父智能体预算的一半，至少 8 轮，且不超过父智能体自己的预算（开启
  `--code-exec` 时为 12，父智能体为 24）。用完预算的智能体会被如实报告，父智能体因此会
  把它的答案当作不完整。
- **重复调用：** 某一轮第三次与上一轮的调用完全相同时，这些调用会被答复而不是被执行
  （“you have made this exact call 3 rounds in a row...”）；再出现一轮完全相同的调用，该
  智能体的任务就此结束。
- **无结果：** 某一轮没有运行任何工具，却把补丁信封当作文本作答，或者什么都没写，会得到
  一次纠正轮次。

这两点都是下面的基准测试发现的：在 Qwen3.5-9B Q8_0 上，有一个智能体连续十轮逐字节地重复
发送同一个失败的 `apply_patch`；另一次运行中，它把父智能体的全部 24 轮都用来修补并重跑一个
坏掉的程序（18 分钟），而它的三个兄弟智能体都在 3–5 轮内完成。有了这一预算，同样的
`code4` 轮次用时从 1565–1736 s 降到 509–551 s，两次运行也都答对了（之前只有一次）——但仍
远慢于该模型在这个任务上由一个智能体独自完成所需的 36 s，这是模型在该任务上的特性：预算
限制了损失，却不能让委派变得划算。

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
  命令。代码工具仍然一次一个地获取工作区的执行锁。共享目录也意味着共享文件*名*：每个
  子智能体都被要求，把为自己用途创建的每个文件的名字以自己的 id 开头（`agent_2_count.py`）
  ——否则，在 gemma-4-E4B 上实测，四个并发智能体中有两个都写了 `solution.py`，一个覆盖了
  另一个，两者报告了同一个数字。
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

## 实测结果

在 Apple M5 Pro（48 GB，Metal）上，用 `eng/validation/sub-agents-e2e.py` 对以
`--code-exec --sub-agents` 启动的服务端测得，温度为 0。每一行是同一个场景在同一个服务端上的
三种模式：**parallel**（提示词要求每个任务一个子智能体，同时启动）、**serial**（同样的子
智能体，一次一个——作为并发的对照组）和 **solo**（一个智能体自己完成所有任务）。`code4`
是四个编程任务，其答案模型无法背出来（已由独立程序核对）；单元格是墙钟时间的中位数，以及
四个数字全部正确的运行次数。

| 模型 | parallel | serial | solo | parallel 相对 serial |
|---|---|---|---|---|
| gemma-4-E2B Q8_0 | 15.7 s，3/3 | 25.9 s，2/3 | 13.2 s，1/3 | 1.65× |
| gemma-4-E4B Q8_0 | 31.0 s，3/3 | 63.6 s，3/3 | 23.4 s，3/3 | 2.05× |
| Qwen3.5-9B IQ4_XS | 44.9 s，3/3 | 63.7 s，3/3 | 22.7 s，3/3 | 1.42× |

这些数字说明：

- **并发运行智能体是有效的。** 同样的委派，并行完成比依次完成快 1.4–2 倍，这就是引擎
  在一个模型上把各智能体的解码批量处理的效果。其上限取决于模型批处理的效率：四条并发的
  gemma-4-E4B 流合计约 68 tok/s，而单条约 41 tok/s；四条 Qwen3.5-9B IQ4_XS 流合计约 52，
  单条约 47。
- **委派有固定开销。** 父智能体要把每个任务写出来，还要读取并合并每个答案。在这类短任务
  上，一个智能体独自完成全部工作更快；只有每个任务很长、而结果很短时，委派才划算。
- **它可以更可靠。** 在 gemma-4-E2B 上，一个智能体在一个上下文里做四个任务，3 次运行中
  只有 1 次全部正确；四个专注的子智能体则是 3 次全对。
- **前缀是共享的。** 在每个模型上，子智能体的第一轮都有 91–95% 的提示词复用了父智能体
  已缓存的“指令加工具”前缀；分叉的智能体从父智能体自己的状态恢复，只需预填充它的任务。

测试工具中的每个子智能体场景——并行计算（`compute3`）、由两个智能体写入并由父智能体读回
的文件（`files2`）、需要父智能体上下文的分叉智能体（`fork`）、模型被告知不要等待的智能体
（`guard`，由宿主收集）、四段文字（`write4`）——在三个模型上的每次运行都通过。

**关闭该功能时没有回退。** 未修改的 `main` 服务端与本分支在关闭 `--sub-agents` 时，使用
相同模型与请求，各运行 4 次：`compute3` 9.11 s 对 9.06 s，`files2` 2.81 s 对 2.81 s，
`write4` 12.80 s 对 12.78 s，答案、轮数与 KV 复用全部逐字节一致（`code4` 的答案在两个构建
中都会逐次变化，因为工具输出里含有每次运行不同的工作区路径）。

## 限制与注意事项

- 只有当每个子任务的工作量足够大、而结果很短时，委派才划算。如果父智能体必须逐字复述
  其智能体的输出——四个智能体写的四段文字，被粘贴进答案——父智能体最后一次生成就与自己
  完成这项工作一样长，委派反而更慢。
- 加速的上限取决于已加载模型跨序列批量解码的效率（见上文）。拒绝批量解码的模型家族
  （gemma-4-E2B 会记录 "declined the default batched fused-decode path ... serving sequences
  round-robin"）会交替地而不是批量地运行其智能体：获得上下文隔离，而不是加速。
- TensorAgent 应用不启用子智能体：为了适应手机内存，它只保留一个对话的状态，而子智能体
  会把它逐出。
