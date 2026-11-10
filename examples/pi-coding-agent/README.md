# Pi coding agent on TensorSharp

[Pi](https://pi.dev) (`@earendil-works/pi-coding-agent`) is a terminal coding agent. It reads files,
runs shell commands and edits code with its own tools, and it can talk to any OpenAI-compatible
endpoint. This example points it at a local TensorSharp server running Qwen3.8-27B in its fastest
Apple Silicon setup:

- the UD-IQ3_XXS GGUF on `ggml_metal`;
- the DFlash2 block drafter, with 7 drafts a step and no confidence gate;
- greedy decoding;
- a 64K-token context with a q8_0 KV cache.

Two scripts run Pi against it: the `pi` command and a short Node.js program using Pi's SDK. Each
fixes the failing tests of a small sample project.

[中文说明](README_zh-cn.md)

| Path | What it is |
|---|---|
| [`tensorsharp-pi.json`](tensorsharp-pi.json) | Server config overlay for Pi. It goes on top of the [`config/qwen3.8-27b-metal-dflash2.json`](../../config/qwen3.8-27b-metal-dflash2.json) preset. |
| [`scripts/start-server.sh`](scripts/start-server.sh) | Starts `TensorSharp.Server.Host` with the preset and the overlay. |
| [`pi-agent/models.json`](pi-agent/models.json) | Pi's provider entry for the server: endpoint, model id, context window, thinking switch. |
| [`pi-agent/settings.json`](pi-agent/settings.json) | Makes that model Pi's default. |
| [`scripts/pi.sh`](scripts/pi.sh) | Runs `pi` with its own config directory (`~/.pi/tensorsharp-agent`), so your `~/.pi/agent` is untouched. |
| [`sdk/run-task.mjs`](sdk/run-task.mjs) | Uses Pi as a library: one coding task, streaming its output and tool calls. |
| [`sample-project/`](sample-project/) | A small Node.js module with three bugs and five tests, three of which fail. |
| [`scripts/e2e.sh`](scripts/e2e.sh) | End-to-end check: the CLI and the SDK each fix a fresh copy of the sample project. |

## Requirements

| | Version or size |
|---|---|
| Mac | Apple Silicon with 32 GB or more of unified memory. The weights take 10.9 GB and the drafter 1.1 GB; the rest is KV cache and Metal buffers. Validated on an M5 Pro with 48 GB, macOS 27.2. |
| .NET | The .NET 10 SDK, CMake 3.20+ and the Xcode command-line tools, to build TensorSharp ([getting started](../../docs/getting_started.md#install-and-first-run)). |
| Node.js | 22.19 or newer, which Pi requires. Validated with 26.8.1. |
| Pi | `@earendil-works/pi-coding-agent` 1.5.1. The older `@mariozechner/pi-coding-agent` package is deprecated. |
| Disk | 12 GB for the two model files, which download on the first start. |

Build from source. Release archives carry the features of their own tag, and this example needs the
server fix that keeps a tool loop's cache (see [How the cache keeps up with Pi](#how-the-cache-keeps-up-with-pi)).

## 1. Build TensorSharp

From the repository root:

```bash
dotnet build TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true
```

This builds the native ggml library with Metal and writes
`TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll`, which `start-server.sh` runs. If that file
does not exist, the script falls back to `dotnet run --project TensorSharp.Server.Host -c Release ...`,
which builds first.

## 2. Start the server

```bash
examples/pi-coding-agent/scripts/start-server.sh
```

The script runs:

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --config config/qwen3.8-27b-metal-dflash2.json \
  --config examples/pi-coding-agent/tensorsharp-pi.json
```

On the first start, the preset downloads the model and the drafter from pinned Hugging Face
revisions and checks their sha256:

- `Qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf`, from unsloth;
- `Qwen/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, from z-lab.

Both land under `$TENSORSHARP_MODELS`, which defaults to `<repo>/models`. If you already have the
files, point the variable at their parent folder:

```bash
TENSORSHARP_MODELS=~/work/models examples/pi-coding-agent/scripts/start-server.sh
```

Arguments after the script name go to the server and win over both files, for example `--port 5001`.

The server is ready when `/v1/models` lists the model:

```bash
curl -s http://127.0.0.1:5000/v1/models
# {"object":"list","data":[{"id":"Qwen3.8-27B-UD-IQ3_XXS","object":"model","owned_by":"local","loras":[]}]}
```

### What the overlay changes

The preset supplies the model, the drafter, the speed settings (7 drafts, `spec-pmin` 0), greedy
sampling and `no-skills`. A later `--config` file replaces the values it sets again, and the server
logs each replacement. The overlay sets:

| Setting | Value | Why |
|---|---|---|
| `host`, `port` | `127.0.0.1`, `5000` | Only this Mac can reach the server. The server has no authentication of its own, so do not expose it. |
| `max-tokens` | `16384` | A `max-tokens` in a config file is a pinned limit, and it also caps every request. The preset's 1,024 would cut off a thinking model in the middle of writing code. Pi asks for its model's `maxTokens`, which is 16,384 here. |
| `mmproj` | `"none"` | Pi sends text only. This skips the 0.93 GB vision projector and its download. |
| `kv-cache-dtype` | `q8_0` | Half the KV memory, at the same decode speed. |
| `environment.MAX_CONTEXT` | `65536` | Pi's system prompt and tool schemas take about 4K tokens before the first message, and file contents and test output add up quickly. 64K tokens of q8_0 KV fit beside the weights on a 32 GB Mac. |

## 3. Install and configure Pi

```bash
npm install -g @earendil-works/pi-coding-agent@1.5.1
pi --version
```

Pi reads its providers from `models.json` in its agent directory. The provider entry for this server
is [`pi-agent/models.json`](pi-agent/models.json):

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

- `id` must be the model name exactly as `GET /v1/models` lists it: the GGUF file name without
  `.gguf`. A wrong id gets HTTP 404.
- `contextWindow` matches the server's `MAX_CONTEXT`. Pi starts compacting older turns when the
  context comes within `compaction.reserveTokens` of it. The default reserve is 16,384, so Pi
  compacts at about 48K tokens.
- `maxTokens` matches the overlay's `max-tokens`.
- `"thinkingFormat": "qwen-chat-template"` turns Qwen's thinking on and off with
  `chat_template_kwargs.enable_thinking`. Qwen3.8 has only that switch, so Pi's `off` level turns
  thinking off and every other level turns it on. `supportsReasoningEffort: false` keeps Pi from also
  sending `reasoning_effort`.
- `supportsDeveloperRole: true`: Pi sends its instructions as a `developer` message, and TensorSharp
  renders it as the system message.
- `apiKey` can be any string. The server does not check it.

[`pi-agent/settings.json`](pi-agent/settings.json) makes this model the default, with thinking on:

```json
{
  "defaultProvider": "tensorsharp",
  "defaultModel": "Qwen3.8-27B-UD-IQ3_XXS",
  "defaultThinkingLevel": "medium",
  "enableInstallTelemetry": false
}
```

There are two ways to install these files:

- **Separate agent directory (what the scripts do).** `scripts/pi.sh` sets
  `PI_CODING_AGENT_DIR=~/.pi/tensorsharp-agent` and copies the two files there on its first run. It
  never overwrites them, so edits you make there are kept. Your usual `~/.pi/agent` setup is untouched.
- **Your normal Pi setup.** Merge the `tensorsharp` provider into `~/.pi/agent/models.json`, then pick
  the model with `pi --model tensorsharp/Qwen3.8-27B-UD-IQ3_XXS` or Pi's model selector.

`pi.sh` also sets `PI_OFFLINE=1`, `PI_SKIP_VERSION_CHECK=1` and `PI_TELEMETRY=0`. Pi then makes no
network calls of its own: no catalog refresh, no version check, no telemetry.

## 4. Use Pi

Interactive, in the project you want to work on:

```bash
cd ~/src/my-project
~/work/TensorSharp/examples/pi-coding-agent/scripts/pi.sh
```

One task, without the interactive UI. Pi prints the final answer. In a script, give it an empty
stdin, because `pi -p` otherwise waits for input:

```bash
cd ~/src/my-project
~/work/TensorSharp/examples/pi-coding-agent/scripts/pi.sh -p "Explain what src/stats.js does" < /dev/null
```

Other useful options:

```bash
pi.sh --thinking off -p "..." < /dev/null              # no thinking: faster, for simple edits
pi.sh --mode json --no-session "..." < /dev/null       # one JSON event per line, nothing saved
pi.sh --continue                                       # continue this directory's last session
```

## 5. Use Pi from code (SDK)

[`sdk/run-task.mjs`](sdk/run-task.mjs) runs one task with Pi as a library, against the same
`models.json`. The core of it:

```js
import { createAgentSession, ModelRuntime, SessionManager } from "@earendil-works/pi-coding-agent";

const modelRuntime = await ModelRuntime.create({
  modelsPath: path.join(agentDir, "models.json"),
  authPath: path.join(agentDir, "auth.json"),
});
const model = modelRuntime.getModel("tensorsharp", "Qwen3.8-27B-UD-IQ3_XXS");

const { session } = await createAgentSession({
  cwd: projectDir,                           // Pi's read/bash/edit/write tools work here
  agentDir,
  modelRuntime,
  model,
  thinkingLevel: "medium",
  sessionManager: SessionManager.inMemory(), // no session files
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

To run it, install the SDK, then point the program at a copy of the sample project, because the agent
edits the files:

```bash
cd examples/pi-coding-agent/sdk
npm install
cp -R ../sample-project /tmp/stats-demo
PI_CODING_AGENT_DIR=~/.pi/tensorsharp-agent PI_OFFLINE=1 node run-task.mjs /tmp/stats-demo
```

`PI_CODING_AGENT_DIR` must hold `models.json`. `scripts/pi.sh` creates `~/.pi/tensorsharp-agent` on
its first run. Without that, create it yourself:

```bash
mkdir -p ~/.pi/tensorsharp-agent && cp ../pi-agent/*.json ~/.pi/tensorsharp-agent/
```

`TENSORSHARP_URL` overrides the provider's `baseUrl`. The program prints the agent's text, each tool
call, the final answer, and the token totals, including how many prompt tokens the server served from
its prefix cache.

## 6. Check it end to end

[`scripts/e2e.sh`](scripts/e2e.sh) gives Pi the same task twice, through the `pi` command and through
the SDK, each on a fresh copy of `sample-project/`. The task: run `npm test`, fix `src/stats.js`
without touching the tests, and run the tests again. The sample has three bugs: `median` sorts
numbers as strings, `median` does not average the two middle values of an even-length list, and
`percentile` is off by one.

The script then checks each copy itself. A copy passes only when the test file is byte-identical to
the original and `node --test` reports 5 of 5 passing.

```bash
examples/pi-coding-agent/scripts/start-server.sh &   # or let e2e.sh start it: START_SERVER=1
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

The script exits 0 only when both runs pass. Copies, Pi's JSON events, test output and logs go to
`.work/`, which git ignores; set `WORK_DIR` to put them elsewhere. Other settings: `TENSORSHARP_URL`
(default `http://127.0.0.1:5000/v1`) and `PI_BIN` (a `pi` that is not on `PATH`).

### Measured

Measured on 2026-11-09 on an M5 Pro with 48 GB, macOS 27.2. Software: TensorSharp `b722bad1`
(ggml `cd6d2b7`, unmodified upstream), Pi 1.5.1, Node.js 26.8.1. One run each. The model's output
differs between runs, so the number of rounds and tool calls varies too.

| Run | Result | Wall time | Model requests | Prompt tokens served from cache |
|---|---|---|---|---|
| `pi` CLI (`--mode json`) | PASS, 5 of 5 tests, test file unchanged | 88 s | 10 (10 tool calls) | 86-99% per request after the first |
| SDK (`run-task.mjs`) | PASS, 5 of 5 tests, test file unchanged | 83.4 s | 8 | 41,962 of 48,166 (87%) |

Per request, from the server log:

- **First request of a session:** a 3.9-4.1K-token prompt (Pi's system prompt, tool schemas and the
  task) is prefilled cold, so its time to first token is 13-14 s.
- **Every later round:** 0.45-3.5 s to the first token, at prompts of 4.2K-9.1K tokens.
- **Decoding:** 20-43 tok/s. DFlash2 drafts are mostly accepted on code. When too many drafts are
  rejected, as in prose and short answers, the speculative governor drops to plain decoding at about
  21 tok/s.

The task and the measurements are a single sample project, not a benchmark of coding ability. They
cover only the Apple Silicon setup above.

## How the cache keeps up with Pi

Pi runs its tools itself. After each tool call it sends the whole conversation again: system prompt,
earlier turns with the model's reasoning, tool calls and tool results. The server reuses the KV state
of the previous round and prefills only the new tokens. This works when the request repeats the
earlier rounds exactly as the server generated them.

Pi sends each round's reasoning back as it streamed it, including the newline before `</think>`. The
server used to compare that with its own trimmed copy, so the two never matched. Every round then
reused only the shared system prefix: about 3.8K tokens, with time to first token growing to 5.3 s by
the sixth round. The server now ignores that surrounding whitespace, and each round continues the
previous one (commit `b722bad1`). Two cases still prefill the conversation:

- a client that changes earlier reasoning or messages;
- a new session in a different directory, because Pi's system prompt names the working directory.

Requests that carry their own `tools` get none of the server's tools (skills, code execution,
sub-agents). The model sees exactly Pi's four tools. See
[Agent clients that run their own tools](../../TensorSharp.Server.Host/API_EXAMPLES.md#agent-clients-that-run-their-own-tools-pi-codex-opencode).

## Other hardware

This example was validated only on Apple Silicon with the preset above. On an NVIDIA GPU the fastest
Qwen3.8-27B preset is [`config/qwen3.8-27b-ninfer.json`](../../config/qwen3.8-27b-ninfer.json): the
NVFP4 NInfer artifact on the direct CUDA engine, with its own DFlash2 weights. Select it with
`TENSORSHARP_PRESET`:

```bash
TENSORSHARP_PRESET=config/qwen3.8-27b-ninfer.json examples/pi-coding-agent/scripts/start-server.sh
```

That preset sets `MAX_CONTEXT` to 40960 in its own `environment`. The overlay's 65536 replaces it,
because among config files the later one wins. A `MAX_CONTEXT` exported in your shell wins over both.
For example, `MAX_CONTEXT=40960 TENSORSHARP_PRESET=... scripts/start-server.sh` keeps the preset's
length. Set `contextWindow` in `models.json` to the length the server actually uses.

This combination has not been run with Pi. That includes the overlay's q8_0 KV cache on the direct
CUDA engine.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pi -p` never returns in a script | Pi waits for stdin. Run it with `< /dev/null`. |
| `404` / `model not found` | The `id` in `models.json` differs from `GET /v1/models`. Use `Qwen3.8-27B-UD-IQ3_XXS`. |
| Answers stop at about 1,024 tokens | The overlay was not applied. It must come after the preset: `--config <preset> --config tensorsharp-pi.json`. |
| `context_length_exceeded` | The conversation outgrew the server's window. Pi compacts and retries on that error by itself. If you lower `MAX_CONTEXT`, lower `contextWindow` too, and set `compaction.reserveTokens` in Pi's `settings.json`, for example to 8192 for a 32K window. |
| The first answer takes a long time | The first request prefills Pi's whole 4K-token system prompt (13-14 s). Later rounds reuse it. A server that was just started also loads 12 GB of weights first. |
| `Address already in use` | Another server is on port 5000. Pass `--port 5001` to `start-server.sh` and change `baseUrl` in `models.json` (or set `TENSORSHARP_URL` for `e2e.sh` and the SDK). |
| Memory pressure on a 32 GB Mac | Use a shorter context: `MAX_CONTEXT=32768 scripts/start-server.sh`. A variable exported in the shell wins over the config files. Change `contextWindow` in `models.json` to match, and lower `compaction.reserveTokens` as described above. |
