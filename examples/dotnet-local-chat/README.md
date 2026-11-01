# Microsoft.Extensions.AI with a TensorSharp GPU server

Use **Microsoft.Extensions.AI.IChatClient** in your .NET application while **TensorSharp** loads and runs the local model. This example defaults to Microsoft's existing adapter and demonstrates streaming, complete responses and a two-turn conversation. The primary model recipe is **Qwen3.8 Flash Next** on the native **GGML Metal** backend.

| Component | Responsibility |
| --- | --- |
| `Microsoft.Extensions.AI` | The application's `IChatClient`, messages, options, streaming and response APIs |
| `Microsoft.Extensions.AI.OpenAI` | Microsoft's `.AsIChatClient()` adapter for the existing HTTP client |
| Official `OpenAI` .NET client | Serializes chat requests and reads HTTP/SSE responses from the configured local endpoint |
| TensorSharp server and native runtime | Load GGUF weights, render the model's chat template, schedule inference and execute on the selected backend |

The client libraries do not run inference. No hosted OpenAI account is needed for this local path. The optional `--client openai` path exercises the underlying transport directly. For a .NET application that should own the engine and `ChatSession` in its own process, use the complementary [direct TensorSharp example](../tensorsharp-native-chat).

## Prerequisites and Qwen GPU recipe

You need the .NET 10 SDK, the repository's native GGML Metal build prerequisites, and an already-installed supported model. See [getting started](../../docs/getting_started.md). This example downloads no weights and provisions no hardware.

The demonstrated model set is `Qwen3.8-Flash-Next-UD-Q2_K_XL`, **three shards totaling 78,869,128,864 bytes**. Keep all three files together and pass the first shard to TensorSharp. The first shard is metadata-only (10,946,624 bytes); the two weight-bearing siblings are required. `UD-Q2_K_XL` is the distribution's mixed-quantization profile, not a statement that every tensor has the same type.

From the repository root on macOS, with an existing GGML Metal build/toolchain:

```bash
MAX_CONTEXT=4096 TS_SCHED_NUM_BLOCKS=16 TS_SCHED_MAX_RUNNING_SEQS=1 \
TS_Q4E_RETAINED_CACHE_MB=0 TS_HOST_MOE_TIMING=3 \
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --backend ggml_metal --host 127.0.0.1 --port 8080 --max-tokens 512 --temperature 0 \
  --no-skills --no-multi-agent --no-prefix-cache --no-webui
```

`TensorSharpSkipMlxNative` skips the separate MLX native build; it does not disable GGML Metal. Selecting `ggml_metal` requires a matching native build. This minimal text-chat configuration disables unrelated agent/UI features and restored prefix state so the E2E check exercises actual generation.

A 78.87 GB model set is larger than the 48 GiB unified memory on the local demonstration Mac. TensorSharp's model-specific host-expert placement can combine CPU MoE experts with Metal computation and memory-mapped weights. This is **hybrid GPU-assisted inference**, not a claim that all weights or layers reside on the GPU. Check the actual server placement and device logs; a requested backend name alone is not GPU execution evidence. Other machines may need a different model or resource configuration.

## Run the Microsoft.Extensions.AI example

In `examples/dotnet-local-chat`:

```bash
dotnet restore --locked-mode --configfile NuGet.Config
dotnet build -c Release --no-restore
dotnet run -c Release --no-build -- --list-models
```

The app discovers the exact ID through `/v1/models` and automatically selects it only when exactly one model is advertised. To choose explicitly, pass `--model ID_FROM_DISCOVERY`.

Stream two turns through `IChatClient.GetStreamingResponseAsync`:

```bash
dotnet run -c Release --no-build -- \
  --prompt "What is 2 + 2? Reply with only the single digit answer." \
  --follow-up "Multiply your previous answer by 3. Reply with only the number." --max-tokens 64
```

Use `IChatClient.GetResponseAsync` to wait for each complete answer:

```bash
dotnet run -c Release --no-build -- --mode response \
  --prompt "What is 2 + 2? Reply with only the single digit answer." \
  --follow-up "Multiply your previous answer by 3. Reply with only the number." --max-tokens 64
```

The intended answers are `4`, then `12`, one answer per turn on stdout. The optional follow-up resends the complete text history: user, assistant, user. History is retained only within this invocation. The HTTP API does not infer previous messages from earlier requests.

`Program.cs` creates the official HTTP client with TensorSharp's endpoint and calls `.AsIChatClient()`. The application then uses Microsoft's `ChatMessage`, `ChatOptions`, `GetStreamingResponseAsync` and `GetResponseAsync`. The same choices can be checked with `--client openai`, which uses `CompleteChatStreamingAsync` and `CompleteChatAsync` directly. No custom TensorSharp HTTP adapter is required.

## Options and behavior

| Option | Default / behavior |
| --- | --- |
| `--client extensions\|openai` | `extensions`, using Microsoft's `IChatClient` adapter |
| `--mode stream\|response` | `stream`; `response` waits for a complete answer |
| `--prompt TEXT` | First user message |
| `--follow-up TEXT` | Optional second user message with full conversation history |
| `--base-url URL` | `http://127.0.0.1:8080/v1`; include `/v1` once |
| `--model ID` | Auto-select only when one model is advertised |
| `--max-tokens N` | `128` per turn; the server's cap also applies |
| `--timeout-seconds N` | `120` for the entire invocation: discovery and both turns |
| `--list-models` | Discovery only; sends no generation request |
| `--allow-remote` | Explicit opt-in to a remote HTTPS endpoint |
| `--help` | Show all options |

`TENSORSHARP_BASE_URL` and `TENSORSHARP_MODEL` provide defaults; CLI options take priority. `TENSORSHARP_API_KEY` can supply an authenticated proxy token. Direct local TensorSharp receives a non-secret placeholder; the app does not read `OPENAI_API_KEY`. Requests go to the configured endpoint, and redirects/retries are disabled.

Diagnostics go to stderr. Exit `0` requires nonempty answer text and completion, `1` reports configuration/request/deadline failure, and `130` is Ctrl+C cancellation. In streaming mode, output before an error can be partial. In response mode, an answer is printed after the complete response arrives.

## Tests and evidence boundaries

From the repository root, after building the client:

```bash
python3 eng/tests/dotnet-local-chat.py
python3 eng/tests/dotnet-local-chat-e2e.py --self-test
python3 eng/tests/dotnet-local-chat-e2e.py --base-url http://127.0.0.1:8080/v1
```

The first command runs **36 deterministic stdlib mock checks** with no model. It retains the original streaming checks and adds complete responses, full two-turn request payloads, empty/malformed/error responses, deadlines and recovery for both SDKs. The oracle self-test covers **6 accepted and 28 rejected fixtures**. It accepts only the complete numeric answers `4` and `12` with optional surrounding whitespace and one trailing period, and rejects contradictory text such as `The answer is 5, not 4.`.

The opt-in live harness starts no server or download. It requires macOS/Linux for SIGINT verification and checks both clients in both modes, two-turn `4` → `12` answers, raw SSE model identity/finish/token metrics, useful errors, cancellation, invocation deadlines and subsequent recovery. Real model output can fail these assertions; that is a failed E2E check, not something the mock can override. These checks are not a performance benchmark or proof of support for every architecture/backend.

### Verified local Qwen Metal run

On **2026-11-01**, all **20 real-endpoint checks** passed against an isolated loopback TensorSharp server on a **Mac M5 Pro, 20 GPU cores, 48 GiB unified memory**, using the three-shard Qwen model above and `ggml_metal`. The server's recorded device/placement diagnostics identified Apple M5 Pro and **33 host-expert layers / 15 Metal-expert layers**. The host-expert path used SSD-backed on-demand weights. This was the hybrid placement described above, not full GPU residency or a measurement of the complete model's resident memory.

- Microsoft.Extensions.AI `GetStreamingResponseAsync` and `GetResponseAsync` each returned exactly `4`, then `12`; the direct SDK paths did the same.
- Raw SSE returned `4`, identified the configured first-shard model, finished with `stop` and `[DONE]`, and reported **27 prompt tokens / 2 completion tokens**.
- Each SDK was interrupted after producing `1`, returned cancellation exit `130`, and then answered `4` on a fresh request.
- Both SDKs in both modes hit a one-second invocation deadline with exit `1`, then recovered with a fresh `4` response. Unknown-model preflight and an incorrect API base also produced the expected errors.
- The final HTTP client built with .NET SDK **10.0.302**, with **0 warnings and 0 errors**. The 36 mock checks and strict oracle self-test also passed.

The managed projects were built in an isolated working tree based on `9a8d4e69e33810ebc039f14e77f1f7ce3ce28a29`, with this PR's focused CI preflight repair applied. The native GGML Metal bridge was freshly built for this source against unchanged GGML `facdc44a67da17df4433adcc468b424f48e9ded2`; tested bridge SHA256: `d9c6453b231451dfcf22b3ec7ea2c87f3167bae37df055246d4018bc31e5ee13`. The server used the recipe's 4096-token context, one running sequence, 16 scheduler blocks, no retained expert cache, and timing diagnostics; expert-offload count overrides were unset so the model's automatic placement policy applied. It listened on loopback port 18763 for the test.

No weights were downloaded, and no existing service or paid runner was used for this run. These results cover the stated model, quantization, hybrid Metal backend and text-only scenarios on this machine. They do not establish general model quality, comparative performance, or correctness for every backend/model. Full machine-local logs remain in ignored `artifacts/` or `docs/validation/`; no private model paths or credentials belong in public evidence.

## Limits and deployment choice

- Discovery advertises configuration, not model readiness. A failed load can appear as ordinary SSE `Error: ...` text with HTTP 200 and a normal finish marker; inspect the server log and answer content as well as the exit status.
- Model IDs normally omit `.gguf`, while response chunks can include it. Use the advertised ID; changing a hosted model requires a server restart.
- The rendered system/tool/skill template consumes context too. Increase the context budget or disable unused features when the server returns `context_length_exceeded`.
- This sample covers two-turn text chat. It does not establish tool calling, Agent Skills, multimodal inputs, distributed operation or full OpenAI API parity.

Use the HTTP path when a .NET application should use Microsoft's common client abstraction while TensorSharp owns inference in a separate process. Use the [direct example](../tensorsharp-native-chat) when the application should own native model loading, session lifetime and runtime metrics. The combination of a native .NET engine, GGUF model support, backend selection and common .NET client APIs is the integration demonstrated here; no exclusivity or comparative speed claim is made.

Keep TensorSharp bound to loopback. The inspected server has no built-in authentication or TLS; use an authenticated TLS proxy and deliberate network controls for remote access. The client requires HTTPS plus `--allow-remote` outside loopback and rejects credentials, query strings and fragments in the base URL. Review server/proxy logging before sending sensitive prompts.

Pinned clients: [OpenAI 2.14.0](https://www.nuget.org/packages/OpenAI/2.14.0) and [Microsoft.Extensions.AI.OpenAI 10.11.0](https://www.nuget.org/packages/Microsoft.Extensions.AI.OpenAI/10.11.0); `packages.lock.json` records the transitive graph. TensorSharp's [routes](../../TensorSharp.Server/Endpoints/OpenAIEndpoints.cs), [chat adapter](../../TensorSharp.Server/ProtocolAdapters/OpenAIChatAdapter.cs) and [SSE response construction](../../TensorSharp.Server/ResponseSerializers/OpenAIResponseFactory.cs) implement the server contract.
