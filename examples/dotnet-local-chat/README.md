# Call a local TensorSharp server from .NET

A small, one-turn console starter for TensorSharp's HTTP server. TensorSharp loads and runs your local GGUF; the official `OpenAI` .NET client and Microsoft's `IChatClient` adapter provide interoperable client APIs. No OpenAI account is required for the local commands.

This path separates the application from the TensorSharp engine process. For C# applications that should own model loading, session lifetime, native backend selection and per-turn KV reuse metrics directly, use the complementary [native TensorSharp .NET example](../tensorsharp-native-chat). That path opens no HTTP listener. These are useful TensorSharp integration choices, not claims of exclusivity over other engines.

## Prerequisites

- .NET 10 SDK and access to the official NuGet registry for the initial restore.
- For real chat: an existing TensorSharp server, a supported model already on disk, and its backend/native build prerequisites. Follow TensorSharp's [getting started guide](https://github.com/zhongkaifu/TensorSharp/blob/72e0fc816da32405ebefdbeae864597cd951c708/docs/getting_started.md). This example contains no model or TensorSharp engine.
- Python 3.9+ only if you want to run the model-free integration checks.

## Run

1. Start TensorSharp if it is not already running. From an existing TensorSharp checkout, use an already-present supported GGUF file:

   ```bash
   dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model /absolute/path/to/model.gguf --backend ggml_cpu --host 127.0.0.1 --port 8080 --max-tokens 512 --temperature 0 --no-skills --no-webui
   ```

   This minimal chat configuration disables the bundled skills catalog and UI; enable them deliberately when you need the server's agent features. Model support, memory needs and native prerequisites depend on your TensorSharp build. Choose a backend supported by your machine. There is no automatic model download in this starter. Keep at least a 4096-token context window for this server recipe; when setting an explicit limit on macOS/Linux, prefix the command with `MAX_CONTEXT=4096` (or set that environment variable in your shell).

2. In `examples/dotnet-local-chat`, restore, build, and discover the server's model ID:

   ```bash
   dotnet restore --locked-mode --configfile NuGet.Config
   dotnet build -c Release --no-restore
   dotnet run -c Release --no-build -- --list-models
   ```

3. Stream a message through `Microsoft.Extensions.AI.IChatClient`:

   ```bash
   dotnet run -c Release --no-build -- --prompt "Explain local inference in one sentence."
   ```

   The app automatically selects the model when exactly one ID is advertised. To choose explicitly, copy an exact ID from discovery:

   ```bash
   dotnet run -c Release --no-build -- --model "ID_FROM_DISCOVERY" --prompt "Hello!"
   ```

   Use the official OpenAI client directly with the same endpoint and request:

   ```bash
   dotnet run -c Release --no-build -- --client openai --prompt "Hello!"
   ```

Model text streams to stdout; endpoint/model information and errors go to stderr. Each invocation sends one user message; it does not retain conversation history. Exit code `0` means the SDK stream completed with nonempty text and a finish reason, `1` means a request/configuration failure, and `130` means Ctrl+C cancellation. Inspect any partial stdout after an error. Transport success alone does not prove inference success; see the server's error-as-text limitation below.

## Configuration

| Option | Default / behavior |
| --- | --- |
| `--base-url URL` | `http://127.0.0.1:8080/v1`; include `/v1` once, not `/chat/completions` |
| `--model ID` | Auto-select only when one model is advertised |
| `--client extensions\|openai` | `extensions` uses `IChatClient`; `openai` uses `ChatClient` directly |
| `--max-tokens N` | `128`; the server's own cap also applies |
| `--timeout-seconds N` | `120` total for discovery plus generation |
| `--list-models` | Calls `GET /v1/models`; sends no generation request |
| `--allow-remote` | Explicit opt-in for an HTTPS endpoint outside loopback |
| `--help` | All options |

`TENSORSHARP_BASE_URL` and `TENSORSHARP_MODEL` provide environment defaults; CLI options take priority. `TENSORSHARP_API_KEY` optionally supplies a token for an authenticated proxy. For direct local TensorSharp, the app sends a non-secret placeholder. It does not read `OPENAI_API_KEY`. By default, requests go only to the local endpoint; explicit remote configuration sends requests to the endpoint you choose.

For a different local port:

```bash
dotnet run -c Release --no-build -- --base-url http://127.0.0.1:5000/v1 --list-models
```

## How the integration works

`GET /v1/models` discovers the exact configured ID; `POST /v1/chat/completions` requests streaming text. `Program.cs` constructs an OpenAI `ChatClient` with a custom endpoint, then calls either `CompleteChatStreamingAsync` or `.AsIChatClient().GetStreamingResponseAsync`. Both paths use the same official HTTP client. SDK retries and redirects are disabled to keep failures explicit.

Pinned dependencies are `OpenAI 2.14.0` and `Microsoft.Extensions.AI.OpenAI 10.11.0`, verified in the official registry and compiled together. `packages.lock.json` also records transitive versions and hashes. The existing TensorSharp examples use curl and Python; this starter adds the .NET client path. The server contract was inspected at TensorSharp commit `72e0fc816da32405ebefdbeae864597cd951c708`: [routes](../../TensorSharp.Server/Endpoints/OpenAIEndpoints.cs), [model discovery and chat handling](../../TensorSharp.Server/ProtocolAdapters/OpenAIChatAdapter.cs), and [SSE chunks](../../TensorSharp.Server/ResponseSerializers/OpenAIResponseFactory.cs). Client APIs and versions are documented by [OpenAI](https://www.nuget.org/packages/OpenAI/2.14.0) and [Microsoft](https://www.nuget.org/packages/Microsoft.Extensions.AI.OpenAI/10.11.0).

## Troubleshooting and limits

- Connection refused: start the server and check host/port. TensorSharp's inspected default is `0.0.0.0:5000`; the explicit recipe above binds `127.0.0.1:8080`.
- HTTP 404: use a base URL ending in `/v1`, with no duplicated `/v1`.
- No model or wrong model: start the server with `--model`; use the exact discovery ID, which normally omits `.gguf`. One server process hosts its startup model; this app does not switch or load models.
- `context_length_exceeded`: the rendered system/tool/skill prompt also consumes context. An actual run with bundled skills and `MAX_CONTEXT=2048` required 2959 protected prompt tokens and correctly returned HTTP 400. Shortening only the user message cannot fix that case; increase the context window or disable unused features. The model-free mock does not detect this deployment constraint.
- A discovered model is **configuration, not a readiness check**. TensorSharp may defer loading until chat. Some load failures are delivered as ordinary SSE text `Error: ...` with HTTP 200 and a normal finish marker. The SDK and this app then return that text with exit `0`; check the server log. The starter deliberately does not guess error status from generated words.
- A malformed stream, an empty reply or a stream ending without a finish reason fails with a diagnostic. A server's normal token-limit finish can leave an incomplete answer; raise both client and server limits as needed. Ctrl+C cancels the request; model loading may need a longer deadline.
- Scope is one-turn text chat. Tool calling, reasoning fields, multimodal inputs, embeddings, Responses API behavior and full OpenAI parity are not validated here.

## When to use / when not to use

Use this deployment when a .NET application should call an independently running local TensorSharp engine over HTTP, you have a supported model that fits your machine, and you want standard client APIs. Choose `IChatClient` when your application uses Microsoft's common chat abstraction; choose `openai` to work directly with the official SDK.

Choose a different deployment when you need a managed service with operational guarantees, authenticated multi-user hosting without building a secure proxy, in-process engine access without HTTP, or API/model capabilities outside this tested text-chat subset. This package supplies no performance comparison or throughput claim; evaluate your own model and hardware.

## Security

Keep TensorSharp bound to `127.0.0.1`. The inspected server has no built-in authentication or TLS; do not expose its default all-interface binding to an untrusted network. A local endpoint is still reachable by other local processes. Use an authenticated TLS proxy and deliberate network controls for remote access; the starter requires HTTPS plus `--allow-remote` for non-loopback URLs. Redirects are refused, and credentials, query strings and fragments in the base URL are rejected. Keep tokens in environment variables, and check server/proxy logging policies before sending sensitive prompts.

## Reproduce the model-free checks

```bash
dotnet restore --locked-mode --configfile NuGet.Config
dotnet build -c Release --no-restore
python3 ../../eng/tests/dotnet-local-chat.py
```

The Python harness binds an ephemeral loopback port, invokes the real compiled clients, checks the HTTP requests and serves deterministic SSE responses. It downloads no model and makes no hosted model call.

The mock harness lives under `eng/tests` following repository conventions. It tests both SDK paths, model selection, request payloads, SSE framing, malformed/truncated/empty responses, HTTP errors, and discovery/generation timeouts. It also reproduces the server's HTTP-200 error-as-text limitation. Keep generated logs under the repository's ignored `artifacts/` or `docs/validation/` directories.

## Real-model checks

After starting your own loopback server with an already-present supported text model and a server token cap of at least 512, run from the repository root:

```bash
python3 eng/tests/dotnet-local-chat-e2e.py --base-url http://127.0.0.1:8080/v1
```

This opt-in harness makes actual generation requests; it starts no server, downloads no model and provisions no hardware. It checks arithmetic output through both clients, raw SSE model identity/finish/usage, discovery and HTTP errors, and SIGINT cancellation followed by another successful generation. It requires macOS/Linux for SIGINT checks. Model answers are checked, so an unsuitable model or unsupported backend correctly fails the run; this is not a hermetic correctness test or benchmark.

### Verified local run

On 2026-10-31, all **10 real-endpoint checks** passed on macOS arm64 with .NET SDK 10.0.302, existing `gemma-4-E2B-it-Q8_0.gguf` (4,967,497,152 bytes), and `ggml_cpu`. Both client paths answered `4`. Raw SSE identified the configured model, finished with `stop` and `[DONE]`, and reported 1263 prompt tokens / 1 generated token. Each client was interrupted after producing `1`, returned exit `130`, and subsequently received `4` from a fresh request. Unknown-model preflight and a wrong API base also produced the expected actionable errors. The separate 22-case mock suite passed on the same final HTTP source.

The tested engine source was `9462445734ed7390accefd6c49a91d0c4a809a6f`, with these example changes applied. Its native bridge was rebuilt CPU-only from that checkout against unchanged GGML `facdc44a67da17df4433adcc468b424f48e9ded2`. Native library SHA256: `861ace737e0ccb1a0a8b990d91e43864957886fa6c2479c908b1036724e1524e`; model SHA256: `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a`.

For this run, the server command above additionally used `--no-prefix-cache` to exercise inference without restored prefix state, with `MAX_CONTEXT=4096`, `TS_SCHED_NUM_BLOCKS=16` and `TS_SCHED_MAX_RUNNING_SEQS=1`. It listened only on loopback port 18763. The server and both sample applications were built in an isolated checkout; no existing service was stopped. Full machine-local logs stay in ignored artifacts. This covers the stated model, CPU backend and text-only cases; it does not validate every backend/model, Agent Skills, multimodal inference or comparative performance.
