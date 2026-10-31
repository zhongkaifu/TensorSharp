# Call TensorSharp directly from .NET

This example loads an existing GGUF into TensorSharp's native inference runtime and streams a local conversation from C#. It opens no HTTP listener and uses no OpenAI client. Use the [HTTP example](../dotnet-local-chat) when the engine should run in a separate process.

The code uses public TensorSharp APIs: `ModelService` and `ChatSession` from `TensorSharp.Chat`, the service's `ChatStreamWithMetricsAsync`, and `SamplingConfig` plus model-family output parsers from `TensorSharp.Runtime`. The public chat types retain the `TensorSharp.Server` namespace; the chat assembly is host-neutral. The project references the checked-out source, so it stays aligned with that checkout's APIs.

## Run from the repository root

Prerequisites: .NET 10 SDK, the native build prerequisites for your platform, and an existing supported autoregressive GGUF. See [getting started](../../docs/getting_started.md). No model is downloaded by this example.

```bash
dotnet build examples/tensorsharp-native-chat -c Release -p:TensorSharpSkipMlxNative=true
dotnet run --project examples/tensorsharp-native-chat -c Release --no-build -- --model /path/to/existing-model.gguf --backend ggml_cpu --prompt "What is 2 + 2? Reply with just the single digit." --follow-up "Multiply that answer by 3. Reply with just the number." --max-tokens 64 --context-tokens 2048 --timeout-seconds 120
```

The build uses the repository's normal native GGML build for your platform. Skipping MLX in this command avoids building the separate MLX backend; this sample selects GGML. For GGML Metal, CUDA or Vulkan, first build the matching native backend with the platform instructions, then select `ggml_metal`, `ggml_cuda` or `ggml_vulkan`. Choosing a backend does not install its toolchain or make an unsupported model work.

If you already have a matching native build for **this checkout**, the repository supports `-p:TensorSharpSkipGgmlNative=true` to avoid rebuilding it. The native library must still be discoverable in the repository's native build tree or beside the app; a managed-only build is not inference validation. Do not mix unrelated managed and native revisions.

`--model` can also come from `TENSORSHARP_MODEL_PATH`. `--max-tokens` defaults to 128; `--context-tokens` defaults to 2048 and sets this console process's `MAX_CONTEXT`. The scheduler pool is sized in 256-token blocks to cover that context and allows one running sequence. Long prompts still need enough context for the rendered template, reply and safety margin. `--timeout-seconds` is a per-generation deadline after synchronous model loading. Ctrl+C cancels generation; the public load API is synchronous and does not accept cancellation.

## What this demonstrates

- Native local inference composed in a .NET application, with an explicit backend rather than an external model server.
- The embedded GGUF chat template plus TensorSharp's family-aware output parser, so model protocol/reasoning markers are not blindly printed as answer text.
- A reusable `ChatSession`, complete conversation history and per-turn terminal metrics: finish reason, prompt/generated tokens and actual KV-cache-reused tokens. The assistant history retains TensorSharp's raw token IDs and prompt-boundary metadata instead of reconstructing them from displayed text. Reuse is reported as observed; a second turn is not a guarantee of a particular cache hit count or speedup.
- Explicit resource lifetime (`IDisposable`), generation cancellation and a bounded token/context budget.

These are verified TensorSharp API capabilities, not claims that competing engines lack them. This small sample does not validate tool execution, Agent Skills, multimodal models, distributed execution or every backend/model combination. Its native finish reasons use TensorSharp's vocabulary (`eos`, `max_tokens`, etc.), rather than HTTP protocol mappings.

The answer streams to the terminal and `[TensorSharp native]` diagnostics report loaded model/backend and terminal metrics. Native runtime diagnostics can also appear. Exit `0` requires nonempty answer text and terminal metrics, `1` reports a configuration/load/generation error, and `130` is Ctrl+C cancellation. Check output content as well as the exit status. This is a runnable example, not a performance benchmark or production hosting service.

After building the example, model-free completion-boundary regression checks can be run from the repository root:

```bash
dotnet run --project eng/tests/NativeChatCompletion -c Release -- examples/tensorsharp-native-chat/bin/Release/net10.0/TensorSharp.NativeChat.dll
```

These seven checks load the compiled completion validator without loading a native backend or weights. They cover an already-cancelled token when terminal metrics are reported or absent, cancellation taking precedence over empty/missing output, ordinary completion, and unrequested runtime cancellation. Ctrl+C retains exit `130`; a generation deadline retains the timeout diagnostic and exit `1`.

## Verified local run

On 2026-10-31, this example was built with .NET SDK 10.0.302 and run on macOS arm64 with an already-installed `gemma-4-E2B-it-Q8_0.gguf` (4,967,497,152 bytes), `ggml_cpu`, a 2048-token context, greedy sampling and a 64-token reply limit. The simple two-turn command above returned `4`, then `12`, with `finish=eos` on both turns.

For a longer synthetic shared prompt, the first turn reported 624 prompt tokens, 1 generated token and 0 reused tokens; the follow-up reported 648 prompt tokens, 2 generated tokens and **626 KV-cache-reused tokens**, again answering `4`, then `12`. The short conversation reported 0 reused tokens: reuse depends on the actual conversation and runtime state. SIGINT after the first generated integer produced the cancellation diagnostic and exit `130`; a one-second generation deadline after partial output produced the timeout diagnostic and exit `1`. A fresh CLI invocation then answered `4`, followed by `12`; this checks cleanup and relaunch. A missing model produced a clear error and exit `1`.

To reproduce that longer input, pass `--prompt` the exact sentence `Background note: TensorSharp runs a local model inside this .NET process. ` repeated 40 times, immediately followed by `What is 2 + 2? Reply with just the single digit.`. Use `--follow-up "Multiply that answer by 3. Reply with just the number." --max-tokens 64`. The trailing space in each repeated sentence is significant for reproducing token counts.

The engine files match `72e0fc816da32405ebefdbeae864597cd951c708`; the final example and regression checks were validated in a worktree based on `6241652e6ecc3f7b2a6fcb15a4eb42409fadd9ca`. The CPU-only native bridge was rebuilt against unchanged GGML `facdc44a67da17df4433adcc468b424f48e9ded2`; the tested bridge SHA256 was `861ace737e0ccb1a0a8b990d91e43864957886fa6c2479c908b1036724e1524e`. Model SHA256: `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a`.

This validates the listed model, backend and text-only scenarios. It does not establish performance, equivalence on other hardware/backends, or support for every architecture. Full machine-local logs remain in ignored validation artifacts; no model files, personal model paths or generated logs are committed.
