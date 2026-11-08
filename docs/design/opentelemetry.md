# OpenTelemetry traces, metrics and logs — Design

**Status:** Implemented on branch `opentelemetry`; unit-tested (§12.1). Verified end to end on one machine against a
local Aspire dashboard and against an Application Insights resource (§12.5); overhead measured in §12.6.
**Date:** 2026-11-06
**Scope:** `TensorSharp.Server.Host`, `TensorSharp.Cli` and the libraries they host (Server, Chat, Runtime,
Models, AgentHost, Studio, and the MLX backend's worker thread). TensorAgent is out of scope: it never references
the OpenTelemetry SDK, and the instrumentation in the shared libraries records nothing there.

Telemetry is off unless an operator turns it on, works with any OTLP collector, and goes to Azure Application
Insights by setting one environment variable. This document describes what was built. It started as a design
draft; a check of that draft against the source found many errors, and the corrections are folded into the
sections below and listed in Appendix A.

---

## What exists

| Component | Files | What it does |
|---|---|---|
| Sources, meter, switches | [`TensorSharpTelemetry`](../../TensorSharp.Runtime/Telemetry/TensorSharpTelemetry.cs) | The activity sources `TensorSharp.Inference`, `TensorSharp.Models`, `TensorSharp.Media` and `TensorSharp.Agent`, the `TensorSharp` meter, the version, the process-wide switches (content capture, step spans, logs exported) and the scrubbing of upload paths and other registered directories. In-box `System.Diagnostics` only. |
| Names | [`TelemetryAttributes`](../../TensorSharp.Runtime/Telemetry/TelemetryAttributes.cs) | Attribute names and values shared by every assembly. |
| Instruments | [`TensorSharpMetrics`](../../TensorSharp.Runtime/Telemetry/TensorSharpMetrics.cs) | Every instrument on the `TensorSharp` meter, engine self-registration, the registries for media counters and GPU memory sources, and the engine step observer hook (`SetEngineStepObserver` / `OnEngineStep`). |
| One generation | [`GenAiRequestTelemetry`](../../TensorSharp.Runtime/Telemetry/GenAiRequestTelemetry.cs) | The span and metrics of one chat, completion or embeddings request; engine phases built after the fact; every exit recorded once. |
| Request context | [`TelemetryContext`](../../TensorSharp.Runtime/Telemetry/TelemetryContext.cs) | `AsyncLocal` protocol, request id, round and origin, set where they are known and read where spans start; `Clear()` for the long-lived workers (§5.4). |
| Content marker | [`LogContent`](../../TensorSharp.Runtime/Logging/LogContent.cs) | Wraps a log argument that is user content (§7.4). |
| SDK (host-only) | [`TensorSharp.Telemetry`](../../TensorSharp.Telemetry/): `TelemetryOptions`, `OtlpSignalSettings`, `TensorSharpTelemetrySetup` (with `TelemetrySession` and the DI extensions), `ContentRedactionProcessor` (with `ContentSpanProcessor` and `ScrubbedException`), `LogContextEnrichmentProcessor`, `TelemetryCliFlags` | Settings, exporters, processors, the `--otel-endpoint` flag. |
| Server host | [`Program.cs`](../../TensorSharp.Server.Host/Program.cs), [`ServerTelemetry`](../../TensorSharp.Server.Host/Hosting/ServerTelemetry.cs), [`ServerUsage`](../../TensorSharp.Server.Host/Hosting/ServerUsage.cs) | Wiring and startup order; the HTTP span filter and capability-path enrichment; usage text. |
| Server library | [`ServerSpan`](../../TensorSharp.Server/Logging/ServerSpan.cs), [`RequestLoggingMiddleware`](../../TensorSharp.Server/Logging/RequestLoggingMiddleware.cs), [`CapabilityPaths`](../../TensorSharp.Server/Logging/CapabilityPaths.cs), [`UploadStaticFiles`](../../TensorSharp.Server/Hosting/UploadStaticFiles.cs), [`ServerOptionsBuilder`](../../TensorSharp.Server/Hosting/ServerOptionsBuilder.cs), [`EmbeddingAdapter`](../../TensorSharp.Server/ProtocolAdapters/EmbeddingAdapter.cs), the OpenAI, Ollama and Web UI adapters, [`HealthEndpoints`](../../TensorSharp.Server/Endpoints/HealthEndpoints.cs), [`WebUiEndpoints`](../../TensorSharp.Server/Endpoints/WebUiEndpoints.cs) | Server span tags, request context, capability-path masking (uploads, code artifacts, stored responses), the embeddings span, response ids, polled routes kept out of HTTP metrics. |
| Chat | [`ChatGenerationPipeline`](../../TensorSharp.Chat/ChatGenerationPipeline.cs), [`InferenceTelemetry`](../../TensorSharp.Chat/InferenceTelemetry.cs) (also `TelemetryContextScope`, `ModelLoadTelemetry`), `ModelService(.Jev, .Agents)`, `ModelLifecycleService`, [`SkillChatLoop`](../../TensorSharp.Chat/Skills/SkillChatLoop.cs), [`WebUiChatService`](../../TensorSharp.Chat/WebUiChatService.cs), `DiffusionBatchScheduler`, `Knowledge/*`, `Jev/JevInference` | Chat, DiffusionGemma and Jev spans; model load; tool calls; images and video on the server. |
| Engine | [`InferenceEngine`](../../TensorSharp.Runtime/Scheduling/InferenceEngine.cs), `ContinuousBatchScheduler`, `SequenceState`, `InferenceRequestHandle`, `Speculative/SpeculativeExecution` | Step metrics, optional step spans, preemption bookkeeping, the engine's finish time, speculation counters. |
| Video | [`VideoGenerationTelemetry`](../../TensorSharp.Models/Models/Video/VideoGenerationTelemetry.cs) (`VideoClipTelemetry` and the helper), [`VideoGenerationProgress`](../../TensorSharp.Models/Models/Video/VideoGenerationProgress.cs), `MiniMaxH3Pipeline`, `Ltx25Pipeline` | The clip span, its phase and stage spans, and the video metrics (§8). |
| GPU memory | [`GpuMemoryTelemetry`](../../TensorSharp.Models/Telemetry/GpuMemoryTelemetry.cs), `ModelBase.DirectAllocatorForTelemetry`, `ModelLifecycleService` | Snapshot-based device memory gauges (§6.3). |
| MLX worker | [`MlxWorker`](../../TensorSharp.Backends.MLX/MlxWorker.cs) | Clears the `Activity.Current` and `TelemetryContext` its thread inherited. |
| Studio | [`StudioTelemetry`](../../TensorSharp.Studio/StudioTelemetry.cs), [`StudioJobs`](../../TensorSharp.Studio/StudioJobs.cs) | Studio's own source and meter; one trace per job (§9). |
| AgentHost | [`ShellRunner`](../../TensorSharp.AgentHost/CodeExec/ShellRunner.cs), `EgressProxy`, `PackageInstaller` | The `codeexec.run` span; content-marked log arguments. |
| CLI | [`CliTelemetry`](../../TensorSharp.Cli/CliTelemetry.cs), [`Program.cs`](../../TensorSharp.Cli/Program.cs), [`CliInferenceSession`](../../TensorSharp.Cli/CliInferenceSession.cs), `Logging/CliLoggingSetup`, `CliUsage`, `InteractiveSession` | The CLI session and its flush paths, chat spans, video clips. |
| Tests | [`InferenceWeb.Tests/Telemetry/`](../../InferenceWeb.Tests/Telemetry/), [`TensorSharp.Studio.Tests/Telemetry/`](../../TensorSharp.Studio.Tests/Telemetry/) | §12.1. |
| End-to-end scripts | [`eng/telemetry/aspire-e2e.ps1`](../../eng/telemetry/aspire-e2e.ps1), [`eng/telemetry/appinsights-e2e.ps1`](../../eng/telemetry/appinsights-e2e.ps1), [`README.md`](../../eng/telemetry/README.md) | §12.3, §12.4. |
| Samples | [`examples/opentelemetry/`](../../examples/opentelemetry/README.md) | The Aspire dashboard, a collector with Jaeger, Prometheus and Grafana, a collector that forwards to Application Insights, and KQL queries. |
| User docs | [`USAGE.md`](../../USAGE.md) ("Telemetry (OpenTelemetry)"), [`docs/env_var_feature_matrix.md`](../env_var_feature_matrix.md), [`config/README.md`](../../config/README.md), `FEATURES.md`, `README.md`, and their `_zh-cn` twins | Settings, quick starts and the `--config` route for operators. |

---

## 1. Goals and non-goals

**Goals**
- Traces, metrics and logs over OTLP to any collector the operator runs.
- Application Insights with one setting, with no collector required.
- Off by default; essentially free when off.
- Useful for both LLM serving and long video generations (MiniMax-H3, LTX 2.5), including Video Studio jobs.
- Prompts, model output, upload names and tool arguments never leave the process unless the operator opts in.

**Non-goals**
- No change to the console or file logs, except the few deliberate changes listed in §7.9.
- No vendor-specific instrumentation in the libraries.
- No telemetry in TensorAgent (desktop/mobile app).
- No tracing inside native kernels, and no trace context over tensor-parallel sockets.

---

## 2. Design principles

1. **Off by default, essentially free when off.** Libraries use only the in-box `System.Diagnostics`
   `ActivitySource` and `Meter`. With no listener, `StartActivity` returns `null` (about 14 ns, by OpenTelemetry
   .NET's own benchmarks) and an instrument drops its measurement. Call sites check `HasListeners()` or
   `IsAllDataRequested` before they build tag collections, span names or captured content.
2. **The standard `OTEL_*` environment variables are the interface.** TensorSharp reads only the variables the
   .NET SDK does not read itself, and the few of its own (§4.1).
3. **The SDK lives in one host-only project**, `TensorSharp.Telemetry`, referenced by the server host and the CLI
   only. NuGet consumers of the libraries, and TensorAgent, never get an OpenTelemetry dependency. It is not
   `TensorSharp.Runtime.Logging`: `TensorSharp.Chat` and `TensorSharp.Server` (and their packages) depend on that,
   and TensorAgent.Maui references it.
4. **Content stays local unless the operator opts in** (§7).
5. **Work that runs on a shared thread gets spans built after the fact**, from timestamps the code already takes:
   the engine phases (§5.2), Studio's queue and stages (§9.2), the wait for the video lock (§8.6).
6. **Every TensorSharp span is `ActivityKind.Internal`.** The GenAI conventions allow INTERNAL for a model in the
   same process. Only ASP.NET Core's span is SERVER, so Application Insights shows one `requests` row per HTTP
   request and the rest as `dependencies` of type InProc.
7. **Telemetry never fails a request.** Post-hoc span construction, the step span, the video helper and the GPU
   snapshot catch their own exceptions.

### 2.1 Why there is no span per token or per engine step

- **A batched step belongs to many requests.** Continuous batching runs one forward pass for every running
  sequence. A span has one parent, so a step span would either be copied into every request's trace, which
  multiplies the volume and makes each request look as if it paid for the whole step, or sit in a trace of its own
  with links, which many backends barely show. Queue, prefill and decode each belong to exactly one request.
- **Volume and cost.** A 1,000-token answer would produce about 1,000 spans instead of about 5. Application
  Insights charges per GB ingested.
- **Dropped spans.** The SDK's batch exporter queues 2,048 spans by default and drops the overflow. A busy server
  producing a span per token would drop spans at random, request spans included.
- **Readability.** "Why was this request slow?" is answered by queue vs prefill vs decode.

CPU is not the reason: a span costs a few hundred nanoseconds with the SDK listening, a decode step milliseconds.
Per-step data goes to metrics instead (step duration, batch size, §6.2), which are aggregated in the process.
DiffusionGemma follows the same rule: its scheduler batches requests, so a request gets one `diffusion.denoise`
span with a span event per preview, not a span per denoising step (§5.9).

### 2.2 Opt-in step spans

For profiling a single run, `TENSORSHARP_OTEL_TRACE_STEPS=1` adds one `inference.step` span per engine step
(§5.11). Each is a root in a trace of its own with links to the requests in its batch. It is off by default and is
the only place a span is created on the engine thread.

### 2.3 The video exception

A clip has tens of denoising steps that each take seconds, and clips run one at a time. So a clip gets a span per
phase and per stage, and a span **event** per finished denoising step (§8).

---

## 3. Architecture

### 3.1 Where the instrumentation lives

| Source or meter | Declared in | Spans or instruments |
|---|---|---|
| `TensorSharp.Inference` | Runtime | `chat <model>`, `text_completion <model>`, `embeddings <model>`, `prompt.render`, `inference.queue`, `inference.prefill`, `inference.decode`, `diffusion.denoise`, `inference.step` |
| `TensorSharp.Models` | Runtime | `model.load`, `model.unload` |
| `TensorSharp.Media` | Runtime | `generate_video <model>` and its children, `generate_image <model>` |
| `TensorSharp.Agent` | Runtime | `execute_tool <name>`, `codeexec.run` |
| `TensorSharp.Studio` | Studio | `studio.job <kind>`, `studio.queue`, `studio.stage <stage>` |
| meter `TensorSharp` | Runtime | everything in §6 except Studio's |
| meter `TensorSharp.Studio` | Studio | `tensorsharp.studio.*` (§9.5) |

`TensorSharp.Studio` has no project references, so it declares its own source and meter by name, and reads the
content-capture variable itself. The SDK subscribes with `AddSource("TensorSharp.*")` and
`AddMeter("TensorSharp", "TensorSharp.Studio", "TensorSharp.*")`.

### 3.2 The `TensorSharp.Telemetry` project

`net10.0`, packable, one project reference (`TensorSharp.Runtime`) and no ASP.NET Core dependency, because the CLI
references it too. It contains:

- `TelemetryOptions.FromEnvironment(TelemetryHost)`: every on/off and exporter rule in §4, in one tested place.
  An overload takes a variable source, for tests.
- `TensorSharpTelemetrySetup`: resource, tracing, metrics and logging configuration shared by both hosts.
  `ApplyProcessWide` pins the library switches (`TensorSharpTelemetry.ConfigureCaptureMessageContent`,
  `ConfigureTraceSteps`, `ConfigureLogsExported`) to the resolved options, so the span side and the log side
  always agree.
- `AddTensorSharpTelemetry(IServiceCollection, ILoggingBuilder, TelemetryOptions, configureTracing,
  configureMetrics)` and `StartTensorSharpTelemetry(IServiceProvider)` for the server. The callbacks let the server
  add the ASP.NET Core instrumentation, which carries a framework reference to `Microsoft.AspNetCore.App` and
  therefore lives in `TensorSharp.Server.Host`, not here.
- `TelemetrySession` for a process without a host (the CLI).
- The log processors (§7) and `TelemetryCliFlags` (§4.5).

### 3.3 Server wiring

In [`Program.cs`](../../TensorSharp.Server.Host/Program.cs), in this order:

1. `TelemetryCliFlags.Consume` removes `--otel-endpoint` right after the code-execution flags are consumed, and
   `ApplyToEnvironment` sets `OTEL_EXPORTER_OTLP_ENDPOINT`.
2. `TelemetryOptions.FromEnvironment(TelemetryHost.Server)` runs after the flag appliers that set the
   `TENSORSHARP_TP_*` variables the resource reports. A malformed value prints one `Configuration error:` line and
   exits with `HostExitCodes.ConfigurationError`.
3. Next to `LoggingSetup.Configure`, `builder.Services.AddTensorSharpTelemetry(...)` adds
   `AddAspNetCoreInstrumentation` for traces (with `ServerTelemetry.ShouldTrace` as the filter and
   `ServerTelemetry.Enrich` as `EnrichWithHttpRequest`) and for metrics.
4. Right after `builder.Build()`, `app.Services.StartTensorSharpTelemetry()` resolves the tracer and meter
   providers. The hosting package would otherwise build them inside `app.RunAsync`, and the Studio service's job
   recovery, the startup model load and the prefix-cache warm-up all run before that. Resolved first, the providers
   are also disposed last, after Studio's shutdown has ended its running job's span (§9.4). The same call registers
   a flush of all three providers on `ApplicationStopped`, which fires once the server and the hosted services have
   stopped and before dependency injection disposes the models. The flush runs on a worker thread, metrics first,
   then traces, then logs, and the host waits for it at most 5 s in all. If an exporter is slow or unreachable, the
   host stops waiting after 5 s and goes on disposing the models while that export finishes or times out in the
   background; the providers' own flush at disposal still runs afterwards. The engine and KV-cache gauges read only
   live engines (§6.2), and the providers' own flush at disposal runs after the models are gone, so without the stop
   flush a server that lived less than one export interval (always 60 s under Azure Monitor, §10.4) never reported
   them. It also exports the last requests' spans, logs and metric deltas before a model's teardown, which can be
   long enough for an orchestrator's grace period to run out.
5. When metrics are exported (telemetry on and `Metrics != None`), `GpuMemoryTelemetry.EnsureRegistered()` and
   `TensorSharpMetrics.SetEngineStepObserver(GpuMemoryTelemetry.Snapshot)` (§6.3). Then
   `TensorSharpTelemetry.RegisterSensitivePath` registers the upload directory and, with `--code-exec`, the
   code-execution scratch and artifact directories for exception scrubbing (§7.5).
6. Right after the `Logging initialized` line, the `Telemetry:` line (§4.6) and any warnings, before any model
   loads, so it appears in every mode, embeddings servers and refused loads included.

Downloads triggered by `--config` happen in `ConfigFileArgs.Expand`, before any of this, so **startup downloads are
not traced**.

### 3.4 CLI wiring

In `Main` ([`Program.cs`](../../TensorSharp.Cli/Program.cs)):

- `TelemetryCliFlags.Consume` reads `--otel-endpoint` without removing it from `args` (the CLI's own parser ignores
  flags it has no case for), then `TelemetryOptions.FromEnvironment(TelemetryHost.Cli)`.
- `TelemetrySession.Start` builds the tracer and meter providers before the logger factory, and
  `CliLoggingSetup.Build(options, telemetry.AddLogging)` adds the OpenTelemetry logger provider after the console
  and file providers. On a normal exit the logger factory is disposed first (flushing logs), then the session
  (flushing traces and metrics, 5 s each). Before that, `TelemetrySession.Flush` exports metrics, then traces, while
  the run's last inference engine is still alive, for the same reason as the server's stop flush (§3.3); a CLI run
  is usually shorter than one export interval. It runs only when that engine is disposed: at the end of a one-shot
  mode, and when the interactive session exits. As on the server, it runs on a worker thread and the CLI waits for
  it at most 5 s. Interactive resets (a new system prompt, a skill change, a failed prefix warm-up) and `/model`
  switches dispose the engine without a flush, so they never hold the prompt on a network export.
- The two exits that skip `using` disposal flush explicitly: the Linux `ggml_vulkan` leave callback disposes the
  session right after the logger factory and before `_exit`, and `RunToExit`'s `catch`, with telemetry on, disposes
  the logger factory (which owns the OpenTelemetry log provider, so the `CliFailed` line is exported too) and then
  the session before rethrowing an unhandled exception. With telemetry off that path is unchanged.
- The `Telemetry:` line is logged after `tensorsharp-cli started`. When metrics are exported,
  `GpuMemoryTelemetry.EnsureRegistered()` runs and `GpuMemoryTelemetry.Snapshot` is registered as the engine step
  observer, as on the server; the CLI tracks the model it loads (`GpuMemoryTelemetry.Track`).

---

## 4. Configuration

### 4.1 Settings

| Setting | Read by | Effect |
|---|---|---|
| nothing set | — | **Off.** No provider is registered and no listener exists. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | SDK; validated by `TelemetryOptions` | OTLP for every signal. Must be an absolute `http://` or `https://` URL. |
| `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_ENDPOINT` | `TelemetryOptions` (§4.3) | OTLP for that signal only, to that URL used as given. |
| `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_{HEADERS,PROTOCOL,TIMEOUT}` | `TelemetryOptions` (§4.3), only for a signal that exports over OTLP | Per-signal overrides of the generic values. |
| `OTEL_EXPORTER_OTLP_PROTOCOL`, `_HEADERS`, `_TIMEOUT` | SDK; protocol validated when a signal exports over OTLP | `grpc` (default) or `http/protobuf`; auth headers such as `x-otlp-api-key=<key>`. |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | `TelemetryOptions` | Every enabled signal also goes to Application Insights, in process. Combines with OTLP. |
| `TENSORSHARP_AZURE_MONITOR_AAD=1` | `TelemetryOptions` | Entra ID authentication (`DefaultAzureCredential`). Needs the connection string, which still names the resource. |
| `TENSORSHARP_OTEL=1` | `TelemetryOptions` | OTLP with no endpoint override, so the SDK's default for the protocol applies: `http://localhost:4317` for gRPC, `http://localhost:4318` for `http/protobuf`. |
| `OTEL_{TRACES,METRICS,LOGS}_EXPORTER` | `TelemetryOptions` (the .NET SDK does not read them) | Comma list of `otlp`, `console`, `none`. `otlp` or `console` turns telemetry on; `none` turns that signal off for every sink, Application Insights included. Unknown names are a warning. |
| `OTEL_SDK_DISABLED=true` | `TelemetryOptions` | Off, whatever else is set. |
| `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES` | SDK environment detector, applied last | Override the defaults of §4.7. |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | SDK; mapped by `TelemetryOptions` under Application Insights | §4.4. |
| `OTEL_BSP_*`, `OTEL_BLRP_*`, `OTEL_METRIC_EXPORT_INTERVAL`, `OTEL_METRIC_EXPORT_TIMEOUT` | SDK | Batch and export intervals. The metric default is 60 s. The Azure Monitor metric reader ignores the two metric variables: exporter 1.10.0 builds it with the SDK's defaults, so it exports every 60 s. |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `TelemetryOptions`; Studio reads it itself | Export prompts and output (§7.7). On for `1`, `true`, `yes` or `on`, any case. The same variable Microsoft.Extensions.AI and the OpenTelemetry Python GenAI instrumentations read. |
| `TENSORSHARP_OTEL_TRACE_STEPS=1` | `TelemetryOptions` | `inference.step` spans (§5.11). |
| `TENSORSHARP_OTEL_LOG_LEVEL` | `TelemetryOptions` | Minimum level exported over OpenTelemetry, server and CLI. A malformed level is a configuration error. Unset keeps the host's level. |
| `Logging__OpenTelemetry__LogLevel__*` | ASP.NET Core configuration (server only) | Standard provider rules for the `OpenTelemetry` provider alias; see §7.3 for the ASP.NET Core exception. |
| `TENSORSHARP_TP_NODE_ID`, `TENSORSHARP_TP_DEGREE` | resource (§4.7) | Tensor-parallel node id and degree as resource attributes. |
| `--otel-endpoint <url>` | `TelemetryCliFlags` | Sets `OTEL_EXPORTER_OTLP_ENDPOINT` (§4.5). |

The backend can change at runtime (a model load), so it goes on spans and metrics, never on the resource.

### 4.2 When telemetry is on

`TelemetryOptions.FromEnvironment` resolves each signal separately:

1. `OTEL_SDK_DISABLED=true` → off.
2. With `OTEL_<SIGNAL>_EXPORTER` set, its list chooses the non-Azure sinks; `none` empties the signal.
   Without it, the signal uses OTLP when the generic endpoint, that signal's endpoint or `TENSORSHARP_OTEL=1` is set.
3. The connection string adds Azure Monitor to every signal not set to `none`.
4. Telemetry is on when any of `TENSORSHARP_OTEL`, the connection string, an explicit `otlp`/`console` sink or an
   endpoint is set, and at least one signal has a sink.

Configuration errors (one `Configuration error:` line, exit `HostExitCodes.ConfigurationError`, before any model
loads): an endpoint that is not an absolute `http(s)` URL; `--otel-endpoint` with no URL or a relative one; with
telemetry on, a malformed `TENSORSHARP_OTEL_LOG_LEVEL`; and, only when at least one signal exports over OTLP, a
protocol other than `grpc` / `http/protobuf` (generic or per signal) or a malformed per-signal timeout. The
per-signal overrides are read only for the signals that use OTLP. The `OTEL_*` variables are shared by every
OpenTelemetry SDK on the machine, and a value set for another program (`http/json` is valid in the specification,
not in .NET) must not stop TensorSharp from starting when its telemetry is off or goes to Application Insights only.
Warnings: an unknown exporter name; a sampler Application Insights cannot honour (§4.4).

### 4.3 Per-signal OTLP, and why it is applied in code

TensorSharp registers one OTLP exporter per signal (`AddOtlpExporter` on each builder), so a signal can go to
OTLP, Azure Monitor, both or neither. The .NET SDK's per-signal `AddOtlpExporter` reads only the generic
`OTEL_EXPORTER_OTLP_*` variables; the cross-cutting `UseOtlpExporter`, which reads the per-signal ones, always turns
every signal on and throws when combined with `AddOtlpExporter`. So
[`OtlpSignalSettings`](../../TensorSharp.Telemetry/OtlpSignalSettings.cs) resolves
`OTEL_EXPORTER_OTLP_<SIGNAL>_{ENDPOINT,HEADERS,PROTOCOL,TIMEOUT}` and applies them to that exporter's options. A
per-signal endpoint is used exactly as given, as the specification says; the generic endpoint keeps the SDK's own
handling, which appends `/v1/traces` and so on for `http/protobuf`.

### 4.4 Sampling, and the mapping under Application Insights

Without Application Insights the SDK reads `OTEL_TRACES_SAMPLER` and `_ARG` itself (default
`parentbased_always_on`). TensorSharp only describes the value on the startup line.

With Application Insights it cannot be left to the SDK. `AddAzureMonitorTraceExporter` installs its own sampler on
the **whole** tracer provider: by default a rate limit of 5 traces per second, which would also throttle OTLP, and
once a sampler is set in code the SDK ignores `OTEL_TRACES_SAMPLER`. So `TelemetryOptions` maps the standard variable
onto the Azure sampler's options (`SamplingRatio`, `TracesPerSecond`), and `ConfigureAzure` sets them explicitly:

| `OTEL_TRACES_SAMPLER` | Azure sampler | Startup line |
|---|---|---|
| unset, `always_on`, `parentbased_always_on` | fixed percentage 1.0 | `azuremonitor fixed_percentage(1)` |
| `always_off`, `parentbased_always_off` | fixed percentage 0 | `azuremonitor fixed_percentage(0)` |
| `traceidratio`, `parentbased_traceidratio`, `microsoft.fixed_percentage` | fixed percentage `_ARG` (clamped to 0–1; 1 when absent) | `azuremonitor fixed_percentage(<ratio>)` |
| `microsoft.rate_limited` | `_ARG` traces per second (5 when absent) | `azuremonitor rate_limited(<n>/s)` |
| anything else | fixed percentage 1.0, plus a warning | as above |

One sampler governs both sinks, so OTLP receives the same traces as Application Insights. The Azure sampler makes
its own decision per trace and does not follow an incoming `traceparent`'s sampled flag, so the `parentbased_*`
names map onto a plain ratio. Metrics are never sampled.

### 4.5 Command-line flag and config files

One flag, `--otel-endpoint <url>` (or `--otel-endpoint=<url>`), sets `OTEL_EXPORTER_OTLP_ENDPOINT`; the last
occurrence wins, and the command line wins over an inherited variable. Every other setting stays an environment
variable. A `--config` file can already set any of them through its reserved `environment` object
(`ConfigFileArgs`; the process environment wins over the file), so the flag is a convenience, not the only way.

- **Server.** `ServerOptionsBuilder` rejects unknown flags, so `Program.cs` consumes the flag before it runs. The
  builder also recognises and skips it (a `Build` handed the raw command line does not throw), lists it in
  `SuggestFlagCorrection` (a typo gets a "Did you mean" hint), and `ServerUsage` documents it in a "Telemetry"
  section and in the environment-variable lines. `otel-endpoint` is in `ConfigFileArgsTests.ServerRecognizedConfigKeys`,
  the set that guards shipped `config/*.json` presets.
- **CLI.** Read in `Main` before the logger factory is built, without removing it. `CliUsage` documents it, which is
  also what lets `NormalizeOptionSpellings` handle the `=` form.

### 4.6 Startup line

Both hosts log one line (event `HostConfiguration`), built by `TelemetryOptions.Describe()`:

```
Telemetry: traces(otlp,azuremonitor) metrics(otlp,azuremonitor) logs(otlp,azuremonitor) -> OTLP http://collector:4317 (grpc), Azure Monitor; service tensorsharp-server; content capture off; sampler azuremonitor fixed_percentage(1)
```

or `Telemetry: off (<reason>)`. It never contains the connection string or OTLP headers, and a `user:password@`
part of an endpoint is removed. `engine step spans ON` and `content capture ON` are shown when set; `Azure Monitor
(Entra ID)` when Entra ID is used. Per-signal OTLP overrides (§4.3) are shown inside the parentheses after the
generic protocol, for example `OTLP http://localhost:4317 (grpc; metrics: http/protobuf)` or
`OTLP http://collector:4317 (grpc; traces: http://tempo:4318/v1/traces http/protobuf)`. Without per-signal
overrides the line is as above.

### 4.7 Resource

`ConfigureResource` builds, in order (later entries win on a collision):

- `service.name` = `tensorsharp-server` or `tensorsharp-cli`; `service.version` = the assembly's informational
  version without the `+<commit>` Source Link suffix; `service.instance.id` = `<machine>:<pid>`, readable in
  Application Insights' `cloud_RoleInstance` and stable across a process's lifetime;
- the SDK's telemetry attributes; `host.name`, `process.pid`; `tensorsharp.tp.node_id` and `tensorsharp.tp.degree`
  when those variables are set;
- the SDK's environment detector **last**, so `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` win over every
  default.

### 4.8 Failure behaviour

Export failures never fail a request: exporters run on their own threads with bounded queues that drop overflow.
The SDK reports them only through its EventSource (self-diagnostics), not through `ILogger`, so a wrong API key or an
unreachable collector is silent in the console (§13). When the host stops, the server waits at most 5 s in all for
its stop flush (§3.3); an exporter that is slow or unreachable then finishes or times out in the background while the
models are disposed. The providers flush again when they are disposed, where an unreachable exporter adds its own
export timeout to the shutdown. The CLI's flush before its last engine goes (§3.4) is bounded the same way, and its
session flushes each provider with a 5 s timeout when it is disposed.

---

## 5. Traces — LLM serving

### 5.1 Span tree

```
POST /v1/chat/completions          ASP.NET Core server span (SERVER)     -> App Insights "requests"
│                                  tensorsharp.request_id, tensorsharp.protocol, gen_ai.response.id
└─ chat <model>                    one per generation (TensorSharp.Inference)
   ├─ prompt.render                engine lookup -> SequenceState constructor (SubmittedAt)
   ├─ inference.queue              SubmittedAt -> FirstScheduledAt          ┐ built when the request
   ├─ inference.prefill            FirstScheduledAt -> FirstTokenAt         │ finishes, from the
   └─ inference.decode             FirstTokenAt -> FinishedAt               ┘ scheduler's timestamps
```

- `<model>` is the model file name without directory or extension (`GenAiRequestTelemetry.ModelNameFromPath`).
- Ollama's `/api/generate` produces `text_completion <model>`: `GenerateStreamAsync` passes
  `ChatTurnContext.OperationName = text_completion`.
- DiffusionGemma replaces the three `inference.*` spans with one `diffusion.denoise` (§5.9).
- A skills or multi-agent turn produces one chat span per round, with `execute_tool` spans beside them (§5.7).
- An incoming W3C `traceparent` is honoured, so a client with its own tracing sees TensorSharp as part of its trace.

### 5.2 Engine phases, built after the fact

The engine runs on its own worker thread, so its phases are not spans that thread opens.
`GenAiRequestTelemetry.RecordEnginePhases` builds them with explicit start and end times once the request is done:

- Every server protocol reaches `engine.SubmitRequest` through `ChatGenerationPipeline.ChatStreamWithMetricsAsync`.
  The CLI submits through `CliInferenceSession.Generate` to an engine of its own and calls the same helpers. The only
  other callers are benchmarks and `eng/` probes.
- `SubmittedAt` is stamped by the `SequenceState` constructor, after all template rendering, tokenizing, compaction
  and media encoding, so `prompt.render` runs from just after the engine lookup to `SubmittedAt`. If preparation
  throws, `prompt.render` ends at the throw.
- The engine records when it finished a request: `InferenceCompletion.FinishedAt`, stamped in `CompleteFinished` and
  `CompleteAborted`. Decode ends there, not when the request thread has drained the token channel, which waits on the
  client. (The chat pipeline falls back to the time it read the completion when `FinishedAt` is absent. The CLI ends
  decode when it has observed the engine's completion; its token loop runs on the calling thread.)
- A phase whose start is unknown is left out: no `FirstScheduledAt` gives only `inference.queue` up to the end; no
  `FirstTokenAt` gives `inference.prefill` up to the end. A phase never starts before its parent: a start earlier
  than the chat span's own (a caller that built its sequence, which stamps `SubmittedAt`, a moment before starting
  the span) is moved to the span's start. An end before its start is clamped to zero length.
- Wall time and compute time differ under continuous batching. Span durations are wall time;
  `PrefillElapsedTicks` / `DecodeElapsedTicks` go on the phase spans as `tensorsharp.compute.prefill_ms` /
  `tensorsharp.compute.decode_ms`.

### 5.3 Preemption

Preemption puts a running sequence back at the front of the waiting queue to be prefilled again. Admission used to
overwrite `FirstScheduledAt`, so a request preempted while decoding got `FirstScheduledAt > FirstTokenAt`: a negative
prefill and a queue span stretched over work already done.

- `ContinuousBatchScheduler` now sets `seq.FirstScheduledAt ??= DateTime.UtcNow`: it keeps the **first** admission.
  The other readers of the field only ask whether the sequence was ever scheduled, so they are unaffected.
- `PreemptSequence` increments `SequenceState.PreemptionCount` (which `ResetForPreemption` leaves alone); the chat
  span gets `tensorsharp.preemptions` when it is non-zero. The wait to be re-admitted and the re-prefill fall into
  whichever phase the request was in: prefill if it had no token yet, decode otherwise.
- As a backstop, a first token earlier than the first admission skips the prefill/decode split instead of drawing a
  negative span.
- `DeferForSequenceSlot` (no device slot for the sequence) also goes through `PreemptSequence`, so it counts as a
  preemption too, and a deferred sequence keeps the `FirstScheduledAt` of its failed admission: its queue phase ends
  early and the wait for a slot shows up in prefill.

### 5.4 Worker threads and `Activity.Current`

The engine's worker is started with plain `Thread.Start`, which flows the `ExecutionContext` of whichever request
lazily built the engine, its server span included, for the engine's whole lifetime. The same holds for
`DiffusionBatchScheduler`'s worker and for `MlxWorker`, whose work items do not carry their caller's context. Each
worker loop therefore starts with `Activity.Current = null` and `TelemetryContext.Clear()`. The second matters as
much as the first: the request id, protocol, round and origin are `AsyncLocal` values that flowed the same way, and
`LogContextEnrichmentProcessor` (§7.2) would otherwise stamp every line the worker writes, for any request, with
the creating request's `tensorsharp.request_id`. They are not switched to `UnsafeStart`, which would also drop the
inherited logging scope and change what their console and file lines print (scopes are not exported, §7.6).

### 5.5 Async iterators, and the wrapper that records every exit once

The chat pipeline is an async iterator. After a `yield return` the iterator resumes under the **consumer's**
`ExecutionContext`, every time, and changes it made to `Activity.Current` never reach the consumer. So:

- The span is held in a `GenAiRequestTelemetry` object, not left in `Activity.Current`; children are parented on
  `requestTelemetry.Context` explicitly, and the object is disposed exactly once (`Interlocked`).
- An iterator cannot put a `catch` around a `yield`. `ChatStreamWithMetricsAsync` (the public entry point every
  protocol calls) is therefore a thin wrapper: it starts the telemetry, drives a private `ChatStreamCoreAsync`
  through `GetAsyncEnumerator()`, and puts each `MoveNextAsync()` in `try/catch`. A
  `PromptContextOverflowException` is recorded as refused `context_overflow`, a `ModelUnloadedException` as refused
  `model_unloaded`, a cancellation as cancelled, anything else as failed with its type. A stream that ends with no
  terminal update and no exception (the consumer stopped reading) is cancelled.
- The core's existing outermost `finally`, which every exit runs through, records the engine phases and usage once,
  and only when the request was submitted. `seq`, `completion`, the token list, the finish reason and the cancel flag
  are hoisted above that `try` so the `finally` sees them.
- Engine errors arrive as an exception from the token loop (`CompleteWithError` completes the token channel with the
  exception), not from the `await handle.Completion` catch; the wrapper sees both the same way.
- `DiffusionChatStreamAsync`, which the Web UI calls directly, has the same wrapper shape around
  `DiffusionChatStreamCoreAsync`; the chat pipeline's diffusion branch calls the core, so a generation has one span.

The span starts in the wrapper before its first `yield`, which is the only moment `TelemetryContext.Round` and
`Origin`, set by a caller just before the call, are reliably visible (§5.7).

### 5.6 Attributes

GenAI semantic conventions, development status:

| Attribute | Source |
|---|---|
| `gen_ai.operation.name` | `chat`, `text_completion`, `embeddings`, `execute_tool` |
| `gen_ai.provider.name` | `tensorsharp` (spans and metrics) |
| `gen_ai.request.model` | the model file name |
| `gen_ai.request.max_tokens` | the `maxTokens` argument (`SamplingConfig.MaxTokens` is not read on this path) |
| `tensorsharp.request.effective_max_tokens` | when fitting the prompt to the context lowered the budget |
| `gen_ai.request.temperature`, `top_p`, `top_k`, `seed` | the effective `SamplingConfig` |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | prompt tokens; generated tokens |
| `gen_ai.usage.cache_read.input_tokens`, `tensorsharp.prefix_cache.reused_tokens` | prefix-cache reuse, when non-zero |
| `gen_ai.response.finish_reasons` | the finish reason; absent after an engine error or an abandoned stream. When the CLI stops a generation itself because a stop sequence matched or a tool call ended the turn, it is `stop_sequence`, as on the server. |
| `gen_ai.response.id` | on the **server** span, through `ServerSpan.SetResponseId`: the OpenAI chat adapter always sets `chatcmpl-…`; the Responses adapter sets `resp_…` only when the request has `store: false` or content capture is on, because a stored response's id retrieves its full output (`GET /v1/responses/{id}`, §5.10). The chat pipeline never sees the id. Ollama and the Web UI have none. |
| `gen_ai.conversation.id` | the session id, only for a session that is one conversation (`!SharedAcrossConversations`, the Web UI). The stateless protocols share one session whose constant id would group unrelated requests. |
| `gen_ai.input.messages`, `gen_ai.output.messages` | content capture only (§7.7) |
| `tensorsharp.engine_request_id` | the engine's own sequence id (`chat-<guid>`), which engine-thread log lines carry (§7.2) |
| `tensorsharp.request_id`, `tensorsharp.protocol` | `TelemetryContext` (§5.8) |
| `tensorsharp.backend` | the loaded backend |
| `tensorsharp.round`, `tensorsharp.origin`, `tensorsharp.internal` | §5.7 |
| `tensorsharp.ttft_ms` | time to first token, for percentiles in Application Insights (§10.7) |
| `tensorsharp.preemptions` | §5.3 |
| `tensorsharp.outcome`, `error.type` | `completed`, `cancelled`, `refused` or `failed`; a refusal code or the exception's full type name, never a message. Only `failed` sets the span status to Error. |

### 5.7 Internal calls and rounds

One HTTP request can run many generations: one per skills-loop round, the final generation, an artifact correction,
each sub-agent round, and the knowledge reranker and media extractor. Each gets its own chat span. Two
`AsyncLocal` values in `TelemetryContext` say which it is, set through `TelemetryContextScope` (Chat) just before the
generation starts and restored afterwards:

- `Round(n)`: `SkillChatLoop` sets the loop round, the final generation's true ordinal and the correction round;
  each child agent's generator in `ModelService.Agents` counts its own calls. The span gets `tensorsharp.round`. Set
  only while spans are listened to.
- `Origin(name)`: `knowledge-rerank` (`ModelKnowledgeReranker`) and `knowledge-media-extract`
  (`ModelKnowledgeMediaExtractor`). The chat pipeline sets `prefix-cache-warmup` itself when
  `PrefixCacheWarmup.IsRunningHere`, and the CLI's one-token system-prefix warm-up sets it too. The CLI's
  `--warmup-runs` passes set `warmup`. The span gets `tensorsharp.origin` and `tensorsharp.internal=true`, and the
  request metrics carry `tensorsharp.origin`, so client dashboards can leave internal calls out.

The request metrics are recorded **per generation**, so a skills turn produces one `gen_ai.server.request.duration`
sample per round (§13).

### 5.8 Protocol and request id

The generation code in Chat cannot see the `HttpContext`, and Chat cannot reference Server. So
`RequestLoggingMiddleware` puts both facts in `TelemetryContext` once per request:

- `TelemetryContext.Protocol` = `RequestLoggingMiddleware.ClassifyClient` (`openai`, `ollama`, `webui`; now
  `internal`, and `/api/embed` is classified as Ollama).
- `TelemetryContext.RequestId` = the request id, cut to 64 characters without splitting a surrogate pair. The id is
  the client's own `X-TensorSharp-Request-Id` when it sends one, of any length; the response header and the local
  log scope keep it whole.
- The server span gets `tensorsharp.request_id` and `tensorsharp.protocol`, found through `ServerSpan.Recorded`,
  which reads `IHttpActivityFeature` (not `Activity.Current`) and accepts only a recorded `ActivityKind.Server`
  activity: with telemetry off, ASP.NET Core still creates a plain activity for log correlation, and tagging it would
  be work nobody reads.

`GenAiRequestTelemetry`, `VideoClipTelemetry` and the image span copy both values onto their spans.

### 5.9 Other spans

| Span | Where | Notes |
|---|---|---|
| `embeddings <model>` | `EmbeddingAdapter.EmbedAsync` (`/v1/embeddings`, `/api/embed`) | One per request, around both the dispatcher path and `EmbedMediaAsync`. Not in `EmbeddingRequestDispatcher`, which batches several requests into one encoder call. `tensorsharp.embedding.inputs`; usage records the prompt tokens (and `output_tokens=0`). Refusals: `model_not_found` (404), `request_too_large` (413), `invalid_request`, `invalid_json`, `invalid_input`. |
| `model.load` | `ModelService.LoadModelCore`, which both `LoadModel` (Web UI, `HostedModelGuard` reloads) and `LoadStartupModel` call; also `GgufKnowledgeEmbeddingProvider` | `tensorsharp.model.file` (file name, never a path), `gen_ai.request.model`, `tensorsharp.backend`, `tensorsharp.model.mmproj_file`, `tensorsharp.model.startup_load`; after the load the architecture, the backend that actually loaded, `tensorsharp.tp.degree` and `tensorsharp.layer_split.degree` when above 1. The distributed no-op is `tensorsharp.model.load.skipped` and records no duration. A refusal (`ModelLoadRefusal.IsRefusal`) is `refused`, cancellation `cancelled`, anything else `failed`; refused and failed both set Error status, because the requested model is not loaded. The span is current during the load, so the loader's log lines are in its trace. |
| `model.unload` | `ModelService.UnloadModelCore`, when a model was loaded | No metric. |
| `diffusion.denoise` | `DiffusionChatStreamCoreAsync` | One per request, from `scheduler.Submit` to its result, with a `preview` span event per preview (`tensorsharp.diffusion.block`, `.step`, `.total_steps`) and `tensorsharp.diffusion.previews`. The scheduler batches requests, so there is no span per denoising step. |
| `chat <model>` for Jev | `ModelService.JevAsync` → `TraceJevAsync` | `POST /v1/systemone` drives DiffusionGemma through `JevInference.Run` and bypasses the pipeline. The span starts before the execution gate, so queueing counts. Usage comes from a callback with the same totals as the response. Refusals: `invalid_request`, `model_not_found`, `model_unavailable`, `queue_full`. |
| `execute_tool <name>` | `SkillChatLoop.ExecuteSkillCallsAsync`, one per host tool call | Started and stopped **inside** the `Task.Run` worker with no `yield` in between, so it is `Activity.Current` for exactly the tool's work and `codeexec.run` nests under it. Its parent is what the request had current (Task.Run carries the context), so it is a sibling of the round's chat span. `gen_ai.operation.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `tensorsharp.round`, `tensorsharp.skill.id`, `tensorsharp.tool.result_bytes`; arguments and result only with capture on. A tool that reports failure is `failed` with `error.type=tool_error`; only an exception sets Error status. |
| `codeexec.run` | `ShellRunner.Run`, after the early refusals that run nothing | Implicit parent (the current `execute_tool`). `tensorsharp.codeexec.shell`, `.sandbox`, `.timed_out`, `.installs`, `.output_bytes`, `.background`, and `.exit_code` only when the command started and was not killed. A command that never started is Error with `error.type=not_started`; a non-zero exit is the model's code failing, which is the tool working, so it is not an error. The command text only with capture on. |
| `generate_image <model>` | `WebUiChatService`, plain and streaming image routes | One per request (`tensorsharp.media.kind=image`, `tensorsharp.image.mode` = `generate` or `edit`). The first outcome marked is kept. Refusals: `invalid_request`, `upload_quota_exceeded`, `payload_too_large`, `not_an_image_model`. No phase spans (Qwen-Image reports only steps) and no prompt, even with capture on. |
| `generate_video <model>` | §8 | |
| outgoing HTTP | `AddHttpClientInstrumentation` | Startup downloads are not traced (§3.3). |

### 5.10 The HTTP span filter and capability-path masking

- [`ServerTelemetry.ShouldTrace`](../../TensorSharp.Server.Host/Hosting/ServerTelemetry.cs), the ASP.NET Core
  instrumentation's `Filter`, leaves out `/health` (any case) and every path in `RequestLoggingOptions.LowNoisePaths`
  (today `/api/queue/status`, which the Web UI polls every second). `/health` is not added to `LowNoisePaths`
  itself: that set also demotes the request's console and file log lines to Debug.
- The trace filter does not reach ASP.NET Core's own `http.server.request.duration` metric, so both routes are mapped
  with `.DisableHttpMetrics()`.
- The host has no authentication middleware, so some routes are bearer capabilities: an unguessable segment of the
  path is the only thing that guards what they serve, and anyone who reads the path can fetch the content.
  [`CapabilityPaths`](../../TensorSharp.Server/Logging/CapabilityPaths.cs) names three:
  - `/uploads/<file>`: uploads and generated media, named by a random id and served by `UseStaticFiles`;
  - `/api/code/artifacts/<run>[/<file>]`: the files a code-execution run produced (the run listing returns every
    file's download URL);
  - `/v1/responses/<id>[/...]`: a stored Responses API response, returned in full to any caller who has its id.

  With content capture off, `ServerTelemetry.Enrich` (`EnrichWithHttpRequest`) sets `url.path` to the placeholder
  and the span name to `<METHOD> <placeholder>`: `/uploads/{file}`, `/api/code/artifacts/{run}` or
  `/api/code/artifacts/{run}/{path}`, and `/v1/responses/{id}`, keeping a fixed sub-route
  (`/v1/responses/{id}/input_items`). Such requests also group as one operation instead of one per id. Matching is by
  segment, ignoring case (`UploadStaticFiles.IsUploadFilePath` does it as the static-file middleware does);
  `/v1/responses` itself is not a capability path. The request log lines are handled in §7.6, the Responses id on
  the server span in §5.6, and in the Responses log lines in §7.4.

### 5.11 Optional step spans

With `TENSORSHARP_OTEL_TRACE_STEPS=1` and a listener, `InferenceEngine.TraceStep` emits one `inference.step` span per
step that ran:

- after `ApplyResults` and `NotifyReleasedSequences`, so the step's tokens go out first, and built from the step's
  measured times (start = now minus the step stopwatch, end = start plus the step's ticks);
- a root (`parentContext: default`, with `Activity.Current` already cleared) with up to 32 `ActivityLink`s to the
  requests' spans: the submitter sets `SequenceState.TelemetryContext` to the chat span's context before
  `SubmitRequest`, and it is never used as an implicit parent;
- attributes `tensorsharp.model.architecture`, `tensorsharp.engine.batch_size`, `tensorsharp.engine.step_kind`
  (`prefill`, `decode` or `mixed`, from `ScheduledSequenceWork.IsPrefill`) and `tensorsharp.engine.scheduled_tokens`;
- inside `try/catch` with a Debug log, so it cannot stop the worker.

The listener check comes first, because `TraceSteps` can read the environment.

---

## 6. Metrics

All histograms are created with `InstrumentAdvice.HistogramBucketBoundaries`. The `gen_ai.server.*` ones use the
boundaries the GenAI conventions recommend; the SDK's default buckets (0, 5, 10, 25, … 10000) would put every sub-5 s
latency in one bucket.

### 6.1 Per-request metrics

| Instrument | Type | Attributes |
|---|---|---|
| `gen_ai.server.request.duration` | histogram, s (0.01 … 81.92) | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `tensorsharp.backend`, `tensorsharp.protocol` (when known), `tensorsharp.origin` (internal calls), `tensorsharp.outcome`, `error.type` |
| `gen_ai.server.time_to_first_token` | histogram, s (0.001 … 10) | the same without outcome and `error.type` |
| `gen_ai.server.time_per_output_token` | histogram, s (0.01 … 2.5) | the same without outcome and `error.type` |
| `tensorsharp.tokens` | counter, `{token}` | base attributes + `tensorsharp.token.type` = `input` \| `output` |
| `tensorsharp.tokens.cache_read` | counter, `{token}` | base attributes; prompt tokens served from the prefix cache, a subset of input |

- `request.duration` is recorded on **every** exit, once, in `GenAiRequestTelemetry.Dispose`, from the start of the
  generation (before the prompt is rendered) to the end of the stream as the consumer read it.
- **TTFT and TPOT are recorded only for a completed request**, as the conventions define them for successful
  responses. TTFT runs from the start of the generation to the engine's `FirstTokenAt`, so it includes rendering,
  queueing and prefill. TPOT is `(FinishedAt − FirstTokenAt) / (output tokens − 1)` and needs at least two output
  tokens. A stopped turn (`wasCancelled`) is cancelled and gets neither; a stop sequence or a thinking budget the
  pipeline enforced is a normal finish.
- Token counters are recorded for every outcome: the tokens were processed either way.
- There is no third `gen_ai.token.type` value for cached tokens: it was never a well-known value, the newer
  conventions removed the attribute, and a cached share of input would be counted twice in a sum. Hence TensorSharp's
  own `tensorsharp.token.type` and a separate `cache_read` counter.
- DiffusionGemma records TTFT at the first preview with any text and no TPOT (a canvas is written all at once). Jev
  records neither. Embeddings record duration and input tokens.

### 6.2 Engine metrics

| Instrument | Type | Source |
|---|---|---|
| `tensorsharp.engine.requests.running`, `.waiting` | observable gauge, `{request}` | `RunningCount`, `WaitingCount` |
| `tensorsharp.engine.steps` | observable counter, `{step}` | `TotalStepsRun`, monotonic across rebuilds |
| `tensorsharp.engine.forward.time` | observable counter, s | `TotalForwardTime`, which includes `Schedule()` as well as the forward |
| `tensorsharp.kv_cache.blocks.used`, `.total` | observable gauge, `{block}` | `PoolStats`: used = total − free (blocks the prefix cache can evict count as free) |
| `tensorsharp.engine.step.duration` | histogram, s (0.001 … 5.12) | per step that ran, `tensorsharp.engine.step_kind` |
| `tensorsharp.engine.batch.size` | histogram, `{sequence}` (1 … 256) | `ScheduledWork.Count`, `tensorsharp.engine.step_kind` |
| `tensorsharp.speculative.tokens.drafted`, `.accepted` | counter, `{token}` | the `SpeculationStats` increment sites (below) |
| `tensorsharp.engine.step.prefill_tokens` | histogram, `{token}` (16 … 65536) | per step with prompt work: the `IsPrefill` work items' `NumScheduledTokens`, `tensorsharp.engine.step_kind` |
| `tensorsharp.engine.inter_token.duration` | histogram, s (0.005 … 30) | per output token after a sequence's first: the time since `SequenceState.LastTokenTimestamp`, a multi-token step's time split evenly, `tensorsharp.engine.step_kind` of the step that emitted it |
| `tensorsharp.prefix_cache.lookups` | counter, `{lookup}` | the scheduler's "Radix prompt reuse" admission log site, `tensorsharp.prefix_cache.outcome` and `.source` |
| `tensorsharp.engine.stage.duration` | histogram, s (0.0001 … 1.024) | the direct-CUDA Qwen 3.8 engine's sampled forwards (`TS_Q4E_PERF_SAMPLE`), `tensorsharp.engine.stage`, `.step_kind`, `.rows` |
| `tensorsharp.slots*`, `tensorsharp.cuda_graph.*`, `tensorsharp.engine.forward.issue_time`, `.wait_time` | observable gauges and counters | `IServingTelemetrySource.ReadServingTelemetry()` of a registered engine's model |

All carry `tensorsharp.model.architecture`.

- **Self-registration.** `InferenceEngine`'s constructor calls `TensorSharpMetrics.RegisterEngine`, `Dispose` calls
  `UnregisterEngine`, and the observable callbacks read only registered engines. They never call
  `InferenceEngineHost.TryGetEngine()`, which builds an engine (allocating its KV block pool and starting its worker)
  when none exists, rebuilds one whose model changed, and does both under a lock that a reset can hold for tens of
  seconds. The CLI's private engine is covered the same way, with no registry to keep.
- **Monotonic totals.** An engine's steps and forward time are per instance and start from zero after a rebuild. On
  `UnregisterEngine` they are folded into retired totals per architecture, and the counters report retired plus live,
  so a model swap never looks like a counter reset. `Dispose` unregisters the engine only after its worker thread has
  stopped, so a step still running when the engine is disposed is counted in the retired totals of
  `tensorsharp.engine.steps` and `tensorsharp.engine.forward.time`, as it is in the step histograms.
- **Step histograms** are recorded in `InferenceEngine.WorkerLoop` from the same `stepTicks` that feed
  `TotalForwardTime`, on exactly the steps `TotalStepsRun` counts (failed and empty steps are not). The tags are cached
  `KeyValuePair` fields, so a step allocates nothing, and the recording returns early unless one of the two
  instruments is enabled. Every waiting-queue admission is marked `IsPrefill`, including the one-token forward after
  a full prefix-cache hit, so those steps are `prefill` or `mixed`.
- **Serving signals.** The prefill-token histogram is recorded with the step histograms and returns early the same
  way. The inter-token histogram is recorded after the step from its results, before they are applied, so a sequence
  that finishes in the step still gets its last gap. It has no stall counter for decodes a step leaves out: the
  scheduler gives every decoding sequence its token before any prompt work, so under prompt traffic the cost shows as
  `mixed` inter-token values, not as skipped decodes. The prefix outcome maps the coordinator's `LastSource`
  and decline reason to fixed values (`PrefixSourceCode`, `PrefixDeclineCode`), never the reason's text, which
  holds token counts.
- **Slot-keeping models.** A model that keeps per-sequence slots in its own engine implements
  `IServingTelemetrySource`; the observable callbacks read it for every registered engine on the collection thread.
  The read is lock-free: the engine's counters are `Interlocked` fields, and the model reads its slot tables'
  `Count` without its lock, which a forward holds for the whole step (a `TryEnter` would never see a running slot
  under load). A read that throws drops that engine from that collection.
  These are per engine and start again after a reload, unlike the step totals.
- **Speculation.** The counters are added at the two places that add to `SpeculationStats`
  (`SpeculativeExecution.DecodeStep` and `CommitBatch`), with a tag cached per execution. That covers the CLI's
  standalone speculative decoder too, and work done before a preemption re-arms a new execution is not lost.

### 6.3 GPU memory

| Instrument | ggml (CUDA, Vulkan, Metal) | direct CUDA | MLX |
|---|---|---|---|
| `tensorsharp.gpu.memory.used` (By) | total − free, device-wide | total − free (`CudaAllocator.GetMemoryInfo`, `cuMemGetInfo`) | `ActiveBytes` |
| `tensorsharp.gpu.memory.total` (By) | total | total | not reported |
| `tensorsharp.gpu.memory.peak` (By) | highest snapshot seen | highest snapshot seen | MLX's own `PeakBytes` (since device init) |

Attributes `tensorsharp.backend` and `tensorsharp.gpu.device`. `GpuMemoryTelemetry.EnsureRegistered` registers one
source per backend with `TensorSharpMetrics.RegisterGpuMemorySource`; `Track(model)` holds a model weakly; and
**no native query ever runs on the metrics collection thread**. The gauges report the latest `Snapshot()`, which runs
on a thread already doing that backend's work, at most every 250 ms, never blocks (a snapshot already running wins)
and never throws. A model stops counting once its unload begins (`ModelBase.IsRetiring`), and a backend stops
reporting once no tracked model on it is alive.

**Which query** is decided by the type of the model's allocator (`ModelBase.DirectAllocatorForTelemetry`), not by the
backend that was requested: LTX, for one, runs a `ggml_cuda` request on direct CUDA. A model on host memory (`cpu`,
`ggml_cpu`) is not tracked. Each backend has its own reason for being asked where it is:

- **ggml — the safety rule.** `TSGgml_DeviceMemoryInfo` (TensorSharp-owned native code over the public ggml API; no
  ggml change) starts with `ensure_backend()`. In a process that has not brought up a GGML backend, that latches
  Metal as the process's backend type and tries to initialize it, so a later `ggml_cuda` load would be refused. And
  `TSGgml_RecreateBackend` / shutdown free the backend under no lock a reader takes. So ggml is asked only from
  `Snapshot()`, only while a tracked model's allocator is a `GgmlAllocator` on a GGML GPU backend (that model's
  context brought the backend up), never through a model being unloaded, and only from a thread the host already
  serializes with teardown (the engine's worker is stopped before an unload or a `RecreateBackend`). One query covers
  every model on the process's one GGML backend. It reports the calling thread's active rank, rank 0 between engine
  steps: other ranks under tensor parallelism or layer split are not reported.
- **Direct CUDA.** `cuMemGetInfo` makes the device's context current on the calling thread (the engine's worker
  already uses it for a direct CUDA model), and an allocator disposed under a collection would hand the driver a
  released context. The pool counters (`CudaAllocatorStats.CachedBytes`) are safe anywhere but count only idle pooled
  blocks, near zero while a denoiser holds most of the card, so they are not used. Each allocator's own device is
  reported.
- **MLX.** Every query is an `MlxWorker.Shared.Invoke`, a FIFO queue behind the running MLX operation; on MLX a whole
  clip runs as one work item, so a call from another thread would wait minutes, and from the engine's worker it would
  stall a step. On the MLX worker `Invoke` runs inline and the figure is read at once (MLX video models report progress
  there). Anywhere else the read is queued with `MlxWorker.Dispatch`, which does not wait, one read at a time, and the
  gauges report it once the worker has run it.

**Only while something collects the gauges.** `Snapshot()` returns at once unless
`TensorSharpMetrics.GpuMemoryObserved` (the `tensorsharp.gpu.memory.used` instrument is enabled: a meter provider that
exports metrics, or a test listener, subscribes to it). With telemetry off, or on with metrics off, every video phase
and engine step returns there and no device query runs.

**When snapshots happen.** The video helper calls `Snapshot()` at every phase and stage boundary, on the thread that
reports progress (§8.5), and tracks the clip's model. For LLM serving, `InferenceEngine` calls
`TensorSharpMetrics.OnEngineStep()` after each step, under the GPU lock and after the step's tokens went out; it runs
the callback registered with `TensorSharpMetrics.SetEngineStepObserver` at most four times a second and swallows its
exceptions. Both hosts register `GpuMemoryTelemetry.Snapshot` there (and call `EnsureRegistered`) only when metrics are
exported (`Metrics != None`); without an observer `OnEngineStep` returns at once. The server's `ModelLifecycleService`
tracks every model it loads, the startup load and a rollback to the previous model included, and the CLI tracks the
model it loads, so LLM serving feeds the gauges on both hosts, not only video clips.

### 6.4 Other instruments

| Instrument | Type | Attributes |
|---|---|---|
| `tensorsharp.model.load.duration` | histogram, s | `gen_ai.request.model`, `tensorsharp.backend` (canonical, `unknown` for an unrecognised name, `ggml_cpu` when none was given), `tensorsharp.outcome`, `error.type` |
| `tensorsharp.tool.duration` | histogram, s | `gen_ai.tool.name` (host tools only, a fixed set), `tensorsharp.outcome`, `error.type` |
| `tensorsharp.image.generation.duration` | histogram, s | `gen_ai.request.model`, `tensorsharp.backend`, `tensorsharp.image.mode`, `tensorsharp.outcome`, `error.type` |
| video and Studio instruments | | §8.8, §9.5 |

### 6.5 Built-in .NET meters

`System.Runtime` (GC, thread pool, exceptions), `System.Net.Http`, and ASP.NET Core's meters through the
instrumentation's `AddAspNetCoreInstrumentation()` on the meter provider (server only).

### 6.6 Cardinality

Metric attributes are limited to: `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`,
`gen_ai.tool.name`, `tensorsharp.backend`, `tensorsharp.protocol`, `tensorsharp.origin`, `tensorsharp.outcome`,
`error.type`, `tensorsharp.token.type`, `tensorsharp.model.architecture`, `tensorsharp.engine.step_kind`,
`tensorsharp.video.mode`, `.audio`, `.phase`, `.stage`, `.cached`, `tensorsharp.image.mode`, `tensorsharp.owner`,
`tensorsharp.gpu.device` and `tensorsharp.studio.job_kind`. `error.type` is always a fixed code or an exception's full
type name, never a message. Video modes are normalized to a fixed set (§8.8). Session ids, request ids, prompts,
resolutions and frame counts go on spans only.

---

## 7. Logs and privacy

### 7.1 Bridge

An OpenTelemetry logger provider is added beside the console and file loggers.

- `IncludeScopes = false`: log scopes are not exported (§7.6). The request id and protocol are added under their own
  names by `LogContextEnrichmentProcessor` (§7.2); the model, backend and session are on the spans the record's trace
  and span ids lead to.
- `IncludeFormattedMessage = true`: records with no content keep their rendered message; records with content lose it
  (§7.4).
- Trace and span ids come from `Activity.Current` where the line is written: usually the server span. `model.load` is
  current during a load; the video helper makes the phase span current for its own line (§8.9); the Studio job span is
  current while a job runs. Lines written on the engine worker, the diffusion worker or the MLX worker carry none, and
  no `tensorsharp.request_id` either: those workers clear `TelemetryContext` (§5.4).
- Over OTLP the `EventId` name is exported as the record's event name; its numeric id only with
  `OTEL_DOTNET_EXPERIMENTAL_OTLP_EMIT_EVENT_LOG_ATTRIBUTES=true`. Azure Monitor gets `EventId` and `EventName`.
- The video request lines, which all used `UploadReceived`, have their own ids: `VideoGenerateStarted` (1640),
  `VideoGenerateCompleted` (1641), `VideoGenerateRejected` (1642), `VideoGeneratePhase` (1643).

### 7.2 Request ids in exported logs

The request logger's scope carries the TensorSharp id as `RequestId`, but ASP.NET Core's hosting scope uses the same
key for `HttpContext.TraceIdentifier` and comes first, so an exported scope would carry two `RequestId` values, and
Application Insights keeps the first. Scopes are not exported at all (§7.6), and a processor cannot rewrite them, so
[`LogContextEnrichmentProcessor`](../../TensorSharp.Telemetry/LogContextEnrichmentProcessor.cs) adds
`tensorsharp.request_id` and `tensorsharp.protocol` from `TelemetryContext` to every record written during a request
(log processors run on the thread that wrote the record, in its async flow). That is the attribute that matches the
`X-TensorSharp-Request-Id` header. The long-lived workers clear `TelemetryContext` when they start (§5.4), so their
lines carry no request id rather than the id of whichever request created the worker.

The engine's own sequence id (`chat-<guid>`) is a different id space. Its log placeholders in `BatchExecutor`,
`InferenceEngine`, `ContinuousBatchScheduler`, `PrefixCacheCoordinator` and `ChatGenerationPipeline` are renamed from
`{RequestId}` to `{EngineRequestId}` (the rendered text is unchanged), and the chat span carries it as
`tensorsharp.engine_request_id`, so engine-thread lines, which carry no trace id and no `tensorsharp.request_id`, can
be joined to their request.

### 7.3 Export level and the ASP.NET Core filter rule

`LoggingSetup` quiets ASP.NET Core with a filter that names no provider (`Microsoft.AspNetCore` → Warning).
`Microsoft.Extensions.Logging` prefers a rule for a specific provider whose category matches over every rule with no
provider, whatever the category length. A `Logging:OpenTelemetry:LogLevel:Default` rule matches every category, so it
would switch that filter off for the OpenTelemetry provider. `AddLoggingFilters` therefore registers
`AddFilter<OpenTelemetryLoggerProvider>("Microsoft.AspNetCore", LogLevel.Warning)`, and `TENSORSHARP_OTEL_LOG_LEVEL`
as a provider rule with no category (so ASP.NET Core stays at Warning under it). The server also honours the standard
`Logging__OpenTelemetry__LogLevel__*` configuration.

### 7.4 Content: marked at the call site

Many log lines carry prompts, model output, upload names, paths and URLs, tool arguments and commands. In a local
file the operator owns that is fine; sending it to a collector or Application Insights is a different decision, and the
Azure Monitor exporter silently truncates long values anyway (a message at 32,768 characters, an attribute at 8,192;
the captured chat messages of §7.7 have limits of their own).

**Why not a list of key names.** Names do not say what a value is. `{Prompt}` is a prompt in one line and a token
count in another (`BatchExecutor`, `ContinuousBatchScheduler`, `CliInferenceSession`); `{Path}` is an HTTP route in the
request logger and an uploaded file elsewhere; `Source`, `Error`, `Reason` and `Tokens` mean different things in
different lines; and content also sits under names no scan would flag (`What`, `Url`, `User`). Whether a value is
content depends on the argument expression, not the placeholder.

**Mechanism.** Every content argument is wrapped at the call site: `LogContent.Of(text)`. `ToString()` returns the
text, so the console formatter and the JSON-line file logger write exactly what they wrote before. OpenTelemetry passes
the original argument objects into `LogRecord.Attributes`, so
[`ContentRedactionProcessor`](../../TensorSharp.Telemetry/ContentRedactionProcessor.cs) replaces every attribute whose
value is a `LogContent`, whatever its key, with `[redacted: N chars]`, and sets `FormattedMessage = null`, so exporters
send the message template (`{OriginalFormat}` is kept) instead of the rendered line. The text of each marked value of
4 characters or more is also removed, replaced by `[redacted]`, from the record's other string attributes and from its
exception's message (§7.5). A CLI line that marks a PDF path, for example, also attaches the exception, whose message
quotes the same path; neither leaves the process.

**Order.** Log processors run in the order added. The pipeline is: `LogContextEnrichmentProcessor`, then
`ContentRedactionProcessor`, then the exporters (OTLP, console, Azure Monitor, each a batch processor that copies the
record as it sees it). The redaction processor is left out when content capture is on.

**Call sites marked** (printed output unchanged everywhere):

| Area | Arguments |
|---|---|
| `InferenceTelemetry` | `chat.start`: `LastUserContent`, `FullInput`, `Uploads`, `StopSequences` (the client's own stop strings; `(none)` is not marked); `chat.cancelled`/`chat.complete`: `AssistantContent`; `generate.*`: `Prompt`, `Uploads`, `StopSequences`, `Completion` (these have no callers today: `/api/generate` logs `chat.*`) |
| Protocol adapters | `LastUser` (OpenAI chat, Responses, Ollama chat); the `ResponseId` of the Responses request, "/v1/responses skills" and "/v1/responses lora" lines when the response is stored (the id retrieves it, §5.6); the `Key` of `RedisConnection`'s "Redis GET failed for key" warning (a stored response's key holds its id); `/api/generate` `Prompt`; `WebUiAdapter` "OpenAI video generation done" `Url` |
| `WebUiChatService` | upload rejections and "Upload received" (`FileName`, `SavedFile`, `SavedPath`); video frame, PDF and HEIC failures (name, path, error); the PDF info lines; "Could not decode `{Source}`" (source, error); the four `{Prompt}` lines; the four media "done" `{Url}` lines; two CSV read errors (I/O messages that quote the upload path) |
| `RequestLoggingMiddleware` | `{Path}` of a capability path: `/uploads/<file>`, `/api/code/artifacts/<run>[/<file>]`, `/v1/responses/<id>[/...]` (§5.10, §7.6) |
| Engine, skills | `InferenceEngine` repetition stop `What` (up to 48 characters of decoded output); `SkillChatLoop` repetition `What` (twice), unknown tool `Tool`, `skills.tool` `Path`, `skills.completion.early`/`corrected` `Artifact` (the artifact's file name), `skills.completion.failed` `Reason`; `SkillsService` upload rejection `FileName` |
| AgentHost | `codeexec.read`/`edited`/`rewrote` `Path`; `codeexec.artifacts` `RunId` (`ShellRunner` and `CodeArtifactStore`: the run id is the capability in the artifact URLs); `codeexec.artifact-evict-failed` `Dir` and `Reason`; `workspace.delete-failed` and `workspace.sweep-failed` `Reason` (I/O messages that quote the user's paths); `EgressProxy` `Target`; `PackageInstaller` `Hosts` |
| CLI | `cli.inference.start` input and uploads; `cli.inference.complete` preview; multi-turn user, thinking, content and raw previews; JSONL batch rendered prompt, `first20` token ids and output; "Tokenized prompt" `first20`/`last10` (token ids decode back to the prompt); `cli.skills.tool` unknown name and resource path; `interactive.skills.tool` path; the input, PDF, image, video, audio, multi-turn and batch file paths |

The `{Prompt}` placeholders that are token counts are renamed `{PromptTokens}` anyway, for readability.
`codeexec.ran` logs only counts and `skills.script.ran` only a byte count, so they need nothing.

### 7.5 Exceptions

Exported exceptions keep their type, message and stack unless the message, or any inner message, mentions a path
under a directory registered with `TensorSharpTelemetry.RegisterSensitivePath` (the server registers the upload
directory and, with `--code-exec`, the code-execution scratch and artifact directories, which hold the user's files
under their own names), an `/uploads/<name>` URL, an upload storage name, or a value the record marked as content
(§7.4). Then the processor replaces `LogRecord.Exception` with a `ScrubbedException`. Its message is the original
one with the inner messages appended and the marked values removed, cleaned by
`TensorSharpTelemetry.ScrubSensitivePaths`:

- a path that starts with a registered directory is replaced whole by `<uploads>`, through the end of the path: to
  the closing quote when the path is quoted, otherwise to the next whitespace or punctuation that ends it. The
  directory alone is not enough: the names below it are what matters (an upload's unguessable name, a code-execution
  artifact's run id, a workspace file's own name);
- an `/uploads/<name>` URL becomes `/uploads/{file}`;
- the 32 hex digits of an upload storage name followed by a file extension (`<32 hex>.mp4`, also inside a name such
  as `video-<32 hex>.mp4`) become `{id}`, anywhere in the message. A 32-digit id with no extension, such as a trace
  id, is left alone.

The `ScrubbedException`'s message starts with the original exception's full type name, for example
`System.IO.IOException: Could not find file '<uploads>'.`, and its stack trace is the original's. Exporters report its
own type, `TensorSharp.Telemetry.ScrubbedException` (OTLP `exception.type`, Application Insights `exceptions.type`),
so the original type is read from the start of the message. An exception that needed no cleaning is exported
unchanged, with its own type. Media exceptions quote the file they failed on (`Failed to open video file: <path>`).
The source check found no thrown exception in Chat, Server, Runtime, AgentHost or Studio that interpolates prompt or
output text; the context-overflow exceptions carry only token counts. Spans never carry exception messages, only
types; Studio keeps messages out of its own diagnostics (`FailureDiagnostic`).

### 7.6 Scopes and request paths

Log scopes are not exported: `TensorSharpTelemetrySetup.ConfigureLoggerOptions` sets `IncludeScopes = false` for both
hosts. A processor cannot rewrite or drop one scope item (scopes are read-only through `ForEachScope`), and what the
scopes hold either must not leave the process or is wrong for the record:

- ASP.NET Core's hosting scope carries the raw `RequestPath`, which on a capability path (§5.10) is an `/uploads` file
  name, a code-artifact run id or a stored response id, and its own `RequestId` under the key TensorSharp's request
  scope uses (§7.2).
- The long-lived workers (the inference engine, the diffusion scheduler, the MLX worker) inherit the scope of whichever
  request created them (§5.4), so every line they write would carry that request's ids.

What an exported record loses is available elsewhere: `tensorsharp.request_id` and `tensorsharp.protocol` come from
`LogContextEnrichmentProcessor` (§7.2), and every record written inside a request carries its trace and span ids, which
lead to the spans that hold the model, the backend and the session. Console and file logs keep their scopes and are
unchanged.

The request path in the message itself is handled at the source: the three `HTTP {Method} {Path} …` lines pass
`LogContent.Of(path)` when `CapabilityPaths.IsCapabilityPath` matches, so the exporter redacts it with capture off;
other paths stay a plain string. The `Operation` scope keeps the real path, locally only.

### 7.7 Content capture

`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is resolved once by `TelemetryOptions` and pinned with
`TensorSharpTelemetry.ConfigureCaptureMessageContent`, which every library reads (Studio reads the variable itself).
With it on:

- the redaction processor is left out, so log content is exported;
- the chat span gets `gen_ai.input.messages` (the history the prompt was rendered from, after compaction) and
  `gen_ai.output.messages`, as JSON strings `[{"role":…,"content":…}]` (the JSON-string form the v1.41.0 conventions
  allow on spans), cut to 64K characters per attribute: a history keeps its newest messages, an answer its beginning.
  The JSON is built only when the span is recorded, and a failure returns nothing rather than throw. The 64K counts
  message text, before the JSON adds quotes and escapes. The Azure Monitor exporter lets these two attributes through
  up to 262,144 characters (exporter 1.10.0's `GenAi_Properties_MaxValueLength`, not the 8,192 of other attributes),
  but Application Insights kept them whole only up to about 64,700 characters of JSON and stored a longer value as its
  first 8,192 (measured in November 2026: 64,701 characters kept whole, 65,007 cut). A history that reaches the 64K
  cap therefore arrives there as the start of the oldest message it kept, while an OTLP collector receives all of it;
- `generate_video` gets `tensorsharp.video.prompt`, `execute_tool` gets `tensorsharp.tool.arguments` and
  `tensorsharp.tool.result`, `codeexec.run` gets `tensorsharp.codeexec.command`.

With it off, [`ContentSpanProcessor`](../../TensorSharp.Telemetry/ContentRedactionProcessor.cs), registered before the
trace exporters, removes `gen_ai.input.messages`, `gen_ai.output.messages` and `tensorsharp.video.prompt` from any span
as a backstop for a call site that forgot to check. Jev, images and Studio record no content even with capture on.

Other places content can travel: capability paths and the Responses API's stored response ids (masked, §5.6, §5.10,
§7.4, §7.6); the Azure Monitor exporter's offline storage,
which with capture on can hold prompts (§10.5); and Studio's `job.json`, which stays local (§9.6).

### 7.8 Console output is not exported

Only `ILogger` output reaches the OpenTelemetry pipeline. These go to `Console.WriteLine` and stay there: MiniMax-H3's
`[h3]` timings (always the text-conditioning, denoiser-load and first-step times; the per-stage split with
`TS_H3_PHASE=1`; latent statistics with `TS_H3_TRACE=1`); LTX's `[ltx25]`, `[ltx25-timing]` and `[ltx25-conditioning]`;
Wan's `[wan-timing]`; Qwen-Image's `[qwen21-timing]`/`[qwen21-step]`; `TS_SCHED_STEP_LOG` step lines; the MP4 export
fallbacks and FFmpeg failures; and the once-per-architecture template, parser and grammar warnings. Video spans and
stage attributes replace the video timings as the exported record. The warnings are not converted (§13).

### 7.9 Changes to local logs

The console and file logs keep their format, levels and content, except:

- one new `Telemetry: …` startup line on both hosts (also when off), plus a warning line per resolution warning;
- placeholder renames `{Prompt}` → `{PromptTokens}` (four lines) and `{RequestId}` → `{EngineRequestId}` (engine
  lines): the rendered text is unchanged, but the JSON-line file log's `template` and `props` key change;
- the video request lines' event ids (1640–1643 instead of `UploadReceived`);
- the `Client` scope of `/api/embed` requests is now `ollama` (it was `webui`);
- the per-phase video line (§8.9) is written only when logs are exported, so it never appears with telemetry off.

---

## 8. Video generation: MiniMax-H3 and LTX 2.5

### 8.1 How a clip runs

- **One callback type.** Every video model reports through `VideoGenerationProgress`. MiniMax-H3 reports the phases
  `text-encode`, `denoise`, `vae-decode`, `audio-decode` (with audio) and `done`; it never reports an `image-encode`
  phase. LTX reports `reference-encode` (Ingredients `ref` mode), `image-encode` (only with image guides),
  `text-encode`, `denoise`, `vae-decode`, `audio-decode` and `done`. Both call `OnProgress` only from the thread running
  the pipeline: the caller's thread, or the MLX worker on MLX. Heartbeats (`Heartbeat=true`) come only from Wan's
  timer thread.
- **One clip at a time.** `RunVideoGeneration` takes the static `_videoGenLock` (a Monitor wait, which cannot be
  cancelled), checks the model is still the one requested, enters a model use, takes `GpuComputeLock`, applies LoRA
  plug-ins, checks the sampling recipe, then generates and saves.
- **Callers.** The plain path (`GenerateVideoCoreAsync`, which `/api/video-generate`, `/v1/videos/generations` and
  Studio's `GenerateVideoAsync` reach), the streaming path (`VideoGenerateStreamAsync`) and the CLI. All three now call
  `VideoGenerationTelemetry.Generate` instead of `GenerateVideo`.
- **Cancellation differs by model.** LTX checks the token in its `Report` and in every sampling step. MiniMax-H3 never
  reads it: it stops only when the caller's `OnStep`/`OnProgress` throws, which only the streaming path does. The plain
  path does not connect the token, and Studio's callbacks never throw on cancel, so an H3 Studio job that is cancelled
  runs to the end and is seen as cancelled only afterwards.
- **Export.** A host that cannot write an MP4 is refused before generation: 503 `video_export_unavailable` on the
  plain path and `GenerateVideoAsync`; on the streaming route a `{done, error}` frame on HTTP 200.
- **LoRA.** The HTTP routes pass no LoRA set (they use the model's load-time plug-ins); only Studio passes one. LTX is
  not an `ILoraPluginHost`.

### 8.2 The progress contract: `Stage` and `Attributes`

[`VideoGenerationProgress`](../../TensorSharp.Models/Models/Video/VideoGenerationProgress.cs) gained two additive
properties:

- `Stage`: a sub-stage within the phase. A model repeats it on every report made inside the stage; a report without one
  ends it. A stage never changes `Phase`, which is what hosts follow; TensorAgent's chat end-to-end script rejects
  phases outside a fixed list and phases that go backwards, which is why new phase names were not an option.
- `Attributes`: facts about the stage, keyed by full OpenTelemetry attribute name (`tensorsharp.video.*`), never user
  content. Each dictionary is built only when `OnProgress` is set.

The wire formats are unchanged: the SSE stream and TensorAgent copy named fields into `VideoFrame`, and Studio maps
reports to `StudioProgress` field by field. Only `eng/validation/Ltx25LatentProbe`, which writes progress objects
straight into its JSON report, now shows the two new members.

Model-specific timings are reported as stages and attributes rather than spans because the model code has no parent
to give a span: on MLX the pipeline runs on the MLX worker, where `Activity.Current` belongs to nobody. The helper
creates every span.

### 8.3 MiniMax-H3 span tree

```
POST /api/video-generate/stream        server span (minutes long)
└─ generate_video <model>              mode, requested then effective width/height/frames/fps/steps/seed, audio, lora_count, backend
   ├─ video.queue                      wait for _videoGenLock (zero-length when nobody waited)
   ├─ lora.apply                       only when the request chose plug-ins (Studio jobs): tensorsharp.lora.count
   ├─ text-encode
   │  ├─ text-encoder                  tensorsharp.video.open_s, .run_s, .teardown_s
   │  ├─ prefault-wait                 only in TS_H3_PREFAULT modes 1-2 (the default is 3)
   │  ├─ dit-load                      tensorsharp.video.cached
   │  ├─ image-encode                  i2v, first/last frame
   │  └─ reference-encode              Ref2VA
   ├─ denoise                          one "step" event per finished step
   ├─ vae-decode
   │  ├─ denoiser-release              tensorsharp.video.release_reason = metal | vram; only when something is released
   │  ├─ vae-open                      tensorsharp.video.cached
   │  └─ vae-decode                    tensorsharp.video.tiles; Detail "tile i/N"
   ├─ audio-decode
   └─ video.export                     tensorsharp.video.codec, .audio_muxed, .bytes
```

H3 sends nothing between its `text-encode` and `denoise` reports, so before stages existed the reference loading, the
prefault wait, the DiT build and the keyframe encodes all counted as text encoding, and the first clip's `text-encode`
included the DiT load while later clips' did not. The stages now separate them. What they mean:

- `dit-load cached=true` means the DiT object was reused (the partition did not change), not that its weights are still
  on the device. Creating the DiT only maps the file; weights upload inside the first denoising forward, and again after
  any `ReleaseDeviceResidency` (before every text encode on Metal; after denoising on CUDA when the VAE would not fit).
  That cost shows in the first denoise step.
- `prefault-wait` exists only in modes 1–2. In the default mode 3 the read-ahead overlaps the first step's upload and is
  joined after the loop, inside `denoise`.
- `teardown_s` is the remainder `TS_H3_PHASE` prints, so it includes the vision-tower encodes and tokenization that run
  between the open and the trunk.
- `denoiser-release` happens on Metal always and on CUDA only when the spare VRAM is below the VAE estimate. It drops
  residency only.
- `vae-open cached=true` (kept from an earlier clip or created for a keyframe encode) is close to free; the VAE's
  weights upload inside the decode either way.

### 8.4 LTX 2.5 span tree

```
└─ generate_video <model>
   ├─ video.queue
   ├─ reference-encode                 ref mode: identity sheet and reference videos
   ├─ image-encode                     only with image guides
   ├─ text-encode                      tensorsharp.video.prompt_cache_hit (a repeated prompt is almost instant)
   ├─ denoise
   │  ├─ first-pass                    half resolution; step events     (single-pass in ref mode: no upscale or refinement)
   │  ├─ upscale                       x2 latent upscale
   │  ├─ refine-image-encode           only with image guides
   │  └─ refinement                    full resolution; step events
   ├─ vae-decode
   ├─ audio-decode
   └─ video.export
```

The first pass, the upscale, the refinement image encode and the refinement all report as phase `denoise`, told apart
only by `Detail`; the `Stage` values now name them. The prompt-cache flag comes from `Ltx25Model.HasCachedConditioning`,
a lookup made before `Conditioning` runs. LTX has no `lora.apply`.

### 8.5 The helper and its threading rules

[`VideoGenerationTelemetry.Generate(model, prompt, parameters, clip, logger)`](../../TensorSharp.Models/Models/Video/VideoGenerationTelemetry.cs)
wraps `parameters.OnProgress` for the duration of the call and restores it afterwards:

1. **The caller's callback runs first, and its exceptions propagate unchanged.** They are how a host stops an H3 clip
   and how Studio refuses on a full disk. The telemetry after it is inside `try/catch` and never throws.
2. **Heartbeats are checked before any shared state** and go to the caller only (Wan's arrive concurrently from a timer
   thread).
3. **The parent is captured once**, on the calling thread: the clip's context. Spans are never parented through
   `Activity.Current`. With no recorded clip span there are no spans at all (metrics and the log line only), so nothing
   starts a stray trace.
4. **`Activity.Current` is saved and restored around every span start and stop.** `StartActivity` would make the span
   current on the reporting thread, the shared MLX worker on MLX, and `Stop()` restores whatever was current where the
   span started, which may be another thread's.
5. **Each phase change** ends the open stage and phase spans, records the finished phase's duration, opens the new
   phase span (`done` only closes the last one), takes a GPU snapshot and writes the phase log line. **Each stage
   change** opens a stage span under the phase span, counts `tensorsharp.video.dit.loads` for `dit-load`, and takes a
   snapshot. A report's `Attributes` are copied onto the current stage span, or the phase span.
6. **Steps.** The first report of `denoise` sets the baseline; a later rise in `Step` is a finished step, timed from the
   previous step or from the start of its stage, whichever is later, so LTX's upscale and refinement image encode are
   not counted in step 9. It becomes a `step` span event (`tensorsharp.video.step`, `.steps`,
   `tensorsharp.video.step_seconds`) and a `tensorsharp.video.denoise.step.duration` sample.
7. **On an exception** the open phase and stage spans get Error status and `error.type`, or `outcome=cancelled`; the
   clip is marked failed or cancelled. **Finally** every open span is ended (LTX can throw from its own `Report` before
   the callback sees that report), and a phase cut short is left out of the phase-duration histogram.
8. After the model returns, the clip span's requested size, length, rate, seed and steps are replaced with the values
   the model actually used (`GeneratedVideo` and the last report's `TotalSteps`): the pipelines pick a random seed, a
   keyframe's canvas or a plug-in's step count themselves.

`VideoClipTelemetry` holds the `generate_video` span and records `tensorsharp.video.generation.duration` once, in
`Dispose`, with the outcome the host marked. `StartChild` and `AddChild` (from UTC timestamps) never start a root.

### 8.6 The clip's lifetime on the server

In [`WebUiChatService`](../../TensorSharp.Chat/WebUiChatService.cs):

- **The generation clip** starts in `GenerateVideoCoreAsync` (an async method, so the span is current there and in the
  `Task.Run` worker, never in the caller's flow) and is disposed in its `finally`. On the streaming route it starts
  just before `Task.Run`, is held in a local, and is disposed by the worker's `finally`: a consumer that stops reading
  disposes the iterator without waiting for the clip. The clip is a child of whatever is current: the HTTP request, or
  the Studio job span (§9.1). Model name: the loaded file name, as for chat spans.
- **Refusals before the queue are recorded where they happen.** `RecordVideoRefusal(code, parameters)` starts a clip,
  marks it refused and disposes it on the calling thread: a zero-length `generate_video` span and a duration sample
  with `tensorsharp.outcome=refused`. The clip carries the request's mode, which is known only once its parameters are
  read, so the generation clip is not opened at the top of the route. Codes: `not_a_video_model`, `model_changed`
  (`GenerateVideoAsync`'s expected model), `invalid_request` (parse and conditioning errors), `video_export_unavailable`,
  `upload_quota_exceeded`.
- **Inside `RunVideoGeneration`** the generation clip is marked at each exit: a `video.queue` child built after the lock
  is held from timestamps taken around it; cancelled while queued; refused `model_changed` or `model_unloading` (three
  places); refused `sampling_recipe_mismatch` (`CheckSamplingRecipe`); anything `generate()` throws (the model, the
  export, the last frame) marked failed, a cancellation as cancelled. The callers' catch blocks do not mark again, so a
  refusal is never overwritten as a failure.
- `ApplyVideoLorasTraced` adds `lora.apply` only when the request chose plug-ins for an `ILoraPluginHost`, and sets
  `tensorsharp.video.lora_count`. `SaveVideoMp4` adds `video.export` with the codec, whether the soundtrack was muxed
  and the file size.
- The queue gauge counts up just before `lock (_videoGenLock)` and down as the first statement inside it; a request
  cancelled while it waits stays counted until it gets the lock.

### 8.7 CLI

`Program.RunVideoGeneration` starts a `VideoClipTelemetry` (a root: nothing is current in the CLI) named after the
model file (`ModelNameFromPath`), as the server names its clips, calls the helper, and wraps saving in
`CliVideoExport`, a `video.export` child that covers both the MP4 and the `.png` frame output (which never goes
through `SaveMp4`; codec `png`, bytes summed over the frame files). `SaveGeneratedVideo` returns the codec. A save
that throws marks the clip failed, as the server records a clip it could not save, and propagates to `RunToExit`;
the export span, disposed without `Saved`, records Error with `error.type=_OTHER`. The helper's logger is passed only
when logs are exported.

### 8.8 Metrics

All video metrics carry `gen_ai.request.model` and, when known, `tensorsharp.backend`.

| Instrument | Type | Further attributes |
|---|---|---|
| `tensorsharp.video.generation.duration` | histogram, s (0.5 … 7200) | `tensorsharp.video.mode` (normalized to `t2v`, `i2v`, `fl2v`, `ref`, `keyframes`, `other`; inferred from the inputs when absent), `tensorsharp.video.audio`, `tensorsharp.outcome` (`completed`, `cancelled`, `refused`, `failed`), `error.type` |
| `tensorsharp.video.phase.duration` | histogram, s | `tensorsharp.video.phase`. Phases that finished only; stage durations are on spans only, so per-phase totals are not counted twice. |
| `tensorsharp.video.denoise.step.duration` | histogram, s | `tensorsharp.video.stage` when there is one |
| `tensorsharp.video.dit.loads` | counter | `tensorsharp.video.cached` (MiniMax-H3 only) |
| `tensorsharp.media.generations.active` | observable gauge | `tensorsharp.owner=webui`. Reads `_mediaJobs`: every picture, clip and image-intent question from the moment it is accepted, so it counts waiting as well as running. |
| `tensorsharp.video.queue.waiting` | observable gauge | `tensorsharp.owner=webui`. The static counter around `_videoGenLock`; it does not cover the CLI or Studio jobs still in Studio's own queue. |
| GPU memory | §6.3 | snapshot at every phase and stage boundary, while the gauges are collected |

Refused clips are part of `generation.duration` with near-zero durations; filter on outcome.

### 8.9 Long clips

A span is exported when it ends, so a 10-minute clip is invisible in a trace backend until it finishes, and lost if the
process dies. To follow one while it runs:

- the `tensorsharp.media.generations.active` gauge;
- one `Information` line per phase change, `video.phase {Phase} model={Model} step={Step}/{TotalSteps}`
  (`VideoGeneratePhase`), on the caller's logger (`TensorSharp.Server.VideoGenerate` on the server, the CLI's logger),
  because TensorSharp.Models has none of its own. That is four or five lines per H3 clip and four to seven per LTX clip;
  stages and repeated reports of the same phase are not changes. The lines are written only when logs are exported.
  Logs are exported in batches (`OTEL_BLRP_*`) as they are written, not when the clip ends; Application Insights adds
  its own ingestion delay. The line carries the phase span's trace when the model reports on the thread that called
  `Generate`, which then runs only this clip; on MLX it carries none.

### 8.10 Privacy

- The prompt goes on `generate_video` only with content capture on. `Detail` is the model's own status text ("tile
  3/7", "first pass at 512x512"), never the prompt or a file name. Failures record the exception type only.
- The Web UI image and video routes log the prompt at `Information`, marked as content. `/v1/videos/generations` and
  Studio jobs do not log it.
- Result URLs are content: the "done" lines mark `{Url}`, and spans carry no result URL.

---

## 9. Video Studio jobs

### 9.1 A trace per job, with a link

Jobs run on one worker started at startup, long after the request that queued them returned, so the request's context
never reaches them.

- `Enqueue` (the only path for both render and generate jobs) calls `StudioTelemetry.CaptureQueuingContext`, which
  stores `Activity.Current.Id` and its trace state on `StudioJobRecord.TraceParent` / `TraceState` when the id format is
  W3C and something records Studio traces. They are public properties marked
  `[JsonIgnore(Condition = WhenWritingNull)]`, so they are persisted in `job.json` and, with telemetry off, the file is
  byte-for-byte unchanged (§9.6).
- `ProcessJobsAsync` calls `StudioTelemetry.StartJob` before its `try`: it clears `Activity.Current`, then starts
  `studio.job <kind>` (`generate` or `render`; fixed names, so a record read from disk cannot add span names) as a root
  dated from `CreatedAt`, with a link to the stored context and `tensorsharp.studio.job_id`, `.job_kind` and
  `.project_id`. It stays current while the job runs, so the generation's `generate_video` tree (§8) nests under it:
  `Task.Run` in `GenerateVideoCoreAsync` carries the worker's context.

### 9.2 Stage spans, built after the fact

`EndJob` runs last in the worker's `finally`, after the work-directory cleanup. It sets the outcome (on failure: Error
status with the exception's short name, `error.type` with its full name, and `tensorsharp.studio.error_code` for a
`StudioException`, one of Studio's fixed codes), then builds children with the job span as explicit parent:

- `studio.queue` from `CreatedAt` to `StartedAt` (it fits inside the job span because the job span starts at
  `CreatedAt`);
- one `studio.stage <stage>` per entry in `job.Stages`, laid end to end from `StartedAt` (each stage starts where the
  previous ended), for **Studio's own stages only**: `starting`, `model-identity` (SHA-256 of the checkpoint and plug-in
  files, only the first time a file is hashed), `encoding` (from the model's `done` until the adapter returns),
  `importing`, `cancelling`, and for render jobs (FFmpeg subprocesses) `rendering` and `validating`.

The other stages of a generate job are the model's phases (the adapter maps `Phase` to the Studio stage), which the
generation's own spans already show. Stage spans are not created in `EnterStage`, which is static, sees only the DTO,
and runs on the generating thread, the worker or an HTTP thread (`cancelling`).

### 9.3 Jobs that never ran

`TraceJobNotRun` emits an instant `studio.job <kind>` span (one tick long, at `FinishedAt`) with the outcome and the
link, in a trace of its own (it saves and restores `Activity.Current`, so it is never a child of a cancel request or of
startup or shutdown), for:

- jobs recovered at startup: found `queued`, `running` or `cancelling` and marked `interrupted`, or a generate job whose
  take was already imported, adopted as `completed`;
- a job cancelled while queued (`CancelJobAsync`);
- jobs still queued at shutdown (`StopJobsAsync`, `interrupted`).

### 9.4 Startup and shutdown ordering

`InitializeJobs` runs in the `StudioService` constructor, which `Program.cs` resolves before `app.RunAsync`, and
`StopJobsAsync` runs from `StudioService.DisposeAsync` during `app.DisposeAsync`. Because the server resolves the
tracer and meter providers right after `builder.Build()` (§3.3), they are listening when startup recovery runs, and DI
disposes them after Studio, so the running job's `interrupted` span (marked by the worker at shutdown) and the queued
jobs' spans are exported.

### 9.5 Metrics

On the `TensorSharp.Studio` meter:

| Instrument | Type | Attributes |
|---|---|---|
| `tensorsharp.studio.job.duration` | histogram, s (0.5 … 14400) | `tensorsharp.studio.job_kind`, `tensorsharp.outcome` = the final state (`completed`, `failed`, `cancelled`, `interrupted`) |
| `tensorsharp.studio.jobs.queued`, `.running` | observable gauge, `{job}` | running includes `cancelling`, which still holds the worker |

The duration is recorded in `Finish`, which every way a job ends goes through, from `CreatedAt` to `FinishedAt`: a job
that ends while queued records its wait. The gauges are summed over the services tracked in a `ConditionalWeakTable`
(`Track` before the worker starts, `Untrack` at shutdown), counted under `jobSync` by `CountActiveJobs`.

### 9.6 Privacy and `job.json` compatibility

- Studio spans carry identifiers, kinds, stages and outcomes only: never the prompt or negative prompt
  (`StudioGenerationRequest`), `StudioJobSummary.Prompt`, take names (built from the prompt), the project snapshot or
  `StudioJob.Error` (an exception message; for render jobs up to 6,000 characters of FFmpeg stderr).
- `job.json` is read and written with `StudioJson.Options` (Web defaults, no `UnmappedMemberHandling`, only an id check),
  so an old file reads the new properties as null and an older host skips them in a new file. Only API payloads use the
  strict options.

---

## 10. Application Insights

### 10.1 Direct, one setting, per-signal exporters

Set `APPLICATIONINSIGHTS_CONNECTION_STRING`. `TensorSharp.Telemetry` adds the Azure Monitor exporter per signal:
`AddAzureMonitorTraceExporter`, `AddAzureMonitorMetricExporter` and `AddAzureMonitorLogExporter`, configured by
`ConfigureAzure` and given the Entra ID credential when `TENSORSHARP_AZURE_MONITOR_AAD=1`.

It uses neither a distro (`UseAzureMonitor()`, or the newer `UseMicrosoftOpenTelemetry()`) nor the exporter package's
own cross-cutting `UseAzureMonitorExporter()`. A distro adds its own ASP.NET Core, HttpClient and other instrumentation
and resource detectors, duplicating what TensorSharp registers. The cross-cutting helpers each own the whole pipeline:
`UseAzureMonitorExporter` cannot be combined with the per-signal calls, turns on formatted messages for every record,
and needs a DI host, which the CLI does not have. Per-signal exporters run side by side with the per-signal OTLP
exporters (§4.3) in both hosts.

### 10.2 Live Metrics is off

`EnableLiveMetrics` defaults to true, but the exporter's own documentation says it applies only to
`UseAzureMonitorExporter`; with the per-signal calls it only logs a warning at each registration. `ConfigureAzure`
sets it to false.

### 10.3 Sampling and log sampling

The trace sampler is mapped as described in §4.4. Two consequences:

- **Count scaling.** The exporter derives the envelope's sample rate from `SamplingRatio`, so fixed-percentage sampling
  scales `itemCount` and the portal's counts. Rate-limited sampling instead tags spans with `microsoft.sample_rate`;
  whether 1.10.0 maps that to the envelope is checked in §12.4.
- **Logs of unsampled traces.** `EnableTraceBasedLogsSampler` is left at its default (on): the Azure Monitor log exporter
  drops every record whose trace was not sampled. Below 100% sampling, `chat.*` lines and the video phase lines of
  unsampled requests do not reach the `traces` table. OTLP logs are unaffected. At the default 100% nothing is dropped.

### 10.4 Where things appear

| OpenTelemetry | Application Insights |
|---|---|
| ASP.NET Core server span | `requests` |
| every TensorSharp span (`ActivityKind.Internal`) | `dependencies`, type InProc |
| span events (denoise `step`, DiffusionGemma `preview`) | `traces`, one row per event; exception events go to `exceptions` |
| log records | `traces` |
| logged exceptions | `exceptions` |
| metrics | `customMetrics` |
| `service.name` / `service.instance.id` | `cloud_RoleName` / `cloud_RoleInstance` (`machine:pid`) |

Span attributes land in `customDimensions`, numbers as strings (`todouble()`). Studio jobs are traces of their own with
no `requests` row; the link to the queuing request is in `customDimensions["_MS.links"]`.

Metrics are exported every 60 s. The exporter's reader is built with the SDK's default interval and does not read
`OTEL_METRIC_EXPORT_INTERVAL` (checked in 1.10.0's source and in the end-to-end run, §12.5), so a process that lives
less than a minute reports its gauges only through the stop flush (§3.3).

### 10.5 Offline storage, Statsbeat, standard metrics

TensorSharp leaves these at the exporter's defaults:

- **Offline storage** is on: while ingestion is unreachable, telemetry is buffered on local disk (by default under a
  `Microsoft\AzureMonitor` directory in the user's local application data or temp folder) and retried. With content
  capture on, prompts can sit in that buffer.
- **Statsbeat**, the exporter's own health statistics sent to Microsoft (no telemetry content), is left as the exporter
  sets it.
- **Standard metrics** stay on: pre-sampling request and dependency rates in the portal.

### 10.6 Entra ID

With `TENSORSHARP_AZURE_MONITOR_AAD=1` the exporters authenticate with `DefaultAzureCredential` (`az login` locally, a
managed identity in Azure); the identity needs the Monitoring Metrics Publisher role. The connection string is still
required: it names the resource and its ingestion endpoint. `Azure.Identity` is a package reference of
`TensorSharp.Telemetry` for this.

### 10.7 Percentiles

Azure Monitor stores histograms as count, sum, min and max, without buckets, so the `gen_ai.server.*` histograms give
averages there, not percentiles. Percentiles come from span durations and span attributes, weighted by `itemCount`:
p95 time to first token is a percentile of `todouble(customDimensions["tensorsharp.ttft_ms"])` over the chat
dependencies, not of `inference.prefill` (which leaves out rendering and queueing). Video phases are selected by
`tensorsharp.video.phase`. [`queries.kql`](../../examples/opentelemetry/application-insights/queries.kql) has the
queries. Prometheus and Grafana users get the full histograms from the same metrics.

### 10.8 Through a collector

OTLP to `otelcol-contrib` with its `azure_monitor` exporter (named `azuremonitor` before v0.159.0) also works and needs
no Azure dependency in TensorSharp: [`collector-azuremonitor.yaml`](../../examples/opentelemetry/collector/collector-azuremonitor.yaml),
pinned to 0.164.0, which converts cumulative metrics to deltas. Set the connection string on the collector only, or
every item arrives twice.

---

## 11. Packaging and dependencies

| Project | Reference | Version |
|---|---|---|
| `TensorSharp.Telemetry` | `Azure.Identity` | 1.21.0 |
| | `Azure.Monitor.OpenTelemetry.Exporter` | 1.10.0 |
| | `Microsoft.Extensions.Logging` | 10.0.6 |
| | `OpenTelemetry.Exporter.Console` | 1.20.1 |
| | `OpenTelemetry.Exporter.OpenTelemetryProtocol` | 1.20.1 |
| | `OpenTelemetry.Extensions.Hosting` | 1.20.1 |
| | `OpenTelemetry.Instrumentation.Http` | 1.20.1 |
| | project `TensorSharp.Runtime` | — |
| `TensorSharp.Server.Host` | `OpenTelemetry.Instrumentation.AspNetCore` | 1.20.1 |
| | project `TensorSharp.Telemetry` | — |
| `TensorSharp.Cli` | project `TensorSharp.Telemetry` | — |

- Every other project takes no new package: the libraries use `System.Diagnostics` only.
- `Azure.Monitor.OpenTelemetry.Exporter` 1.10.0 (`net10.0`) depends on `Azure.Core` 1.63.0,
  `OpenTelemetry.Extensions.Hosting` 1.18.0 (lifted to 1.20.1 by the direct reference, so one OpenTelemetry version
  resolves) and `OpenTelemetry.PersistentStorage.FileSystem` 1.1.1.
- `TensorSharp.Telemetry` sets `IsPackable=true` (the repo default is false) and has `InternalsVisibleTo` for
  `InferenceWeb.Tests`. It is in `TensorSharp.slnx`.
- [`eng/verify-packages.ps1`](../../eng/verify-packages.ps1) checks each nuspec's direct `TensorSharp.*` dependencies
  exactly. It has a `TensorSharp.Telemetry` entry with `@("TensorSharp.Runtime")`, and `TensorSharp.Telemetry` is in the
  `TensorSharp.Server.Host` and `TensorSharp.Cli` lists.

---

## 12. Verification

### 12.1 Unit tests

In [`InferenceWeb.Tests/Telemetry/`](../../InferenceWeb.Tests/Telemetry/) and
[`TensorSharp.Studio.Tests/Telemetry/`](../../TensorSharp.Studio.Tests/Telemetry/), observed through real
`ActivityListener` and `MeterListener` instances and real OpenTelemetry logger pipelines; tests that change
process-wide switches or the environment share a non-parallel collection.

| File | Classes | Covers |
|---|---|---|
| [`TelemetryOptionsTests.cs`](../../InferenceWeb.Tests/Telemetry/TelemetryOptionsTests.cs) | `TelemetryOptionsTests`, `TelemetryCliFlagsTests` | Off with nothing set; generic endpoint; Application Insights alone (and the connection string never printed); OTLP plus Application Insights; `OTEL_SDK_DISABLED` wins; `TENSORSHARP_OTEL=1` default endpoint per protocol; a per-signal endpoint turns on only that signal and is used as given; per-signal headers, protocol and timeout reach the exporter options; `none` turns a signal off for every sink; `console` alone turns telemetry on; unknown exporter is a warning; malformed values are configuration errors; an OTLP value set for another program never stops startup with telemetry off, and the protocol is not validated with Application Insights only; endpoint credentials not printed; service name defaults and override; capture off unless asked; log level parsing; Entra ID needs a connection string; sampler description without Application Insights, mapping with it, and the unsupported-sampler warning; `--otel-endpoint` in both spellings, last one wins, pass-through without it, missing or relative URL is an error; per-signal endpoints and protocols on the startup line, without credentials. |
| [`TelemetryRedactionTests.cs`](../../InferenceWeb.Tests/Telemetry/TelemetryRedactionTests.cs) | `TelemetryRedactionTests` | `LogContent` redacted whatever its key and the rendered message dropped; a record without content keeps its message; an unmarked placeholder is left alone; with capture on content is exported; local logs print exactly what they printed before; `tensorsharp.request_id` added under its own name; log scopes are never exported; `TelemetryContext.Clear` forgets every request fact; an exception quoting an upload path is replaced but keeps its type name and stack; a scrubbed exception's message starts with the original type name; an exception with nothing sensitive is exported as is; upload URLs and registered directories scrubbed from text; a registered directory replaced through the end of the path; upload storage names replaced wherever they appear; marked content removed from the record's other attributes and its exception; capability paths masked and nothing else; the span backstop removes content attributes. |
| [`ContentContractTests.cs`](../../InferenceWeb.Tests/Telemetry/ContentContractTests.cs) | `ContentContractTests` | A source scan of every `ILogger` call in the production projects: each argument that carries content is wrapped in `LogContent.Of` (`ResponseId` placeholders count as content), each constant template has one argument per placeholder, and every allow-list entry still matches a call site. |
| [`TelemetryShutdownTests.cs`](../../InferenceWeb.Tests/Telemetry/TelemetryShutdownTests.cs) | `TelemetryShutdownTests` | The server's stop flush: stopping the host exports the gauges before the services they read are disposed; without the stop flush the gauge of a disposed engine is never exported (a control that reproduces the bug); a flush with nothing to flush returns at once; a stalled exporter does not hold the stop flush past its deadline. |
| [`CliTelemetryTests.cs`](../../InferenceWeb.Tests/Telemetry/CliTelemetryTests.cs) | `CliTelemetryTests` | The CLI's chat span: a stop the CLI makes itself (its `onToken` ends the generation) is recorded as `stop_sequence` with a completed outcome, not the engine's `aborted`; a run to its limit keeps the engine's finish reason. |
| [`ServerTelemetryTests.cs`](../../InferenceWeb.Tests/Telemetry/ServerTelemetryTests.cs) | `RequestLoggingTelemetryTests`, `ServerTelemetryContentSwitchTests`, `ServerOtelEndpointFlagTests` | `ClassifyClient`; request-id truncation (whole in the header, bounded for telemetry, no split surrogate pair); a capability path logged as content, an ordinary route as plain text, also on failure; an exported request log carries neither the upload name nor the scope that holds it, and the local `Operation` scope is unchanged; `ServerTelemetry.Enrich` masks each capability path with capture off and keeps it with capture on; the server's `--otel-endpoint` handling, typo hint and usage entry. |
| [`GenAiRequestTelemetryTests.cs`](../../InferenceWeb.Tests/Telemetry/GenAiRequestTelemetryTests.cs) | `GenAiRequestTelemetryTests`, `GenAiRequestTelemetryWithoutListenerTests` | A completed chat's span, phases, usage and metrics; a preempted sequence never gets a negative prefill; a phase never starts before its parent; a request aborted before scheduling has only a queue span; a failure records the type, never the message, and timing only for success; cancellation is not a failure; the duration is recorded once; content reaches the span only with capture on; internal calls and rounds are marked; model names never carry a path; nothing is recorded without a listener. |
| [`EngineTelemetryTests.cs`](../../InferenceWeb.Tests/Telemetry/EngineTelemetryTests.cs) | `EngineTelemetryTests`, `EngineStepObserverTests` | The engine's finish time on completed and aborted requests; slot and block preemptions counted with the first admission kept; engine counters monotonic across a rebuild; step duration and batch size per step; the step observer at most four times a second, exceptions swallowed, run by the worker between steps; step spans as roots linked to their requests. |
| [`VideoTelemetryTests.cs`](../../InferenceWeb.Tests/Telemetry/VideoTelemetryTests.cs) | `VideoTelemetryTests` | The video helper over recorded progress sequences: an H3 clip's phase and stage spans, steps and metrics; DiT loads by reuse; LTX passes and upscale as stages, with and without heartbeats; step timing after another stage; callback exceptions propagate and fail the clip; caller cancellation; a model failure marks its stage and phase; no spans without a clip; the prompt only with capture on; phase lines only when logs are exported; refused clips; effective values; child nesting; mode normalization. |
| [`TelemetryResourceTests.cs`](../../InferenceWeb.Tests/Telemetry/TelemetryResourceTests.cs) | `TelemetryResourceTests`, `ServerTelemetryFilterTests` | Default service name, version and `machine:pid` instance; `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` win; `/health` and low-noise paths are not traced and everything else is. |
| [`StudioTelemetryTests.cs`](../../TensorSharp.Studio.Tests/Telemetry/StudioTelemetryTests.cs) | `StudioTelemetryTests`, `StudioTelemetryWithoutListenerTests` | Jobs that never ran; fixed span names; the stored link and `job.json` compatibility; queue and stage spans from recorded times; failure codes without messages; a job found unfinished at startup; the duration metric; the content switch's spellings; nothing kept or started without a listener. |

[`ConfigFileArgsTests.cs`](../../InferenceWeb.Tests/ConfigFileArgsTests.cs) has `otel-endpoint` in its recognised
keys. Existing tests that assert rendered log text (for example the "Radix prompt reuse" lines in
`DeepSeek41ThinkingTurnReuseTests`) are unaffected by the placeholder renames, which change no rendered text.

**Not covered by unit tests** (§13): `GpuMemoryTelemetry` (including the `GpuMemoryObserved` gate), the chat
pipeline's wrapper and outcome mapping (including the DiffusionGemma branch producing exactly one span),
`CapturedMessagesJson` bounds, the Responses API's `gen_ai.response.id` rule, the workers' `TelemetryContext.Clear`
calls themselves, the CLI's crash-path flush, the embeddings span, `ShellRunner`, `execute_tool`, `model.load`, and a
`MeterListener` test that every metric tag key is in the §6.6 list; the engine unregistering only after its worker has stopped (§6.2), the streaming image route recording a client that leaves during the upload read as cancelled, and the `warmup` origin on the CLI's `--warmup-runs` passes.

### 12.2 What end-to-end runs must show

Unit tests cannot show the SDK wiring, the exporters, the startup order, real models on real backends, or what a
backend makes of the data. Those are the two end-to-end runs below. AGENTS.md applies: evidence goes to the ignored
`docs/validation/` or `artifacts/`, is never force-added, and records the TensorSharp commit, the ggml revision of
`ExternalProjects/ggml`, the resolved package versions, model, backend, GPU and OS; a skipped or MANUAL check is not a
pass.

### 12.3 Aspire dashboard, end to end

[`eng/telemetry/aspire-e2e.ps1`](../../eng/telemetry/aspire-e2e.ps1) ([README](../../eng/telemetry/README.md)), PowerShell
7.2+:

- **Setup.** Starts `mcr.microsoft.com/aspire/dashboard:13.6.2` (or a given `Aspire.Dashboard.dll`, or a running
  dashboard) with OTLP API-key authentication and the telemetry HTTP API on, both keys random per run, on loopback
  (18888 UI and API, 4317 gRPC, 4318 HTTP). `ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS` is never set: it also turns
  OTLP authentication off. Each server run starts with every inherited `OTEL_*`, `TENSORSHARP_OTEL*` and
  `APPLICATIONINSIGHTS_CONNECTION_STRING` variable removed, a unique `OTEL_SERVICE_NAME`, and short export intervals
  (`OTEL_BSP_SCHEDULE_DELAY=1000`, `OTEL_BLRP_SCHEDULE_DELAY=1000`, `OTEL_METRIC_EXPORT_INTERVAL=5000`). Every prompt
  carries a random marker, so the script can prove no exported log or span carries prompt text.
- **Runs.** `otlp-grpc`, `otlp-http`, `content-capture` (through `--otel-endpoint`), `embeddings`, `video-ltx`,
  `video-minimax-h3`, `negative-off`, `negative-sdk-disabled`, `negative-wrong-key`; each starts only when one of its
  scenarios is selected and runnable.
- **Scenarios.** OpenAI chat streamed and not; Ollama `/api/generate` and `/api/chat/ollama`; Web UI chat; a client
  `traceparent`; a cancelled stream; a 400; an 8-request burst; `/api/queue/status` polling; embeddings over
  `/v1/embeddings` and `/api/embed`; LTX (stages `first-pass`, `upscale`, `refinement`); MiniMax-H3 twice in a row
  (`dit-load cached=false`, then `true`); a cancelled clip; a Studio generate job (own trace, link, `studio.queue`,
  stage spans, `generate_video` under it); step spans with `-TraceSteps`; content capture; and the three negative runs
  (nothing arrives; the server keeps serving with a rejected key, within a latency margin of the baseline).
- **Per-run checks.** The startup line; resource attributes (`service.instance.id` = `machine:pid`); request spans and
  logs joined by `tensorsharp.request_id` = the response header; `/health` not traced; content redaction; the OTLP and
  API keys absent from the server's output and logs. Spans must lie inside their parents within a small tolerance;
  token counts on spans must equal the responses' usage blocks.
- **Results.** `docs/validation/otel-e2e-<timestamp>/`: `results.md` and `results.json` (PASS, FAIL, SKIPPED, MANUAL),
  `environment.json` (commit, ggml revision, dashboard image digest, OS, GPUs, SDK), per-run server output, requests and
  exported telemetry, and the dashboard's final snapshot. `login.txt` and `keys.json` hold live credentials and must
  stay out of commits.
- **Limits.** The dashboard's telemetry API has no metrics route, so metrics are MANUAL (read in the dashboard UI). The
  script exercises the OTLP path only; Application Insights has its own script (§12.4).

### 12.4 Application Insights, end to end

[`eng/telemetry/appinsights-e2e.ps1`](../../eng/telemetry/appinsights-e2e.ps1)
([README](../../eng/telemetry/README.md#application-insights-appinsights-e2eps1)), PowerShell 7.2+ and the Azure CLI:

- **Setup.** `-AppInsightsResourceId` names the `microsoft.insights/components` resource. The script reads its
  connection string and AppId with `az resource show` at run time, and checks that the query API answers
  (`print ok = 1`) before any server starts. The signed-in identity needs Reader or Monitoring Reader on the resource;
  for the Entra ID run, the identity `DefaultAzureCredential` picks inside the server also needs Monitoring Metrics
  Publisher.
- **Secrets.** The connection string reaches the server processes through their environment only and is never printed,
  logged or written. Every file the script writes, every result row and every console line goes through a filter that
  replaces the connection string, its instrumentation key, the AppId, the resource id, the subscription id and any
  dashboard key; results name the resource by its name. `connection-string-not-logged` fails if the server's own output
  or file logs contain the connection string or the key, and a last `harness / evidence-scrub` row scans the whole
  evidence folder.
- **Runs.** `main` (every `llm-*` scenario and `queue-status-poll`), `content-capture`, `embeddings`, `video-ltx`,
  `video-minimax-h3` and, with `-EntraId`, `entra-id` (`TENSORSHARP_AZURE_MONITOR_AAD=1`, no model); each starts only
  when one of its scenarios is selected and runnable. Every run starts with each inherited `OTEL_*`,
  `TENSORSHARP_OTEL*`, `APPLICATIONINSIGHTS_*` and `TENSORSHARP_AZURE_MONITOR_AAD` variable removed, a unique
  `OTEL_SERVICE_NAME` (`tensorsharp-e2e-<tag>-<run>`, its `cloud_RoleName`), short export intervals and no sampler
  variable, so the sampler is Azure Monitor's at 100%. `-BothSinks` also sends OTLP to an Aspire dashboard when one
  listens. All runs are driven first and read back afterwards, so their ingestion delays overlap.
- **Stopping.** The exporters flush at shutdown, so each server is stopped, not killed: SIGTERM on Linux and macOS,
  CTRL_BREAK raised on the server's own console on Windows. `server-stop` passes only for an exit by itself with code 0.
- **Reading.** `az rest` against `https://api.applicationinsights.io/v1/apps/<AppId>/query`, every query filtered by the
  run's `cloud_RoleName` and start time (so a shared resource does not disturb the checks), repeated with backoff from
  15 to 60 s until every expected row is there or `-IngestionTimeoutMinutes` (12) has passed since the server stopped.
  The absence checks run once, after the presence checks.
- **Checks.** Per request: a `requests` row whose `tensorsharp.request_id` is the response header, with the status,
  the protocol, `itemCount` 1 and no parent (the client's trace and span for `llm-client-traceparent`); the generation
  dependency (InProc) with `prompt.render` and the `inference.*` children, the outcome, a positive
  `tensorsharp.ttft_ms` and usage equal to the response's; `traces` rows with the same request id in the request's
  operation, and none there with another id; the LTX and H3 trees with the H3 DiT cached on the second clip; a Studio
  job as an operation of its own with its `_MS.links` entry; all eight burst requests (no rate-limited sampling). Per
  run: the startup line (sinks, `sampler azuremonitor fixed_percentage(1)`, no Live Metrics warning); no `/health` or
  `/api/queue/status` row; `cloud_RoleInstance` = `<machine>:<pid>`; no prompt marker in any row with capture off,
  and the marker in `traces` with it on; the presence of the `customMetrics` instruments; ingestion with Entra ID.
- **Results.** `docs/validation/appinsights-e2e-<timestamp>/`, with the PASS / FAIL / SKIPPED / MANUAL rules and result
  files of §12.3, plus each query and its last result per run.
- **Limits.** Histograms are kept as count, sum, min and max, so the metrics check is presence only; an absence check
  queries once, so a row ingested later is not seen; the Azure Monitor metric reader exports every 60 s whatever
  `OTEL_METRIC_EXPORT_INTERVAL` says, so in runs shorter than that the gauges arrive only through the stop flush
  (§3.3), and their presence checks it; whether the trace and log exporters honour `OTEL_BSP_SCHEDULE_DELAY` and
  `OTEL_BLRP_SCHEDULE_DELAY` is not verified.

The script does not cover these; they stay manual, with the queries in
[`queries.kql`](../../examples/opentelemetry/application-insights/queries.kql). The connection string comes from the
resource into the environment of the run; it is never written to the repository, to a config file used by the run,
or to the evidence.

1. Sampling: rerun a burst with `OTEL_TRACES_SAMPLER=traceidratio`, `OTEL_TRACES_SAMPLER_ARG=0.25`, and compare
   `union requests, dependencies | summarize sum(itemCount)` and `100/avg(itemCount)` with the client's request count;
   check that logs of unsampled traces are missing from `traces` and present over OTLP; repeat with
   `microsoft.rate_limited` to see whether its counts are scaled.
2. Once with ingestion blocked: telemetry is buffered in the offline storage directory and sent after it is unblocked.
3. A restart with a queued Studio job: the recovered job's instant span and its link.
4. Run `queries.kql` and keep the outputs.

### 12.5 Results

Every run used one machine: an NVIDIA RTX Spark N1X (32 GB, memory shared with the CPU), Windows 11 on Arm (build
28020), .NET 10.0.14, Debug builds of the parent commit `92bd7204` plus this change, and ggml `57e2c2ca` (v0.29.0)
from upstream with no patches (this change touches no native code). Models: Qwen3.5-0.8B Q8_0 on `ggml_cuda` (chat),
all-MiniLM-L6-v2 Q8_0 on `ggml_cpu` (embeddings), LTX 2.5 22B distilled UD-Q4_K_XL and MiniMax-H3 FL2VA pruned Q4_K
on `ggml_cuda` (video). The evidence folders under `docs/validation/` are not committed.

Final results, on the code as submitted:

| Run | Result | Evidence |
|---|---|---|
| Aspire dashboard 13.6.2 (§12.3), gRPC and `http/protobuf`, every scenario | 69 pass, 1 skipped (`engine-step-spans`, which needs `-TraceSteps`), 1 manual (`metrics`) | `otel-e2e-aspire-final2-20261106-214543` |
| Aspire, `-TraceSteps`: `engine-step-spans`, `llm-cancel` | 15 pass, 1 manual (`metrics`); 7 `inference.step` spans link to the request | `otel-e2e-aspire-steps2-20261106-214543` |
| Application Insights (§12.4), every scenario, `-EntraId` | 70 pass | `appinsights-e2e-final-20261106-214543` |
| `InferenceWeb.Tests`, CI's filter (`Category!=Bench&Requires!=Cuda&Requires!=Mlx&Requires!=Models`) | 11,895 pass, 177 skipped, 6 fail; the same 6 fail on the parent commit (symbolic-link privilege, file locking in `Qwen35ServingTraceTests`, a skill bundle's pinned provenance, a Qwen3.5 template rendering) | `otel-tests-final-20261106-214543` |
| `TensorSharp.Studio.Tests` | 195 pass, 4 skipped, in four full runs; one earlier run had 2 failures from a Windows file-sharing race in `StudioStorage` (unchanged here), and the parent commit's suite fails the same way about once in four runs | `otel-tests-final-20261106-214543` |

The `metrics` row of the Aspire runs is MANUAL because the dashboard's telemetry API has no metrics route. It was
checked by eye on the run before the last fixes (`otel-e2e-aspire-final-20261106-210156/dashboard/manual-metrics-check.md`):
every TensorSharp and TensorSharp.Studio instrument was there with data. On the final code, the Application Insights
run checks the instruments, the engine and KV-cache gauges included.
What the runs cover: OpenAI chat streamed and not, Ollama generate and chat, the Web UI chat, a client `traceparent`,
a 400, an 8-request burst (each in its own trace, none sampled away), `/api/queue/status` polling (not traced), a
cancelled stream, embeddings, an LTX clip with its `first-pass`, `upscale` and `refinement` stages, a cancelled LTX
clip, a Studio generate job (its own trace, linked to the request), two MiniMax-H3 clips (`dit-load` uncached, then
cached), content capture on and off (no prompt marker in any exported row with it off), the three negative runs (off,
`OTEL_SDK_DISABLED`, a wrong OTLP key that must not break serving), graceful shutdown of every server (1.2 to 5.9 s after the stop signal, the stop flush included),
and Entra ID ingestion, where the token carried spans, logs and 21 TensorSharp metric names. In the dashboard's
Metrics page every TensorSharp and TensorSharp.Studio instrument is there with data; `tensorsharp.tokens` has the
`tensorsharp.token.type` values `input` and `output`, the protocols `openai`, `ollama` and `webui`, and the
`prefix-cache-warmup` origin.

What the runs found, and what changed:

- **Engine gauges never reached Application Insights** (a product bug). The Azure Monitor metric reader exports every
  60 s whatever `OTEL_METRIC_EXPORT_INTERVAL` says, each server ran for less than a minute, and the providers' own
  flush ran after the models were disposed, when the engine and KV-cache gauges have nothing to report. The stop flush
  (§3.3) and the CLI's flush before its engine goes (§3.4) fix it; `TelemetryShutdownTests` has the failing control.
  The metrics check now also requires `.waiting`, both KV-cache gauges, the step instruments and GPU memory.
- **`llm-cancel` failed on both OTLP protocols** because of the harness. `SocketsHttpHandler` drains up to 1 MB or 2 s
  of an unread response before closing it, so the server streamed the whole 0.5 s reply and never saw the client go;
  with `MaxResponseDrainSize = 0` the chat span ends `cancelled` after 4 output tokens. The check also expected every
  round to be cancelled, although a round that called a skill tool had completed; now only the last round must match,
  and the request sends `tool_choice: "none"` (the 0.8B model once spent all 8 skill rounds before streaming).
- **Studio's media probe timed out** in the Application Insights runs. Studio runs `ffprobe -version` with a 10 s
  limit; right after the cancelled clip, LTX's 12 GiB denoiser read-ahead was still filling memory and the disk, and
  the probe ran out of time (ffprobe starts in 60 ms on an idle machine). The Aspire script happened to wait for the
  cancelled clip's span first. Both scripts now wait for the cancelled request to end and ask Studio again after a
  timed-out probe, which Studio does not cache.

### 12.6 Performance

Release builds of the parent commit and of this change, on the machine of §12.5, Qwen3.5-0.8B Q8_0 on `ggml_cuda`
through Ollama's `/api/generate` (no skill rounds): three requests to warm up, then seven sequential requests (110
output tokens each) and three bursts of eight concurrent ones, per server start. Each configuration ran twice,
interleaved with the others, so 14 sequential samples and 6 bursts each. "OTLP on" exported traces, logs and
metrics over gRPC to a local Aspire dashboard at 100% sampling.

| Configuration | Decode tokens/s, median [range] | Request wall time, median [range] | 8-request burst, tokens/s, median [range] |
|---|---|---|---|
| Parent commit, telemetry off | 218.5 [211.7 – 223.0] | 568 ms [553 – 586] | 530 [493 – 548] |
| This change, telemetry off | 216.4 [206.6 – 228.3] | 574 ms [539 – 602] | 527 [492 – 556] |
| This change, OTLP on | 217.2 [206.3 – 226.1] | 578 ms [548 – 597] | 515 [495 – 554] |

The medians differ by 1 to 3%, inside the spread of single runs (±5%), so this measurement shows no cost it can
separate from noise; it cannot rule out a difference of a few percent. Not measured: video clips with telemetry on
and off, longer prompts and outputs, other models and backends, other machines, and the periodic metric export
itself (each server lived less than one 60 s interval and was killed, not stopped; recording on the request path is
included). The measurement predates the last fixes (§12.1's redaction cases, the threaded stop flush, the CLI's
flush rule). Of those, only the redaction one runs while serving: a log record that carries content now also has its
other string attributes searched for the marked text, on the thread that logs it, when logs are exported. Evidence:
`docs/validation/otel-overhead-20261106` (not committed).

---

## 13. Known limitations and follow-ups

| Area | Limitation | Follow-up |
|---|---|---|
| GPU memory | ggml reports rank 0 only, so tensor-parallel and layer-split ranks are missing. Readings are device-wide, so used memory includes other processes. An MLX figure read through `Dispatch` can lag behind a busy worker. | A TensorSharp-owned per-rank export. |
| Request metrics | Recorded per generation: a skills or multi-agent turn gives one `request.duration` sample per round; internal calls are on the same instruments, told apart by `tensorsharp.origin`. | Record one sample per HTTP request at the adapter if dashboards need it. |
| Outcomes | Pre-submission errors with generic types ("No model is loaded", an image with no vision encoder, unsupported audio, LoRA on DiffusionGemma) are `failed`, not `refused`; so are video model rejections (`ArgumentException`, `InvalidOperationException`, `NotSupportedException` inside `GenerateVideo`) that the plain route answers with 400. A consumer that disposes the stream early is `cancelled`. Two theoretical video exits are unmarked (`InvalidCastException` from the model cast, an exception from `TryEnterUse`). | Typed exceptions. |
| Spans | `gen_ai.request.stream` and `gen_ai.response.model` are not set; `gen_ai.response.id` is only on the server span, and is absent for a stored Responses API response with capture off (§5.6). The embeddings span carries `output_tokens=0`, and no batch size (the dispatcher would have to report it). | |
| Spans | DiffusionGemma's `prompt.render` is missing when its preparation throws; its TTFT is the first preview even for clients that receive only the final answer; preview event times include consumer backpressure. Jev and images record no content even with capture on. | |
| Spans | `ServerSpan.Recorded` assumes ASP.NET Core's span has kind Server (true on .NET 8+). The embeddings dispatcher's pump runs in the first caller's context (harmless while nothing below it traces). | Clear `Activity.Current` and `TelemetryContext` in the pump if it is instrumented. |
| Model load | Refusals `StartupModelLoader` raises before `LoadStartupModel` get no span; the embedding-model startup load (`EmbeddingHosting`) and CLI loads have no `model.load`; load duration includes the Jev gate wait, engine teardown and unloading the previous model. Threads started during a load (`CpuWorkerPool`, `GgmlTensorParallelGroup`, `Dsv4SpinPool`, `MappedFileReadAhead`) now inherit `model.load` as `Activity.Current`. | `ModelLoadTelemetry` is reusable from Server and CLI; clear `Activity.Current` in those worker loops. |
| Tools | Sub-agent tool calls in `AgentHost/Skills/SkillAgentLoop` have no `execute_tool` span; the CLI's skill loops have none either, and each CLI chat round and `codeexec.run` is the root of its own trace. A tool that reports failure is not Error status. | |
| Engine | `PreemptionCount` also counts slot deferrals, and a deferred sequence's queue phase ends early (§5.3). `tensorsharp.engine.forward.time` includes `Schedule()`. | |
| Logs | Log scopes are not exported (§7.6), so an exported record has no `SessionId`, `Model`, `Backend` or `Operation` attribute; for a record written inside a request, the spans its trace and span ids lead to hold the model, backend and session. Engine, diffusion and MLX worker lines carry neither a trace id nor `tensorsharp.request_id`; only `{EngineRequestId}` joins them to a request (§7.2). | Pass a value an exported record needs as a template argument. |
| Logs | The ASP.NET Core filter for the OpenTelemetry provider is registered unconditionally, after configuration rules, so `Logging:OpenTelemetry:LogLevel:Microsoft.AspNetCore` cannot raise it. | Skip it when that key is configured. |
| Logs | Exception messages are exported with only paths under the registered directories, `/uploads` URLs, upload storage names and the record's own marked values removed; content in another kind of message would pass. The span backstop does not cover `tensorsharp.tool.*` or `tensorsharp.codeexec.command` (their call sites check the switch). Console-only warnings (MP4 fallbacks, FFmpeg failures, template and grammar warnings) are not exported. | |
| Video | H3 now calls `OnProgress` five to eight more times per clip; a cancelled streaming H3 clip can stop at these stage boundaries (all before any state change), and in prefault modes 1–2 the read-ahead then finishes in the background. Studio may save the job slightly more often (its 300 ms throttle applies). | |
| Video | Wan's `Step` names the step that is starting, so its step events are one step late and its last step is not recorded. `video.queue` is on every clip, zero-length when nobody waited. On the plain path the WAV sidecar and `RecordFile` run after the clip ends. Refused clips are separate zero-length spans in the duration histogram. | |
| Studio | The job span is `ActivityKind.Internal`, a root with no `requests` row in Application Insights. Instant spans for jobs recovered at startup use the job's last `FinishedAt`, which can be old. If `SaveJob` throws right after `Finish("completed")`, the duration is recorded twice. A job queued with telemetry off and run after a restart with it on has no link. The gauge callback takes `jobSync`, and instant spans for cancel and shutdown are emitted under it (with the console exporter, its write happens under the lock). | Consider `Consumer` kind and a `queued_by_trace_id` attribute. |
| Configuration | A tensor-parallel node id set by a CLI flag is applied after telemetry starts, so it is not on the resource (the variable works). TP workers export as `tensorsharp-cli`. `--otel-endpoint` given as the literal value of another flag is read as the flag. The server usage line says `TENSORSHARP_OTEL=1 (localhost:4317)`, which is 4318 with `http/protobuf`. | |
| Operations | Export failures are silent (SDK EventSource only); the startup line does not print the offline storage directory; `GET /` with `--no-webui` is traced. An engine rebuilt on a model swap (the CLI's interactive `/model` included) or on a CLI interactive reset is not flushed first, so its gauges for the last partial interval are lost (only the server's stop flush and the CLI's flush before its last engine goes cover the last engine). | A rate-limited export-failure warning to the console and file logs. |
| Application Insights | `EnableTraceBasedLogsSampler`, Statsbeat, standard metrics and offline storage are left at the exporter's defaults; rate-limited count scaling on 1.10.0 is unverified. | §12.4. |
| Conventions | The GenAI conventions moved from `semantic-conventions` to `semantic-conventions-genai` (core v1.42.0), which has no release or schema URL yet; names follow core v1.41.0. Known differences pending there: `gen_ai.request.top_k` becomes an integer, token metrics change, content attributes must be structured on spans. | Record the targeted version in code. |
| Docs | The project tree in `DEVELOPMENT.md` and `DEVELOPMENT_zh-cn.md` does not show `TensorSharp.Telemetry` (their package count and package table do). `eng/NuGet.dotnet-public.config` was not checked for the new packages (it has `<clear />`, so nuget.org is not a fallback). | |
| Tests | §12.1 lists what has no unit test. | |

---

## Appendix A. Corrections the source check found

The draft was checked against the code before implementation. "Built" says what the code does instead.

| Draft said | The code or the specification showed | Built |
|---|---|---|
| **Architecture and configuration** | | |
| Do not use `Runtime.Logging` because TensorAgent references it | The stronger reason: Chat and Server, and their packages, depend on it | Host-only `TensorSharp.Telemetry` (§2) |
| ASP.NET Core instrumentation in `TensorSharp.Telemetry` | It carries a framework reference to `Microsoft.AspNetCore.App`; the CLI references the project | In `Server.Host`, added through a callback (§3.2) |
| One call next to `LoggingSetup.Configure` | The hosting package starts providers inside `app.RunAsync`, after the startup load, the warm-up and Studio recovery | `StartTensorSharpTelemetry` right after `Build()` (§3.3) |
| CLI providers disposed on exit | `_exit` on Linux `ggml_vulkan` and a rethrown exception skip `using`; the log provider belongs to the logger factory | Flush in the leave callback, and in `RunToExit`'s catch the logger factory, then the session (§3.4) |
| `--otel-endpoint` so a config file can carry it | Config files already set any variable through `environment` | Kept as a convenience; command line wins (§4.5) |
| Add the key to `ServerRecognizedConfigKeys` | That set guards shipped presets only; acceptance is the early strip and the builder | Strip, builder skip, typo hint, usage entry; key also added (§4.5) |
| CLI flag parsing unspecified | The CLI ignores unknown flags; `=` spelling is normalized only for documented options | Read in `Main`; documented in `CliUsage` (§4.5) |
| Banner line in `StartupBanner.Emit` | `Emit` returns early for embeddings servers and runs after the model load | Logged after `Logging initialized` (§4.6) |
| Banner shows the SDK sampler | The SDK exposes no effective sampler; under Azure the SDK's is replaced | Describes the value; under Azure the effective Azure sampler (§4.4) |
| `TENSORSHARP_OTEL=1` → `localhost:4317` | Wrong for `http/protobuf`; a code endpoint also disables path appending | No endpoint override; the SDK default per protocol (§4.1) |
| `service.version` from `TensorSharpVersion` | An MSBuild property only | Informational version without `+<commit>` (§4.7) |
| Default service name | `AddService` would override `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` | Environment detector last; tested (§4.7) |
| `service.instance.id` per process | A GUID replaces the machine name in `cloud_RoleInstance` | `machine:pid` (§4.7) |
| Per-signal endpoint turns OTLP on | `AddOtlpExporter` ignores per-signal variables; `UseOtlpExporter` turns every signal on and does not combine | `OtlpSignalSettings` applied per exporter (§4.3) |
| `OTEL_*_EXPORTER` / `OTEL_SDK_DISABLED` | Not implemented by the .NET SDK for this setup | Enforced by `TelemetryOptions` (§4.2) |
| Settings table complete | Log level, Entra ID and SDK batch variables were only in later sections | One table (§4.1) |
| `verify-packages.ps1` lines 98–118 | The manifest is `$PublicPackages`; dependency lists must match exactly | New entry plus the two host lists (§11) |
| "No change to logs" | Placeholder renames change the JSON file log's template and props | Listed exceptions (§7.9) |
| Request id from the middleware | It reuses an inbound header verbatim, any length | Bounded to 64 characters for telemetry (§5.8) |
| No failure behaviour | Invalid values throw at provider build; export failures are silent | Configuration errors before start (§4.2); silence recorded (§13) |
| Validate every OTLP setting | The `OTEL_*` variables are shared by every SDK on the machine; `http/json` is valid elsewhere | Protocol and per-signal settings validated only when a signal exports over OTLP (§4.2) |
| **LLM traces** | | |
| Queue and prefill from the engine's timestamps | Re-admission after preemption overwrote `FirstScheduledAt`: negative prefill | `??=`, `PreemptionCount`, guard (§5.3) |
| Record metrics on three exits (line 900 = submission failure) | Line 900 is the completion await; engine errors come from the token loop; more exits exist; an iterator cannot catch around `yield` | Wrapper plus the core's `finally`, once (§5.5) |
| `Activity.Current` does not reach the engine | `Thread.Start` flows the first request's context, `TelemetryContext` included | Both cleared in the engine, diffusion and MLX workers (§5.4) |
| Span in `ChatStreamAsync` | A thin wrapper on no server path | `ChatStreamWithMetricsAsync` (§5.5) |
| One chat span per request | Skills rounds, sub-agents, knowledge reranking and the warm-up each generate | One per generation, with round and origin (§5.7) |
| Decode ends at completion | The engine recorded no end time; draining waits on the client | `InferenceCompletion.FinishedAt` (§5.2) |
| Protocol and request id on the chat span | Chat cannot see them; `ClassifyClient` was private and misread `/api/embed` | `TelemetryContext`; classifier fixed (§5.8) |
| `gen_ai.response.id` on the chat span | The ids are created in Server; a stored response's id retrieves its output | On the server span; not for a stored response with capture off (§5.6) |
| `gen_ai.conversation.id` = session id | The stateless protocols share one constant session | Non-shared sessions only (§5.6) |
| `max_tokens` from `SamplingConfig` | `SamplingConfig.MaxTokens` is not read on this path | The argument, plus the effective budget (§5.6) |
| Embeddings span in the dispatcher | The dispatcher batches requests; media skips it | Per request in `EmbeddingAdapter` (§5.9) |
| `model.load` at `LoadModel` | Both entry points meet in `LoadModelCore`; a distributed no-op exists | `LoadModelCore`, with `skipped` (§5.9) |
| `execute_tool` per round | Execution is per call, on a `Task.Run` worker | Per call, inside the worker, so `codeexec.run` nests (§5.9) |
| `codeexec.run`: shell, exit code, sandbox | The log line has seven fields and a not-started path | All fields; `not_started` (§5.9) |
| DiffusionGemma `denoise` children | Its scheduler batches steps across requests | One `diffusion.denoise` with preview events, both entry points (§5.9) |
| Every protocol through the pipeline | Web UI DiffusionGemma, Jev and the CLI bypass it | Each instrumented (§5.2, §5.9) |
| Add `/health` to `LowNoisePaths` | That set also demotes console and file lines | The trace filter adds it (§5.10) |
| Step spans with links | The engine had no request context to link to | `SequenceState.TelemetryContext` (§5.11) |
| Starting a span costs about 1 µs | About 14 ns without a listener, a few hundred with the SDK | §2 |
| **Metrics** | | |
| TTFT and TPOT with `error.type`, on every exit | Defined for successful responses, without `error.type` | Completed requests only (§6.1) |
| Attribute set | `gen_ai.provider.name` is required | Added (§6.1) |
| Histograms in seconds | The SDK's default buckets put sub-5 s values in one bucket | Bucket advice (§6) |
| `gen_ai.token.type` with `cached` | Not a valid value; removed upstream; double counting | `tensorsharp.token.type` and a `cache_read` counter (§6.1) |
| Engine gauges through `TryGetEngine()` | It builds and disposes engines under a lock; totals reset per engine; the CLI's engine is invisible | Self-registration, retired totals (§6.2) |
| Step histograms on the engine thread | Location, and which steps count, unspecified | Next to the step counters, same steps (§6.2) |
| `blocks.used` from `PoolStats` | No used field; forward time includes scheduling | total − free; documented (§6.2) |
| Engine-wide speculation total | Per-sequence stats are replaced when a preempted sequence re-arms | Counted at the increment sites (§6.2) |
| Trace filter keeps polling out | Built-in HTTP metrics ignore it | `DisableHttpMetrics` (§5.10) |
| Cardinality list | Incomplete; `error.type` formation unspecified | Full list; codes or type names only (§6.6) |
| **GPU memory** | | |
| CUDA `CachedBytes` / `PeakCachedBytes` as used and peak | Idle pooled blocks only, on a stats struct | Used = total − free from `GetMemoryInfo` (§6.3) |
| `Ltx25Model.DirectAllocator` accessor | H3 had none, other models' allocators are protected, and LTX maps backends itself | `ModelBase.DirectAllocatorForTelemetry`; query chosen by allocator type (§6.3) |
| ggml from a metrics callback | `ensure_backend()` latches Metal in a process with no GGML backend; teardown takes no reader lock | Snapshot only, only through a live `GgmlAllocator` on a GPU backend (§6.3) |
| ggml device index | Rank 0 only; CPU reports host RAM | Device 0; host-memory models not tracked (§6.3) |
| MLX waits behind one step | On MLX a whole clip is one work item | Inline on the MLX worker, `Dispatch` elsewhere (§6.3) |
| GPU snapshots at phase boundaries | LLM serving has no phases; a snapshot nobody collects is wasted native work | Engine step observer, registered only when metrics are exported; `Snapshot` gated on `GpuMemoryObserved` (§6.3) |
| used/total/peak for every backend | No backend has all three | Per-backend nulls (§6.3) |
| **Logs and privacy** | | |
| Content-key table "audit of every template" | Upload names and paths, result URLs, repetition text, tool arguments and many CLI lines were missing; two CLI rows logged fixed strings | Call-site marking (§7.4) |
| Key-based redaction with `PromptTokens` renames | Names collide throughout (`Path`, `Source`, `Error`, `Tokens`); content hides under `What`, `Url`, `User` | `LogContent`; renames kept for readability (§7.4) |
| `generate.*` lines carry content | They have no callers | Marked anyway (§7.4) |
| Processor redacts paths | Scope values cannot be rewritten; ASP.NET Core's scope holds the raw request path | Scopes not exported; `{Path}` marked at the source (§7.6) |
| Exceptions not mentioned | Exported with messages; media messages quote paths | Upload-path scrubbing (§7.5) |
| Processor placement unspecified | Processors run in order; an earlier exporter sees unredacted data | Redaction before every exporter (§7.4) |
| Clear `FormattedMessage` → body is the template | Body and message semantics differ per exporter | Clear it only on records with content; keep `{OriginalFormat}` (§7.4) |
| `RequestId` scope matches the header | ASP.NET Core's hosting scope uses the same key and comes first; worker threads inherit their creator's scope | `tensorsharp.request_id` enrichment; scopes not exported (§7.2, §7.6) |
| One request id space | Engine lines log their own id as `{RequestId}` | `{EngineRequestId}` and `tensorsharp.engine_request_id` (§7.2) |
| Event ids exported | Video lines shared `UploadReceived`; OTLP exports the name only | New ids; documented (§7.1) |
| ASP.NET Core filter: any rule disables it | Only a provider rule whose category matches; ties go to the later rule | Registered for the provider (§7.3) |
| Capture flag in `TelemetryOptions` | Libraries cannot see that project | Pinned static switch; Studio reads the variable; span backstop (§7.7) |
| `/uploads` unauthenticated per Program.cs:362 | That line is the knowledge store; `/uploads` is `UseStaticFiles`; code artifacts and stored responses are guarded only by their ids too | Capability paths masked in spans and request log lines (§5.10, §7.6) |
| Exported records may exceed the size limit | The exporter truncates silently (32,768 / 8,192); Application Insights cuts captured chat messages over about 64,700 characters to 8,192 | §7.4, §7.7 |
| Console-only output is the H3 and LTX timings | Also `[h3]` lines, warnings, MP4 fallbacks, Wan, Qwen-Image, grammar warnings | Listed (§7.8) |
| **Video** | | |
| Both pipelines check the token while reporting | Only LTX does; H3 stops only when a callback throws | §8.1 |
| H3 tree of sibling phases | Prefault, DiT load and keyframe encodes fall inside `text-encode`; release and VAE open inside `vae-decode` | Stages under their phases (§8.3) |
| `dit.load cached` | Reuse of the object, not residency; upload happens in step 1 | Documented (§8.3) |
| `weights.prefault_wait` | Never runs in the default mode 3 | Stage only in modes 1–2 (§8.3) |
| `denoiser.release` | Metal always, CUDA when it would not fit; residency only | `release_reason` (§8.3) |
| `lora.apply` on HTTP routes and LTX | HTTP routes pass no set; LTX takes none | Only for an `ILoraPluginHost` with a set (§8.6) |
| Phase list | LTX also reports `reference-encode`; ref mode is single-pass; H3 never reports `image-encode` | §8.1, §8.4 |
| Outer span and refusals in `RunVideoGeneration` | Every refusal happens before it; the stream route answers 200 | Refusal clips where they happen (§8.6) |
| Model code creates child spans | No parent on MLX | Stages and attributes; the helper creates spans (§8.2) |
| Callback exceptions unspecified | Host cancellation and Studio's disk check depend on them; Wan heartbeats come from a timer | Caller first, propagate; heartbeats skipped (§8.5) |
| Heartbeats from H3 and LTX | Only Wan sends them | §8.1 |
| CLI gets `video.export` through `SaveMp4` | The CLI has no outer span; PNG output skips `SaveMp4` | CLI clip and `CliVideoExport` (§8.7) |
| Request values as span attributes | Seed, size and steps are decided inside the pipelines | Requested, then effective (§8.5) |
| A step event per finished step | Some `denoise` reports are not steps; step 9 would include the upscale | Step rise, timed from the stage start (§8.5) |
| `open_s`, `run_s`, `teardown_s`, `open_s`/`decode_s`/`tiles` | Teardown is a remainder; uploads happen in run and decode | Documented (§8.3) |
| LTX text-encode | A repeated prompt hits a small cache | `prompt_cache_hit` (§8.4) |
| Helper in `TensorSharp.Models/Video` | The folder is `TensorSharp.Models/Models/Video` | §8.5 |
| One phase line per change, exported immediately, with the trace id | Logs are batched; trace id needs `Activity.Current`; Models has no logger | Caller's logger, phase span current on the calling thread, only when logs are exported (§8.9) |
| `media.generations.active` counts generations | It counts accepted requests, waiting included, and image-intent questions | Documented (§8.8) |
| Queue counter around the lock | The lock is static and its wait cannot be cancelled | Static counter (§8.6) |
| Phase duration by phase and stage | Everything between reports counted as one phase | Stages on spans; phases on the metric (§8.8) |
| `refused` outcome on the duration | Impossible from `RunVideoGeneration` | Refusal clips (§8.6) |
| Result URLs are content | No `Url` key in the redaction list | Marked (§7.4) |
| **Studio** | | |
| Startup marks running jobs interrupted | A generate job with an imported take is adopted as completed; the worker marks the running job at shutdown | §9.3 |
| The link survives a restart | Startup and shutdown fell outside the SDK's lifetime | Early provider start; disposal order (§9.4) |
| Stage spans from `EnterStage` | `EnterStage` is static; generate stages repeat the video phases; queue would start before its job span | After the fact, Studio's own stages, job span from `CreatedAt` (§9.2) |
| Render stage spans | Names not given | `starting`, `rendering`, `validating` (§9.2) |
| `traceparent` on the record | Record fields are not serialized | Public properties (§9.1) |
| Metrics by kind and outcome | Running includes cancelling; queued-only jobs have no start | §9.5 |
| `StudioJobSummary.Prompt` excluded | Prompts, take names and error messages also exist | All excluded (§9.6) |
| **Application Insights** | | |
| Decide sampling later | The exporter installs a 5 traces/s sampler on the whole provider and the SDK then ignores `OTEL_TRACES_SAMPLER` | Mapping (§4.4) |
| Live Metrics on | Not supported by per-signal exporters | Off (§10.2) |
| Log sampling unmentioned | Logs of unsampled traces are dropped | Documented (§10.3) |
| Confirm count scaling later | Fixed percentage scales; rate-limited uses a span attribute | §10.3, §12.4 |
| Distro "does not combine with OTLP" | Microsoft documents both together; the real reasons are duplicated instrumentation and pipeline ownership | §10.1 |
| Mapping table | Span events become `traces` rows; kind decides InProc | §10.4 |
| p95 TTFT from prefill durations | Prefill leaves out rendering and queueing | `tensorsharp.ttft_ms` (§10.7) |
| Collector `azuremonitor` exporter | Renamed `azure_monitor` | §10.8 |
| Entra ID with only the exporter package | `DefaultAzureCredential` is in `Azure.Identity` | Added (§11) |
| Offline storage "document where" | On by default under `Microsoft\AzureMonitor` | §10.5 |
| Statsbeat unmentioned | The exporter sends SDK health statistics | §10.5 |
| **Verification** | | |
| Aspire API on by default? | The docs disagree | The script turns it on explicitly; pinned 13.6.2 (§12.3) |
| Histogram counts equal requests | Only `request.duration` covers every request | §6.1, §12.3 |
| Short bursts move the gauges | Default metric export interval is 60 s | Short intervals in the script (§12.3) |
| Promises without checks | Many sections had none | Script checks plus §12.1's gap list |

## Appendix B. Not verified

- The end-to-end runs (§12.5) cover one machine: Windows 11 on Arm with an NVIDIA GPU, the `ggml_cuda` and `ggml_cpu`
  backends, and the models listed there. Not run end to end: x64, Linux and macOS, the `cuda`, `ggml_vulkan`,
  `ggml_metal` and `mlx` backends, tensor parallelism, the CLI (unit tests only), other video and image models
  (Wan, Qwen-Image), DiffusionGemma, Jev, and an OpenTelemetry Collector between TensorSharp and a backend (the
  `examples/opentelemetry/collector` configurations were not started).
- The Azure Monitor exporter facts were checked against 1.7.0's source and documentation and spot-checked in 1.10.0's
  source and documentation (Live Metrics, sampler options, trace-based log sampling, the metric reader's fixed
  interval). Count scaling under rate-limited sampling and offline storage were not exercised: every run sampled 100%
  and ingestion was reachable.
- The GenAI semantic-convention names are at development status and may change (§13).
- Performance (§12.6): one small model on one GPU; no video, long-context, other-backend or other-machine measurement.
