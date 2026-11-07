# OpenTelemetry examples

TensorSharp can export traces, metrics and logs with OpenTelemetry: over OTLP to any collector or
dashboard, directly to Azure Application Insights, or both at once. It is off by default and costs
nothing until something asks for it. This directory has ready-made backends to point it at, and
queries for Application Insights.

| Path | What it is for |
|---|---|
| [`aspire/docker-compose.yml`](aspire/docker-compose.yml) | The standalone Aspire dashboard, pinned to 13.6.2, with OTLP API-key authentication and the telemetry HTTP API on. The quickest way to look at a trace. |
| [`aspire/.env.example`](aspire/.env.example) | The two keys the compose file needs. Copy it to `.env` (git-ignored). |
| [`collector/docker-compose.yml`](collector/docker-compose.yml) | An OpenTelemetry Collector with Jaeger, Prometheus and Grafana behind it. |
| [`collector/collector.yaml`](collector/collector.yaml) | The collector configuration for that stack: OTLP in; traces to Jaeger, metrics to Prometheus, logs to the collector's console. |
| [`collector/prometheus.yml`](collector/prometheus.yml) | Prometheus scraping the collector's metrics endpoint. |
| [`collector/grafana/provisioning/datasources/datasources.yaml`](collector/grafana/provisioning/datasources/datasources.yaml) | Grafana's Prometheus and Jaeger data sources. |
| [`collector/collector-azuremonitor.yaml`](collector/collector-azuremonitor.yaml) | A collector that forwards all three signals to Application Insights with the contrib Azure Monitor exporter. |
| [`application-insights/queries.kql`](application-insights/queries.kql) | Commented KQL for request and failure rates, time to first token, queue/prefill/decode time, token throughput, video phases, Studio jobs, model loads and logs per request. |

The commands below run from the repository root unless they `cd` somewhere. Each backend
publishes OTLP on the standard ports 4317 (gRPC) and 4318 (HTTP), so run one at a time.

## Turning telemetry on

Telemetry is configured with the standard `OTEL_*` environment variables, plus a few of
TensorSharp's own. It turns on when any of these is set:

| Setting | Effect |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT=<url>` | OTLP for all three signals. `--otel-endpoint <url>` on the server or CLI sets the same variable (and wins over an inherited value), so a `--config` file can carry it. |
| `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_ENDPOINT=<url>` | OTLP for one signal, to its own endpoint. |
| `APPLICATIONINSIGHTS_CONNECTION_STRING=<connection string>` | Every enabled signal also goes to Application Insights, from inside the process. |
| `TENSORSHARP_OTEL=1` | OTLP to the SDK's default endpoint for the protocol: `http://localhost:4317` for `grpc`, `http://localhost:4318` for `http/protobuf`. |
| `OTEL_{TRACES,METRICS,LOGS}_EXPORTER=otlp,console,none` | Chooses one signal's sinks; `otlp` or `console` turns telemetry on, `none` turns that signal off everywhere, Application Insights included. |

Other settings:

| Setting | Effect |
|---|---|
| `OTEL_SDK_DISABLED=true` | Off, whatever else is set. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` (default) or `http/protobuf`. Checked only when a signal exports over OTLP, so a value meant for another program does not stop TensorSharp when its telemetry is off or goes to Application Insights only. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Headers on every export, e.g. `x-otlp-api-key=<key>` for the Aspire dashboard. |
| `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES` | Override the service name (default `tensorsharp-server`, or `tensorsharp-cli`) and add resource attributes. `service.instance.id` is `machine:pid`. |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | Trace sampling; default is everything. With Application Insights on, the value is mapped onto the Azure Monitor sampler (see below). |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` | Export prompts and outputs. Off by default; see [Privacy](#privacy). |
| `TENSORSHARP_AZURE_MONITOR_AAD=1` | Authenticate to Application Insights with Entra ID instead of the connection string's key. |
| `TENSORSHARP_OTEL_LOG_LEVEL=<level>` | Minimum level of exported logs (`Information`, `Warning`, ...). Console and file logs keep their own level. |
| `TENSORSHARP_OTEL_TRACE_STEPS=1` | One `inference.step` span per engine step, for profiling a single run. Noisy; leave it off in production. |

The server and the CLI print one startup line that starts with `Telemetry: `, for example

```text
Telemetry: traces(otlp) metrics(otlp) logs(otlp) -> OTLP http://localhost:4317 (grpc); service tensorsharp-server; content capture off; sampler parentbased_always_on
```

or `Telemetry: off (...)` with the reason. It never shows OTLP headers or the connection string.
The settings are read in [`TensorSharp.Telemetry/TelemetryOptions.cs`](../../TensorSharp.Telemetry/TelemetryOptions.cs)
and the SDK is wired up in [`TensorSharpTelemetrySetup.cs`](../../TensorSharp.Telemetry/TensorSharpTelemetrySetup.cs).
Span and metric names are in [`TelemetryAttributes.cs`](../../TensorSharp.Runtime/Telemetry/TelemetryAttributes.cs)
and [`TensorSharpMetrics.cs`](../../TensorSharp.Runtime/Telemetry/TensorSharpMetrics.cs).

## 1. Aspire dashboard on this machine

The [Aspire dashboard](https://aspire.dev/dashboard/standalone/) shows traces, metrics and
structured logs in one browser page and needs nothing but Docker. Pick a random key of at least
128 bits (`openssl rand -hex 32`) and start it with OTLP authentication on:

```sh
docker run --rm -d --name aspire-dashboard -p 127.0.0.1:18888:18888 -p 127.0.0.1:4317:18889 -p 127.0.0.1:4318:18890 -e Dashboard__Otlp__AuthMode=ApiKey -e Dashboard__Otlp__PrimaryApiKey=<key> mcr.microsoft.com/aspire/dashboard:13.6.2
```

The container listens for OTLP/gRPC on 18889 and OTLP/HTTP on 18890; the line maps them to the
standard 4317 and 4318. `docker logs aspire-dashboard` prints a login URL ending in
`login?t=<token>`; open it as `http://localhost:18888/login?t=<token>`.

Then run TensorSharp with the endpoint and the key (bash):

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
OTEL_EXPORTER_OTLP_HEADERS=x-otlp-api-key=<key> \
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda
```

PowerShell:

```powershell
$env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4317"
$env:OTEL_EXPORTER_OTLP_HEADERS = "x-otlp-api-key=<key>"
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda
```

`TENSORSHARP_OTEL=1` instead of the endpoint variable sends to the same `localhost:4317`. The CLI
takes the same variables, or `--otel-endpoint http://localhost:4317`, and reports as
`tensorsharp-cli`.

The compose file in [`aspire/`](aspire/) starts the same dashboard and also turns on its telemetry
HTTP API, which lets scripts read telemetry back:

```sh
cd examples/opentelemetry/aspire
cp .env.example .env          # PowerShell: Copy-Item .env.example .env; then put two random keys in it
docker compose up -d
docker compose logs aspire-dashboard          # the login?t= line
curl -H "x-api-key: <ASPIRE_API_KEY>" http://localhost:18888/api/telemetry/resources
curl -H "x-api-key: <ASPIRE_API_KEY>" "http://localhost:18888/api/telemetry/traces?limit=5"
```

The API answers `GET /api/telemetry/resources`, `/spans`, `/logs`, `/traces` and
`/traces/{traceId}` in OTLP JSON. The compose file refuses to start until both keys are in
`.env`. Do not use `ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS=true` to skip the login: it turns
off OTLP and API authentication too.

The dashboard keeps telemetry in a temporary SQLite database that is deleted when the container
stops. Everything shown is gone after `docker stop` or `docker compose down`.

## 2. Your own collector: Jaeger, Prometheus and Grafana

[`collector/`](collector/) runs the OpenTelemetry Collector (contrib distribution) with a trace
store, a metrics store and a dashboard behind it:

```sh
cd examples/opentelemetry/collector
docker compose up -d
```

Run TensorSharp with `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317` (no headers: this
stack does not authenticate, so every port is bound to `127.0.0.1`). Then:

- **Traces**: Jaeger at <http://localhost:16686>. Pick the service `tensorsharp-server`. To find
  the trace of one request, search by the tag `tensorsharp.request_id=<id>`, the id the server
  returns in the `X-TensorSharp-Request-Id` header.
- **Metrics**: Prometheus at <http://localhost:9090>, and Grafana at <http://localhost:3000>
  (`admin` / `admin` on first login) with Prometheus and Jaeger already added as data sources.
  The exporter's default translation turns dots into underscores and adds unit suffixes, for
  example:

  ```promql
  # p95 time to first token per model
  histogram_quantile(0.95, sum by (le, gen_ai_request_model) (rate(gen_ai_server_time_to_first_token_seconds_bucket[5m])))
  # tokens per second, input and output
  sum by (tensorsharp_token_type) (rate(tensorsharp_tokens_total[1m]))
  ```

- **Logs**: `docker compose logs -f otel-collector`. The debug exporter prints one line per log
  record. Replace it with your log store's exporter in `collector.yaml`.

To use a collector you already run, copy the `receivers`, `processors` and `pipelines` you need
from [`collector/collector.yaml`](collector/collector.yaml). TensorSharp only needs the OTLP
receiver.

## 3. Application Insights

### Directly from TensorSharp

The simplest path is the in-process Azure Monitor exporter. Set the connection string from the
Application Insights resource's Overview page:

```sh
APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=https://...;LiveEndpoint=https://...;ApplicationId=..." \
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda
```

- **Entra ID.** With `TENSORSHARP_AZURE_MONITOR_AAD=1` the exporter authenticates with
  `DefaultAzureCredential` (your `az login` locally, a managed identity in Azure). The identity
  needs the Monitoring Metrics Publisher role on the Application Insights resource. The connection
  string is still required: it names the resource and its ingestion endpoint.
- **Sampling.** Every trace is kept by default. `OTEL_TRACES_SAMPLER=traceidratio` (or
  `parentbased_traceidratio`) with `OTEL_TRACES_SAMPLER_ARG=0.25` keeps 25%;
  `microsoft.rate_limited` with an argument in traces per second is honoured too. The Azure
  sampler decides for the whole process, so OTLP gets the same traces when both are on.
- **Together with OTLP.** Setting both the connection string and an OTLP endpoint sends every
  signal to both, for example the Aspire dashboard during development and Application Insights
  for history.
- **Live Metrics** is not available: the per-signal exporters TensorSharp uses do not support it.

[`eng/telemetry/appinsights-e2e.ps1`](../../eng/telemetry/appinsights-e2e.ps1) checks this path end
to end against a real resource, reading the connection string with the Azure CLI at run time; see
[`eng/telemetry/README.md`](../../eng/telemetry/README.md#application-insights-appinsights-e2eps1).

### Through a collector

[`collector/collector-azuremonitor.yaml`](collector/collector-azuremonitor.yaml) receives OTLP
and forwards all three signals with the contrib `azure_monitor` exporter. It reads the
connection string from the collector's own environment:

```sh
cd examples/opentelemetry/collector
export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=https://..."
docker run --rm -p 127.0.0.1:4317:4317 -p 127.0.0.1:4318:4318 -e APPLICATIONINSIGHTS_CONNECTION_STRING -v "$PWD/collector-azuremonitor.yaml:/etc/otelcol-contrib/config.yaml:ro" otel/opentelemetry-collector-contrib:0.164.0
```

Point TensorSharp at it with `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317` and do not set
`APPLICATIONINSIGHTS_CONNECTION_STRING` on TensorSharp as well, or every item arrives twice. The
configuration converts TensorSharp's cumulative OTLP metrics to the deltas Application Insights
expects.

### Where the data lands

| OpenTelemetry | Application Insights |
|---|---|
| The ASP.NET Core server span of each HTTP request | `requests` |
| TensorSharp's own spans (all `ActivityKind.Internal`) | `dependencies`, type `InProc` |
| Log records | `traces` |
| Logged exceptions | `exceptions` |
| Metrics | `customMetrics` |
| `service.name` | `cloud_RoleName` |
| `service.instance.id` (`machine:pid`) | `cloud_RoleInstance` |

Three things to know before writing queries:

- Histograms arrive as count, sum, min and max, without buckets, so a metric gives averages but
  not percentiles. Percentiles come from span durations, or from span attributes such as
  `tensorsharp.ttft_ms` on the chat span.
- Span attributes are in `customDimensions`. The in-process exporter stores numbers there as
  strings (use `todouble()`); the collector's exporter stores them as numbers in
  `customMeasurements`. The queries read both.
- Studio jobs are traces of their own, so their spans have no row in `requests`; each job links
  to the request that queued it (`customDimensions["_MS.links"]`).

[`application-insights/queries.kql`](application-insights/queries.kql) has the queries. Run them
from the Logs page of the Application Insights resource, which uses the table and column names
above; the Log Analytics workspace stores the same data under other names (`AppRequests`,
`AppDependencies`, ...).

## What you will see

A chat request (`POST /v1/chat/completions`, Ollama's `/api/chat/ollama`, or the Web UI's
`/api/chat`):

```text
POST /v1/chat/completions              server span  (App Insights: requests)
└─ chat gemma-4-E4B-it-Q8_0            gen_ai.operation.name, gen_ai.request.model, gen_ai.provider.name=tensorsharp,
   │                                   gen_ai.usage.input_tokens / output_tokens / cache_read.input_tokens,
   │                                   tensorsharp.ttft_ms, tensorsharp.protocol, tensorsharp.request_id, tensorsharp.outcome
   ├─ prompt.render                    chat template rendering
   ├─ inference.queue                  submitted until first scheduled
   ├─ inference.prefill                until the first token; tensorsharp.compute.prefill_ms
   └─ inference.decode                 the rest; tensorsharp.compute.decode_ms
```

Ollama's `/api/generate` names the span `text_completion <model>`
(`gen_ai.operation.name=text_completion`). Skill and agent turns add one chat span per round
(`tensorsharp.round`), one `execute_tool <tool>` span per tool call, and `codeexec.run` when a
tool runs code. Embeddings are `embeddings <model>`, images `generate_image <model>`, DiffusionGemma
adds `diffusion.denoise`, and loading a model is `model.load`.

A MiniMax-H3 or LTX 2.5 clip (`POST /v1/videos/generations`, `/api/video-generate`, a Studio job
or the CLI):

```text
generate_video <model>                 gen_ai.request.model, tensorsharp.video.mode / width / height / frames / steps, tensorsharp.outcome
├─ video.queue                         wait for the one-clip-at-a-time video lock
├─ lora.apply                          when LoRA adapters change
├─ reference-encode | image-encode     reference or image conditioning, when used
├─ text-encode                         MiniMax-H3: text-encoder, prefault-wait, dit-load, image-encode, reference-encode stages
├─ denoise                             LTX 2.5: first-pass, upscale, refinement (or single-pass); one "step" event per step
├─ vae-decode                          MiniMax-H3: denoiser-release, vae-open, vae-decode stages
├─ audio-decode
└─ video.export
```

The model name is the loaded model's file name without its extension (for example
`minimax_h3_fl2va_pruned-Q4_K` or `ltx-2.5-22b-distilled-UD-Q4_K_XL`), the same in the server and
the CLI. Phase spans carry `tensorsharp.video.phase`, stage spans
also `tensorsharp.video.stage`. A Studio job is its own trace, `studio.job generate` or
`studio.job render`, with a `studio.queue` child and its stages (`studio.stage encoding`, ...),
and the generation's spans nested inside.

Metrics come from the meters `TensorSharp` and `TensorSharp.Studio`, plus .NET's `System.Runtime`,
`System.Net.Http` and ASP.NET Core's meters: request duration, time to first token and time per
output token (`gen_ai.server.*`), token counters, engine queue depth, steps, batch size and KV
cache use, speculative decoding, model load, tool, video, image and Studio job durations, and GPU
memory. The full list is in [`TensorSharpMetrics.cs`](../../TensorSharp.Runtime/Telemetry/TensorSharpMetrics.cs).
Health probes and the Web UI's `/api/queue/status` polling get no spans.

## Privacy

Content stays out of exported telemetry by default. With
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` unset:

- Spans carry ids, model names, sizes, counts, timings and outcomes, never prompts, model output,
  tool arguments or results, or video prompts. Failures are recorded by exception type, not
  message.
- In exported log records every content value (prompts, chat messages, output, uploaded file
  names, paths and URLs, tool arguments, commands) reads `[redacted: N chars]`, and a record that
  had content is sent as its message template instead of the rendered line. Log scopes are not
  exported at all: ASP.NET Core's scope holds the raw request path. A record's trace and span ids
  lead to the spans with the model, backend and session, and `tensorsharp.request_id` is added to
  it directly.
- Some URLs need no sign-in and are guarded only by an id in their path: uploads and generated
  files (`/uploads/<file>`), a code-execution run's files (`/api/code/artifacts/<run>/...`) and a
  stored Responses API response (`/v1/responses/<id>`). Their HTTP spans use `/uploads/{file}`,
  `/api/code/artifacts/{run}[/{path}]` and `/v1/responses/{id}` in `url.path` and the span name, the
  request log lines' path is redacted, and a stored response's id is not put on its span.
  Exception messages have paths under the upload directory (and, with `--code-exec`, the
  code-execution directories) replaced by `<uploads>`, and upload URLs by `/uploads/{file}`.
- Console and file logs on the machine are unchanged.

Setting `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` exports all of it, including
the real request paths and response ids, and `gen_ai.input.messages` and `gen_ai.output.messages` on
chat spans. Only do that when every place
the telemetry goes (the dashboard, the collector's backends, the Application Insights resource and
its retention) is allowed to hold your users' prompts.

If telemetry is part of validation evidence, keep exported logs, reports and screenshots in the
git-ignored `docs/validation/` or `artifacts/`, as for every other validation run.

## Pinned versions

Checked on 2026-11-06 against the registries and release pages.

| Image | Tag |
|---|---|
| `mcr.microsoft.com/aspire/dashboard` | `13.6.2` |
| `otel/opentelemetry-collector-contrib` | `0.164.0` |
| `jaegertracing/jaeger` | `2.22.0` |
| `prom/prometheus` | `v3.15.1` |
| `grafana/grafana` | `13.3.0` |

The collector files use the component names of 0.164.0: `otlp_grpc`, `azure_monitor` and
`cumulative_to_delta` (older releases call them `otlp`, `azuremonitor` and `cumulativetodelta`).
