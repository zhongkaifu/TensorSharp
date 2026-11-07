<!--
Copyright (c) Zhongkai Fu. All rights reserved.
https://github.com/zhongkaifu/TensorSharp

This file is part of TensorSharp.

TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
-->

# OpenTelemetry end-to-end check

Two scripts check TensorSharp.Server.Host's telemetry end to end, each against a real backend:

- `aspire-e2e.ps1`: the OTLP path, against a local Aspire dashboard. Everything above the
  Application Insights section is about this script.
- `appinsights-e2e.ps1`: the in-process Application Insights path
  (`APPLICATIONINSIGHTS_CONNECTION_STRING`), against a real resource. See
  [Application Insights](#application-insights-appinsights-e2eps1).

`aspire-e2e.ps1` checks the whole OpenTelemetry path of TensorSharp.Server.Host against a
standalone [Aspire dashboard](https://aspire.dev/dashboard/standalone/): it starts the dashboard,
starts the server several times with different telemetry settings, sends real requests, and then
reads what the dashboard received through its telemetry HTTP API. It checks the span trees, the
token counts, the request ids on logs, what must not be traced, and what must not be exported.

Each check ends as PASS, FAIL, SKIPPED or MANUAL. SKIPPED means its model or setup was not given;
MANUAL means only a person looking at the dashboard UI can confirm it. Neither counts as passed.
The script exits with 1 when anything FAILED, and with 0 otherwise.

## What you need

- PowerShell 7.2 or later, on Windows, Linux or macOS. No extra modules.
- A built TensorSharp.Server.Host. By default the script runs the newest
  `TensorSharp.Server.Host.dll` under `TensorSharp.Server.Host/bin` with `dotnet`; pass
  `-ServerHost` for another build (the .dll, the executable, or a folder to search).
- A dashboard, one of:
  - Docker. The script runs `mcr.microsoft.com/aspire/dashboard:13.6.2` and pulls it if it is
    missing.
  - An `Aspire.Dashboard.dll` (`-DashboardDll`), run with `dotnet`.
  - A dashboard that is already running (`-SkipDocker -DashboardUrl ... -DashboardApiKey ...
    -OtlpApiKey ...`).
- Free local ports: 5055 for the server (`-Port`), and 18888, 4317 and 4318 for the dashboard
  (`-DashboardUiPort`, `-OtlpGrpcPort`, `-OtlpHttpPort`).
- Models for the scenarios you want. Without any model the script still runs the model-free
  checks (startup line, resource attributes, request spans and logs, filtered paths and the
  negative runs).

## Running it

```powershell
# Everything a chat model can cover, over gRPC and HTTP/protobuf, plus the negative runs.
./eng/telemetry/aspire-e2e.ps1 -ChatConfig D:\models\qwen3-4b.json

# A model file, or server arguments in one string, work too.
./eng/telemetry/aspire-e2e.ps1 -ChatConfig "--model D:\models\gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda"

# Only some scenarios.
./eng/telemetry/aspire-e2e.ps1 -ChatConfig D:\models\qwen3-4b.json -Scenarios llm-openai,llm-cancel

# Video, Studio and embeddings.
./eng/telemetry/aspire-e2e.ps1 -LtxConfig D:\models\ltx25.json -MiniMaxH3Config D:\models\minimax-h3.json `
    -EmbeddingConfig D:\models\snowflake-arctic-embed-l-v2.0-q8_0.gguf

# Keep the dashboard afterwards to look at the data (the login link is in dashboard/login.txt).
./eng/telemetry/aspire-e2e.ps1 -ChatConfig D:\models\qwen3-4b.json -KeepDashboard
```

`-ChatConfig`, `-EmbeddingConfig`, `-LtxConfig` and `-MiniMaxH3Config` each take a `--config`
JSON file, a model file (passed as `--model`; an embedding GGUF also gets `--embeddings`), or a
string of server arguments that starts with `-`. Relative paths resolve against the current
folder, which is also the server's working folder. `-ExtraServerArgs "--backend ggml_cuda"` is
added to every server command line. With `pwsh -File`, pass lists as one comma-separated string
(`-Scenarios "llm-openai,llm-cancel"`).

The evidence goes to `docs/validation/otel-e2e-<yyyyMMdd-HHmmss>/` unless `-OutDir` names another
fresh folder under `docs/validation/` or `artifacts/`. Both are git-ignored; the script refuses
any other place.

## What it does

1. Starts the dashboard with OTLP API-key authentication
   (`Dashboard__Otlp__AuthMode=ApiKey`, `Dashboard__Otlp__PrimaryApiKey`) and the telemetry API
   on (`Dashboard__Api__Enabled=true`, `Dashboard__Api__Disabled=false`,
   `Dashboard__Api__AuthMode=ApiKey`, `Dashboard__Api__PrimaryApiKey`). Both keys are random for
   each run. The container publishes 18888 (UI and API), 18889 (OTLP/gRPC) and 18890 (OTLP/HTTP)
   on loopback as 18888, 4317 and 4318. With `-DashboardDll` the same settings, plus the endpoint
   settings, are environment variables. `ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS` is never set,
   because it would also turn OTLP authentication off.
2. Waits until the API answers, and saves the `login?t=` link from the dashboard log to
   `dashboard/login.txt`.
3. Starts TensorSharp.Server.Host once per run (below), bound to 127.0.0.1. Every inherited
   `OTEL_*`, `TENSORSHARP_OTEL*`, `APPLICATIONINSIGHTS_CONNECTION_STRING`, `PORT`, `HOST` and
   `ASPNETCORE_URLS` variable is removed first, so only what the run sets applies. A run sets
   `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
   `OTEL_EXPORTER_OTLP_HEADERS=x-otlp-api-key=<key>`, a unique `OTEL_SERVICE_NAME`
   (`ts-e2e-<tag>-<run>`), `TENSORSHARP_LOG_DIR` and `TS_STUDIO_DIRECTORY` in the run's evidence
   folder, and short export intervals (`OTEL_BSP_SCHEDULE_DELAY=1000`,
   `OTEL_BLRP_SCHEDULE_DELAY=1000`, `OTEL_METRIC_EXPORT_INTERVAL=5000`). Server output goes to
   files; the script waits for `/health`.
4. Checks the `Telemetry: ` startup line, sends the run's requests, and reads the result back from
   the dashboard.
5. Saves a snapshot of everything the dashboard holds, stops the servers and the dashboard (its
   data is deleted when it stops), and writes the result files.

Every prompt contains a marker (`TSOTEL` and twelve hex digits, in `environment.json`), so the
script can prove that no exported log or span carries prompt text.

### Server runs

| Run | Telemetry | Model | Scenarios |
| --- | --- | --- | --- |
| `otlp-grpc` | OTLP/gRPC to 4317 | chat model if an `llm-*` scenario runs | `llm-*`, `queue-status-poll`, `engine-step-spans` |
| `otlp-http` | OTLP `http/protobuf` to 4318 | same | `llm-*`, `queue-status-poll` |
| `content-capture` | `--otel-endpoint` flag, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` | chat | `content-capture` |
| `embeddings` | OTLP/gRPC | embedding | `embeddings` |
| `video-ltx` | OTLP/gRPC | LTX 2.5 | `video-ltx`, `video-cancel`, `studio-generate` |
| `video-minimax-h3` | OTLP/gRPC | MiniMax-H3 | `video-minimax-h3` (and `video-cancel`, `studio-generate` when no LTX) |
| `negative-off` | no telemetry variables at all | none | `negative-off` |
| `negative-sdk-disabled` | OTLP plus `OTEL_SDK_DISABLED=true` | none | `negative-sdk-disabled` |
| `negative-wrong-key` | OTLP/gRPC with a wrong `x-otlp-api-key` | chat, if given | `negative-wrong-key` |

A run starts only when one of its scenarios is selected and runnable. `-Protocols grpc` leaves
out the HTTP/protobuf run.

### Scenarios

| Scenario | Needs | What it checks |
| --- | --- | --- |
| `llm-openai` | chat | `POST /v1/chat/completions`. The server span carries `tensorsharp.request_id` = the `X-TensorSharp-Request-Id` header and `tensorsharp.protocol=openai`. Under it, `chat <model>` has `gen_ai.provider.name=tensorsharp`, `tensorsharp.outcome=completed`, a positive `tensorsharp.ttft_ms`, and the children `prompt.render`, `inference.queue`, `inference.prefill`, `inference.decode` (or `diffusion.denoise` for DiffusionGemma). `gen_ai.usage.input_tokens`, `output_tokens` and `cache_read.input_tokens` equal the response's usage block. Every child lies inside its parent. The log records of the trace carry the same `tensorsharp.request_id`. |
| `llm-openai-stream` | chat | The same for a streamed response; usage comes from the last chunk. |
| `llm-ollama-generate` | chat | `POST /api/generate`: a `text_completion <model>` span, `tensorsharp.protocol=ollama`, tokens equal `prompt_eval_count` / `eval_count`. |
| `llm-ollama-chat` | chat | `POST /api/chat/ollama`: a `chat <model>` span with the Ollama counts. |
| `llm-webui` | chat | `POST /api/chat` (server-sent events): `tensorsharp.protocol=webui`, tokens equal the done frame's `promptTokens` / `tokenCount` / `kvReusedTokens`. |
| `llm-client-traceparent` | chat | Sends a W3C `traceparent`; the server span must be in that trace, under that span. |
| `llm-cancel` | chat | Asks for a long reply with `tool_choice: "none"` (no skill rounds, so the reply streams from its first token) and closes the connection after three content chunks, without draining the rest. The last round's chat span ends with `tensorsharp.outcome=cancelled` (an earlier tool-call round completed), and `/api/queue/status` shows nothing processing within 60 s. A reply that finished before the disconnect fails the check instead of passing it. |
| `llm-bad-request` | chat | A request without `messages` gets HTTP 400, a server span with that status and no chat span, and logs with its request id. |
| `llm-burst` | chat | Eight concurrent requests: distinct request ids, one trace each, each with a complete chat tree. |
| `queue-status-poll` | none | Polls `/api/queue/status` 20 times; no server span may exist for it. A traced control request afterwards must arrive, so "no span" means filtered, not lost. |
| `embeddings` | embedding | `POST /v1/embeddings` and `POST /api/embed`: an `embeddings <model>` span under each server span, with input tokens equal to the usage. |
| `video-ltx` | LTX | `POST /v1/videos/generations` (512x320, 17 frames by default): `generate_video <model>` under the server span, with `text-encode`, `denoise` (stages `first-pass`, `upscale`, `refinement`), `vae-decode` and `video.export`, all inside their parents. |
| `video-minimax-h3` | MiniMax-H3 | Two clips in a row (384x224, 22 frames by default), each with `text-encode`, `denoise`, `vae-decode` and `video.export`. The first clip's `dit-load` has `tensorsharp.video.cached=false`, the second `true`. If the first clip released the DiT (`denoiser-release`), the cached path was not exercised and the row is MANUAL. |
| `video-cancel` | LTX or MiniMax-H3 | Drops the `POST /api/video-generate/stream` progress stream at the first denoise event (or after `-VideoCancelAfterSec`). `generate_video` must end cancelled, the request must end within 600 s (the clip stops at its next token check), and the server must keep answering. The next scenario starts only after that. |
| `studio-generate` | LTX or MiniMax-H3 | A Video Studio job (`POST /api/studio`, `jobs/generate`). Its `studio.job generate` span starts its own trace, links to the request that queued it, has `studio.queue` and `studio.stage ...` children and `generate_video` under it. SKIPPED when Studio reports no available text-to-video model (for example without FFmpeg). Studio does not cache a media probe that ran out of time (its 10 s `ffprobe -version` limit can pass on a machine still busy with a cancelled clip), so such an answer is asked again up to three times, 15 s apart. |
| `engine-step-spans` | chat, `-TraceSteps` | With `TENSORSHARP_OTEL_TRACE_STEPS=1`: `inference.step` spans are roots, link to the request's trace, and carry `tensorsharp.engine.batch_size`. |
| `content-capture` | chat | With capture on, the prompt marker appears in the exported logs; whether it is also on spans is noted. This run also uses `--otel-endpoint` in place of the variable. |
| `negative-off` | none | The startup line says `Telemetry: off (...)`, and after 12 s the dashboard has no resource, span or log from this process (matched by `service.instance.id` = `<machine>:<pid>`). |
| `negative-sdk-disabled` | none | The line says `Telemetry: off (OTEL_SDK_DISABLED=true)`, and nothing arrives under the run's service name. |
| `negative-wrong-key` | none | The server keeps serving with a rejected key: its median latency stays within `-LatencyMarginPercent` (50) plus `-LatencyMarginMs` (500) of the baseline measured in the first OTLP run, and nothing arrives under its service name. |
| `metrics` | none | Always MANUAL: the dashboard's telemetry API has no metrics route. The row lists the instruments to look for in the dashboard's Metrics page. |

Every server run also reports these rows:

| Row | What it checks |
| --- | --- |
| `server-start` | Only appears as FAIL, when the server exited or never answered `/health`. |
| `startup-line` | The `Telemetry: ` line names OTLP for all three signals, the endpoint and protocol, the service name, and content capture off (ON in `content-capture`); or `off` for the negative runs. |
| `resource-attributes` | `service.name`, `service.instance.id` (`<machine>:<pid>`), `host.name` and `process.pid` match the process; `service.version` is present; `/api/telemetry/resources` lists the service. |
| `request-span-and-logs` | `GET /api/models` has a root server span with its request id and protocol, and its log records carry the request id. |
| `health-not-traced` | No server span for `/health`, which the script polled while the server started. |
| `content-redaction` | No exported log record or span of the run contains the prompt marker. SKIPPED when the run sent no prompt. |
| `otlp-key-not-logged` | The OTLP and API keys appear in none of the server's output or log files. |

## Results

`-OutDir` holds:

- `results.md` and `results.json`: one row per check (run, scenario, status, seconds, details),
  with counts. SKIPPED and MANUAL rows are not passes.
- `environment.json`: the TensorSharp commit, branch and number of uncommitted changes; the ggml
  revision from `ExternalProjects/ggml` (or why there is none); the server build that ran; the
  dashboard mode, image tag, image id and repository digest; OS, architecture, GPUs
  (`nvidia-smi`), .NET SDK and runtimes; and for every run its arguments, service name, startup
  line, loaded model, backend, architecture, `service.version`, process id and exit code.
- `runs/<run>/`: `server-launch.json` (arguments and variables, key redacted), the server's
  stdout and stderr, its file logs (`logs/`), every request and response (`requests/`), and the
  run's exported spans and logs (`telemetry/`, with one file per checked trace in
  `telemetry/traces/`).
- `dashboard/`: `login.txt`, the dashboard's own log, and the final snapshot of everything it
  received (`resources.json`, `spans-all.json`, `logs-all.json`). With `-KeepDashboard`, also
  `keys.json` with the API and OTLP keys.

`login.txt` and `keys.json` hold credentials that work until the dashboard stops. Like the rest
of the evidence, keep them out of commits.

## How the dashboard is read

The script uses only these routes, on the UI port with the `X-API-Key` header. They, and their
query parameters, were checked against [the dashboard API page](https://aspire.dev/dashboard/apis/),
[its configuration page](https://aspire.dev/dashboard/configuration/) and the dashboard's own
`DashboardEndpointsBuilder.cs` and `TelemetryApiService.cs` at tag v13.6.2:

| Route | Used for |
| --- | --- |
| `GET /api/telemetry/resources` | readiness, and which services sent anything |
| `GET /api/telemetry/spans?resource=<service>&limit=<n>` | finding a request's server span by `tensorsharp.request_id` |
| `GET /api/telemetry/traces/{traceId}` | the full span tree of one request |
| `GET /api/telemetry/logs?resource=<service>&traceId=<id>&limit=<n>` | a request's log records |

List routes return the newest `limit` items (200 when not given), wrapped as
`{ data: { resourceSpans | resourceLogs }, totalCount, returnedCount }` in OTLP JSON; the script
always passes `limit`. The dashboard stores every attribute value as a string, so numbers are
parsed from strings. An unknown resource is 404, which the script reads as "nothing arrived".

If the API answers 401 or 404 (wrong key, API disabled, or an older dashboard), the server checks
still run, and every telemetry check becomes MANUAL with what to look for in the UI. Rerun with
`-KeepDashboard` and open the link in `dashboard/login.txt` to do those by hand.

## Things to know

- The dashboard keeps telemetry only while it runs. The script saves the snapshot before it
  stops it; use `-KeepDashboard` to look around afterwards, and `docker rm -f <container>` (the
  name is printed) when done.
- On Windows the server is killed at the end of a run, after its telemetry has been read; on Linux
  and macOS it gets SIGTERM first. A file log's last lines can be missing on Windows.
- A `--config` file's `environment` object only fills variables the process does not already
  have, so it cannot override the script's settings. It can still add ones the script does not
  set, such as `APPLICATIONINSIGHTS_CONNECTION_STRING`; keep those out of the configs used here.
  The `negative-off` and `negative-sdk-disabled` runs load no model and no config.
- LTX needs width and height divisible by 64 and 8k+1 frames; MiniMax-H3 needs multiples of 32
  and 17k+5 frames at a fixed 24 fps. Change `-LtxSize`, `-LtxFrames`, `-MiniMaxH3Size` and
  `-MiniMaxH3Frames` together with that in mind. A single-stage LTX setup shows `single-pass`
  under `denoise`; pass `-LtxExpectStages single-pass` for it.
- `-ContainmentToleranceMs` (5) is how far a child span may stick out of its parent. Spans built
  after the fact from recorded timestamps can be a few microseconds off; the LLM phase spans are
  clamped to start no earlier than their chat span. For
  `video-cancel`, a child that outlives the server span is noted, not failed: that stream's
  handler returns before the generation unwinds.
- `aspire-e2e.ps1` checks the OTLP path only. Application Insights
  (`APPLICATIONINSIGHTS_CONNECTION_STRING`) is covered by `appinsights-e2e.ps1`, below.

## Application Insights (`appinsights-e2e.ps1`)

`appinsights-e2e.ps1` checks the in-process Azure Monitor exporters against a real Application
Insights resource. It reads the resource's connection string with the Azure CLI, starts the server
several times with `APPLICATIONINSIGHTS_CONNECTION_STRING` set, sends the same kinds of requests as
`aspire-e2e.ps1`, stops each server gracefully so its exporters flush, and then queries the
resource through the Application Insights query API until the rows arrive. It uses the same
PASS / FAIL / SKIPPED / MANUAL rules, result files and exit code as `aspire-e2e.ps1`.

### What you need

- PowerShell 7.2 or later and a built TensorSharp.Server.Host, as above.
- The Azure CLI (`az`), signed in (`az login`) as an identity that can read the resource: it reads
  `properties.ConnectionString` and `properties.AppId` with `az resource show`, and queries with
  `az rest --resource https://api.applicationinsights.io`. Reader or Monitoring Reader on the
  resource (or its resource group) is enough. Only the public Azure cloud's query endpoint
  (`api.applicationinsights.io`) is used.
- For `-EntraId`: the identity `DefaultAzureCredential` picks inside the server process (normally
  the same `az login`; `AZURE_*` variables are left in the server's environment) needs the
  **Monitoring Metrics Publisher** role on the resource. A role assignment can take several minutes
  to apply; until then the run's data is simply missing.
- If the resource has local authentication disabled, only the Entra ID run can ingest; every other
  run then fails with no data.
- Models are optional, as with `aspire-e2e.ps1`.

### Running it

```powershell
$id = '<the ARM id of the microsoft.insights/components resource>'

# A chat model: the main run (every llm-* scenario and queue-status-poll) and content-capture.
./eng/telemetry/appinsights-e2e.ps1 -AppInsightsResourceId $id -ChatModel D:\models\qwen3-4b-Q8_0.gguf

# Add the Entra ID run, and also send OTLP to an Aspire dashboard if one listens on 4317.
./eng/telemetry/appinsights-e2e.ps1 -AppInsightsResourceId $id -ChatModel D:\models\qwen3-4b-Q8_0.gguf `
    -EntraId -BothSinks -OtlpApiKey <dashboard OTLP key> -DashboardApiKey <dashboard API key>

# Video and Studio.
./eng/telemetry/appinsights-e2e.ps1 -AppInsightsResourceId $id -LtxConfig D:\models\ltx25.json `
    -MiniMaxH3Config D:\models\minimax-h3.json -Scenarios video-ltx,video-minimax-h3,video-cancel,studio-generate,metrics
```

`-ChatModel`, `-EmbeddingModel`, `-LtxConfig` and `-MiniMaxH3Config` take a model file, a
`--config` JSON file, or server arguments in one string, like the `-*Config` parameters of
`aspire-e2e.ps1`. The evidence goes to `docs/validation/appinsights-e2e-<yyyyMMdd-HHmmss>/` unless
`-OutDir` names another fresh folder under `docs/validation/` or `artifacts/`.

### Secrets

- The connection string is read at run time and passed only through each server process's
  environment. It is never printed, logged or written: `server-launch.json` shows a placeholder,
  and every file the script writes, every result row and every console line is passed through a
  filter that replaces the connection string, its instrumentation key, the AppId, the resource id,
  the subscription id and any `-OtlpApiKey` / `-DashboardApiKey`.
- The `connection-string-not-logged` row fails if the server's own output contains the connection
  string or the instrumentation key (that copy is then masked), and the last row,
  `harness / evidence-scrub`, scans the whole evidence folder once more.
- The resource id never reaches the evidence; results name the resource by its name.

### Runs

Each run starts the server with every inherited `OTEL_*`, `TENSORSHARP_OTEL*`,
`APPLICATIONINSIGHTS_*`, `TENSORSHARP_AZURE_MONITOR_AAD`, `PORT`, `HOST` and `ASPNETCORE_URLS`
variable removed, then sets `APPLICATIONINSIGHTS_CONNECTION_STRING`, a unique
`OTEL_SERVICE_NAME` (`tensorsharp-e2e-<tag>-<run>`, which becomes `cloud_RoleName`), short export
intervals (`OTEL_BSP_SCHEDULE_DELAY=2000`, `OTEL_BLRP_SCHEDULE_DELAY=2000`,
`OTEL_METRIC_EXPORT_INTERVAL=5000`), and `TENSORSHARP_LOG_DIR` and `TS_STUDIO_DIRECTORY` in the
run's folder. No sampler variable is set, so the sampler is Azure Monitor's at 100%.

| Run | Adds | Model | Scenarios |
| --- | --- | --- | --- |
| `main` | | chat model if an `llm-*` scenario runs | `llm-*`, `queue-status-poll` |
| `content-capture` | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` | chat | `content-capture` |
| `embeddings` | | embedding | `embeddings` |
| `video-ltx` | | LTX 2.5 | `video-ltx`, `video-cancel`, `studio-generate` |
| `video-minimax-h3` | | MiniMax-H3 | `video-minimax-h3` (and `video-cancel`, `studio-generate` when no LTX) |
| `entra-id` (with `-EntraId`) | `TENSORSHARP_AZURE_MONITOR_AAD=1` | chat when given, otherwise none | `entra-id` |

With `-BothSinks`, every run also gets `OTEL_EXPORTER_OTLP_ENDPOINT` (`-OtlpEndpoint`, default
`http://localhost:4317`), `OTEL_EXPORTER_OTLP_PROTOCOL` and, with `-OtlpApiKey`,
`OTEL_EXPORTER_OTLP_HEADERS=x-otlp-api-key=<key>`, but only when something listens on that
endpoint; otherwise a `both-sinks` row is SKIPPED and the runs go to Application Insights only.

All runs are driven first and read back afterwards, so their ingestion delays overlap.

### Stopping the server

The exporters flush when the host shuts down, so each server is stopped, not killed. On Linux and
macOS it gets SIGTERM. On Windows the server is started with its own console without a window,
and a small helper (a second `pwsh`) attaches to that console and raises CTRL_BREAK, which the
host's console lifetime handles as a stop request (SIGQUIT). The script first waits `-SettleSec`
(8) seconds after the run's last request, then up to `-StopGraceSec` (90) seconds for the exit.
The `server-stop` row is PASS only for an exit by itself with code 0; a kill is a FAIL, because
whatever was still queued in the exporters may be missing.

### How Application Insights is read

Every query goes through
`az rest --method post --url https://api.applicationinsights.io/v1/apps/<AppId>/query --resource https://api.applicationinsights.io --body @<file>`,
with `{"query": ..., "timespan": "P2D"}` in a temporary UTF-8 file that is deleted afterwards.
`az` is always run from PowerShell, never through a POSIX shell, which would rewrite the leading
`/subscriptions` of the resource id as a path. Before any server starts, `print ok = 1` checks
that the query API is readable.

Every query is filtered with `cloud_RoleName == "<the run's service name>"` and
`timestamp > <the run's start>`. Per run the script asks for:

| Query | Columns it reads |
| --- | --- |
| `requests` | `id`, `name`, `url`, `resultCode`, `itemCount`, `operation_Id`, `operation_ParentId`, `customDimensions["tensorsharp.request_id"]` and `["tensorsharp.protocol"]` |
| `dependencies` | the span tree (`id`, `operation_ParentId`, `operation_Id`, `name`, `type`, `itemCount`) and `customDimensions`: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.*` and `tensorsharp.ttft_ms` (strings, read with `todouble`), `tensorsharp.outcome`, `tensorsharp.internal`, `tensorsharp.video.*`, `tensorsharp.studio.job_id`, `_MS.links` |
| `traces` | row counts by `customDimensions["tensorsharp.request_id"]` and `operation_Id` |
| `metrics` | `customMetrics` rows by `name`, with the `tensorsharp.token.type` values |
| `markers` | rows of `requests`, `dependencies`, `traces` and `exceptions` whose name, url, data, message, exception text or `tostring(customDimensions)` contains the run's prompt marker |
| `role-instance` | row counts by `itemType` and `cloud_RoleInstance` |

A query is repeated, with backoff from 15 to 60 seconds, until every row the run expects is there
or `-IngestionTimeoutMinutes` (12) after the server stopped; ingestion usually takes 1 to 5
minutes. A check whose rows did not arrive in time is FAIL and names the saved query. The absence
checks (no `/health` row, no prompt marker) run once, after the presence checks: by then the rows
they look for would have arrived too. A query or permission error ends the polling at once.

### Checks

Every scenario row combines the HTTP side (status, usage block, stream) with what reached
Application Insights. For each request the run drove:

- a `requests` row whose `customDimensions["tensorsharp.request_id"]` is the
  `X-TensorSharp-Request-Id` header, for the driven path, with the response's `resultCode`, the
  expected `tensorsharp.protocol`, `itemCount` 1, and no parent (or, for
  `llm-client-traceparent`, the client's trace id as `operation_Id` and its span as
  `operation_ParentId`);
- for chat (`chat <model>`) and Ollama generate (`text_completion <model>`): a dependency with
  the same request id and `operation_Id`, under the `requests` row, of type `InProc`, with
  `gen_ai.provider.name=tensorsharp`, the expected `tensorsharp.outcome`, `itemCount` 1 (100%
  sampling), a positive `tensorsharp.ttft_ms`, and the children `prompt.render`,
  `inference.queue`, `inference.prefill` and `inference.decode` (dependencies whose
  `operation_ParentId` is its `id`); `gen_ai.usage.input_tokens`, `output_tokens` and
  `cache_read.input_tokens` equal the response's usage block. A cancelled stream needs only
  `inference.queue` and `tensorsharp.outcome=cancelled`; a 400 and `GET /api/models` must have no
  generation dependency; embeddings get an `embeddings <model>` dependency with the input tokens;
- `traces` rows carrying the same request id, in the request's operation, and none in that
  operation carrying another request id;
- for video: `generate_video <model>` under the `requests` row with `text-encode`, `denoise` (and
  under it LTX's `first-pass`, `upscale`, `refinement`) and `vae-decode`, and `video.export`;
  MiniMax-H3's `dit-load` with `tensorsharp.video.cached` false, then true; a cancelled clip ends
  cancelled; a Studio job is a `studio.job generate` operation of its own with `studio.queue`,
  `studio.stage ...` and `generate_video` under it and a `_MS.links` entry to the queuing request's
  operation;
- `llm-burst`: all eight requests arrive, each in its own operation (no rate-limited sampling);
- `queue-status-poll`: no `requests` row for `/api/queue/status`, while the control request after
  the polls arrives;
- `content-capture`: the prompt marker appears in `traces`; whether it is also on `dependencies` is
  noted;
- `entra-id`: the run's `requests` rows arrive at all, which proves token-based ingestion. With a
  chat model the run also sends one chat request, whose dependency tree, token counts, `traces`
  rows and TensorSharp `customMetrics` must arrive too: the trace, log and metric exporters each
  use the token. Lines of the server output that look like exporter or credential warnings are saved to
  `exporter-messages.txt` and quoted in the row. The exporter itself reports through EventSource;
  `-ExporterDiagnostics` writes an `OTEL_DIAGNOSTICS.json` into each run folder and starts the
  server there (string server arguments must then use absolute paths), so the SDK's
  self-diagnostics land in `otel-diagnostics/`.

Every run also reports:

| Row | What it checks |
| --- | --- |
| `server-start` | Only appears as FAIL, when the server exited or never answered `/health`. |
| `startup-line` | `traces(azuremonitor)`, `metrics(azuremonitor)`, `logs(azuremonitor)` (with `otlp` too under `-BothSinks`, and not otherwise), `-> Azure Monitor` (`Azure Monitor (Entra ID)` in the `entra-id` run only), the service name, content capture off (ON in `content-capture`), `sampler azuremonitor fixed_percentage(1)`, and no Live Metrics warning in the output. |
| `server-stop` | The server shut down by itself, with exit code 0 (see above). |
| `connection-string-not-logged` | Neither the connection string nor its instrumentation key is in the server's output or file logs. |
| `request-rows` | Every driven request has a `requests` row; lists the routes found. |
| `request-row-and-logs` | `GET /api/models`: its `requests` row (protocol `webui`), no generation dependency, and `traces` rows with its request id. |
| `untraced-paths` | No `requests` row for `/health` (polled while the server started) or `/api/queue/status`. |
| `role-instance` | Every row's `cloud_RoleInstance` is `<machine>:<pid>` of the server process. |
| `content-redaction` | No row of the run contains a prompt marker, and `traces` rows did arrive. SKIPPED when the run sent no prompt. |
| `metrics` | After LLM traffic, `customMetrics` has `gen_ai.server.request.duration`, `gen_ai.server.time_to_first_token`, `tensorsharp.tokens` (with `tensorsharp.token.type` input and output), the engine gauges `tensorsharp.engine.requests.running` and `.waiting`, `tensorsharp.kv_cache.blocks.used` and `.total`, `tensorsharp.engine.steps` and `tensorsharp.engine.step.duration`; after a completed clip or Studio job, `tensorsharp.video.generation.duration` and `tensorsharp.video.phase.duration`; on a GPU backend, `tensorsharp.gpu.memory.used`. A run is shorter than the exporter's 60 s metric interval, so the gauges arrive only through the flush when the host stops, while the model is still loaded: their presence checks that flush. Only presence: Application Insights keeps histograms as count, sum, min and max. |
| `both-sinks` | With `-BothSinks` and `-DashboardApiKey`, the dashboard lists the run's service and its spans carry every request id; without the key it is MANUAL. |

### Results

`-OutDir` holds `results.md`, `results.json` and `environment.json` as for `aspire-e2e.ps1`
(environment.json adds the resource name, the `az` version, the resolved `OpenTelemetry*` and
`Azure.*` package versions from the server's `deps.json`, the names, not values, of `AZURE_*`
variables, and each run's service name, start and stop time and how the server stopped), plus,
for every run in `runs/<run>/`:

- `server-launch.json`, the server's stdout, stderr and file logs, and `requests/` as before;
- `requests-sent.json`: every request id and prompt marker the run sent, by scenario;
- `appinsights/NN-<query>.kql` and `.json`: each query and its last result (rows, number of
  attempts, whether it completed, what was still missing);
- `exporter-messages.txt` when the output had exporter or credential warnings, and
  `otlp/spans.json` under `-BothSinks` with `-DashboardApiKey`.

### Things to know

- **It sends real telemetry.** Every run ingests into the resource and is billed like any other
  data; it stays for the resource's retention period (only a purge request removes it earlier).
  Each run is small, but video runs add a `traces` row per denoising step.
- **The resource can be shared.** Every query is filtered by the run's unique service name and
  start time, so other data in the resource does not affect the checks. The other way round, the
  e2e rows have `cloud_RoleName` `tensorsharp-e2e-...`, which the `startswith "tensorsharp-"`
  filter of `examples/opentelemetry/application-insights/queries.kql` includes; add
  `| where cloud_RoleName !startswith "tensorsharp-e2e-"` there to leave them out.
- **Ingestion is slow.** Rows usually appear 1 to 5 minutes after they are sent, sometimes later.
  A run with several servers takes at least that long after the last one stops; raise
  `-IngestionTimeoutMinutes` on a slow day rather than reading a timeout as a product failure.
- The Azure Monitor metric reader exports every 60 seconds whatever `OTEL_METRIC_EXPORT_INTERVAL`
  says (in exporter 1.10.0 it is built with the SDK's default interval). Whether the trace and log
  exporters honour `OTEL_BSP_SCHEDULE_DELAY` and `OTEL_BLRP_SCHEDULE_DELAY` is not verified. The
  graceful stop flushes whatever is still queued, which is why a killed server fails
  `server-stop`.
- A `--config` file's `environment` object only fills variables the process does not already
  have; keep `APPLICATIONINSIGHTS_CONNECTION_STRING` and OTLP settings out of the configs used
  here.
