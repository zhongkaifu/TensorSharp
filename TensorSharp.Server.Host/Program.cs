// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.AspNetCore.Http.Features;
using TensorSharp.GGML;
using TensorSharp.Runtime.Logging;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Logging;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;
using TensorSharp.Runtime.Redis;
using TensorSharp.Server.Host.Hosting;

const long MaxRequestBodyBytes = 500L * 1024L * 1024L;

Console.OutputEncoding = System.Text.Encoding.UTF8;

// Merge in options from a --config <file.json> before anything reads argv.
// File-derived tokens are spliced in ahead of the real command line, so any
// option also passed on the command line overrides the file (every option
// pass below is last-one-wins). The --config flag itself is stripped here.
try
{
    args = ConfigFileArgs.Expand(args);
}
catch (Exception ex) when (ex is ArgumentException or FileNotFoundException)
{
    Console.Error.WriteLine("Configuration error: " + ex.Message);
    Environment.ExitCode = 1;
    return;
}

bool showSarah = Array.Exists(args, a => a == "--xzf");
ConsoleBanner.Print(showSarah);

// Informational invocations print and exit before the web host is built. A
// bare `TensorSharp.Server` shows the usage page instead of silently starting
// a model-less server. Passing another option can still start a status-only
// process, but inference requires --model at startup.
if (args.Length == 0 || ServerUsage.IsHelpRequested(args))
{
    ServerUsage.PrintUsage(Console.Out);
    return;
}

if (ServerUsage.IsListGpusRequested(args))
{
    ServerUsage.PrintVulkanGpus(Console.Out);
    return;
}

string baseDirectory = AppContext.BaseDirectory;

// Parsed BEFORE the server's own options and removed from the argument list, because
// ServerOptionsBuilder.ParseArgs rejects any flag it does not recognise — a deliberate
// guard against typos that would otherwise be ignored. Feature-owned flags therefore have
// to be consumed first rather than handed to it.
//
// Running code the MODEL wrote is its own decision, separate from skills: a skill's
// script is a file an operator put on disk and can read first, this is not.
// The retired spellings are checked against the ORIGINAL line, because Parse consumes
// what it recognises and a retired flag is by definition not recognised.
string[] originalArgs = (string[])args.Clone();
CodeExecOptions codeExecOptions;
List<string> remainingArgs;
try
{
    codeExecOptions = CodeExecOptions.Parse(args, out List<string> parsedRemaining);
    remainingArgs = parsedRemaining;
}
catch (ArgumentException ex)
{
    Console.Error.WriteLine("Configuration error: " + ex.Message);
    Environment.ExitCode = 1;
    return;
}
codeExecOptions.ApplyEnvironment();

// Consume them, exactly as the config-file expansion above rewrites `args`. A dozen
// helpers downstream each re-parse the array, and every one of them would reject an
// unrecognised flag; filtering once here is what keeps that guard useful for real typos.
args = remainingArgs.ToArray();

ServerHostingOptions hostingOptions;
try
{
    // Removed flag spellings (e.g. --mtp-spec) would otherwise die in the
    // unknown-option trap with no pointer; reject them first so the error names
    // the surviving spelling. Then let Build validate everything else.
    TensorSharp.Runtime.Speculative.SpeculativeCliFlags.RejectRemoved(args);
    // Same reason, for the code-execution family: --code-exec-packages and
    // --code-exec-languages could not be enforced once the tool surface became a shell,
    // so they are refused by name with a pointer at what replaced them rather than
    // being silently ignored.
    if (CodeExecOptions.RejectRemoved(originalArgs) is { } removedCodeExecFlag)
        throw new ArgumentException(removedCodeExecFlag);
    hostingOptions = ServerOptionsBuilder.Build(args, baseDirectory);
}
catch (ArgumentException ex)
{
    // A configuration mistake is the operator's to fix; a stack trace buries
    // the one line they need.
    Console.Error.WriteLine("Configuration error: " + ex.Message);
    Environment.ExitCode = 1;
    return;
}
codeExecOptions.ArtifactUriPrefix = CodeArtifactEndpoints.RoutePrefix;
codeExecOptions.ScratchDirectory ??= Path.Combine(baseDirectory, "code-scratch");
// --code-exec-unconfined used to be refused here outright, "available on
// TensorSharp.Cli only". That was not a policy so much as a platform outage.
//
// The reasoning was sound for a server reachable from elsewhere, but it was applied
// unconditionally, and on Windows there IS no confining sandbox to fall back to: the
// job object bounds a process tree and cannot restrict a single file or socket, so
// ShellRunner.CanRun was false forever and --code-exec was a flag that did nothing at
// all. An operator running the server on their own desktop had no way to say yes,
// while --skills-allow-exec — which lets the model run any script in any installed
// skill with this process's privileges, the same class of decision — was accepted on
// the same command line with a warning. One of the two had to move, and refusing the
// weaker one while permitting the stronger was the wrong way round.
//
// So it is accepted and said plainly. Other agents' Windows mechanisms and setup
// requirements differ; this decision describes TensorSharp's CURRENT job-object
// backend only. What is not relaxed is the reporting: the startup warning states the
// gap and ShellRunner repeats it in every tool result.
if (codeExecOptions.Enabled && codeExecOptions.Unconfined)
{
    Console.Error.WriteLine(
        $"{CodeExecOptions.UnconfinedFlag} is set: commands this model writes may fall back to this "
        + "process's privileges — full filesystem access and network — when confinement is unavailable "
        + "(every run on Windows). That is a decision about "
        + "the machine this server runs on, so do not leave it on for a server others can reach.");
}
if (codeExecOptions.Enabled && codeExecOptions.AllowNetwork)
{
    Console.Error.WriteLine(
        $"{CodeExecOptions.AllowNetworkFlag} is set: commands this model writes have unrestricted "
        + "IP network access (subject to the host OS and firewall), including LAN/loopback services and "
        + "listening sockets. Remote content can contain prompt injections, and generated code can send "
        + "workspace or other host-readable data away. Package/install-domain allow-lists constrain only "
        + "the host installer, not direct downloads by a command. On macOS, a deliberately detached child "
        + "may outlive its request; tool results report that gap. Enable this only for users and tasks you trust.");
}

// --list-skills answers "what does this deployment have?" without starting the
// host or loading a model, and prints the LOAD ERRORS too — a skill whose
// SKILL.md will not parse is otherwise simply absent from every listing, which
// is the hardest kind of problem for its author to diagnose.
if (SkillHostOptions.Parse(args).ListOnly)
{
    var listing = new SkillRegistry(new SkillRegistryOptions
    {
        Roots = hostingOptions.SkillDirectories,
        InstallDirectory = hostingOptions.SkillsEnabled
            ? Path.Combine(baseDirectory, SkillHostOptions.DefaultDirectoryName)
            : null,
    });
    ServerUsage.PrintSkills(Console.Out, listing, hostingOptions.SkillsEnabled);
    return;
}

LogLevel resolvedLogLevel = LoggingSetup.ResolveMinimumLevel();
string configuredBackendInput = ServerOptionsBuilder.ReadConfiguredBackendInput(args);
// Translate --paged-kv* flags into env vars before startup logging reads
// PagedKvCacheConfig.FromEnvironment().
bool pagedKvFlagsApplied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(args);
// Translate --redis-url into TS_KV_CACHE_REDIS_URL and
// TS_RESPONSES_STORE_REDIS_URL so a single flag enables Redis for both the
// paged KV cache tier and the Responses API store.
bool redisFlagsApplied = ServerOptionsBuilder.ApplyRedisCliFlags(args);
// Translate --continuous-batching / --no-continuous-batching into env vars
// that gate BatchExecutor (TS_SCHED_DISABLE_BATCHED) and Qwen3.5 ForwardBatch
// (TS_QWEN35_BATCHED). Must run before InferenceEngine constructs its
// BatchExecutor and the per-model batched-paged adapters initialise.
bool continuousBatchingFlagApplied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args);
// Translate --spec / --spec-draft / --spec-pmin / --draft-model into the env vars
// read by SchedulerConfig.FromEnvironment when the engine is constructed.
bool specFlagsApplied = ServerOptionsBuilder.ApplySpeculativeCliFlags(args);
// Translate --qwen-image-vae / --qwen-image-vl / --qwen-image-mmproj into the
// TS_QWEN_IMAGE_* env vars QwenImageModel reads to locate the VAE, Qwen2.5-VL
// text-encoder, and mmproj GGUFs. Must run before the startup model is loaded.
bool qwenImageFlagsApplied = ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args);
// Translate --kv-cache-dtype into the process-wide KvCacheDtypeConfig (or honor
// the KV_CACHE_DTYPE env var) so block-quantized / half-precision KV caches are
// selectable on the server, mirroring the CLI. The fused native decode path used
// by the scheduler is the one that supports block-quantized (q8_0 / q4_0) caches.
// Must run before the startup model is loaded so InitKVCache sees the choice.
TensorSharp.Models.KvCacheDtypeConfig.ConfigureFromEnvironment();
bool kvCacheDtypeFlagApplied = ServerOptionsBuilder.ApplyKvCacheDtypeCliFlag(args);
// Translate --n-cpu-moe / --cpu-moe into MoeCpuOffloadConfig (or honor the
// TS_N_CPU_MOE / TS_CPU_MOE env vars). Must run before the startup model is
// loaded: weight residency is decided while preparing the quantized weights.
TensorSharp.Models.MoeCpuOffloadConfig.ConfigureFromEnvironment();
bool moeCpuOffloadFlagsApplied = ServerOptionsBuilder.ApplyMoeCpuOffloadCliFlags(args);
// Translate --gpu-device into TS_GGML_VULKAN_DEVICE so multi-GPU hosts can pick
// which Vulkan device the ggml_vulkan backend initializes on. Must run before
// the startup model is loaded (the device is fixed at first backend init).
bool gpuDeviceFlagApplied = ServerOptionsBuilder.ApplyGpuDeviceCliFlag(args);
// Translate --tp / --tp-node-id / --tp-peers into the TENSORSHARP_TP_* env vars
// the model loader reads (ModelBase.Create for the local degree,
// DistributedTpConfig for the multi-node pair). Must run before the startup
// model is loaded so the very first load is sharded across the GPUs.
bool tensorParallelFlagsApplied = ServerOptionsBuilder.ApplyTensorParallelCliFlags(args);

var builder = WebApplication.CreateBuilder(args);
LoggingSetup.Configure(builder.Logging, hostingOptions, resolvedLogLevel);

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = MaxRequestBodyBytes;
});

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = MaxRequestBodyBytes;
});

builder.Services.AddSingleton(hostingOptions);
// Constructed eagerly: the constructor scans the upload directory once so the
// quota tally starts from what is already on disk.
var uploadPolicy = new UploadStoragePolicy(
    hostingOptions.UploadDirectory,
    hostingOptions.UploadMaxFileBytes,
    hostingOptions.UploadQuotaBytes,
    hostingOptions.UploadTtl);
builder.Services.AddSingleton(uploadPolicy);
if (hostingOptions.UploadTtl.HasValue)
    builder.Services.AddHostedService<UploadCleanupService>();
// Agent Skills, constructed eagerly for the same reason as the upload policy: the
// constructor walks the configured roots once, so the on-disk state seeds the
// in-memory index and the first chat request never pays for the scan.
//
// Uploads land in ONE dedicated directory next to the binary, never in a root the
// operator named with --skills-dir. That separation is what makes DELETE safe: a
// skill discovered under an operator root is refused by SkillRegistry.Remove, so
// pointing --skills-dir at a checkout of somebody's skills repository cannot turn
// the management API - or the delete button in the Web UI - into a way to erase it.
// The registry scans the install directory first, so an uploaded fix shadows a stale
// copy of the same name in a read-only root rather than being shadowed by it.
var skillRegistry = new SkillRegistry(
    new SkillRegistryOptions
    {
        Roots = hostingOptions.SkillDirectories,
        InstallDirectory = hostingOptions.SkillsEnabled
            ? Path.Combine(baseDirectory, SkillHostOptions.DefaultDirectoryName)
            : null,
    });
builder.Services.AddSingleton(skillRegistry);
builder.Services.AddSingleton<SkillsAdapter>();

// Where a command's output files are kept so a user can download them.
//
// Deliberately NOT the scratch directory the run itself used: that one is deleted the
// moment the call returns, which is right for an interpreter and its installed packages
// and wrong for the PDF the user asked the model to produce. Registered as null when the
// feature is off, so the download route can answer "not enabled" rather than 404 and
// leave the caller guessing.
CodeArtifactStore? codeArtifactStore = codeExecOptions.Enabled
    ? new CodeArtifactStore(
        codeExecOptions.ArtifactDirectory
            ?? Path.Combine(baseDirectory, "code-artifacts"),
        new CodeArtifactLimits())
    : null;
builder.Services.AddSingleton(_ => codeArtifactStore!);

// One persistent execution workspace per chat session: the files, installed packages
// and working directory that the shell tool and skill scripts share for the length of a
// conversation — generate.pptx in one step, validate it in the next. Registered null
// when code execution is off. A restart orphans every session, so old workspaces are
// swept at startup rather than leaked.
SessionWorkspaceManager? workspaceManager = codeExecOptions.Enabled
    ? new SessionWorkspaceManager(codeExecOptions.ScratchDirectory!)
    : null;
workspaceManager?.SweepOrphans();
builder.Services.AddSingleton(_ => workspaceManager!);

// The thing that answers a shell call. Registered as null when the feature is off, so
// every adapter can take it unconditionally and SkillRequestPlan simply does not offer
// the tool — rather than each adapter having to know whether the feature exists.
// Built lazily so the runner gets a real logger: its codeexec.ran line (shell, exit
// code, sandbox, elapsed) is the one place a failed run is diagnosable after the fact.
builder.Services.AddSingleton<ICodeRunner>(sp => codeExecOptions.Enabled
    ? new CodeRunnerAdapter(
        new ShellRunner(
            codeExecOptions,
            sp.GetRequiredService<ILoggerFactory>().CreateLogger("TensorSharp.Server.CodeExec"),
            codeArtifactStore),
        codeExecOptions)
    : null!);
builder.Services.AddSingleton<ModelService>();
builder.Services.AddSingleton<InferenceQueue>();
builder.Services.AddSingleton<SessionManager>();
// Engine is owned by ModelService now (so its lifecycle is tied to the
// loaded model). Re-export it as a DI service for adapters that wish to
// submit requests directly.
builder.Services.AddSingleton<InferenceEngineHost>(sp =>
    sp.GetRequiredService<ModelService>().EngineHost);

// Demote the high-frequency status-polling endpoints to Debug so the
// default Information-level log isn't dominated by their request entries.
// Set TENSORSHARP_LOG_LEVEL=Debug to see them when troubleshooting.
builder.Services.AddTensorSharpRequestLogging(options =>
{
    options.LowNoisePaths.Add("/api/queue/status");
});

// One adapter per protocol; instances are stateless and free to share between requests.
builder.Services.AddSingleton<WebUiAdapter>();
builder.Services.AddSingleton<OllamaAdapter>();
builder.Services.AddSingleton<OpenAIChatAdapter>();
// Responses API store: use Redis when TS_RESPONSES_STORE_REDIS_URL is set,
// otherwise fall back to the bounded in-memory cache.
string? responsesRedisUrl = Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL")?.Trim();
if (!string.IsNullOrEmpty(responsesRedisUrl))
{
    builder.Services.AddSingleton<IResponsesStore>(sp =>
    {
        var logger = sp.GetRequiredService<ILoggerFactory>().CreateLogger("TensorSharp.Server.Responses.RedisResponsesStore");
        var redis = new RedisConnection(responsesRedisUrl, logger);
        return new RedisResponsesStore(redis, logger);
    });
}
else
{
    builder.Services.AddSingleton<IResponsesStore, InMemoryResponsesStore>();
}
builder.Services.AddSingleton<OpenAIResponsesAdapter>();

WebRootSetup.Resolve(builder.Environment, baseDirectory);

var app = builder.Build();

ILogger startupLogger = app.Services.GetRequiredService<ILoggerFactory>()
    .CreateLogger("TensorSharp.Server.Startup");
startupLogger.LogInformation(LogEventIds.LoggingInitialized,
    "Logging initialized: minimumLevel={MinimumLevel} fileLogging={FileLogging} logDir={LogDir}",
    resolvedLogLevel, hostingOptions.FileLoggingEnabled,
    hostingOptions.FileLoggingEnabled ? hostingOptions.LogDirectory : "(disabled)");

// --code-exec was asked for but the runner would refuse every call (no usable
// sandbox on this host). Said once at startup rather than discovered one request
// at a time: the shell tool is simply never offered to the model, and nothing else in
// any response says why.
if (codeExecOptions.Enabled)
{
    ICodeRunner startupCodeRunner = app.Services.GetRequiredService<ICodeRunner>();
    if (!startupCodeRunner.CanRun)
    {
        startupLogger.LogWarning(LogEventIds.HostConfiguration,
            "--code-exec is set but code execution is unavailable: {Reason}. The shell tool will NOT be offered to the model; " +
            "requests are answered without running code. Install an OS sandbox to enable it " +
            "(Linux: install/update bubblewrap/bwrap 0.12.0 or newer; macOS: sandbox-exec ships with the OS).",
            startupCodeRunner.UnavailableReason);
    }
}

if (pagedKvFlagsApplied)
{
    var pagedCfg = PagedKvCacheConfig.FromEnvironment();
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "paged-kv configured via CLI: enabled={Enabled} blockSize={BlockSize} ramMB={RamMB} ssdDir={SsdDir} maxSsdMB={MaxSsdMB}",
        pagedCfg.Enabled, pagedCfg.BlockSize, pagedCfg.MaxRamBytes / (1024 * 1024),
        string.IsNullOrEmpty(pagedCfg.SsdDirectory) ? "(disabled)" : pagedCfg.SsdDirectory,
        pagedCfg.MaxSsdBytes / (1024 * 1024));
}

if (redisFlagsApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Redis configured via CLI: kvCacheUrl={KvRedisUrl} responsesStoreUrl={ResponsesRedisUrl}",
        Environment.GetEnvironmentVariable("TS_KV_CACHE_REDIS_URL") ?? "(disabled)",
        Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL") ?? "(disabled)");
}

if (specFlagsApplied)
{
    var schedCfg = TensorSharp.Runtime.Scheduling.SchedulerConfig.FromEnvironment();
    string? blockDraft = Environment.GetEnvironmentVariable("TS_DSV4_DSPARK");
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Speculative decoding configured via CLI: enabled={Enabled} algorithm={Algorithm} maxDraft={MaxDraft} " +
        "pMin={PMin} draftModel={DraftModel} (engages for solo sequences)",
        schedCfg.Speculation.Enabled, schedCfg.Speculation.SpeculatorName, schedCfg.Speculation.MaxDraftTokens,
        schedCfg.Speculation.MinDraftProb.HasValue
            ? schedCfg.Speculation.MinDraftProb.Value.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture)
            : "auto (per algorithm)",
        string.IsNullOrEmpty(blockDraft) ? "(none)" : Path.GetFileName(blockDraft));
}

if (gpuDeviceFlagApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Vulkan GPU device configured via CLI: --gpu-device {DeviceIndex} (applies when the ggml_vulkan backend initializes)",
        Environment.GetEnvironmentVariable(GgmlBasicOps.VulkanDeviceEnvVar));
}

if (tensorParallelFlagsApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Tensor parallelism configured via CLI: degree={TpDegree} nodeId={TpNodeId} peers={TpPeers}",
        Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE") ?? "1",
        Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID") ?? "(single-node)",
        Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS") ?? "(none)");
}

if (moeCpuOffloadFlagsApplied || TensorSharp.Models.MoeCpuOffloadConfig.IsEnabled)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "MoE CPU offload active: routed experts of {Layers} stay in system RAM and run on the host ({Threads} threads)",
        TensorSharp.Models.MoeCpuOffloadConfig.Describe() ?? "no layers",
        TensorSharp.Models.MoeCpuOffloadConfig.CpuThreads > 0
            ? TensorSharp.Models.MoeCpuOffloadConfig.CpuThreads.ToString(System.Globalization.CultureInfo.InvariantCulture)
            : "auto");
}

if (qwenImageFlagsApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Qwen-Image-Edit companions configured via CLI: vae={Vae} vl={Vl} mmproj={Mmproj}",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_VAE") ?? "(scan)",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE") ?? "(scan)",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ") ?? "(scan)");
}

if (hostingOptions.UploadMaxFileBytes != UploadStoragePolicy.DefaultMaxFileBytes
    || uploadPolicy.QuotaEnabled
    || hostingOptions.UploadTtl.HasValue)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Upload storage limits: maxFileMB={MaxFileMB} quotaMB={QuotaMB} ttlHours={TtlHours} usedMB={UsedMB}",
        hostingOptions.UploadMaxFileBytes / (1024 * 1024),
        uploadPolicy.QuotaEnabled ? (hostingOptions.UploadQuotaBytes / (1024 * 1024)).ToString() : "(off)",
        hostingOptions.UploadTtl.HasValue ? hostingOptions.UploadTtl.Value.TotalHours.ToString("0.##") : "(off)",
        uploadPolicy.UsedBytes / (1024 * 1024));
}

if (hostingOptions.SkillsEnabled && (skillRegistry.Skills.Count > 0 || skillRegistry.Errors.Count > 0))
{
    startupLogger.LogInformation(LogEventIds.SkillsScanned,
        "Agent skills: loaded={SkillCount} errors={ErrorCount} roots={Roots} scripts={AllowScripts} maxRounds={MaxRounds}",
        skillRegistry.Skills.Count,
        skillRegistry.Errors.Count,
        string.Join(", ", skillRegistry.Roots),
        hostingOptions.SkillsAllowScripts,
        // The EFFECTIVE cap, not the configured one. A host that offers code execution
        // raises its own default, and printing 8 while the loop enforces 24 sends an
        // operator reading their own startup log looking for a bug that is not there.
        hostingOptions.SkillsMaxRoundsSpecified
            ? hostingOptions.SkillsMaxRounds
            : Math.Max(hostingOptions.SkillsMaxRounds,
                codeExecOptions.Enabled ? SkillHostOptions.CodeExecutionRounds : 0));
    foreach (var skillError in skillRegistry.Errors)
    {
        startupLogger.LogWarning(LogEventIds.SkillRejected,
            "Agent skill not loaded: {Path} - {Reason}", skillError.Path, skillError.Message);
    }
    if (hostingOptions.SkillsAllowScripts)
    {
        // Whether the flag can actually DO anything here, said at startup rather than
        // only inside a tool result the model may never surface.
        //
        // The startup line above prints scripts=True from the flag alone. On Windows
        // that was a lie in the one direction that costs a session: the default
        // --skills-sandbox required cannot be satisfied by a job object, which confines
        // no file and no socket, so every skills_run refused - and the only place that
        // refusal appeared was the tool result, where a model is free to summarise it
        // as "done". The reported failure did exactly that: four rounds, then "the
        // README has been converted into a slide deck", with nothing on disk.
        //
        // --code-exec already had this line. Skills did not.
        var probe = new SkillScriptRunner(new SkillScriptRunnerOptions
        {
            Sandbox = hostingOptions.SkillsSandbox,
        });
        if (!probe.CanRun)
        {
            startupLogger.LogWarning(LogEventIds.HostConfiguration,
                "--skills-allow-exec is set but skill scripts cannot run on this host: {Reason}",
                probe.UnavailableReason);
        }
        else
        {
            // Worth its own line at Warning: this is the one setting that turns a skill
            // upload into code execution, and an operator who set it by copying a command
            // line should see it said plainly at every start.
            startupLogger.LogWarning(LogEventIds.HostConfiguration,
                "--skills-allow-exec is set: the model may RUN scripts bundled with any installed skill, " +
                "with this process's privileges. Do not leave this on for a server that accepts skill uploads." +
                (probe.Sandbox is { } box && box.Capabilities.Gaps().Count > 0
                    ? " This host's sandbox (" + box.Name + ") does not confine: "
                      + string.Join("; ", box.Capabilities.Gaps()) + "."
                    : string.Empty));
        }
    }
}
else if (!hostingOptions.SkillsEnabled)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Agent skills disabled (--no-skills / TS_NO_SKILLS): /v1/skills and /api/skills are not mapped");
}

StartupBanner.EmitBackendFallback(startupLogger, hostingOptions, configuredBackendInput);

// Outermost application middleware, so it handles an escaping exception before
// the framework's Developer Exception Page can answer with the throwing source
// file. Request logging sits just inside it and has already recorded the
// failure in full by the time it rethrows here, so every API surface fails as
// JSON without losing a single log line.
app.UseApiExceptionHandling();
app.UseTensorSharpRequestLogging();
// Convert a prompt-doesn't-fit-context failure into a 400. After request
// logging so the rejection is still traced; before the endpoints so it covers
// every protocol surface.
app.UsePromptOverflowHandling();
// Serve the bundled static UI. GET / sends index.html too (see
// HealthEndpoints), so a bare http://host:port/ opens the chat UI; the plain
// liveness response moved to GET /health and still answers / on headless
// deployments that ship no wwwroot content. --no-webui skips the wwwroot
// middleware entirely for API-only deployments; /uploads stays served below
// because the image and video APIs return result URLs under it.
if (hostingOptions.WebUiEnabled)
{
    app.UseDefaultFiles();
    app.UseStaticFiles();
}
else
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Web UI disabled (--no-webui / TS_NO_WEBUI): wwwroot is not served; API endpoints and /uploads remain available");
}
// /uploads holds user-supplied files, so its content types come from the
// UploadContentPolicy allow-list: media keeps real types, text/code always
// comes back as text/plain (an uploaded .html page must never execute in the
// server's origin), unlisted extensions 404, and every response carries
// X-Content-Type-Options: nosniff.
app.UseStaticFiles(UploadContentPolicy.BuildStaticFileOptions(hostingOptions.UploadDirectory));

app.MapHealthEndpoints(app.Environment, hostingOptions.WebUiEnabled);
app.MapSessionEndpoints();
app.MapUploadEndpoints();
if (hostingOptions.SkillsEnabled)
    app.MapSkillEndpoints();
if (codeExecOptions.Enabled)
    app.MapCodeArtifactEndpoints();
app.MapWebUiEndpoints();
app.MapOllamaEndpoints();
app.MapOpenAIEndpoints();

StartupModelLoader.LoadIfConfigured(
    hostingOptions,
    app.Services.GetRequiredService<ModelService>(),
    configuredBackendInput,
    startupLogger);

StartupBanner.Emit(startupLogger, hostingOptions, hostingOptions.ListenUrls);

// Tear down the process-global GGML backend after the host stops. On macOS
// the ggml-metal device's C++ static destructor asserts that its resource
// set is empty; if g_backend (and its MTLBuffer wrappers) outlive the .NET
// host the assertion aborts the process during exit. ApplicationStopped
// fires after all hosted services have shut down, so all in-flight
// inference is already complete. The shutdown call is idempotent and a
// no-op when no GGML backend was ever initialised. Also hooked onto
// ProcessExit as a safety net for non-graceful exits.
app.Lifetime.ApplicationStopped.Register(static () => GgmlBasicOps.Shutdown());
AppDomain.CurrentDomain.ProcessExit += static (_, _) => GgmlBasicOps.Shutdown();

// Bind the address resolved by ServerOptionsBuilder (--port / --host / --urls,
// then PORT / HOST / ASPNETCORE_URLS, then http://0.0.0.0:5000). Passing it to
// Run() overrides anything the host builder configured, so ASPNETCORE_URLS is
// folded into that resolution rather than being silently discarded here.
app.Run(hostingOptions.ListenUrls);
