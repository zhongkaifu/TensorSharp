// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;
using TensorSharp.GGML;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Logging;
using TensorSharp.Server.ProtocolAdapters;

const string ListenAddress = "http://0.0.0.0:5000";
const long MaxRequestBodyBytes = 500L * 1024L * 1024L;

Console.OutputEncoding = System.Text.Encoding.UTF8;
bool showSarah = Array.Exists(args, a => a == "--xzf");
ConsoleBanner.Print(showSarah);

// Informational invocations print and exit before the web host is built. A
// bare `TensorSharp.Server` shows the usage page instead of silently starting
// a model-less server; passing any option (e.g. --backend ggml_cpu) still
// starts headless for operators who load models from the web UI.
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
ServerHostingOptions hostingOptions = ServerOptionsBuilder.Build(args, baseDirectory);
LogLevel resolvedLogLevel = LoggingSetup.ResolveMinimumLevel();
string configuredBackendInput = ServerOptionsBuilder.ReadConfiguredBackendInput(args);
// Translate --paged-kv* flags into env vars before startup logging reads
// PagedKvCacheConfig.FromEnvironment().
bool pagedKvFlagsApplied = ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(args);
// Translate --continuous-batching / --no-continuous-batching into env vars
// that gate BatchExecutor (TS_SCHED_DISABLE_BATCHED) and Qwen3.5 ForwardBatch
// (TS_QWEN35_BATCHED). Must run before InferenceEngine constructs its
// BatchExecutor and the per-model batched-paged adapters initialise.
bool continuousBatchingFlagApplied = ServerOptionsBuilder.ApplyContinuousBatchingCliFlag(args);
// Translate --mtp-spec / --mtp-draft / --mtp-pmin into the TS_MTP_* env vars
// read by SchedulerConfig.FromEnvironment when the engine is constructed.
bool mtpSpecFlagsApplied = ServerOptionsBuilder.ApplyMtpSpeculativeCliFlags(args);
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
// Translate --gpu-device into TS_GGML_VULKAN_DEVICE so multi-GPU hosts can pick
// which Vulkan device the ggml_vulkan backend initializes on. Must run before
// the startup model is loaded (the device is fixed at first backend init).
bool gpuDeviceFlagApplied = ServerOptionsBuilder.ApplyGpuDeviceCliFlag(args);

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

WebRootSetup.Resolve(builder.Environment, baseDirectory);

var app = builder.Build();

ILogger startupLogger = app.Services.GetRequiredService<ILoggerFactory>()
    .CreateLogger("TensorSharp.Server.Startup");
startupLogger.LogInformation(LogEventIds.LoggingInitialized,
    "Logging initialized: minimumLevel={MinimumLevel} fileLogging={FileLogging} logDir={LogDir}",
    resolvedLogLevel, hostingOptions.FileLoggingEnabled,
    hostingOptions.FileLoggingEnabled ? hostingOptions.LogDirectory : "(disabled)");

if (pagedKvFlagsApplied)
{
    var pagedCfg = PagedKvCacheConfig.FromEnvironment();
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "paged-kv configured via CLI: enabled={Enabled} blockSize={BlockSize} ramMB={RamMB} ssdDir={SsdDir} maxSsdMB={MaxSsdMB}",
        pagedCfg.Enabled, pagedCfg.BlockSize, pagedCfg.MaxRamBytes / (1024 * 1024),
        string.IsNullOrEmpty(pagedCfg.SsdDirectory) ? "(disabled)" : pagedCfg.SsdDirectory,
        pagedCfg.MaxSsdBytes / (1024 * 1024));
}

if (mtpSpecFlagsApplied)
{
    var schedCfg = TensorSharp.Runtime.Scheduling.SchedulerConfig.FromEnvironment();
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "MTP speculative decoding configured via CLI: enabled={Enabled} maxDraft={MaxDraft} pMin={PMin} (engages on NextN/MTP draft-head models only)",
        schedCfg.MtpSpeculativeEnabled, schedCfg.MtpMaxDraftTokens, schedCfg.MtpMinDraftProb);
}

if (gpuDeviceFlagApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Vulkan GPU device configured via CLI: --gpu-device {DeviceIndex} (applies when the ggml_vulkan backend initializes)",
        Environment.GetEnvironmentVariable(GgmlBasicOps.VulkanDeviceEnvVar));
}

if (qwenImageFlagsApplied)
{
    startupLogger.LogInformation(LogEventIds.HostConfiguration,
        "Qwen-Image-Edit companions configured via CLI: vae={Vae} vl={Vl} mmproj={Mmproj}",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_VAE") ?? "(scan)",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_TE") ?? "(scan)",
        Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ") ?? "(scan)");
}

StartupBanner.EmitBackendFallback(startupLogger, hostingOptions, configuredBackendInput);

app.UseTensorSharpRequestLogging();
// Rewrite "/" to "/index.html" so the bare root serves the chat UI (the
// MapGet("/") health route below would otherwise win). UseDefaultFiles is a
// no-op when wwwroot/index.html is absent, so headless/text-only deployments
// still get the plain-text liveness string at "/".
app.UseDefaultFiles();
app.UseStaticFiles();
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(hostingOptions.UploadDirectory),
    RequestPath = "/uploads",
});

app.MapHealthEndpoints(app.Environment);
app.MapSessionEndpoints();
app.MapUploadEndpoints();
app.MapWebUiEndpoints();
app.MapOllamaEndpoints();
app.MapOpenAIEndpoints();

StartupModelLoader.LoadIfConfigured(
    hostingOptions,
    app.Services.GetRequiredService<ModelService>(),
    configuredBackendInput,
    startupLogger);

StartupBanner.Emit(startupLogger, hostingOptions, ListenAddress);

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

app.Run(ListenAddress);
