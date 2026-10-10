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
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Server.Hosting;

/// <summary>
/// Reads CLI arguments and environment variables and produces a fully
/// resolved <see cref="ServerHostingOptions"/>. Pure (no I/O beyond <see cref="Path"/>
/// helpers and probing the host for supported backends), which makes it easy
/// to test without spinning up a web app.
/// </summary>
public static class ServerOptionsBuilder
{
    private const int DefaultMaxTokensFallback = 20000;

    public static ServerHostingOptions Build(string[] args, string baseDirectory)
    {
        ArgumentNullException.ThrowIfNull(args);
        if (string.IsNullOrEmpty(baseDirectory)) throw new ArgumentNullException(nameof(baseDirectory));

        // A removed option is refused by name with what to do instead, before the
        // unknown-option trap in ParseArgs can reduce it to a bare "Unknown option".
        // Program.cs checks the same table before this is called; checking here too
        // keeps every caller of Build on the same message.
        TensorSharp.Runtime.RemovedCliFlags.RejectRemoved(args);
        var parallelismArgs = new List<string>();

        ParseArgs(args,
            out string? configuredModel,
            out string? configuredMmProj,
            out string? configuredBackend,
            out int? configuredMaxTokens,
            out int? configuredVideoFrames,
            out int? configuredVideoFps,
            out int? configuredVideoWidth,
            out int? configuredVideoHeight,
            out int? configuredVideoSteps,
            out string? configuredVideoMode,
            out SamplingOverrides configuredSampling,
            out SamplingPrecedence? configuredPrecedence,
            out ListenOverrides configuredListen,
            out UploadLimitOverrides configuredUploads,
            out bool configuredNoWebUi,
            out bool configuredNoPrefixCache,
            parallelismArgs);
        TensorSharp.Distributed.ModelParallelismOptions.Parse(parallelismArgs.ToArray());

        if (!string.IsNullOrWhiteSpace(configuredMmProj) && string.IsNullOrWhiteSpace(configuredModel))
            throw new ArgumentException("--mmproj requires --model.");

        string? startupModelPath = ResolveConfiguredModelPath(configuredModel);
        string? startupMmProjPath = ResolveConfiguredMmProjPath(configuredMmProj, startupModelPath);
        bool embeddingsEnabled = args.Any(a => string.Equals(a, "--embeddings", StringComparison.OrdinalIgnoreCase));
        int embeddingThreads = ReadEmbeddingIntOption(args, "--embedding-threads");
        int embeddingContextSize = ReadEmbeddingIntOption(args, "--embedding-context-size");
        if (embeddingsEnabled && string.IsNullOrWhiteSpace(startupModelPath))
            throw new ArgumentException("--embeddings requires --model.");
        if (embeddingsEnabled && startupMmProjPath != null)
            throw new ArgumentException("--embeddings does not support --mmproj.");
        if (!embeddingsEnabled && (embeddingThreads > 0 || embeddingContextSize > 0))
            throw new ArgumentException("--embedding-threads and --embedding-context-size require --embeddings.");

        string? backendInput = configuredBackend ?? Environment.GetEnvironmentVariable("BACKEND");
        string requestedBackend = backendInput ?? PlatformDefaultBackend;
        if (embeddingsEnabled)
            EmbeddingHosting.ResolveModelBackend(requestedBackend);

        // A managed embedding deployment does not need native backend libraries,
        // including the probes that normally discover the host's GPU choices.
        var supportedBackends = embeddingsEnabled && BackendCatalog.Canonicalize(requestedBackend) == "cpu"
            ? new[] { new BackendOption("cpu", "CPU (Pure C#)") }
            : BackendCatalogProbes.GetSupportedBackends()
                .Where(backend => !embeddingsEnabled || backend.Value is "cpu" or "ggml_cpu" or "ggml_metal" or "ggml_cuda")
                .ToArray();
        string defaultBackend = BackendCatalog.ResolveDefaultBackend(requestedBackend, supportedBackends);

        bool maxTokensPinned = configuredMaxTokens.HasValue;
        int defaultMaxTokens;
        if (configuredMaxTokens.HasValue)
        {
            defaultMaxTokens = configuredMaxTokens.Value;
        }
        else if (TryParsePositiveInt(Environment.GetEnvironmentVariable("MAX_TOKENS"), out int envMaxTokens))
        {
            defaultMaxTokens = envMaxTokens;
            maxTokensPinned = true;
        }
        else
        {
            defaultMaxTokens = DefaultMaxTokensFallback;
        }

        // Keep mutable media outside a pinned/read-only application deployment
        // when configured, just as logs and prefix checkpoints can be relocated.
        string? uploadDirectory = Environment.GetEnvironmentVariable("TENSORSHARP_UPLOAD_DIR");
        if (string.IsNullOrWhiteSpace(uploadDirectory))
            uploadDirectory = Path.Combine(baseDirectory, "uploads");
        Directory.CreateDirectory(uploadDirectory);

        string? logDirectory = Environment.GetEnvironmentVariable("TENSORSHARP_LOG_DIR");
        if (string.IsNullOrWhiteSpace(logDirectory))
            logDirectory = Path.Combine(baseDirectory, "logs");

        bool fileLoggingEnabled = !string.Equals(
            Environment.GetEnvironmentVariable("TENSORSHARP_LOG_FILE"),
            "0",
            StringComparison.Ordinal);

        SamplingDefaults defaultSampling = ResolveDefaultSamplingConfig(configuredSampling, configuredPrecedence);

        string listenUrls = ResolveListenUrls(configuredListen);

        long uploadMaxFileBytes = ResolveUploadMb(
            configuredUploads.MaxFileMb, "TS_UPLOAD_MAX_MB", UploadStoragePolicy.DefaultMaxFileBytes);
        long uploadQuotaBytes = ResolveUploadMb(
            configuredUploads.QuotaMb, "TS_UPLOAD_QUOTA_MB", 0);
        TimeSpan? uploadTtl = ResolveUploadTtl(configuredUploads.TtlHours, "TS_UPLOAD_TTL_HOURS");

        // Agent Skills. Parsed by the shared reader so the CLI and this host accept
        // exactly the same spellings, then layered with its env vars and defaulted to
        // the skills/ directory next to the binary (created if absent, so dropping a
        // skill directory in and restarting is all it takes).
        SkillHostOptions skillOptions = SkillHostOptions.Parse(args)
            .ApplyEnvironmentAndDefaults(baseDirectory);
        bool skillDirectoriesAreDefault = skillOptions.Roots.Count == 1
            && string.Equals(
                skillOptions.Roots[0],
                Path.Combine(baseDirectory, SkillHostOptions.DefaultDirectoryName),
                StringComparison.Ordinal);
        if (skillOptions.Enabled)
            skillOptions.ValidateRoots(createDefault: skillDirectoriesAreDefault);

        // TS_NO_WEBUI follows the TENSORSHARP_LOG_FILE convention: set to
        // anything but "0" counts as on.
        string? noWebUiEnv = Environment.GetEnvironmentVariable("TS_NO_WEBUI");
        bool webUiEnabled = !configuredNoWebUi
            && (string.IsNullOrWhiteSpace(noWebUiEnv) || string.Equals(noWebUiEnv.Trim(), "0", StringComparison.Ordinal));

        return new ServerHostingOptions(
            startupModelPath,
            startupMmProjPath,
            defaultBackend,
            supportedBackends,
            defaultMaxTokens,
            maxTokensPinned,
            configuredVideoFrames ?? 0,
            configuredVideoFps ?? 0,
            configuredVideoWidth ?? 0,
            configuredVideoHeight ?? 0,
            configuredVideoSteps ?? 0,
            configuredVideoMode,
            uploadDirectory,
            logDirectory,
            fileLoggingEnabled,
            defaultSampling,
            listenUrls,
            uploadMaxFileBytes,
            uploadQuotaBytes,
            uploadTtl,
            webUiEnabled,
            skillOptions.Roots,
            skillOptions.Enabled,
            skillOptions.Discovery,
            skillOptions.AllowScripts,
            // Zero means "the operator did not choose", which is what lets a
            // code-execution host raise its own default without overriding a
            // number somebody set on purpose.
            skillOptions.MaxRoundsSpecified ? skillOptions.MaxRounds : 0,
            skillOptions.Selected,
            skillOptions.Sandbox,
            skillOptions.AllowNetwork,
            // The flag is the only way to turn it OFF, which is why the switch is the
            // negation: a config file emits nothing for `false`, so a positive
            // "prefix-cache": false would silently leave it on.
            prefixCacheEnabled: !configuredNoPrefixCache,
            prefixCacheDirectory: ResolvePrefixCacheDirectory(baseDirectory, startupModelPath),
            embeddingsEnabled: embeddingsEnabled,
            embeddingThreads: embeddingThreads,
            embeddingContextSize: embeddingContextSize,
            multiAgent: ReadMultiAgentOptions(args));
    }

    private static readonly string[] AgentValueFlags =
    {
        "--agents-max-concurrent", "--agents-max-count", "--agents-max-depth",
        "--agents-max-rounds", "--agents-max-generations", "--agents-timeout",
        "--agents-max-result-chars",
    };

    private static MultiAgentOptions ReadMultiAgentOptions(string[] args)
    {
        var defaults = new MultiAgentOptions();
        int Read(string flag, int fallback)
        {
            int result = fallback;
            for (int i = 0; i < args.Length; i++)
                if (TryReadOption(args, ref i, flag, out string? value))
                {
                    if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out result))
                        throw new ArgumentException($"Invalid value for {flag}: '{value}'. Expected an integer.");
                }
            return result;
        }
        string? disabled = Environment.GetEnvironmentVariable("TS_NO_MULTI_AGENT");
        var result = new MultiAgentOptions
        {
            Enabled = !args.Any(a => string.Equals(a, "--no-multi-agent", StringComparison.OrdinalIgnoreCase))
                && (string.IsNullOrEmpty(disabled) || disabled == "0"),
            AllowWorkerTools = args.Any(a => string.Equals(a, "--agents-allow-worker-tools", StringComparison.OrdinalIgnoreCase)),
            MaxConcurrentAgents = Read("--agents-max-concurrent", defaults.MaxConcurrentAgents),
            MaxAgents = Read("--agents-max-count", defaults.MaxAgents),
            MaxDepth = Read("--agents-max-depth", defaults.MaxDepth),
            MaxRoundsPerAgent = Read("--agents-max-rounds", defaults.MaxRoundsPerAgent),
            MaxTotalChildGenerations = Read("--agents-max-generations", defaults.MaxTotalChildGenerations),
            AgentTimeoutSeconds = Read("--agents-timeout", defaults.AgentTimeoutSeconds),
            MaxResultCharacters = Read("--agents-max-result-chars", defaults.MaxResultCharacters),
        };
        result.Validate();
        return result;
    }

    private static int ReadEmbeddingIntOption(string[] args, string flag)
    {
        int value = 0;
        for (int i = 0; i < args.Length; i++)
            if (TryReadOption(args, ref i, flag, out string? raw))
            {
                if (!TryParsePositiveInt(raw, out value))
                    throw new ArgumentException($"Invalid value for {flag}: '{raw}'. Expected a positive integer.");
            }
        return value;
    }

    /// <summary>
    /// Where shared-prefix checkpoints are kept: the environment variable when the
    /// operator set one, otherwise a directory beside the binary. A path is a
    /// deployment detail rather than a behaviour, so it follows TENSORSHARP_LOG_DIR
    /// rather than adding a second flag for one behaviour.
    /// </summary>
    internal static string? ResolvePrefixCacheDirectory(string baseDirectory, string? startupModelPath)
    {
        string? root = Environment.GetEnvironmentVariable("TENSORSHARP_PREFIX_CACHE_DIR");
        if (string.IsNullOrWhiteSpace(root))
        {
            if (string.IsNullOrWhiteSpace(baseDirectory))
                return null;
            root = Path.Combine(baseDirectory, "prefix-cache");
        }

        // ONE DIRECTORY PER MODEL, which is the layout PrefixCheckpointFileStore
        // documents and the only one its retention policy makes sense in: it evicts
        // beyond two files by last-access time across every checkpoint in the
        // directory it was given, with no notion of which model wrote one. Pointing
        // several models at one directory turns a two-file-per-model budget into a
        // two-file GLOBAL budget, so alternating launches of two configs from the same
        // install would delete each other's checkpoints — each launch destroying the
        // several hundred megabytes the previous one had just spent twenty seconds
        // computing, and neither ever restoring. It is also the layout SweepOrphans
        // walks.
        string? model = ModelCacheKey(startupModelPath);
        return string.IsNullOrEmpty(model) ? root : Path.Combine(root, model);
    }

    /// <summary>
    /// A directory-safe name for the weights, distinct enough that two models never
    /// share a checkpoint directory.
    /// </summary>
    /// <remarks>
    /// The file name alone is not enough — two builds of one model can be called the
    /// same thing in different directories — so a short hash of the full path is
    /// appended. The readable half is kept in front because an operator clearing one
    /// model's cache by hand should be able to tell which directory is which.
    /// </remarks>
    internal static string? ModelCacheKey(string? startupModelPath)
    {
        if (string.IsNullOrWhiteSpace(startupModelPath))
            return null;

        string name = Path.GetFileNameWithoutExtension(startupModelPath) ?? string.Empty;
        var safe = new StringBuilder(name.Length);
        foreach (char c in name)
            safe.Append(char.IsAsciiLetterOrDigit(c) || c is '-' or '.' ? c : '_');
        if (safe.Length > 48)
            safe.Length = 48;

        byte[] hash = System.Security.Cryptography.SHA256.HashData(
            Encoding.UTF8.GetBytes(Path.GetFullPath(startupModelPath)));
        string suffix = Convert.ToHexString(hash, 0, 4).ToLowerInvariant();
        return (safe.Length == 0 ? "model" : safe.ToString()) + "-" + suffix;
    }

    /// <summary>
    /// The backend asked for when neither <c>--backend</c> nor <c>BACKEND</c> names one:
    /// ggml_metal on macOS, ggml_cpu elsewhere. <see cref="Build"/> still falls back from it
    /// to a backend this machine actually has; the startup banner compares against it to
    /// say so.
    /// </summary>
    public static string PlatformDefaultBackend => OperatingSystem.IsMacOS() ? "ggml_metal" : "ggml_cpu";

    /// <summary>Backend originally requested via <c>--backend</c> / <c>BACKEND</c> (without the OS-default fallback).</summary>
    public static string? ReadConfiguredBackendInput(string[] args)
    {
        ParseArgs(args, out _, out _, out string? configuredBackend, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _);
        return configuredBackend ?? Environment.GetEnvironmentVariable("BACKEND");
    }

    /// <summary>Upload storage-limit overrides captured from the CLI (see <see cref="UploadStoragePolicy"/>).</summary>
    private struct UploadLimitOverrides
    {
        public int? MaxFileMb;
        public int? QuotaMb;
        public double? TtlHours;
    }

    /// <summary>
    /// The request-body limit every route has: 500 MB, the size the server has always
    /// accepted. The JSON endpoints (chat, Ollama, Responses, image and video requests)
    /// buffer the whole body and decode a base64 attachment out of one .NET string, so a
    /// bigger limit there buys memory pressure and a 500 at the string-length ceiling, not
    /// a bigger usable attachment. Only <c>POST /api/upload</c> goes above it (see
    /// <see cref="ResolveUploadRequestBodyBytes"/>).
    /// </summary>
    public const long DefaultMaxRequestBodyBytes = UploadStoragePolicy.DefaultMaxFileBytes;

    /// <summary>
    /// The request-body limit of the multipart <c>POST /api/upload</c> route (and the
    /// multipart body limit) for a per-file upload cap: the cap itself, never below
    /// <see cref="DefaultMaxRequestBodyBytes"/>. That route streams the file to disk, so
    /// it is the one place a larger cap can be honoured.
    /// </summary>
    /// <remarks>
    /// The limit used to be a hard-coded 500 MB everywhere, so <c>--upload-max-mb 2000</c>
    /// was accepted, logged and then unreachable: Kestrel refused every body over 500 MB
    /// before the upload policy ever saw the file. Raising it server-wide instead let every
    /// JSON request buffer up to the cap. The floor stays because lowering the per-FILE cap
    /// is not a statement about request size, and because at the default the two were
    /// always equal, which keeps the default deployment exactly as it was.
    /// </remarks>
    public static long ResolveUploadRequestBodyBytes(long uploadMaxFileBytes)
        => Math.Max(DefaultMaxRequestBodyBytes, uploadMaxFileBytes);

    /// <summary>Resolve one MB-denominated upload limit: CLI flag, then env var, then <paramref name="fallbackBytes"/>.</summary>
    private static long ResolveUploadMb(int? cliMb, string envVar, long fallbackBytes)
    {
        if (cliMb.HasValue)
            return cliMb.Value * 1024L * 1024L;
        if (TryReadEnvInt(envVar, out int envMb) && envMb > 0)
            return envMb * 1024L * 1024L;
        return fallbackBytes;
    }

    private static TimeSpan? ResolveUploadTtl(double? cliHours, string envVar)
    {
        if (cliHours.HasValue)
            return TimeSpan.FromHours(cliHours.Value);
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(raw)
            && double.TryParse(raw.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double hours)
            && hours > 0)
        {
            return TimeSpan.FromHours(hours);
        }
        return null;
    }

    /// <summary>Listen address overrides captured from the CLI.</summary>
    private struct ListenOverrides
    {
        public int? Port;
        public string Host;
        public string Urls;
    }

    /// <summary>
    /// Resolve the address Kestrel binds. Highest precedence first:
    /// <list type="number">
    ///   <item><c>--port</c> / <c>--host</c> — the most specific operator intent.</item>
    ///   <item><c>--urls</c> — full control (multiple bindings, https).</item>
    ///   <item><c>PORT</c> / <c>HOST</c> env vars — the convention most container
    ///         platforms (Cloud Run, Heroku, Hugging Face Spaces) inject.</item>
    ///   <item><c>ASPNETCORE_URLS</c> — the stock ASP.NET Core variable.</item>
    ///   <item><see cref="ServerHostingOptions.DefaultListenUrls"/>.</item>
    /// </list>
    /// A partial choice still resolves: <c>--port</c> alone keeps the default
    /// host and vice versa, so <c>--host 127.0.0.1</c> binds loopback on 5000.
    /// The result is handed to <c>app.Run(url)</c>, which is why
    /// <c>ASPNETCORE_URLS</c> has to be read here — that call overrides
    /// anything the host builder picked up on its own, so leaving it out
    /// would silently ignore the variable.
    /// </summary>
    private static string ResolveListenUrls(ListenOverrides cli)
    {
        if (cli.Port.HasValue || !string.IsNullOrWhiteSpace(cli.Host))
            return BuildListenUrl(cli.Host ?? ReadEnvHost() ?? DefaultHost,
                                  cli.Port ?? ReadEnvPort() ?? ServerHostingOptions.DefaultPort);

        if (!string.IsNullOrWhiteSpace(cli.Urls))
            return cli.Urls.Trim();

        int? envPort = ReadEnvPort();
        string? envHost = ReadEnvHost();
        if (envPort.HasValue || envHost != null)
            return BuildListenUrl(envHost ?? DefaultHost, envPort ?? ServerHostingOptions.DefaultPort);

        string? aspnetUrls = Environment.GetEnvironmentVariable("ASPNETCORE_URLS");
        if (!string.IsNullOrWhiteSpace(aspnetUrls))
            return aspnetUrls.Trim();

        return ServerHostingOptions.DefaultListenUrls;
    }

    private const string DefaultHost = "0.0.0.0";

    /// <summary>
    /// Compose <c>http://host:port</c>. A bare IPv6 literal is bracketed
    /// (<c>::1</c> -&gt; <c>[::1]</c>) so the result is a well-formed URL;
    /// a host that already carries a scheme is honoured as written, which
    /// lets <c>--host https://0.0.0.0</c> serve TLS.
    /// </summary>
    private static string BuildListenUrl(string host, int port)
    {
        host = host.Trim();

        string scheme = "http://";
        int schemeIndex = host.IndexOf("://", StringComparison.Ordinal);
        if (schemeIndex >= 0)
        {
            scheme = host[..(schemeIndex + 3)];
            host = host[(schemeIndex + 3)..];
        }

        // Bracket an unbracketed IPv6 literal. Detected by a second colon:
        // "::1" and "fe80::1" have one, "localhost" and "10.0.0.1" do not.
        if (host.IndexOf(':') != host.LastIndexOf(':') && !host.StartsWith('['))
            host = "[" + host + "]";

        return $"{scheme}{host}:{port.ToString(CultureInfo.InvariantCulture)}";
    }

    /// <summary>Read <c>PORT</c>, the variable container platforms inject.</summary>
    private static int? ReadEnvPort()
    {
        string? raw = Environment.GetEnvironmentVariable("PORT");
        if (string.IsNullOrWhiteSpace(raw))
            return null;
        if (!TryParsePort(raw.Trim(), out int port))
            throw new ArgumentException($"Invalid PORT environment variable: '{raw}'. Expected a port between 1 and 65535.");
        return port;
    }

    private static string? ReadEnvHost()
    {
        string? raw = Environment.GetEnvironmentVariable("HOST");
        return string.IsNullOrWhiteSpace(raw) ? null : raw.Trim();
    }

    private static bool TryParsePort(string value, out int port)
    {
        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out port)
            && port >= 1
            && port <= 65535;
    }

    /// <summary>
    /// Translate <c>--continuous-batching</c> / <c>--no-continuous-batching</c>
    /// into <c>TS_SCHED_DISABLE_BATCHED</c>, which gates the batched path (the
    /// scheduler falls through to per-sequence KV-swap when set). Batching is on
    /// by default, so operators get paged-attention continuous batching without
    /// setting any env var or passing any flag; <c>--continuous-batching</c> is
    /// idempotent with the default, kept for explicit operator intent.
    /// <c>--no-continuous-batching</c> forces the per-seq path for every model,
    /// speculation included.
    ///
    /// Must run before <see cref="InferenceEngine"/> is constructed; the
    /// <c>BatchExecutor</c> reads the variable at runtime on each step.
    /// </summary>
    public static bool ApplyContinuousBatchingCliFlag(string[] args)
    {
        if (args == null || args.Length == 0)
            return false;

        bool changed = false;
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (string.Equals(a, "--continuous-batching", StringComparison.OrdinalIgnoreCase))
            {
                Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");
                changed = true;
                continue;
            }
            if (string.Equals(a, "--no-continuous-batching", StringComparison.OrdinalIgnoreCase))
            {
                Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
                changed = true;
                continue;
            }
            // Tune mixed-step chunked-prefill granularity. Each prefill chunk runs
            // as a single ExecuteStep that holds ModelBase.GpuComputeLock
            // for the duration of its forward pass, so smaller chunks
            // give parallel decode requests more frequent turns at the
            // GPU. Default 256 (see SchedulerConfig.MaxPrefillChunkSize).
            if (TryReadOption(args, ref i, "--prefill-chunk-size", out string? chunkOpt))
            {
                if (!int.TryParse(chunkOpt, out int chunk) || chunk <= 0)
                    throw new ArgumentException($"Invalid value for --prefill-chunk-size: '{chunkOpt}'.");
                Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", chunk.ToString(CultureInfo.InvariantCulture));
                changed = true;
                continue;
            }
        }
        return changed;
    }

    /// <summary>
    /// Translate <c>--kv-cache-dtype &lt;f32|f16|q8_0|q4_0&gt;</c> into the
    /// process-wide <see cref="TensorSharp.Models.KvCacheDtypeConfig"/> so the
    /// startup model picks it up at <c>InitKVCache</c>. Overrides any value
    /// already applied from the <c>KV_CACHE_DTYPE</c> env var. Block-quantized
    /// caches (q8_0 / q4_0) require the fused native decode path the scheduler
    /// uses. Returns true when the flag was present.
    /// </summary>
    public static bool ApplyKvCacheDtypeCliFlag(string[] args)
    {
        if (args == null || args.Length == 0)
            return false;

        // Last one wins, like every other option: a --config file's tokens come first and
        // the command line's after them, so stopping at the first occurrence let a file
        // beat the command line.
        bool applied = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (TryReadOption(args, ref i, "--kv-cache-dtype", out string? dtypeOpt))
            {
                if (!TensorSharp.Models.KvCacheDtypeConfig.TryParse(dtypeOpt, out var dtype))
                    throw new ArgumentException(
                        $"Unknown --kv-cache-dtype value '{dtypeOpt}'. Valid: f32, f16, q8_0, q4_0.");
                TensorSharp.Models.KvCacheDtypeConfig.Set(dtype);
                applied = true;
            }
        }
        return applied;
    }

    /// <summary>
    /// Translate <c>--n-cpu-moe &lt;N&gt;</c> / <c>-ncmoe</c> / <c>--cpu-moe</c> /
    /// <c>-cmoe</c> / <c>--cpu-moe-threads &lt;N&gt;</c> into the process-wide
    /// <see cref="TensorSharp.Models.MoeCpuOffloadConfig"/>, so the routed
    /// experts of the selected layers are never uploaded and their FFN runs on
    /// the host. Overrides any value already applied from <c>TS_N_CPU_MOE</c> /
    /// <c>TS_CPU_MOE</c>. Must run before the startup model is loaded, because
    /// weight residency is decided during model preparation. Returns true when
    /// any of the flags was present.
    /// </summary>
    public static bool ApplyMoeCpuOffloadCliFlags(string[] args)
    {
        if (args == null || args.Length == 0)
            return false;

        bool applied = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (TryReadOption(args, ref i, "--n-cpu-moe", out string? ncmoe) ||
                TryReadOption(args, ref i, "-ncmoe", out ncmoe))
            {
                if (!TensorSharp.Models.MoeCpuOffloadConfig.TryParse(ncmoe, out int layers, out bool all))
                    throw new ArgumentException(
                        $"Invalid --n-cpu-moe value '{ncmoe}'. Expected a non-negative integer or 'all'.");
                if (all) TensorSharp.Models.MoeCpuOffloadConfig.SetAllLayers();
                else TensorSharp.Models.MoeCpuOffloadConfig.SetLayers(layers);
                applied = true;
                continue;
            }
            if (TryReadOption(args, ref i, "--cpu-moe-threads", out string? threads))
            {
                if (!int.TryParse(threads, out int n) || n <= 0)
                    throw new ArgumentException($"Invalid --cpu-moe-threads value '{threads}'. Expected a positive integer.");
                TensorSharp.Models.MoeCpuOffloadConfig.SetCpuThreads(n);
                applied = true;
                continue;
            }
            if (string.Equals(args[i], "--cpu-moe", StringComparison.Ordinal) ||
                string.Equals(args[i], "-cmoe", StringComparison.Ordinal))
            {
                TensorSharp.Models.MoeCpuOffloadConfig.SetAllLayers();
                applied = true;
            }
        }
        return applied;
    }

    /// <summary>
    /// Translate <c>--gpu-device &lt;index&gt;</c> into the env var that
    /// <c>GgmlNative</c> reads when the GGML Vulkan backend initializes
    /// (<c>TS_GGML_VULKAN_DEVICE</c>), so operators on multi-GPU hosts
    /// (e.g. an integrated Intel GPU next to a discrete NVIDIA one) can pick
    /// which Vulkan device serves inference. Only the ggml_vulkan backend
    /// consumes the value; it is inert for every other backend. Must run
    /// before the startup model is loaded. Returns true when the flag was
    /// present so the caller can emit a startup-log line.
    /// </summary>
    public static bool ApplyGpuDeviceCliFlag(string[] args)
    {
        if (args == null || args.Length == 0)
            return false;

        // Last one wins (see ApplyKvCacheDtypeCliFlag).
        bool applied = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (TryReadOption(args, ref i, "--gpu-device", out string? gpuOpt))
            {
                if (!int.TryParse(gpuOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int gpuIndex) || gpuIndex < 0)
                    throw new ArgumentException($"Invalid value for --gpu-device: '{gpuOpt}'. Expected a non-negative Vulkan device index.");
                Environment.SetEnvironmentVariable(
                    TensorSharp.GGML.GgmlBasicOps.VulkanDeviceEnvVar,
                    gpuIndex.ToString(CultureInfo.InvariantCulture));
                applied = true;
            }
        }
        return applied;
    }

    /// <summary>Apply the shared, validated tensor-parallel or layer-split configuration.</summary>
    public static bool ApplyTensorParallelCliFlags(string[] args)
    {
        var parallelismArgs = new List<string>();
        // Use the host's existing operand consumption rather than scanning values
        // as options. Unknown flags are left to Build's normal diagnostic pass.
        ParseArgs(args ?? Array.Empty<string>(), out _, out _, out _, out _, out _, out _, out _,
            out _, out _, out _, out _, out _, out _, out _, out _, out _,
            parallelismArgs, ignoreUnknownOptions: true);
        return TensorSharp.Distributed.ModelParallelismOptions.Parse(parallelismArgs.ToArray()).ApplyEnvironment();
    }

    /// <summary>
    /// Translate <c>--spec</c> / <c>--no-spec</c> /
    /// <c>--spec-draft N</c> / <c>--spec-pmin X</c> / <c>--draft-model PATH</c> into the
    /// env vars read by <c>SchedulerConfig.FromEnvironment</c> when the
    /// inference engine is constructed. Speculation is off by default; it
    /// engages for a drafter the checkpoint embeds (<c>--spec</c>), one named on
    /// <c>--draft-model</c>, or the weight-free n-gram algorithm
    /// (<c>--spec --spec-type ngram</c>) on a trunk that can verify a draft
    /// window - <c>--spec-type</c> alone turns nothing on. Returns true when at
    /// least one flag was applied so the caller can emit a startup-log line.
    /// </summary>
    public static bool ApplySpeculativeCliFlags(string[] args)
    {
        if (args == null || args.Length == 0)
            return false;

        // The speculative-decoding flags mean the same thing in both hosts,
        // so they are parsed and validated in one shared place
        // (TensorSharp.Runtime.Speculative) rather than kept in step by hand.
        return SpeculativeCliFlags.Apply(args);
    }

    /// <summary>Disable scheduler prefix reuse when the host's prefix-cache opt-out
    /// is present. Startup preparation and persistence use the same parsed flag.</summary>
    public static bool ApplyPrefixCacheCliFlag(string[] args)
    {
        if (args == null || !args.Any(a => string.Equals(a, "--no-prefix-cache", StringComparison.OrdinalIgnoreCase)))
            return false;
        Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", "0");
        return true;
    }

    /// <summary>
    /// Translate <c>--redis-url &lt;url&gt;</c> into
    /// <c>TS_RESPONSES_STORE_REDIS_URL</c>, which backs the Responses API store with
    /// Redis instead of the bounded in-memory cache. An already-set variable is left
    /// untouched. Returns true when the flag was present.
    /// </summary>
    public static bool ApplyRedisCliFlags(string[] args)
    {
        if (args == null || args.Length == 0)
            return false;

        // Last one wins (see ApplyKvCacheDtypeCliFlag): the value is chosen first and
        // applied once, because applying each occurrence would let the first fill the
        // variables the later ones then leave alone.
        string? redisUrl = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (TryReadOption(args, ref i, "--redis-url", out string? value))
                redisUrl = value;
        }
        if (redisUrl == null)
            return false;
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL")))
            Environment.SetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL", redisUrl);
        return true;
    }

    /// <summary>
    /// Translate the Qwen-Image-2.1 companion flags
    /// (<c>--qwen-image-vae</c> / <c>--qwen-image-vl</c> /
    /// <c>--qwen-image-mmproj</c>) into the env vars that
    /// <c>QwenImageModel</c> reads (<c>TS_QWEN_IMAGE_VAE</c> /
    /// <c>TS_QWEN_IMAGE_TE</c> / <c>TS_QWEN_IMAGE_MMPROJ</c>), the checkpoint
    /// declaration (<c>--qwen-image-variant</c>, <c>TS_QWEN_IMAGE_VARIANT</c>) and the LoRA plug-ins
    /// (<c>--lora</c> / <c>--lora-scale</c> / <c>--lora-config</c>, published as
    /// <c>TS_LORAS</c>) — the existing
    /// override mechanism for the three networks the qwen_image DiT GGUF does
    /// not itself contain — plus the output-size defaults and the video
    /// companions, which use the same env-var mechanism. Each path is validated
    /// here so a typo fails fast at startup instead of silently falling back to
    /// the same-directory scan. Must run before the startup model is loaded.
    /// Returns true when at least one flag was applied so the caller can emit a
    /// startup-log line.
    /// </summary>
    public static bool ApplyQwenImageCompanionCliFlags(string[] args)
    {
        if (args == null || args.Length == 0)
            return false;

        bool changed = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (TryReadOption(args, ref i, "--qwen-image-vae", out string? vaeOpt))
            {
                SetQwenImageCompanionEnv("--qwen-image-vae", "TS_QWEN_IMAGE_VAE", vaeOpt);
                changed = true;
                continue;
            }
            if (TryReadOption(args, ref i, "--qwen-image-vl", out string? vlOpt))
            {
                SetQwenImageCompanionEnv("--qwen-image-vl", "TS_QWEN_IMAGE_TE", vlOpt);
                changed = true;
                continue;
            }
            if (TryReadOption(args, ref i, "--qwen-image-mmproj", out string? mmprojOpt))
            {
                SetQwenImageCompanionEnv("--qwen-image-mmproj", "TS_QWEN_IMAGE_MMPROJ", mmprojOpt);
                changed = true;
                continue;
            }
            // Which 2.1 checkpoint the DiT GGUF holds (base / turbo): the files carry no
            // metadata, and Turbo samples its own 8-step schedule. Last one wins.
            if (TryReadOption(args, ref i, QwenImageVariantFlag.Flag, out string? variantOpt))
            {
                Environment.SetEnvironmentVariable(QwenImageVariantFlag.EnvironmentVariable,
                    QwenImageVariantFlag.Name(QwenImageVariantFlag.Parse(variantOpt, QwenImageVariantFlag.Flag)));
                changed = true;
                continue;
            }
            // Video-generation companions (same env-var override mechanism).
            if (TryReadOption(args, ref i, "--video-vae", out string? videoVaeOpt))
            {
                SetQwenImageCompanionEnv("--video-vae", "TS_VIDEO_VAE", videoVaeOpt);
                changed = true;
                continue;
            }
            if (TryReadOption(args, ref i, "--video-text-encoder", out string? videoTeOpt))
            {
                SetQwenImageCompanionEnv("--video-text-encoder", "TS_VIDEO_TEXT_ENCODER", videoTeOpt);
                changed = true;
                continue;
            }
            // Dual-expert models (Wan 2.2 A14B) ship as a PAIR of GGUFs and need both.
            // They are auto-resolved by name when they sit together, but a config file
            // has to be able to name the second one explicitly — that is the only way
            // its auto-download entry can exist at all.
            if (TryReadOption(args, ref i, "--video-dit2", out string? videoDit2Opt))
            {
                SetQwenImageCompanionEnv("--video-dit2", "TS_VIDEO_DIT2", videoDit2Opt);
                changed = true;
                continue;
            }
            // Audio VAE for models that generate an audio track jointly with the video.
            if (TryReadOption(args, ref i, "--audio-vae", out string? audioVaeOpt))
            {
                SetQwenImageCompanionEnv("--audio-vae", "TS_VIDEO_AUDIO_VAE", audioVaeOpt);
                changed = true;
                continue;
            }
            // Default output size for image requests that name neither a size nor a
            // target area of their own. Read by QwenImage21Pipeline.ResolveDimensions as
            // TS_QWEN_IMAGE_WIDTH/HEIGHT, which needs both (a half-set or unparsable pair is
            // ignored with a warning) and snaps an off-grid side down to a multiple of 32,
            // never below 32, warning once. Per-request sizes and areas from the Web UI /
            // API still override this default. Neither half is refused here: --width/--height
            // are also the video size aliases (ParseArgs), and video rounds to its own grid
            // and takes a missing side from the conditioning image. What the image default
            // makes of an incomplete or off-grid pair is said once at startup instead
            // (DescribeQwenImageSizeDefaultWarnings).
            if (TryReadOption(args, ref i, "--width", out string? widthOpt))
            {
                SetQwenImageSizeEnv("--width", QwenImageWidthEnvVar, widthOpt);
                changed = true;
                continue;
            }
            if (TryReadOption(args, ref i, "--height", out string? heightOpt))
            {
                SetQwenImageSizeEnv("--height", QwenImageHeightEnvVar, heightOpt);
                changed = true;
                continue;
            }
        }
        // LoRA plug-ins for the Qwen-Image-2.1 transformer, in one ordered pass of their
        // own: --lora-scale and --lora-config bind to the --lora before them. The files
        // are checked now, so a typo fails at startup rather than on the first request.
        var loras = LoraCliFlags.Resolve(LoraCliFlags.Parse(args));
        if (loras.Count > 0)
        {
            Environment.SetEnvironmentVariable(LoraCliFlags.EnvironmentVariable, LoraCliFlags.ToJson(loras));
            changed = true;
        }
        return changed;
    }

    private static void SetQwenImageSizeEnv(string flag, string envVar, string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"Missing value for option '{flag}'.");
        if (!int.TryParse(value, CultureInfo.InvariantCulture, out int px) || px <= 0)
            throw new ArgumentException($"Option '{flag}' needs a positive integer (pixels), got '{value}'.");
        Environment.SetEnvironmentVariable(envVar, px.ToString(CultureInfo.InvariantCulture));
    }

    /// <summary>The Qwen-Image-2.1 default-size variables <c>--width</c> / <c>--height</c> publish.</summary>
    internal const string QwenImageWidthEnvVar = "TS_QWEN_IMAGE_WIDTH";

    /// <inheritdoc cref="QwenImageWidthEnvVar"/>
    internal const string QwenImageHeightEnvVar = "TS_QWEN_IMAGE_HEIGHT";

    /// <summary>
    /// What <c>--width</c> / <c>--height</c> will NOT do as the Qwen-Image-2.1 default image
    /// size, one warning per problem, or none. The host logs these once at startup when
    /// the hosted model is a Qwen-Image model.
    /// </summary>
    /// <remarks>
    /// Warned rather than refused: both flags are also the video size aliases, and video
    /// legitimately takes one side alone (MiniMax-H3 fills the other from the conditioning
    /// image) and rounds to its own grid. For the image default, though, an incomplete pair
    /// is ignored outright and an off-grid value cannot be used as given - and neither used
    /// to be said anywhere, so an operator who set <c>--width 1000</c> saw 2048x2048 images
    /// (or a refused request) with no clue why. The other half may come from the
    /// environment variable, which the pipeline reads the same way.
    /// </remarks>
    public static IReadOnlyList<string> DescribeQwenImageSizeDefaultWarnings(string[] args)
    {
        var warnings = new List<string>();
        if (args == null || args.Length == 0)
            return warnings;

        // Last one wins, as in ApplyQwenImageCompanionCliFlags, which already validated
        // each value as a positive integer.
        string? widthFlag = null;
        string? heightFlag = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (TryReadOption(args, ref i, "--width", out string? w))
                widthFlag = w;
            else if (TryReadOption(args, ref i, "--height", out string? h))
                heightFlag = h;
        }
        if (widthFlag == null && heightFlag == null)
            return warnings;

        int? width = ParsePixels(widthFlag ?? Environment.GetEnvironmentVariable(QwenImageWidthEnvVar));
        int? height = ParsePixels(heightFlag ?? Environment.GetEnvironmentVariable(QwenImageHeightEnvVar));
        if (width == null || height == null)
        {
            string given = width != null ? "--width" : "--height";
            string missing = width != null ? "--height" : "--width";
            warnings.Add(
                $"{given} was given without {missing}: the Qwen-Image-2.1 default image size needs both, so "
                + $"{given} is ignored for images, and image requests that name neither a size nor an area keep "
                + $"the automatic size (a 2048x2048 area; 1024x1024 on the cpu backend). Pass {missing} as well "
                + $"to set it (video requests still use "
                + $"{given} on its own).");
            return warnings;
        }

        foreach ((string flag, string? raw, int px) in new[]
                 {
                     ("--width", widthFlag, width.Value),
                     ("--height", heightFlag, height.Value),
                 })
        {
            if (raw == null || px % 32 == 0)
                continue;
            // The pipeline's own rule for an off-grid default: down to the grid, and never
            // below one 32-pixel tile.
            string outcome = px >= 32
                ? "snapped down to a multiple of 32 (" + (px / 32 * 32).ToString(CultureInfo.InvariantCulture) + ")"
                : "raised to 32, the smallest size";
            warnings.Add(
                $"{flag} {px.ToString(CultureInfo.InvariantCulture)} is not a multiple of 32, which Qwen-Image-2.1 "
                + $"requires: as its default image size it is {outcome} for image requests that name neither a "
                + "size nor an area. Pass a multiple of 32 to get exactly that size (video requests round it to "
                + "their own grid either way).");
        }
        return warnings;

        static int? ParsePixels(string? value) =>
            int.TryParse(value?.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) && parsed > 0
                ? parsed
                : null;
    }

    private static void SetQwenImageCompanionEnv(string flag, string envVar, string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException($"Missing value for option '{flag}'.");
        if (!File.Exists(path))
            throw new FileNotFoundException($"{flag} file not found: {path}", path);
        Environment.SetEnvironmentVariable(envVar, Path.GetFullPath(path));
    }

    /// <summary>
    /// Bag of nullable sampling overrides captured from the CLI. We track
    /// each field separately (as <see cref="Nullable{T}"/>) so the caller
    /// can distinguish "operator pinned this value" from "operator didn't
    /// supply any sampling flags" - that distinction matters for the
    /// CLI &gt; env var &gt; type-default precedence.
    /// </summary>
    private struct SamplingOverrides
    {
        public float? Temperature;
        public int? TopK;
        public float? TopP;
        public float? MinP;
        public float? RepetitionPenalty;
        public int? PenaltyLastN;
        public float? PresencePenalty;
        public float? FrequencyPenalty;
        public int? Seed;
        public List<string> StopSequences;
    }

    private static void ParseArgs(
        string[] args,
        out string? configuredModel,
        out string? configuredMmProj,
        out string? configuredBackend,
        out int? configuredMaxTokens,
        out int? configuredVideoFrames,
        out int? configuredVideoFps,
        out int? configuredVideoWidth,
        out int? configuredVideoHeight,
        out int? configuredVideoSteps,
        out string? configuredVideoMode,
        out SamplingOverrides configuredSampling,
        out SamplingPrecedence? configuredPrecedence,
        out ListenOverrides configuredListen,
        out UploadLimitOverrides configuredUploads,
        out bool configuredNoWebUi,
        out bool configuredNoPrefixCache,
        List<string>? parallelismArgs = null,
        bool ignoreUnknownOptions = false)
    {
        configuredModel = null;
        configuredMmProj = null;
        configuredBackend = null;
        configuredMaxTokens = null;
        configuredVideoFrames = null;
        configuredVideoFps = null;
        configuredVideoWidth = null;
        configuredVideoHeight = null;
        configuredVideoSteps = null;
        configuredVideoMode = null;
        configuredSampling = default;
        configuredPrecedence = null;
        configuredListen = default;
        configuredUploads = default;
        configuredNoWebUi = false;
        configuredNoPrefixCache = false;

        for (int i = 0; i < args.Length; i++)
        {
            if (TryReadOption(args, ref i, "--model", out string? modelOption))
            {
                configuredModel = modelOption;
                continue;
            }

            if (string.Equals(args[i], "--embeddings", StringComparison.OrdinalIgnoreCase)
                || TryReadOption(args, ref i, "--embedding-threads", out _)
                || TryReadOption(args, ref i, "--embedding-context-size", out _))
                continue;

            if (TryReadOption(args, ref i, "--mmproj", out string? mmProjOption))
            {
                configuredMmProj = mmProjOption;
                continue;
            }

            if (TryReadOption(args, ref i, "--backend", out string? backendOption))
            {
                configuredBackend = backendOption;
                continue;
            }

            if (TryReadOption(args, ref i, "--port", out string? portOption))
            {
                if (!TryParsePort(portOption, out int parsedPort))
                    throw new ArgumentException(
                        $"Invalid value for --port: '{portOption}'. Expected a port between 1 and 65535.");
                configuredListen.Port = parsedPort;
                continue;
            }

            if (TryReadOption(args, ref i, "--host", out string? hostOption))
            {
                if (string.IsNullOrWhiteSpace(hostOption))
                    throw new ArgumentException("Missing value for option '--host'.");
                configuredListen.Host = hostOption;
                continue;
            }

            if (TryReadOption(args, ref i, "--urls", out string? urlsOption))
            {
                if (string.IsNullOrWhiteSpace(urlsOption))
                    throw new ArgumentException("Missing value for option '--urls'.");
                configuredListen.Urls = urlsOption;
                continue;
            }

            if (TryReadOption(args, ref i, "--max-tokens", out string? maxTokensOption))
            {
                if (!TryParsePositiveInt(maxTokensOption, out int parsedMaxTokens))
                    throw new ArgumentException($"Invalid value for --max-tokens: '{maxTokensOption}'.");
                configuredMaxTokens = parsedMaxTokens;
                continue;
            }

            // --video-width / --video-height seed the size of every video request.
            // --width / --height ALSO seed them: an operator who starts the server
            // with a size reasonably expects video to use it, and the Web UI sends
            // no size of its own. (They keep their Qwen-Image meaning too;
            // that pass reads them separately and leaves them in argv.)
            if (TryReadOption(args, ref i, "--video-width", out string? videoWidthOption)
                || TryReadOption(args, ref i, "--width", out videoWidthOption))
            {
                if (!TryParsePositiveInt(videoWidthOption, out int parsedVideoWidth))
                    throw new ArgumentException(
                        $"Invalid value for --video-width: '{videoWidthOption}'. Expected a positive integer.");
                configuredVideoWidth = parsedVideoWidth;
                continue;
            }

            if (TryReadOption(args, ref i, "--video-height", out string? videoHeightOption)
                || TryReadOption(args, ref i, "--height", out videoHeightOption))
            {
                if (!TryParsePositiveInt(videoHeightOption, out int parsedVideoHeight))
                    throw new ArgumentException(
                        $"Invalid value for --video-height: '{videoHeightOption}'. Expected a positive integer.");
                configuredVideoHeight = parsedVideoHeight;
                continue;
            }

            if (TryReadOption(args, ref i, "--video-steps", out string? videoStepsOption))
            {
                if (!TryParsePositiveInt(videoStepsOption, out int parsedVideoSteps))
                    throw new ArgumentException(
                        $"Invalid value for --video-steps: '{videoStepsOption}'. Expected a positive integer.");
                configuredVideoSteps = parsedVideoSteps;
                continue;
            }

            if (TryReadOption(args, ref i, "--video-mode", out string? videoModeOption))
            {
                // Validate at startup rather than on the first request: a typo here
                // should stop the server coming up, not surface an hour later.
                TensorSharp.Models.MiniMaxH3.MiniMaxH3ModeResolver.Parse(videoModeOption);
                configuredVideoMode = videoModeOption;
                continue;
            }

            if (TryReadOption(args, ref i, "--video-frames", out string? videoFramesOption))
            {
                if (!TryParsePositiveInt(videoFramesOption, out int parsedVideoFrames))
                    throw new ArgumentException(
                        $"Invalid value for --video-frames: '{videoFramesOption}'. Expected a positive integer.");
                configuredVideoFrames = parsedVideoFrames;
                continue;
            }

            if (TryReadOption(args, ref i, "--fps", out string? fpsOption))
            {
                if (!TryParsePositiveInt(fpsOption, out int parsedFps))
                    throw new ArgumentException(
                        $"Invalid value for --fps: '{fpsOption}'. Expected a positive integer.");
                configuredVideoFps = parsedFps;
                continue;
            }

            if (TryReadOption(args, ref i, "--temperature", out string? tempOption))
            {
                configuredSampling.Temperature = ParseFloat("--temperature", tempOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--top-k", out string? topKOption))
            {
                configuredSampling.TopK = ParseInt("--top-k", topKOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--top-p", out string? topPOption))
            {
                configuredSampling.TopP = ParseFloat("--top-p", topPOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--min-p", out string? minPOption))
            {
                configuredSampling.MinP = ParseFloat("--min-p", minPOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--repeat-penalty", out string? repPenOption))
            {
                configuredSampling.RepetitionPenalty = ParseFloat("--repeat-penalty", repPenOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--repeat-last-n", out string? repeatLastNOption))
            {
                configuredSampling.PenaltyLastN = ParseInt("--repeat-last-n", repeatLastNOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--presence-penalty", out string? presPenOption))
            {
                configuredSampling.PresencePenalty = ParseFloat("--presence-penalty", presPenOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--frequency-penalty", out string? freqPenOption))
            {
                configuredSampling.FrequencyPenalty = ParseFloat("--frequency-penalty", freqPenOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--seed", out string? seedOption))
            {
                configuredSampling.Seed = ParseInt("--seed", seedOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--stop", out string? stopOption))
            {
                // The flag is repeatable so operators can pin multiple stop
                // sequences (e.g. `--stop "</s>" --stop "<|eot|>"`).
                configuredSampling.StopSequences ??= new List<string>();
                configuredSampling.StopSequences.Add(stopOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--sampling-precedence", out string? precedenceOption))
            {
                configuredPrecedence = ParseSamplingPrecedence("--sampling-precedence", precedenceOption);
                continue;
            }

            if (TryReadOption(args, ref i, "--upload-max-mb", out string? uploadMaxOption))
            {
                if (!TryParsePositiveInt(uploadMaxOption, out int parsedUploadMax))
                    throw new ArgumentException(
                        $"Invalid value for --upload-max-mb: '{uploadMaxOption}'. Expected a positive number of megabytes.");
                configuredUploads.MaxFileMb = parsedUploadMax;
                continue;
            }

            if (TryReadOption(args, ref i, "--upload-quota-mb", out string? uploadQuotaOption))
            {
                if (!TryParsePositiveInt(uploadQuotaOption, out int parsedUploadQuota))
                    throw new ArgumentException(
                        $"Invalid value for --upload-quota-mb: '{uploadQuotaOption}'. Expected a positive number of megabytes.");
                configuredUploads.QuotaMb = parsedUploadQuota;
                continue;
            }

            if (string.Equals(args[i], "--no-webui", StringComparison.OrdinalIgnoreCase))
            {
                configuredNoWebUi = true;
                continue;
            }

            if (string.Equals(args[i], "--no-prefix-cache", StringComparison.OrdinalIgnoreCase))
            {
                configuredNoPrefixCache = true;
                continue;
            }

            if (string.Equals(args[i], "--no-multi-agent", StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[i], "--agents-allow-worker-tools", StringComparison.OrdinalIgnoreCase)
                || TryReadAnyOption(args, ref i, AgentValueFlags))
                continue;

            // Agent Skills. The VALUES are read by SkillHostOptions.Parse, which
            // lives in TensorSharp.Runtime so this host and the CLI cannot drift
            // apart on a spelling — a config file's keys ARE CLI flags, and the same
            // file is expected to drive either. This block only consumes them so the
            // unknown-option trap below does not refuse to start.
            if (string.Equals(args[i], SkillHostOptions.DisableFlag, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[i], SkillHostOptions.NoDiscoveryFlag, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[i], SkillHostOptions.AllowScriptsFlag, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[i], SkillHostOptions.ListFlag, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[i], SkillHostOptions.AllowNetworkFlag, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            if (TryReadOption(args, ref i, SkillHostOptions.RootsFlag, out _)
                || TryReadOption(args, ref i, SkillHostOptions.SelectFlag, out _)
                || TryReadOption(args, ref i, SkillHostOptions.MaxRoundsFlag, out _)
                || TryReadOption(args, ref i, SkillHostOptions.SandboxFlag, out _))
            {
                continue;
            }

            if (TryReadOption(args, ref i, "--upload-ttl-hours", out string? uploadTtlOption))
            {
                if (!double.TryParse(uploadTtlOption, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsedTtlHours)
                    || parsedTtlHours <= 0)
                {
                    throw new ArgumentException(
                        $"Invalid value for --upload-ttl-hours: '{uploadTtlOption}'. Expected a positive number of hours.");
                }
                configuredUploads.TtlHours = parsedTtlHours;
                continue;
            }

            // Hidden easter-egg flag consumed by the entry point to
            // enable the animated mascot banner. Recognised here so it
            // doesn't trip the unknown-arg trap below.
            if (string.Equals(args[i], "--xzf", StringComparison.Ordinal))
            {
                continue;
            }

            // Continuous-batching flags are also consumed by an earlier pass
            // (ApplyContinuousBatchingCliFlag, including --prefill-chunk-size).
            // Skip here so ParseArgs doesn't trip the unknown-arg trap.
            if (string.Equals(args[i], "--continuous-batching", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[i], "--no-continuous-batching", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            if (TryReadOption(args, ref i, "--prefill-chunk-size", out _))
            {
                continue;
            }
            // --redis-url is consumed by ApplyRedisCliFlags(args) in a
            // separate earlier pass; skip it (and its value) here.
            if (TryReadOption(args, ref i, "--redis-url", out _))
            {
                continue;
            }
            // --kv-cache-dtype is consumed by ApplyKvCacheDtypeCliFlag(args)
            // in a separate earlier pass; skip it (and its value) here so it
            // doesn't trip the unknown-arg trap below.
            if (TryReadOption(args, ref i, "--kv-cache-dtype", out _))
            {
                continue;
            }
            // The MoE CPU offload flags are consumed by
            // ApplyMoeCpuOffloadCliFlags(args) in a separate earlier pass.
            if (TryReadOption(args, ref i, "--n-cpu-moe", out _) ||
                TryReadOption(args, ref i, "-ncmoe", out _) ||
                TryReadOption(args, ref i, "--cpu-moe-threads", out _))
            {
                continue;
            }
            if (string.Equals(args[i], "--cpu-moe", StringComparison.Ordinal) ||
                string.Equals(args[i], "-cmoe", StringComparison.Ordinal))
            {
                continue;
            }
            // --gpu-device is consumed by ApplyGpuDeviceCliFlag(args) in a
            // separate earlier pass; skip it (and its value) here so it
            // doesn't trip the unknown-arg trap below.
            if (TryReadOption(args, ref i, "--gpu-device", out _))
            {
                continue;
            }
            // Collect at option boundaries after other options have consumed
            // their operands; --stop "--layer-split=2" is literal stop text.
            if (TensorSharp.Distributed.ModelParallelismOptions.TryCollect(args, ref i, parallelismArgs))
            {
                continue;
            }
            // --list-gpus / --help exit in Program.cs before Build runs;
            // recognise them here anyway so a Build with them present
            // (tests, future reordering) doesn't trip the unknown-arg trap.
            if (string.Equals(args[i], "--list-gpus", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[i], "--help", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            // Speculative-decoding flags are consumed by
            // ApplySpeculativeCliFlags(args) in a separate earlier pass, which
            // READS argv without removing anything. Recognise + skip them here
            // so they don't trip the unknown-arg trap below.
            //
            // Driven off SpeculativeCliFlags' own tables rather than a second
            // hand-written list: the two lists used to be maintained
            // separately, so when the flags were renamed --mtp-* -> --spec*
            // the applier learned the new spellings and this trap did not.
            // Every documented --spec* flag then made the server refuse to
            // start with "Unknown option '--spec-draft'". A copy of a list is
            // a drift bug; consume the source of truth.
            if (MatchesAny(args[i], SpeculativeCliFlags.SwitchFlags))
            {
                continue;
            }
            if (TryReadAnyOption(args, ref i, SpeculativeCliFlags.ValueFlags)
                || TryReadOption(args, ref i, "--draft-model", out _))
            {
                continue;
            }
            // Code-execution flags are consumed by CodeExecOptions.Parse in
            // Program.cs, which REMOVES them from argv before Build ever runs.
            // Recognise + skip them here anyway so a Build handed the raw command
            // line — a test, a future reordering, an embedder — doesn't trip the
            // unknown-arg trap on a flag the server documents and accepts.
            //
            // Driven off CodeExecOptions' own constants for the same reason the
            // speculative block above is: a hand-copied list is a drift bug waiting
            // to happen, and this one already bit — the flags reached --help while
            // this trap still rejected them.
            if (MatchesAny(args[i], CodeExecSwitchFlags))
            {
                continue;
            }
            if (TryReadAnyOption(args, ref i, CodeExecValueFlags))
            {
                continue;
            }
            // Qwen-Image-2.1 companion flags are consumed by
            // ApplyQwenImageCompanionCliFlags(args) in a separate
            // pass. Recognise + skip them here so they don't trip the
            // unknown-arg trap below.
            if (TryReadOption(args, ref i, "--qwen-image-vae", out _)
                || TryReadOption(args, ref i, "--qwen-image-vl", out _)
                || TryReadOption(args, ref i, "--qwen-image-mmproj", out _)
                || TryReadOption(args, ref i, QwenImageVariantFlag.Flag, out _)
                || TryReadOption(args, ref i, "--width", out _)
                || TryReadOption(args, ref i, "--height", out _)
                || TryReadOption(args, ref i, LoraCliFlags.LoraFlag, out _)
                || TryReadOption(args, ref i, LoraCliFlags.ScaleFlag, out _)
                || TryReadOption(args, ref i, LoraCliFlags.ConfigFlag, out _))
            {
                continue;
            }
            // Video companions, consumed by the same earlier pass. Each one must be
            // listed here too, or it reaches the unknown-flag trap below and the
            // server refuses to start.
            if (TryReadOption(args, ref i, "--video-vae", out _)
                || TryReadOption(args, ref i, "--video-text-encoder", out _)
                || TryReadOption(args, ref i, "--video-dit2", out _)
                || TryReadOption(args, ref i, "--audio-vae", out _)
                || TryReadOption(args, ref i, "--video-width", out _)
                || TryReadOption(args, ref i, "--video-height", out _)
                || TryReadOption(args, ref i, "--video-steps", out _)
                || TryReadOption(args, ref i, "--video-mode", out _))
            {
                continue;
            }

            if (ignoreUnknownOptions)
                continue;

            // Anything else that starts with `--` is an unknown flag and we
            // refuse to start. Previously these were silently dropped, so a
            // typo like `--mproj <path>` (instead of `--mmproj`) would launch
            // the server with no vision projector and produce image-unrelated
            // output later. Fail fast so the operator sees the typo at
            // startup, not as a confusing inference bug.
            if (args[i].StartsWith("--", StringComparison.Ordinal))
            {
                string? suggestion = SuggestFlagCorrection(args[i]);
                string suffix = suggestion != null ? $" Did you mean '{suggestion}'?" : string.Empty;
                throw new ArgumentException($"Unknown option '{args[i]}'.{suffix}");
            }

            // Bare positional arg (no '--' prefix) — also unsupported but
            // produce a clearer error so the operator knows it's not a
            // value attached to an above option.
            throw new ArgumentException($"Unexpected positional argument '{args[i]}'.");
        }
    }

    /// <summary>Suggest a known flag that differs from the typo by one
    /// character (insertion, deletion, or substitution) — covers `--mproj`
    /// → `--mmproj`, `--temprature` → `--temperature`, etc. Returns null
    /// when no flag is within edit distance 2.</summary>
    private static string? SuggestFlagCorrection(string typo)
    {
        var knownFlags = new List<string>
        {
            "--model", "--mmproj", "--backend", "--max-tokens", "--video-frames", "--fps",
            "--embeddings", "--embedding-threads", "--embedding-context-size",
            "--port", "--host", "--urls", "--no-webui", "--no-prefix-cache",
            "--temperature", "--top-k", "--top-p", "--min-p",
            "--repeat-penalty", "--repeat-last-n", "--presence-penalty", "--frequency-penalty",
            "--seed", "--stop", "--sampling-precedence",
            "--continuous-batching", "--no-continuous-batching", "--prefill-chunk-size",
            // Speculative flags come from SpeculativeCliFlags' tables below
            // (appended after this literal) so a new spelling is suggestible
            // the moment it is accepted.
            "--draft-model",
            "--redis-url",
            "--n-cpu-moe", "--cpu-moe", "--cpu-moe-threads",
            "--qwen-image-vae", "--qwen-image-vl", "--qwen-image-mmproj", QwenImageVariantFlag.Flag,
            "--video-vae", "--video-text-encoder", "--video-dit2", "--audio-vae",
            "--video-width", "--video-height", "--video-steps", "--video-mode",
            "--kv-cache-dtype", "--gpu-device", "--list-gpus", "--help",
            "--tp", "--tp-node-id", "--tp-peers",
            "--upload-max-mb", "--upload-quota-mb", "--upload-ttl-hours",
            SkillHostOptions.RootsFlag, SkillHostOptions.SelectFlag, SkillHostOptions.ListFlag,
            SkillHostOptions.DisableFlag, SkillHostOptions.NoDiscoveryFlag,
            SkillHostOptions.AllowScriptsFlag, SkillHostOptions.MaxRoundsFlag,
            SkillHostOptions.SandboxFlag, SkillHostOptions.AllowNetworkFlag,
            SkillHostOptions.SelectFlag, SkillHostOptions.ListFlag,
            // Without these, a typo like --code-exe fell through to a bare
            // "Unknown option" with no hint, because CodeExecOptions.Parse does not
            // match a misspelling and nothing else knew the names.
            "--config",
        };
        // Taken from CodeExecOptions' own tables rather than copied: without these a
        // typo like --code-exe fell through to a bare "Unknown option" with no hint,
        // because CodeExecOptions.Parse does not match a misspelling and nothing else
        // knew the names.
        knownFlags.AddRange(CodeExecOptions.SwitchFlags);
        knownFlags.AddRange(CodeExecOptions.ValueFlags);
        knownFlags.AddRange(LoraCliFlags.Flags);
        string? best = null;
        int bestDist = int.MaxValue;
        foreach (var flag in knownFlags)
        {
            int d0 = LevenshteinDistance(typo, flag);
            if (d0 < bestDist) { bestDist = d0; best = flag; }
        }
        foreach (var flag in SpeculativeCliFlags.SwitchFlags)
        {
            int d1 = LevenshteinDistance(typo, flag);
            if (d1 < bestDist) { bestDist = d1; best = flag; }
        }
        foreach (var flag in SpeculativeCliFlags.ValueFlags)
        {
            int d = LevenshteinDistance(typo, flag);
            if (d < bestDist) { bestDist = d; best = flag; }
        }
        // Only suggest if it's a near-miss (≤ 2 edits) — beyond that the
        // suggestion is more confusing than helpful.
        return bestDist <= 2 ? best : null;
    }

    /// <summary>True when <paramref name="arg"/> is exactly one of
    /// <summary>
    /// The code-execution flags, taken from CodeExecOptions rather than copied, so
    /// adding one there cannot leave this parser (or the typo hints) behind.
    /// </summary>
    internal static readonly string[] CodeExecSwitchFlags =
        CodeExecOptions.SwitchFlags.ToArray();

    internal static readonly string[] CodeExecValueFlags =
        CodeExecOptions.ValueFlags.ToArray();

    /// <paramref name="flags"/> (case-insensitive). Used to consume the
    /// valueless switches an earlier applier pass already handled.</summary>
    private static bool MatchesAny(string arg, string[] flags)
    {
        foreach (var flag in flags)
        {
            if (string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    /// <summary>Consume the first of <paramref name="flags"/> that matches at
    /// <paramref name="index"/>, in the order given (longest names first, so
    /// a shorter flag can never eat a longer flag's value).</summary>
    private static bool TryReadAnyOption(string[] args, ref int index, string[] flags)
    {
        foreach (var flag in flags)
        {
            if (TryReadOption(args, ref index, flag, out _))
                return true;
        }
        return false;
    }

    private static int LevenshteinDistance(string a, string b)
    {
        if (string.IsNullOrEmpty(a)) return b?.Length ?? 0;
        if (string.IsNullOrEmpty(b)) return a.Length;
        int[,] d = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
        for (int j = 0; j <= b.Length; j++) d[0, j] = j;
        for (int i = 1; i <= a.Length; i++)
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + cost);
            }
        return d[a.Length, b.Length];
    }

    private static float ParseFloat(string flag, string value)
    {
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
            throw new ArgumentException($"Invalid value for {flag}: '{value}'.");
        return parsed;
    }

    private static int ParseInt(string flag, string value)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed))
            throw new ArgumentException($"Invalid value for {flag}: '{value}'.");
        return parsed;
    }

    /// <summary>
    /// Layer environment-variable fallbacks under the CLI overrides. CLI wins
    /// (CLI args are the operator's most explicit intent), then env vars,
    /// then the type's built-in <see cref="SamplingConfig"/> defaults.
    /// Returning a fresh instance instead of <c>null</c> lets adapters call
    /// <see cref="SamplingConfig.Clone"/> on it without worrying about it
    /// being missing.
    ///
    /// Every value that came from a flag or an env var — as opposed to the
    /// type's built-in default — is also recorded in the returned
    /// <see cref="SamplingDefaults.Pinned"/> mask, because only those
    /// represent a decision the operator actually made and therefore only
    /// those outrank a client's request under
    /// <see cref="SamplingPrecedence.Config"/>.
    /// </summary>
    private static SamplingDefaults ResolveDefaultSamplingConfig(
        SamplingOverrides overrides,
        SamplingPrecedence? configuredPrecedence)
    {
        var resolved = new SamplingConfig();
        var pinned = SamplingField.None;

        if (overrides.Temperature.HasValue) { resolved.Temperature = overrides.Temperature.Value; pinned |= SamplingField.Temperature; }
        else if (TryReadEnvFloat("TENSORSHARP_TEMPERATURE", out float envTemp)) { resolved.Temperature = envTemp; pinned |= SamplingField.Temperature; }

        if (overrides.TopK.HasValue) { resolved.TopK = overrides.TopK.Value; pinned |= SamplingField.TopK; }
        else if (TryReadEnvInt("TENSORSHARP_TOP_K", out int envTopK)) { resolved.TopK = envTopK; pinned |= SamplingField.TopK; }

        if (overrides.TopP.HasValue) { resolved.TopP = overrides.TopP.Value; pinned |= SamplingField.TopP; }
        else if (TryReadEnvFloat("TENSORSHARP_TOP_P", out float envTopP)) { resolved.TopP = envTopP; pinned |= SamplingField.TopP; }

        if (overrides.MinP.HasValue) { resolved.MinP = overrides.MinP.Value; pinned |= SamplingField.MinP; }
        else if (TryReadEnvFloat("TENSORSHARP_MIN_P", out float envMinP)) { resolved.MinP = envMinP; pinned |= SamplingField.MinP; }

        if (overrides.RepetitionPenalty.HasValue) { resolved.RepetitionPenalty = overrides.RepetitionPenalty.Value; pinned |= SamplingField.RepetitionPenalty; }
        else if (TryReadEnvFloat("TENSORSHARP_REPEAT_PENALTY", out float envRep)) { resolved.RepetitionPenalty = envRep; pinned |= SamplingField.RepetitionPenalty; }

        if (overrides.PenaltyLastN.HasValue) { resolved.PenaltyLastN = overrides.PenaltyLastN.Value; pinned |= SamplingField.PenaltyLastN; }
        else if (TryReadEnvInt("TENSORSHARP_REPEAT_LAST_N", out int envLastN)) { resolved.PenaltyLastN = envLastN; pinned |= SamplingField.PenaltyLastN; }

        if (overrides.PresencePenalty.HasValue) { resolved.PresencePenalty = overrides.PresencePenalty.Value; pinned |= SamplingField.PresencePenalty; }
        else if (TryReadEnvFloat("TENSORSHARP_PRESENCE_PENALTY", out float envPres)) { resolved.PresencePenalty = envPres; pinned |= SamplingField.PresencePenalty; }

        if (overrides.FrequencyPenalty.HasValue) { resolved.FrequencyPenalty = overrides.FrequencyPenalty.Value; pinned |= SamplingField.FrequencyPenalty; }
        else if (TryReadEnvFloat("TENSORSHARP_FREQUENCY_PENALTY", out float envFreq)) { resolved.FrequencyPenalty = envFreq; pinned |= SamplingField.FrequencyPenalty; }

        if (overrides.Seed.HasValue) { resolved.Seed = overrides.Seed.Value; pinned |= SamplingField.Seed; }
        else if (TryReadEnvInt("TENSORSHARP_SEED", out int envSeed)) { resolved.Seed = envSeed; pinned |= SamplingField.Seed; }

        // Stop sequences only support CLI overrides for now: the env var
        // would need an unambiguous list separator and that's overkill.
        if (overrides.StopSequences != null)
        {
            resolved.StopSequences = new List<string>(overrides.StopSequences);
            pinned |= SamplingField.StopSequences;
        }

        SamplingPrecedence precedence = configuredPrecedence
            ?? ReadEnvSamplingPrecedence()
            ?? SamplingPrecedence.Config;

        return new SamplingDefaults(resolved, pinned, precedence);
    }

    /// <summary>Parse the <c>config</c>/<c>request</c> value of <c>--sampling-precedence</c>.</summary>
    private static SamplingPrecedence ParseSamplingPrecedence(string flag, string value)
    {
        if (string.Equals(value, "config", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "server", StringComparison.OrdinalIgnoreCase))
            return SamplingPrecedence.Config;
        if (string.Equals(value, "request", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "client", StringComparison.OrdinalIgnoreCase))
            return SamplingPrecedence.Request;
        throw new ArgumentException(
            $"Invalid value for {flag}: '{value}'. Expected 'config' (server-pinned values win) or 'request' (client values win).");
    }

    private static SamplingPrecedence? ReadEnvSamplingPrecedence()
    {
        string? raw = Environment.GetEnvironmentVariable("TENSORSHARP_SAMPLING_PRECEDENCE");
        if (string.IsNullOrWhiteSpace(raw))
            return null;
        return ParseSamplingPrecedence("TENSORSHARP_SAMPLING_PRECEDENCE", raw.Trim());
    }

    private static bool TryReadEnvFloat(string name, out float value)
    {
        string? raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            value = 0f;
            return false;
        }
        return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    private static bool TryReadEnvInt(string name, out int value)
    {
        string? raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            value = 0;
            return false;
        }
        return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    private static bool TryReadOption(string[] args, ref int index, string option, [NotNullWhen(true)] out string? value)
    {
        string arg = args[index];
        if (string.Equals(arg, option, StringComparison.OrdinalIgnoreCase))
        {
            if (index + 1 >= args.Length)
                throw new ArgumentException($"Missing value for option '{option}'.");

            value = args[++index];
            return true;
        }

        string prefix = option + "=";
        if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            value = arg[prefix.Length..];
            return true;
        }

        value = null;
        return false;
    }

    private static bool TryParsePositiveInt(string? value, out int parsed)
    {
        if (int.TryParse(value, out parsed) && parsed > 0)
            return true;

        parsed = 0;
        return false;
    }

    private static string? ResolveConfiguredModelPath(string? configuredPath)
    {
        if (string.IsNullOrWhiteSpace(configuredPath))
            return null;

        return Path.GetFullPath(configuredPath);
    }

    private static string? ResolveConfiguredMmProjPath(string? configuredPath, string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(configuredPath))
            return null;

        if (string.Equals(configuredPath, "none", StringComparison.OrdinalIgnoreCase))
            return null;

        if (Path.IsPathRooted(configuredPath) ||
            configuredPath.Contains(Path.DirectorySeparatorChar) ||
            configuredPath.Contains(Path.AltDirectorySeparatorChar) ||
            File.Exists(configuredPath))
        {
            return Path.GetFullPath(configuredPath);
        }

        string? preferredDirectory = Path.GetDirectoryName(modelPath);
        if (string.IsNullOrWhiteSpace(preferredDirectory))
            return Path.GetFullPath(configuredPath);

        return Path.GetFullPath(Path.Combine(preferredDirectory, configuredPath));
    }
}
