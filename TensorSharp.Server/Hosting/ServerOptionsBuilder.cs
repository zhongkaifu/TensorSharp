// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.Runtime.Speculative;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Runtime;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Reads CLI arguments and environment variables and produces a fully
    /// resolved <see cref="ServerHostingOptions"/>. Pure (no I/O beyond <see cref="Path"/>
    /// helpers and probing the host for supported backends), which makes it easy
    /// to test without spinning up a web app.
    /// </summary>
    internal static class ServerOptionsBuilder
    {
        private const int DefaultMaxTokensFallback = 20000;

        public static ServerHostingOptions Build(string[] args, string baseDirectory)
        {
            if (args == null) throw new ArgumentNullException(nameof(args));
            if (string.IsNullOrEmpty(baseDirectory)) throw new ArgumentNullException(nameof(baseDirectory));

            ParseArgs(args,
                out string configuredModel,
                out string configuredMmProj,
                out string configuredBackend,
                out int? configuredMaxTokens,
                out int? configuredVideoFrames,
                out int? configuredVideoFps,
                out int? configuredVideoWidth,
                out int? configuredVideoHeight,
                out int? configuredVideoSteps,
                out string configuredVideoMode,
                out SamplingOverrides configuredSampling,
                out SamplingPrecedence? configuredPrecedence,
                out ListenOverrides configuredListen,
                out UploadLimitOverrides configuredUploads,
                out bool configuredNoWebUi);

            if (!string.IsNullOrWhiteSpace(configuredMmProj) && string.IsNullOrWhiteSpace(configuredModel))
                throw new ArgumentException("--mmproj requires --model.");

            string startupModelPath = ResolveConfiguredModelPath(configuredModel);
            string startupMmProjPath = ResolveConfiguredMmProjPath(configuredMmProj, startupModelPath);

            string backendInput = configuredBackend ?? Environment.GetEnvironmentVariable("BACKEND");
            string requestedBackend = backendInput ?? (OperatingSystem.IsMacOS() ? "ggml_metal" : "ggml_cpu");

            var supportedBackends = BackendCatalog.GetSupportedBackends().ToArray();
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

            string uploadDirectory = Path.Combine(baseDirectory, "uploads");
            Directory.CreateDirectory(uploadDirectory);

            string logDirectory = Environment.GetEnvironmentVariable("TENSORSHARP_LOG_DIR");
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
            string noWebUiEnv = Environment.GetEnvironmentVariable("TS_NO_WEBUI");
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
                skillOptions.AllowNetwork);
        }

        /// <summary>Backend originally requested via <c>--backend</c> / <c>BACKEND</c> (without the OS-default fallback).</summary>
        public static string ReadConfiguredBackendInput(string[] args)
        {
            ParseArgs(args, out _, out _, out string configuredBackend, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _, out _);
            return configuredBackend ?? Environment.GetEnvironmentVariable("BACKEND");
        }

        /// <summary>Upload storage-limit overrides captured from the CLI (see <see cref="UploadStoragePolicy"/>).</summary>
        private struct UploadLimitOverrides
        {
            public int? MaxFileMb;
            public int? QuotaMb;
            public double? TtlHours;
        }

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
            string raw = Environment.GetEnvironmentVariable(envVar);
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
            string envHost = ReadEnvHost();
            if (envPort.HasValue || envHost != null)
                return BuildListenUrl(envHost ?? DefaultHost, envPort ?? ServerHostingOptions.DefaultPort);

            string aspnetUrls = Environment.GetEnvironmentVariable("ASPNETCORE_URLS");
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
                scheme = host.Substring(0, schemeIndex + 3);
                host = host.Substring(schemeIndex + 3);
            }

            // Bracket an unbracketed IPv6 literal. Detected by a second colon:
            // "::1" and "fe80::1" have one, "localhost" and "10.0.0.1" do not.
            if (host.IndexOf(':') != host.LastIndexOf(':') && !host.StartsWith("[", StringComparison.Ordinal))
                host = "[" + host + "]";

            return $"{scheme}{host}:{port.ToString(CultureInfo.InvariantCulture)}";
        }

        /// <summary>Read <c>PORT</c>, the variable container platforms inject.</summary>
        private static int? ReadEnvPort()
        {
            string raw = Environment.GetEnvironmentVariable("PORT");
            if (string.IsNullOrWhiteSpace(raw))
                return null;
            if (!TryParsePort(raw.Trim(), out int port))
                throw new ArgumentException($"Invalid PORT environment variable: '{raw}'. Expected a port between 1 and 65535.");
            return port;
        }

        private static string ReadEnvHost()
        {
            string raw = Environment.GetEnvironmentVariable("HOST");
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
        /// into the two env vars that gate the batched path:
        /// <c>TS_SCHED_DISABLE_BATCHED</c> (scheduler — falls through to
        /// per-sequence KV-swap when set) and <c>TS_QWEN35_BATCHED</c>
        /// (model — Qwen3.5 ForwardBatch gate; default ON, set to 0 to force
        /// the per-seq fallback). Both default to ON, so operators get
        /// paged-attention continuous batching without setting any env vars
        /// and without passing any flag; <c>--continuous-batching</c> is
        /// idempotent with the default, kept for explicit operator intent.
        /// <c>--no-continuous-batching</c> forces the per-seq path for every
        /// model.
        ///
        /// Must run before <see cref="InferenceEngine"/> is constructed because
        /// <c>BatchExecutor</c> and Qwen3.5's <c>SupportsBatchedMultimodal</c>
        /// read the env vars at runtime on each step.
        /// </summary>
        public static bool ApplyContinuousBatchingCliFlag(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            bool changed = false;
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                if (string.Equals(a, "--continuous-batching", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(a, "--paged-batching", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");
                    Environment.SetEnvironmentVariable("TS_QWEN35_BATCHED", "1");
                    changed = true;
                    continue;
                }
                if (string.Equals(a, "--no-continuous-batching", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(a, "--no-paged-batching", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
                    Environment.SetEnvironmentVariable("TS_QWEN35_BATCHED", "0");
                    changed = true;
                    continue;
                }
                // Tune chunked-prefill granularity. Each prefill chunk runs
                // as a single ExecuteStep that holds ModelBase.GpuComputeLock
                // for the duration of its forward pass, so smaller chunks
                // give parallel decode requests more frequent turns at the
                // GPU. Default 1024 (see SchedulerConfig.MaxPrefillChunkSize).
                if (TryReadOption(args, ref i, "--prefill-chunk-size", out string chunkOpt))
                {
                    if (!int.TryParse(chunkOpt, out int chunk) || chunk <= 0)
                        throw new ArgumentException($"Invalid value for --prefill-chunk-size: '{chunkOpt}'.");
                    Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", chunk.ToString());
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

            for (int i = 0; i < args.Length; i++)
            {
                if (TryReadOption(args, ref i, "--kv-cache-dtype", out string dtypeOpt))
                {
                    if (!TensorSharp.Models.KvCacheDtypeConfig.TryParse(dtypeOpt, out var dtype))
                        throw new ArgumentException(
                            $"Unknown --kv-cache-dtype value '{dtypeOpt}'. Valid: f32, f16, q8_0, q4_0.");
                    TensorSharp.Models.KvCacheDtypeConfig.Set(dtype);
                    return true;
                }
            }
            return false;
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
                if (TryReadOption(args, ref i, "--n-cpu-moe", out string ncmoe) ||
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
                if (TryReadOption(args, ref i, "--cpu-moe-threads", out string threads))
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

            for (int i = 0; i < args.Length; i++)
            {
                if (TryReadOption(args, ref i, "--gpu-device", out string gpuOpt))
                {
                    if (!int.TryParse(gpuOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int gpuIndex) || gpuIndex < 0)
                        throw new ArgumentException($"Invalid value for --gpu-device: '{gpuOpt}'. Expected a non-negative Vulkan device index.");
                    Environment.SetEnvironmentVariable(
                        TensorSharp.GGML.GgmlBasicOps.VulkanDeviceEnvVar,
                        gpuIndex.ToString(CultureInfo.InvariantCulture));
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Translate the tensor-parallelism flags (<c>--tp N</c> /
        /// <c>--tp-node-id N</c> / <c>--tp-peers host:port,...</c>) into the
        /// <c>TENSORSHARP_TP_*</c> env vars the model loader already reads:
        /// <c>ModelBase.Create</c> picks up the local degree and
        /// <c>DistributedTpConfig.TryFromEnvironment</c> picks up the multi-node
        /// pair when the startup model is loaded. Mirrors the CLI's flags so a
        /// single command line drives multi-GPU serving without env vars. Must
        /// run before <see cref="StartupModelLoader"/> so the very first model
        /// load is sharded. Returns true when at least one flag was applied so
        /// the caller can emit a startup-log line.
        /// </summary>
        public static bool ApplyTensorParallelCliFlags(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            bool changed = false;
            bool nodeIdSeen = false;
            bool peersSeen = false;
            for (int i = 0; i < args.Length; i++)
            {
                if (TryReadOption(args, ref i, "--tp", out string tpOpt))
                {
                    if (!int.TryParse(tpOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int tpDegree) || tpDegree < 1)
                        throw new ArgumentException($"Invalid value for --tp: '{tpOpt}'. Expected the number of local GPUs to split the model across (an integer >= 1).");
                    Environment.SetEnvironmentVariable("TENSORSHARP_TP_DEGREE", tpDegree.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--tp-node-id", out string nodeIdOpt))
                {
                    if (!int.TryParse(nodeIdOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int nodeId) || nodeId < 0)
                        throw new ArgumentException($"Invalid value for --tp-node-id: '{nodeIdOpt}'. Expected this node's 0-based ID within the distributed TP cluster.");
                    Environment.SetEnvironmentVariable("TENSORSHARP_TP_NODE_ID", nodeId.ToString(CultureInfo.InvariantCulture));
                    nodeIdSeen = true;
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--tp-peers", out string peersOpt))
                {
                    // Validate the host:port list now so a malformed endpoint
                    // fails at startup with the flag name, not later inside the
                    // model load with a bare parse error.
                    try
                    {
                        TensorSharp.Distributed.DistributedTpConfig.ParsePeers(peersOpt);
                    }
                    catch (Exception ex) when (ex is ArgumentException or FormatException)
                    {
                        throw new ArgumentException(
                            $"Invalid value for --tp-peers: '{peersOpt}'. Expected a comma-separated host:port list " +
                            $"(e.g. 192.168.1.10:9500,192.168.1.11:9500). {ex.Message}");
                    }
                    Environment.SetEnvironmentVariable("TENSORSHARP_TP_PEERS", peersOpt);
                    peersSeen = true;
                    changed = true;
                    continue;
                }
            }

            // Distributed mode needs BOTH the node id and the peer list;
            // DistributedTpConfig silently stays single-node when either is
            // missing, so a half-configured pair would run without TP and the
            // operator would only notice via slow/OOM inference. Fail fast
            // instead. Checked against the resulting env state so one half may
            // legitimately come from the environment.
            if (nodeIdSeen || peersSeen)
            {
                bool haveNodeId = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID"));
                bool havePeers = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS"));
                if (haveNodeId != havePeers)
                {
                    throw new ArgumentException(haveNodeId
                        ? "--tp-node-id requires --tp-peers (comma-separated host:port list of all nodes in the cluster)."
                        : "--tp-peers requires --tp-node-id (this node's 0-based ID within the cluster).");
                }
            }

            return changed;
        }

        /// <summary>
        /// Translate <c>--spec</c> / <c>--no-spec</c> /
        /// <c>--spec-draft N</c> / <c>--spec-pmin X</c> / <c>--draft-model PATH</c> into the
        /// env vars read by <c>SchedulerConfig.FromEnvironment</c> when the
        /// inference engine is constructed. NextN/MTP speculative decoding only
        /// engages on models that ship a draft head (Qwen3.6) and is off by
        /// default. Returns true when at least one flag was applied so the
        /// caller can emit a startup-log line.
        /// </summary>
        public static bool ApplySpeculativeCliFlags(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            // The speculative-decoding flags mean the same thing in both hosts,
            // so they are parsed and validated in one shared place
            // (TensorSharp.Runtime.Speculative) rather than kept in step by hand.
            bool changed = SpeculativeCliFlags.Apply(args);

            for (int i = 0; i < args.Length; i++)
            {
                // The one --draft-model also has to reach the model FACTORY for the
                // block drafters that must be resident before the layer split runs
                // (DeepSeek V4's DSpark, the DFlash / DFlash2 drafters for
                // Muse-Glimmer and Qwen 3.8). SpeculativeCliFlags.Apply above
                // already validated the path, published it for the attach-after-load
                // path, and enabled speculation; these env vars are how the factory
                // and a runtime model switch see it.
                //
                // All THREE are set, because each architecture reads its own and
                // only the loaded model reads any of them. Setting just the DSpark
                // one - which is what this did - meant --draft-model was silently
                // ignored on the server for every DFlash target, the flag's most
                // common use. A file that is not a block drafter at all (Gemma 4's
                // per-token assistant head) is simply never read from these: the
                // loaders probe the GGUF's own declared architecture, so the
                // operator never has to know which kind their file is.
                if (SpeculativeCliFlags.TryReadOption(args, ref i, "--draft-model", out string dsparkOpt))
                {
                    Environment.SetEnvironmentVariable("TS_DSV4_DSPARK", dsparkOpt);
                    Environment.SetEnvironmentVariable("TS_QWEN35_DFLASH", dsparkOpt);
                    Environment.SetEnvironmentVariable("TS_MUSE_GLIMMER_DFLASH", dsparkOpt);
                    Environment.SetEnvironmentVariable("TS_NEMOTRON_DFLASH", dsparkOpt);
                    changed = true;
                }
            }
            return changed;
        }

        /// <summary>
        /// Translate <c>--paged-kv*</c> CLI flags into the env vars consumed by
        /// <see cref="PagedKvCacheConfig.FromEnvironment"/>. Returns true when at
        /// least one flag was applied so the caller can emit a startup-log line.
        /// These flags are retained for CLI compatibility with older server
        /// builds; the continuous-batching engine reads its scheduler knobs
        /// from <c>TS_SCHED_*</c>.
        /// </summary>
        public static bool ApplyPagedKvCacheCliFlags(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            bool changed = false;
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                if (string.Equals(a, "--paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "--paged-kv-cache", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_KV_PAGED_CACHE", "1");
                    changed = true;
                    continue;
                }
                if (string.Equals(a, "--no-paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "--no-paged-kv-cache", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_KV_PAGED_CACHE", "0");
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-block-size", out string blockSizeOpt))
                {
                    if (!int.TryParse(blockSizeOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int blockSize) || blockSize <= 0)
                        throw new ArgumentException($"Invalid value for --paged-kv-block-size: '{blockSizeOpt}'.");
                    Environment.SetEnvironmentVariable("TS_KV_BLOCK_SIZE", blockSize.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-ram-mb", out string ramOpt))
                {
                    if (!long.TryParse(ramOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out long ramMb) || ramMb <= 0)
                        throw new ArgumentException($"Invalid value for --paged-kv-ram-mb: '{ramOpt}'.");
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_MAX_RAM_MB", ramMb.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-ssd-dir", out string ssdDirOpt))
                {
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_SSD_DIR", ssdDirOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-ssd-mb", out string ssdMbOpt))
                {
                    if (!long.TryParse(ssdMbOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out long ssdMb) || ssdMb <= 0)
                        throw new ArgumentException($"Invalid value for --paged-kv-ssd-mb: '{ssdMbOpt}'.");
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_MAX_SSD_MB", ssdMb.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-quant-bits", out string quantBitsOpt))
                {
                    // Accepts 0 (explicit off), 2, 4, or 8 - the same set the CLI
                    // takes and the same set TurboQuantKvCodec implements (2-bit
                    // uses its affine min+scale layout rather than the symmetric
                    // one 4/8-bit use). The paged manager gates the codec a second
                    // time on model.RequiresPerBlockCapture, so requesting a codec
                    // on a recurrent-state model still falls back to passthrough at
                    // runtime - the value here just records the operator's intent.
                    if (!int.TryParse(quantBitsOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int quantBits))
                        throw new ArgumentException($"Invalid value for --paged-kv-quant-bits: '{quantBitsOpt}'. Expected 0 (off), 2, 4, or 8.");
                    if (quantBits != 0 && quantBits != 2 && quantBits != 4 && quantBits != 8)
                        throw new ArgumentException($"Invalid value for --paged-kv-quant-bits: {quantBits}. Expected 0 (off), 2, 4, or 8.");
                    Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", quantBits.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-redis-url", out string redisUrlOpt))
                {
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_URL", redisUrlOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-redis-ttl", out string redisTtlOpt))
                {
                    if (!int.TryParse(redisTtlOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int redisTtl) || redisTtl < 0)
                        throw new ArgumentException($"Invalid value for --paged-kv-redis-ttl: '{redisTtlOpt}'. Expected minutes (0 = no TTL).");
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_TTL_MINUTES", redisTtl.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
            }
            return changed;
        }

        /// <summary>
        /// Translate <c>--redis-url &lt;url&gt;</c> into both
        /// <c>TS_KV_CACHE_REDIS_URL</c> and
        /// <c>TS_RESPONSES_STORE_REDIS_URL</c> so a single flag enables Redis
        /// for both the paged KV cache tier and the Responses API store.
        /// If either env var is already set, it is left untouched so that
        /// split configurations (different Redis instances per subsystem)
        /// are preserved.
        /// Returns true when the flag was present.
        /// </summary>
        public static bool ApplyRedisCliFlags(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            for (int i = 0; i < args.Length; i++)
            {
                if (TryReadOption(args, ref i, "--redis-url", out string redisUrl))
                {
                    if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_KV_CACHE_REDIS_URL")))
                        Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_URL", redisUrl);
                    if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL")))
                        Environment.SetEnvironmentVariable("TS_RESPONSES_STORE_REDIS_URL", redisUrl);
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Translate the Qwen-Image-Edit companion-model flags
        /// (<c>--qwen-image-vae</c> / <c>--qwen-image-vl</c> /
        /// <c>--qwen-image-mmproj</c>) into the env vars that
        /// <c>QwenImageModel</c> reads (<c>TS_QWEN_IMAGE_VAE</c> /
        /// <c>TS_QWEN_IMAGE_TE</c> / <c>TS_QWEN_IMAGE_MMPROJ</c>) — the existing
        /// override mechanism for the three networks the qwen_image DiT GGUF does
        /// not itself contain. Each path is validated here so a typo fails fast at
        /// startup instead of silently falling back to the same-directory scan.
        /// Must run before the startup model is loaded. Returns true when at least
        /// one flag was applied so the caller can emit a startup-log line.
        /// </summary>
        public static bool ApplyQwenImageCompanionCliFlags(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            bool changed = false;
            for (int i = 0; i < args.Length; i++)
            {
                if (TryReadOption(args, ref i, "--qwen-image-vae", out string vaeOpt))
                {
                    SetQwenImageCompanionEnv("--qwen-image-vae", "TS_QWEN_IMAGE_VAE", vaeOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--qwen-image-vl", out string vlOpt))
                {
                    SetQwenImageCompanionEnv("--qwen-image-vl", "TS_QWEN_IMAGE_TE", vlOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--qwen-image-mmproj", out string mmprojOpt))
                {
                    SetQwenImageCompanionEnv("--qwen-image-mmproj", "TS_QWEN_IMAGE_MMPROJ", mmprojOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--qwen-image-lora", out string loraOpt))
                {
                    // DiT LoRA (e.g. a lightx2v Lightning step-distillation checkpoint),
                    // merged into the quantized weights at model load. A Lightning LoRA
                    // also switches the sampling defaults (its step count, cfg 1.0,
                    // fixed timestep shift 3).
                    SetQwenImageCompanionEnv("--qwen-image-lora", "TS_QWEN_IMAGE_LORA", loraOpt);
                    changed = true;
                    continue;
                }
                // Video-generation companions (same env-var override mechanism). Each path
                // is published under the generic TS_VIDEO_* name AND the historical TS_WAN_*
                // one, so WanVideoModel keeps reading what it always did. The --wan-*
                // spellings predate the second video model and stay accepted; config files
                // in the wild use them.
                if (TryReadOption(args, ref i, "--video-vae", out string videoVaeOpt)
                    || TryReadOption(args, ref i, "--wan-vae", out videoVaeOpt))
                {
                    SetQwenImageCompanionEnv("--video-vae", "TS_VIDEO_VAE", videoVaeOpt);
                    SetQwenImageCompanionEnv("--video-vae", "TS_WAN_VAE", videoVaeOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--video-text-encoder", out string videoTeOpt)
                    || TryReadOption(args, ref i, "--video-te", out videoTeOpt)
                    || TryReadOption(args, ref i, "--wan-te", out videoTeOpt))
                {
                    SetQwenImageCompanionEnv("--video-text-encoder", "TS_VIDEO_TEXT_ENCODER", videoTeOpt);
                    SetQwenImageCompanionEnv("--video-text-encoder", "TS_WAN_TE", videoTeOpt);
                    changed = true;
                    continue;
                }
                // Dual-expert models (Wan 2.2 A14B) ship as a PAIR of GGUFs and need both.
                // They are auto-resolved by name when they sit together, but a config file
                // has to be able to name the second one explicitly — that is the only way
                // its auto-download entry can exist at all.
                if (TryReadOption(args, ref i, "--video-dit2", out string videoDit2Opt)
                    || TryReadOption(args, ref i, "--wan-dit2", out videoDit2Opt))
                {
                    SetQwenImageCompanionEnv("--video-dit2", "TS_VIDEO_DIT2", videoDit2Opt);
                    SetQwenImageCompanionEnv("--video-dit2", "TS_WAN_DIT2", videoDit2Opt);
                    changed = true;
                    continue;
                }
                // Audio VAE for models that generate an audio track jointly with the video.
                if (TryReadOption(args, ref i, "--audio-vae", out string audioVaeOpt))
                {
                    SetQwenImageCompanionEnv("--audio-vae", "TS_VIDEO_AUDIO_VAE", audioVaeOpt);
                    changed = true;
                    continue;
                }
                // Fixed output size for every edit (bypasses the auto VRAM area clamp, but is still
                // capped at the hardware memory ceiling so an oversized request can't OOM into
                // garbage). Read by QwenImagePipeline as TS_QWEN_IMAGE_WIDTH/HEIGHT. Per-request
                // sizes from the Web UI / API still override this default.
                if (TryReadOption(args, ref i, "--width", out string widthOpt))
                {
                    SetQwenImageSizeEnv("--width", "TS_QWEN_IMAGE_WIDTH", widthOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--height", out string heightOpt))
                {
                    SetQwenImageSizeEnv("--height", "TS_QWEN_IMAGE_HEIGHT", heightOpt);
                    changed = true;
                    continue;
                }
                // CPU offload (sd.cpp --offload-to-cpu equivalent): stream the DiT weights
                // from RAM per block instead of holding them resident in VRAM, so high
                // (native ~1 MP) resolutions fit on VRAM-limited cards. Without the flag the
                // pipeline engages offload automatically only when the target resolution
                // does not fit beside the resident weights; the flag forces it always on.
                if (string.Equals(args[i], "--offload-cpu", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_OFFLOAD_CPU", "1");
                    changed = true;
                    continue;
                }
            }
            return changed;
        }

        private static void SetQwenImageSizeEnv(string flag, string envVar, string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException($"Missing value for option '{flag}'.");
            if (!int.TryParse(value, out int px) || px <= 0)
                throw new ArgumentException($"Option '{flag}' needs a positive integer (pixels), got '{value}'.");
            Environment.SetEnvironmentVariable(envVar, px.ToString());
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
            out string configuredModel,
            out string configuredMmProj,
            out string configuredBackend,
            out int? configuredMaxTokens,
            out int? configuredVideoFrames,
            out int? configuredVideoFps,
            out int? configuredVideoWidth,
            out int? configuredVideoHeight,
            out int? configuredVideoSteps,
            out string configuredVideoMode,
            out SamplingOverrides configuredSampling,
            out SamplingPrecedence? configuredPrecedence,
            out ListenOverrides configuredListen,
            out UploadLimitOverrides configuredUploads,
            out bool configuredNoWebUi)
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

            for (int i = 0; i < args.Length; i++)
            {
                if (TryReadOption(args, ref i, "--model", out string modelOption))
                {
                    configuredModel = modelOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--mmproj", out string mmProjOption))
                {
                    configuredMmProj = mmProjOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--backend", out string backendOption))
                {
                    configuredBackend = backendOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--port", out string portOption))
                {
                    if (!TryParsePort(portOption, out int parsedPort))
                        throw new ArgumentException(
                            $"Invalid value for --port: '{portOption}'. Expected a port between 1 and 65535.");
                    configuredListen.Port = parsedPort;
                    continue;
                }

                if (TryReadOption(args, ref i, "--host", out string hostOption))
                {
                    if (string.IsNullOrWhiteSpace(hostOption))
                        throw new ArgumentException("Missing value for option '--host'.");
                    configuredListen.Host = hostOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--urls", out string urlsOption))
                {
                    if (string.IsNullOrWhiteSpace(urlsOption))
                        throw new ArgumentException("Missing value for option '--urls'.");
                    configuredListen.Urls = urlsOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--max-tokens", out string maxTokensOption))
                {
                    if (!TryParsePositiveInt(maxTokensOption, out int parsedMaxTokens))
                        throw new ArgumentException($"Invalid value for --max-tokens: '{maxTokensOption}'.");
                    configuredMaxTokens = parsedMaxTokens;
                    continue;
                }

                // --video-width / --video-height seed the size of every video request.
                // --width / --height ALSO seed them: an operator who starts the server
                // with a size reasonably expects video to use it, and the Web UI sends
                // no size of its own. (They keep their Qwen-Image-Edit meaning too;
                // that pass reads them separately and leaves them in argv.)
                if (TryReadOption(args, ref i, "--video-width", out string videoWidthOption)
                    || TryReadOption(args, ref i, "--width", out videoWidthOption))
                {
                    if (!TryParsePositiveInt(videoWidthOption, out int parsedVideoWidth))
                        throw new ArgumentException(
                            $"Invalid value for --video-width: '{videoWidthOption}'. Expected a positive integer.");
                    configuredVideoWidth = parsedVideoWidth;
                    continue;
                }

                if (TryReadOption(args, ref i, "--video-height", out string videoHeightOption)
                    || TryReadOption(args, ref i, "--height", out videoHeightOption))
                {
                    if (!TryParsePositiveInt(videoHeightOption, out int parsedVideoHeight))
                        throw new ArgumentException(
                            $"Invalid value for --video-height: '{videoHeightOption}'. Expected a positive integer.");
                    configuredVideoHeight = parsedVideoHeight;
                    continue;
                }

                if (TryReadOption(args, ref i, "--video-steps", out string videoStepsOption))
                {
                    if (!TryParsePositiveInt(videoStepsOption, out int parsedVideoSteps))
                        throw new ArgumentException(
                            $"Invalid value for --video-steps: '{videoStepsOption}'. Expected a positive integer.");
                    configuredVideoSteps = parsedVideoSteps;
                    continue;
                }

                if (TryReadOption(args, ref i, "--video-mode", out string videoModeOption))
                {
                    // Validate at startup rather than on the first request: a typo here
                    // should stop the server coming up, not surface an hour later.
                    TensorSharp.Models.MiniMaxH3.MiniMaxH3ModeResolver.Parse(videoModeOption);
                    configuredVideoMode = videoModeOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--video-frames", out string videoFramesOption))
                {
                    if (!TryParsePositiveInt(videoFramesOption, out int parsedVideoFrames))
                        throw new ArgumentException(
                            $"Invalid value for --video-frames: '{videoFramesOption}'. Expected a positive integer.");
                    configuredVideoFrames = parsedVideoFrames;
                    continue;
                }

                if (TryReadOption(args, ref i, "--fps", out string fpsOption))
                {
                    if (!TryParsePositiveInt(fpsOption, out int parsedFps))
                        throw new ArgumentException(
                            $"Invalid value for --fps: '{fpsOption}'. Expected a positive integer.");
                    configuredVideoFps = parsedFps;
                    continue;
                }

                if (TryReadOption(args, ref i, "--temperature", out string tempOption))
                {
                    configuredSampling.Temperature = ParseFloat("--temperature", tempOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--top-k", out string topKOption))
                {
                    configuredSampling.TopK = ParseInt("--top-k", topKOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--top-p", out string topPOption))
                {
                    configuredSampling.TopP = ParseFloat("--top-p", topPOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--min-p", out string minPOption))
                {
                    configuredSampling.MinP = ParseFloat("--min-p", minPOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--repeat-penalty", out string repPenOption))
                {
                    configuredSampling.RepetitionPenalty = ParseFloat("--repeat-penalty", repPenOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--repeat-last-n", out string repeatLastNOption))
                {
                    configuredSampling.PenaltyLastN = ParseInt("--repeat-last-n", repeatLastNOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--presence-penalty", out string presPenOption))
                {
                    configuredSampling.PresencePenalty = ParseFloat("--presence-penalty", presPenOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--frequency-penalty", out string freqPenOption))
                {
                    configuredSampling.FrequencyPenalty = ParseFloat("--frequency-penalty", freqPenOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--seed", out string seedOption))
                {
                    configuredSampling.Seed = ParseInt("--seed", seedOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--stop", out string stopOption))
                {
                    // The flag is repeatable so operators can pin multiple stop
                    // sequences (e.g. `--stop "</s>" --stop "<|eot|>"`).
                    configuredSampling.StopSequences ??= new List<string>();
                    configuredSampling.StopSequences.Add(stopOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--sampling-precedence", out string precedenceOption))
                {
                    configuredPrecedence = ParseSamplingPrecedence("--sampling-precedence", precedenceOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--upload-max-mb", out string uploadMaxOption))
                {
                    if (!TryParsePositiveInt(uploadMaxOption, out int parsedUploadMax))
                        throw new ArgumentException(
                            $"Invalid value for --upload-max-mb: '{uploadMaxOption}'. Expected a positive number of megabytes.");
                    configuredUploads.MaxFileMb = parsedUploadMax;
                    continue;
                }

                if (TryReadOption(args, ref i, "--upload-quota-mb", out string uploadQuotaOption))
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

                if (TryReadOption(args, ref i, "--upload-ttl-hours", out string uploadTtlOption))
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

                // Paged-KV flags are consumed by ApplyPagedKvCacheCliFlags(args)
                // in a separate earlier pass. They still appear in args[] when
                // ParseArgs walks the list, so recognise + skip them here to
                // keep them out of the unknown-arg trap below.
                if (string.Equals(args[i], "--paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--paged-kv-cache", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-paged-kv-cache", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                // Continuous-batching flags are also consumed by an earlier pass
                // (ApplyContinuousBatchingCliFlag, including --prefill-chunk-size).
                // Skip here so ParseArgs doesn't trip the unknown-arg trap.
                if (string.Equals(args[i], "--continuous-batching", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--paged-batching", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-continuous-batching", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-paged-batching", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                if (TryReadOption(args, ref i, "--prefill-chunk-size", out _))
                {
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-block-size", out _)
                    || TryReadOption(args, ref i, "--paged-kv-ram-mb", out _)
                    || TryReadOption(args, ref i, "--paged-kv-ssd-dir", out _)
                    || TryReadOption(args, ref i, "--paged-kv-ssd-mb", out _)
                    || TryReadOption(args, ref i, "--paged-kv-quant-bits", out _)
                    || TryReadOption(args, ref i, "--paged-kv-redis-url", out _)
                    || TryReadOption(args, ref i, "--paged-kv-redis-ttl", out _))
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
                // Tensor-parallelism flags are consumed by
                // ApplyTensorParallelCliFlags(args) in a separate earlier pass;
                // skip them (and their values) here so they don't trip the
                // unknown-arg trap below.
                if (TryReadOption(args, ref i, "--tp", out _)
                    || TryReadOption(args, ref i, "--tp-node-id", out _)
                    || TryReadOption(args, ref i, "--tp-peers", out _))
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
                // Qwen-Image-Edit companion-model flags are consumed by
                // ApplyQwenImageCompanionCliFlags(args) in a separate earlier
                // pass. Recognise + skip them here so they don't trip the
                // unknown-arg trap below.
                if (TryReadOption(args, ref i, "--qwen-image-vae", out _)
                    || TryReadOption(args, ref i, "--qwen-image-vl", out _)
                    || TryReadOption(args, ref i, "--qwen-image-mmproj", out _)
                    || TryReadOption(args, ref i, "--qwen-image-lora", out _)
                    || TryReadOption(args, ref i, "--width", out _)
                    || TryReadOption(args, ref i, "--height", out _))
                {
                    continue;
                }
                // Video companions, consumed by the same earlier pass. They were once
                // missing here, so `--wan-vae`/`--wan-te` — both documented in --help —
                // reached the unknown-flag trap below and the server refused to start.
                // Every spelling, old and new, must be listed or that regression returns.
                if (TryReadOption(args, ref i, "--video-vae", out _)
                    || TryReadOption(args, ref i, "--video-text-encoder", out _)
                    || TryReadOption(args, ref i, "--video-te", out _)
                    || TryReadOption(args, ref i, "--video-dit2", out _)
                    || TryReadOption(args, ref i, "--audio-vae", out _)
                    || TryReadOption(args, ref i, "--wan-vae", out _)
                    || TryReadOption(args, ref i, "--wan-te", out _)
                    || TryReadOption(args, ref i, "--wan-dit2", out _)
                    || TryReadOption(args, ref i, "--video-width", out _)
                    || TryReadOption(args, ref i, "--video-height", out _)
                    || TryReadOption(args, ref i, "--video-steps", out _)
                    || TryReadOption(args, ref i, "--video-mode", out _))
                {
                    continue;
                }
                if (string.Equals(args[i], "--offload-cpu", StringComparison.OrdinalIgnoreCase))
                {
                    continue;   // consumed by ApplyQwenImageCompanionCliFlags (boolean flag)
                }

                // Anything else that starts with `--` is an unknown flag and we
                // refuse to start. Previously these were silently dropped, so a
                // typo like `--mproj <path>` (instead of `--mmproj`) would launch
                // the server with no vision projector and produce image-unrelated
                // output later. Fail fast so the operator sees the typo at
                // startup, not as a confusing inference bug.
                if (args[i].StartsWith("--", StringComparison.Ordinal))
                {
                    string suggestion = SuggestFlagCorrection(args[i]);
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
        private static string SuggestFlagCorrection(string typo)
        {
            var knownFlags = new List<string>
            {
                "--model", "--mmproj", "--backend", "--max-tokens", "--video-frames", "--fps",
                "--port", "--host", "--urls", "--no-webui",
                "--temperature", "--top-k", "--top-p", "--min-p",
                "--repeat-penalty", "--repeat-last-n", "--presence-penalty", "--frequency-penalty",
                "--seed", "--stop", "--sampling-precedence",
                "--paged-kv", "--paged-kv-cache", "--no-paged-kv", "--no-paged-kv-cache",
                "--paged-kv-block-size", "--paged-kv-ram-mb",
                "--paged-kv-ssd-dir", "--paged-kv-ssd-mb", "--paged-kv-quant-bits",
                "--continuous-batching", "--no-continuous-batching",
                "--paged-batching", "--no-paged-batching", "--prefill-chunk-size",
                // Speculative flags come from SpeculativeCliFlags' tables below
                // (appended after this literal) so a new spelling is suggestible
                // the moment it is accepted.
                "--draft-model",
                "--redis-url", "--paged-kv-redis-url", "--paged-kv-redis-ttl",
                "--n-cpu-moe", "--cpu-moe", "--cpu-moe-threads",
                "--qwen-image-vae", "--qwen-image-vl", "--qwen-image-mmproj", "--qwen-image-lora",
                "--video-vae", "--video-text-encoder", "--video-te", "--video-dit2", "--audio-vae",
                "--video-width", "--video-height", "--video-steps", "--video-mode",
                "--wan-vae", "--wan-te", "--wan-dit2",
                "--offload-cpu",
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
            string best = null;
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
            string raw = Environment.GetEnvironmentVariable("TENSORSHARP_SAMPLING_PRECEDENCE");
            if (string.IsNullOrWhiteSpace(raw))
                return null;
            return ParseSamplingPrecedence("TENSORSHARP_SAMPLING_PRECEDENCE", raw.Trim());
        }

        private static bool TryReadEnvFloat(string name, out float value)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw))
            {
                value = 0f;
                return false;
            }
            return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryReadEnvInt(string name, out int value)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw))
            {
                value = 0;
                return false;
            }
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryReadOption(string[] args, ref int index, string option, out string value)
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
                value = arg.Substring(prefix.Length);
                return true;
            }

            value = null;
            return false;
        }

        private static bool TryParsePositiveInt(string value, out int parsed)
        {
            if (int.TryParse(value, out parsed) && parsed > 0)
                return true;

            parsed = 0;
            return false;
        }

        private static string ResolveConfiguredModelPath(string configuredPath)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
                return null;

            return Path.GetFullPath(configuredPath);
        }

        private static string ResolveConfiguredMmProjPath(string configuredPath, string modelPath)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
                return null;

            if (string.Equals(configuredPath, "none", StringComparison.OrdinalIgnoreCase))
                return null;

            if (Path.IsPathRooted(configuredPath) ||
                configuredPath.IndexOf(Path.DirectorySeparatorChar) >= 0 ||
                configuredPath.IndexOf(Path.AltDirectorySeparatorChar) >= 0 ||
                File.Exists(configuredPath))
            {
                return Path.GetFullPath(configuredPath);
            }

            string preferredDirectory = Path.GetDirectoryName(modelPath);
            if (string.IsNullOrWhiteSpace(preferredDirectory))
                return Path.GetFullPath(configuredPath);

            return Path.GetFullPath(Path.Combine(preferredDirectory, configuredPath));
        }
    }
}
