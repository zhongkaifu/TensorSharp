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
using System.Diagnostics;
using System.IO;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Pre-loads the model named on the command line into the singleton
    /// <see cref="ModelService"/> before the host starts accepting requests.
    /// Throws when configuration is internally inconsistent (e.g. the requested
    /// backend isn't available) so the process fails fast rather than serving
    /// 4xx/5xx responses for every request. A load that is REFUSED (see
    /// <see cref="ModelLoadRefusal"/>) is reported by the host through
    /// <see cref="ReportRefusal"/>: one error line and
    /// <see cref="HostExitCodes.ModelLoadRefused"/>, never an unhandled exception.
    /// </summary>
    public static class StartupModelLoader
    {
        public static void LoadIfConfigured(
            ServerHostingOptions options,
            ModelService modelService,
            string configuredBackendInput,
            ILogger logger)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));
            if (modelService == null) throw new ArgumentNullException(nameof(modelService));
            if (logger == null) throw new ArgumentNullException(nameof(logger));

            if (string.IsNullOrWhiteSpace(options.StartupModelPath))
            {
                logger.LogInformation(LogEventIds.HostConfiguration,
                    "No startup model configured. Launch with --model <path.gguf> --backend <type> [--mmproj <path>] [--tp N] [--max-tokens 20000]" +
                    " [--temperature F] [--top-k N] [--top-p F] [--min-p F] [--repeat-penalty F]" +
                    " [--presence-penalty F] [--frequency-penalty F] [--seed N] [--stop <text>]" +
                    " [--prefill-chunk-size N]" +
                    " [--qwen-image-vae <vae.gguf>] [--qwen-image-vl <qwen2.5-vl.gguf>] [--qwen-image-mmproj <mmproj.gguf>]" +
                    " to use the Web UI.");
                return;
            }

            if (!BackendSelector.TryResolveSupportedBackend(options, configuredBackendInput, out string startupBackend, out string startupBackendError))
                throw new ModelLoadRefusedException(startupBackendError);

            if (!File.Exists(options.StartupModelPath))
                throw new FileNotFoundException($"Configured model file not found: {options.StartupModelPath}", options.StartupModelPath);

            if (!string.IsNullOrWhiteSpace(options.StartupMmProjPath) && !File.Exists(options.StartupMmProjPath))
                throw new FileNotFoundException($"Configured mmproj file not found: {options.StartupMmProjPath}", options.StartupMmProjPath);

            // BEFORE the load, because the engine reads the store at admission and the
            // very first request is the one that most needs it. Attaching it afterwards
            // would leave exactly the launch this exists for — a fresh process whose first
            // message would otherwise prefill the whole shared prompt — with nothing to
            // restore from.
            AttachPrefixCheckpointStore(options, modelService, logger);

            modelService.LoadModel(options.StartupModelPath, options.StartupMmProjPath, startupBackend);

            // An explicit --draft-model that couldn't be activated (missing,
            // wrong architecture, an incompatible/incomplete draft) used to be
            // swallowed as a warning, leaving the server up with speculation
            // silently off. Promote it to a fail-fast startup error so the operator
            // sees exactly why MTP didn't engage instead of discovering it later.
            string mtpFatal = SpeculationStartupValidation.GetFatalActivationError(
                modelService.DraftHeadActivationError, modelService.DraftHeadRefusedByModel);
            if (mtpFatal != null)
                throw new ModelLoadRefusedException(mtpFatal);

            logger.LogInformation(LogEventIds.ModelLoadCompleted,
                "Startup model loaded: {Model} architecture={Architecture} backend={Backend} mmproj={MmProj}",
                modelService.LoadedModelName,
                modelService.Architecture ?? "unknown",
                modelService.LoadedBackend,
                modelService.LoadedMmProjName ?? "(none)");

            if (modelService.Model != null)
            {
                var warmupSw = Stopwatch.StartNew();
                modelService.Model.WarmUpKernels();
                warmupSw.Stop();
                logger.LogInformation(LogEventIds.HostConfiguration,
                    "Kernel warmup completed in {ElapsedMs:F1} ms", warmupSw.Elapsed.TotalMilliseconds);
                modelService.Model.LogVramSnapshot("after kernel warmup");

                // Multi-node tensor parallelism: from here on this server is the
                // driver (node 0) — every forward pass broadcasts to the worker
                // nodes so their weight shards join each AllReduce. Must run
                // AFTER WarmUpKernels (warmup executes symmetrically on every
                // node and must not broadcast). No-op on single-node groups.
                modelService.Model.BeginDistributedDriver();
            }
        }

        /// <summary>
        /// Report a refused startup load the way every host does: the stack trace goes to
        /// the log at Debug only (<c>TENSORSHARP_LOG_LEVEL=Debug</c> shows it), then
        /// <paramref name="releaseResources"/> runs, then exactly one line goes to
        /// <paramref name="stderr"/>. Returns the exit code to leave with.
        /// </summary>
        /// <remarks>
        /// The error line is written AFTER the release so it is the last thing on stderr
        /// rather than being followed by teardown chatter, and a release that itself
        /// fails cannot swallow it: that failure is written first, as its own line.
        /// </remarks>
        public static int ReportRefusal(Exception refusal, string reason, TextWriter stderr, ILogger logger,
            Action releaseResources = null)
        {
            if (refusal == null) throw new ArgumentNullException(nameof(refusal));
            if (stderr == null) throw new ArgumentNullException(nameof(stderr));

            logger?.LogDebug(LogEventIds.ModelLoadFailed, refusal, "Startup model load refused");
            try
            {
                releaseResources?.Invoke();
            }
            catch (Exception releaseEx)
            {
                stderr.WriteLine($"warning: releasing resources after the refused load failed: " +
                    $"{releaseEx.GetType().Name}: {ModelLoadRefusal.ToSingleLine(releaseEx.Message)}");
            }
            stderr.WriteLine(ModelLoadRefusal.FormatErrorLine(reason ?? refusal.Message));
            stderr.Flush();
            return HostExitCodes.ModelLoadRefused;
        }

        /// <summary>
        /// Give the engine somewhere to keep the shared-prefix checkpoint between
        /// launches, or explain why it has nowhere.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Deliberately not fatal, unlike everything else in this class. The contract of
        /// <see cref="LoadIfConfigured"/> is to throw on configuration that cannot work, so
        /// the process dies at startup rather than failing every request — but a prefix
        /// cache that cannot be written is a server that is merely as slow as it was
        /// before, and refusing to start over a read-only directory would turn a latency
        /// optimisation into an outage. The store's constructor touches the disk (it
        /// sweeps stray temporaries), so this is also where an unwritable path is found.
        /// </para>
        /// </remarks>
        private static void AttachPrefixCheckpointStore(
            ServerHostingOptions options, ModelService modelService, ILogger logger)
        {
            if (!options.PrefixCacheEnabled)
            {
                logger.LogInformation(LogEventIds.HostConfiguration,
                    "Prefix cache disabled (--no-prefix-cache): the first message of this process "
                    + "will prefill the whole shared prompt, and nothing is kept between launches.");
                return;
            }

            if (string.IsNullOrWhiteSpace(options.PrefixCacheDirectory))
                return;

            try
            {
                string identity = PrefixCheckpointFileStore.WeightsIdentityOf(
                    options.StartupModelPath, options.StartupMmProjPath);
                var store = new PrefixCheckpointFileStore(options.PrefixCacheDirectory, identity, logger);
                modelService.EngineHost.PrefixCheckpointStore = store;
                logger.LogInformation(LogEventIds.HostConfiguration,
                    "Prefix cache: keeping shared-prefix checkpoints in {Directory} ({Bytes} bytes already there). "
                    + "Turn it off with --no-prefix-cache; move it with TENSORSHARP_PREFIX_CACHE_DIR.",
                    store.Directory, store.TotalBytes());
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException)
            {
                logger.LogWarning(LogEventIds.HostConfiguration,
                    "Prefix cache unavailable at {Directory}: {Reason}. Checkpoints will live only in "
                    + "memory, so every launch pays for the shared prompt once.",
                    options.PrefixCacheDirectory, ex.Message);
            }
        }
    }
}
