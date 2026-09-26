// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Globalization;
using TensorSharp.Models;
using TensorSharp.Runtime.Logging;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.Host.Hosting
{
    /// <summary>
    /// Emits the structured "what is this server doing?" banner to the logger.
    /// Centralised here so we can iterate on the operator-facing summary
    /// without touching <c>Program.cs</c>.
    /// </summary>
    public static class StartupBanner
    {
        private static readonly string[] EndpointSummary =
        {
            "GET  /health                    - Health check",
            "GET  /api/tags                  - List hosted models (Ollama)",
            "POST /api/show                  - Show model details (Ollama)",
            "POST /api/generate              - Generate text (Ollama)",
            "POST /api/chat/ollama           - Chat completion (Ollama)",
            "POST /v1/chat/completions       - Chat completion (OpenAI)",
            "POST /v1/systemone             - Typed Jev decisions (DiffusionGemma)",
            "GET  /v1/models                 - List hosted models (OpenAI)",
            "POST /api/chat                  - Chat (Web UI SSE)",
            "POST /api/models/load           - Reload hosted model (Web UI)",
            "GET  /api/models                - Show hosted model state (Web UI)",
        };

        private static readonly string[] EmbeddingEndpointSummary =
        {
            "GET  /health                    - Health check",
            "POST /v1/embeddings             - Embeddings (OpenAI; float or base64)",
            "POST /api/embed                 - Embeddings (Ollama; string or batch)",
            "POST /api/embeddings            - Embeddings (legacy Ollama)",
            "GET  /v1/models                 - List embedding model (OpenAI)",
            "GET  /api/tags                  - List embedding model (Ollama)",
            "POST /api/show                  - Embedding model details (Ollama)",
            "GET  /api/models                - Show embedding model state",
        };

        internal static IEnumerable<string> DescribeEndpoints(ServerHostingOptions options)
        {
            yield return options.WebUiEnabled
                ? "GET  /                          - Web UI (index.html)"
                : "GET  /                          - Health check";
            foreach (string endpoint in options.EmbeddingsEnabled ? EmbeddingEndpointSummary : EndpointSummary)
                yield return endpoint;
        }

        private static void EmitEndpoints(ILogger logger, ServerHostingOptions options, string listenAddress)
        {
            logger.LogInformation(LogEventIds.HostStarting,
                "Starting TensorSharp.Server on {ListenAddress}", listenAddress);
            foreach (string endpoint in DescribeEndpoints(options))
                logger.LogInformation(LogEventIds.HostConfiguration, "Endpoint: {Endpoint}", endpoint);
        }

        public static void Emit(ILogger logger, ServerHostingOptions options, string listenAddress)
        {
            if (logger == null) throw new ArgumentNullException(nameof(logger));
            if (options == null) throw new ArgumentNullException(nameof(options));

            if (options.SupportedBackends.Count == 0)
            {
                logger.LogWarning(LogEventIds.BackendUnavailable,
                    "No supported backends detected on this machine.");
            }
            else
            {
                logger.LogInformation(LogEventIds.BackendDetected,
                    "Supported backends: {SupportedBackends}",
                    string.Join(", ", options.SupportedBackends.Select(b => b.Value)));
            }

            // Why a probed backend is missing from the list above: the probe threw and
            // the exception was swallowed into "unavailable" during discovery.
            foreach (string probeFailure in options.UsesManagedEmbeddingBackend
                ? Array.Empty<string>() : BackendCatalogProbes.DescribeProbeFailures())
            {
                logger.LogInformation(LogEventIds.BackendUnavailable,
                    "Backend probe failed, so that backend is not offered: {ProbeFailure}", probeFailure);
            }

            if (options.EmbeddingsEnabled)
            {
                logger.LogInformation(LogEventIds.HostConfiguration,
                    "Embedding server configuration: hostedModel={HostedModel} backend={Backend} execution={Execution} threads={Threads} contextLimit={ContextLimit} listen={ListenAddress}",
                    options.StartupModelPath, options.DefaultBackend, options.UsesManagedEmbeddingBackend ? "pure-csharp" : "native-ggml",
                    options.EmbeddingThreads > 0 ? options.EmbeddingThreads.ToString(CultureInfo.InvariantCulture) : "backend-default",
                    options.EmbeddingContextSize > 0 ? options.EmbeddingContextSize.ToString(CultureInfo.InvariantCulture) : "model-default",
                    listenAddress);
                EmitEndpoints(logger, options, listenAddress);
                return;
            }

            logger.LogInformation(LogEventIds.HostConfiguration,
                "Server configuration: hostedModel={HostedModel} hostedMmProj={HostedMmProj} defaultMaxTokens={DefaultMaxTokens}{MaxTokensPinned} videoFrames={VideoFrames} videoFps={VideoFps} videoSize={VideoSize} videoSteps={VideoSteps} videoMode={VideoMode} videoSampleFps={VideoSampleFps} videoMaxFrames={VideoMaxFrames} jevMaxBodyBytes={JevMaxBodyBytes} listen={ListenAddress}",
                options.StartupModelPath ?? "(none)",
                options.StartupMmProjPath ?? "(none)",
                options.DefaultMaxTokens,
                options.MaxTokensPinned ? " (server cap)" : string.Empty,
                options.DefaultVideoFrames > 0
                    ? options.DefaultVideoFrames.ToString(CultureInfo.InvariantCulture)
                    : "model-default",
                options.DefaultVideoFps > 0
                    ? options.DefaultVideoFps.ToString(CultureInfo.InvariantCulture)
                    : "model-default",
                options.DefaultVideoWidth > 0 || options.DefaultVideoHeight > 0
                    ? $"{options.DefaultVideoWidth}x{options.DefaultVideoHeight}"
                    : "model-default",
                options.DefaultVideoSteps > 0
                    ? options.DefaultVideoSteps.ToString(CultureInfo.InvariantCulture)
                    : "model-default",
                string.IsNullOrWhiteSpace(options.DefaultVideoMode) ? "auto" : options.DefaultVideoMode,
                MediaHelper.GetConfiguredVideoSampleFps().ToString("0.###", CultureInfo.InvariantCulture),
                MediaHelper.GetConfiguredMaxVideoFrames(),
                // Reading it here also validates TS_JEV_MAX_BODY_MB at startup instead of on
                // the first /v1/systemone request; images ride inside that body.
                ProtocolAdapters.JevAdapter.MaxRequestBodyBytes,
                listenAddress);

            // Surface the resolved sampling defaults so operators can confirm
            // the CLI flags / env vars they passed actually took effect.
            // We log the structured fields (rather than just one big string)
            // so log scrapers can pull individual values.
            var sampling = options.DefaultSamplingConfig;
            logger.LogInformation(LogEventIds.HostConfiguration,
                "Default sampling: temperature={Temperature} topK={TopK} topP={TopP} minP={MinP} repeatPenalty={RepeatPenalty} repeatLastN={RepeatLastN} presencePenalty={PresencePenalty} frequencyPenalty={FrequencyPenalty} seed={Seed} stopSequences={StopSequences}",
                sampling.Temperature.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.TopK,
                sampling.TopP.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.MinP.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.RepetitionPenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.PenaltyLastN,
                sampling.PresencePenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.FrequencyPenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.Seed,
                sampling.StopSequences != null && sampling.StopSequences.Count > 0
                    ? "[" + string.Join(", ", sampling.StopSequences.Select(s => "\"" + s + "\"")) + "]"
                    : "(none)");

            // Which of those defaults a request can talk the server out of.
            // Without this line an operator whose client hardcodes temperature
            // (VS Code Copilot Chat does) has no way to tell whether their
            // config is in charge — see issue #113.
            logger.LogInformation(LogEventIds.HostConfiguration,
                "Sampling precedence: {SamplingPrecedence}",
                options.SamplingDefaults.DescribePolicy());

            EmitEndpoints(logger, options, listenAddress);
        }

        /// <summary>
        /// Say, once and before the startup load, when the backend this server will use is
        /// not the one that was asked for - and what happens as a result, which depends on
        /// who asked.
        /// </summary>
        /// <remarks>
        /// This used to print "Requested default backend 'X' is unavailable. Falling back
        /// to 'Y'" in two cases where it was false. With no <c>--backend</c> at all it
        /// compared the resolved backend against nothing and fired on every launch, naming
        /// an empty request. And with an explicit <c>--backend</c> plus <c>--model</c> the
        /// startup load does NOT fall back - it resolves the requested backend itself and
        /// refuses the load (exit 2) - so the line promised the opposite of what the next
        /// one reported. Only a model-less process really falls back; a missing platform
        /// default is a fallback too, and is named as that.
        /// </remarks>
        public static void EmitBackendFallback(ILogger logger, ServerHostingOptions options, string? requestedBackendInput)
        {
            if (logger == null) throw new ArgumentNullException(nameof(logger));
            if (options == null) throw new ArgumentNullException(nameof(options));
            if (string.IsNullOrWhiteSpace(options.DefaultBackend))
                return;

            bool explicitRequest = !string.IsNullOrWhiteSpace(requestedBackendInput);
            string requested = explicitRequest
                ? BackendCatalog.Canonicalize(requestedBackendInput) ?? requestedBackendInput!
                : ServerOptionsBuilder.PlatformDefaultBackend;
            if (string.Equals(options.DefaultBackend, requested, StringComparison.OrdinalIgnoreCase))
                return;

            string available = options.SupportedBackends.Count == 0
                ? "none"
                : string.Join(", ", options.SupportedBackends.Select(b => b.Value));

            if (!explicitRequest)
            {
                logger.LogWarning(LogEventIds.BackendUnavailable,
                    "The platform default backend '{PlatformDefaultBackend}' is unavailable on this machine (available: {AvailableBackends}). " +
                    "Using '{ResolvedBackend}' instead; pass --backend to choose another.",
                    requested, available, options.DefaultBackend);
                return;
            }

            if (!string.IsNullOrWhiteSpace(options.StartupModelPath))
            {
                logger.LogWarning(LogEventIds.BackendUnavailable,
                    "Requested backend '{RequestedBackend}' is not available on this machine (available: {AvailableBackends}). " +
                    "The startup model is not moved to another backend: its load is refused. Pass one of the available backends with --backend.",
                    requestedBackendInput, available);
                return;
            }

            logger.LogWarning(LogEventIds.BackendUnavailable,
                "Requested default backend '{RequestedBackend}' is not available on this machine (available: {AvailableBackends}). " +
                "Falling back to '{ResolvedBackend}' as this model-less server's default backend.",
                requestedBackendInput, available, options.DefaultBackend);
        }
    }
}
