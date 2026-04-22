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
using System.Globalization;
using System.Linq;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Emits the structured "what is this server doing?" banner to the logger.
    /// Centralised here so we can iterate on the operator-facing summary
    /// without touching <c>Program.cs</c>.
    /// </summary>
    internal static class StartupBanner
    {
        private static readonly string[] EndpointSummary =
        {
            "GET  /                         - Health check",
            "GET  /api/tags                  - List hosted models (Ollama)",
            "POST /api/show                  - Show model details (Ollama)",
            "POST /api/generate              - Generate text (Ollama)",
            "POST /api/chat/ollama           - Chat completion (Ollama)",
            "POST /v1/chat/completions       - Chat completion (OpenAI)",
            "GET  /v1/models                 - List hosted models (OpenAI)",
            "POST /api/chat                  - Chat (Web UI SSE)",
            "POST /api/models/load           - Reload hosted model (Web UI)",
            "GET  /api/models                - Show hosted model state (Web UI)",
        };

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

            logger.LogInformation(LogEventIds.HostConfiguration,
                "Server configuration: hostedModel={HostedModel} hostedMmProj={HostedMmProj} defaultWebMaxTokens={DefaultWebMaxTokens} videoMaxFrames={VideoMaxFrames} listen={ListenAddress}",
                options.StartupModelPath ?? "(none)",
                options.StartupMmProjPath ?? "(none)",
                options.DefaultWebMaxTokens,
                MediaHelper.GetConfiguredMaxVideoFrames(),
                listenAddress);

            // Surface the resolved sampling defaults so operators can confirm
            // the CLI flags / env vars they passed actually took effect.
            // We log the structured fields (rather than just one big string)
            // so log scrapers can pull individual values.
            var sampling = options.DefaultSamplingConfig;
            logger.LogInformation(LogEventIds.HostConfiguration,
                "Default sampling: temperature={Temperature} topK={TopK} topP={TopP} minP={MinP} repeatPenalty={RepeatPenalty} presencePenalty={PresencePenalty} frequencyPenalty={FrequencyPenalty} seed={Seed} stopSequences={StopSequences}",
                sampling.Temperature.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.TopK,
                sampling.TopP.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.MinP.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.RepetitionPenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.PresencePenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.FrequencyPenalty.ToString("0.###", CultureInfo.InvariantCulture),
                sampling.Seed,
                sampling.StopSequences != null && sampling.StopSequences.Count > 0
                    ? "[" + string.Join(", ", sampling.StopSequences.Select(s => "\"" + s + "\"")) + "]"
                    : "(none)");

            logger.LogInformation(LogEventIds.HostStarting,
                "Starting TensorSharp.Server on {ListenAddress}", listenAddress);

            foreach (string ep in EndpointSummary)
                logger.LogInformation(LogEventIds.HostConfiguration, "Endpoint: {Endpoint}", ep);
        }

        public static void EmitBackendFallback(ILogger logger, ServerHostingOptions options, string requestedBackendInput)
        {
            if (logger == null) throw new ArgumentNullException(nameof(logger));
            if (options == null) throw new ArgumentNullException(nameof(options));

            string canonicalRequested = BackendCatalog.Canonicalize(requestedBackendInput);
            if (!string.Equals(options.DefaultBackend, canonicalRequested, StringComparison.OrdinalIgnoreCase) &&
                !string.IsNullOrWhiteSpace(options.DefaultBackend))
            {
                logger.LogWarning(LogEventIds.BackendUnavailable,
                    "Requested default backend '{RequestedBackend}' is unavailable. Falling back to '{ResolvedBackend}'.",
                    requestedBackendInput, options.DefaultBackend);
            }
        }
    }
}
