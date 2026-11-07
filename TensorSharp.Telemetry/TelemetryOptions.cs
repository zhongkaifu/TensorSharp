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
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Telemetry
{
    /// <summary>Which TensorSharp process is setting telemetry up; picks the default service name.</summary>
    public enum TelemetryHost
    {
        Server,
        Cli,
    }

    /// <summary>Where one signal is exported to.</summary>
    [Flags]
    public enum TelemetrySinks
    {
        None = 0,
        Otlp = 1,
        Console = 2,
        AzureMonitor = 4,
    }

    /// <summary>
    /// Telemetry settings, resolved once from the standard <c>OTEL_*</c> environment
    /// variables plus TensorSharp's few of its own. Every rule about when telemetry is on
    /// lives here, so it is tested in one place:
    /// <list type="bullet">
    ///   <item>Off unless something asks for it: an OTLP endpoint (general or per signal),
    ///         <c>APPLICATIONINSIGHTS_CONNECTION_STRING</c>, <c>TENSORSHARP_OTEL=1</c>, or an
    ///         <c>OTEL_{TRACES,METRICS,LOGS}_EXPORTER</c> naming a sink.</item>
    ///   <item><c>OTEL_SDK_DISABLED=true</c> turns it off whatever else is set.</item>
    ///   <item><c>OTEL_{SIGNAL}_EXPORTER</c> (<c>otlp</c>, <c>console</c>, <c>none</c>, comma
    ///         separated) chooses the non-Azure sinks of one signal; <c>none</c> turns the signal
    ///         off for every sink, Azure Monitor included. The .NET SDK does not read these
    ///         variables itself.</item>
    ///   <item>Azure Monitor receives every enabled signal while the connection string is set.</item>
    /// </list>
    /// Endpoint, protocol, headers, timeouts, the sampler and resource attributes are read by
    /// the OpenTelemetry SDK itself from the same variables; they are only inspected here to
    /// validate them and to describe the result in one startup line.
    /// </summary>
    public sealed class TelemetryOptions
    {
        public const string EnableEnvironmentVariable = "TENSORSHARP_OTEL";
        public const string LogLevelEnvironmentVariable = "TENSORSHARP_OTEL_LOG_LEVEL";
        public const string AzureMonitorEntraIdEnvironmentVariable = "TENSORSHARP_AZURE_MONITOR_AAD";
        public const string AzureMonitorConnectionStringEnvironmentVariable = "APPLICATIONINSIGHTS_CONNECTION_STRING";
        public const string SdkDisabledEnvironmentVariable = "OTEL_SDK_DISABLED";
        public const string OtlpEndpointEnvironmentVariable = "OTEL_EXPORTER_OTLP_ENDPOINT";
        public const string OtlpProtocolEnvironmentVariable = "OTEL_EXPORTER_OTLP_PROTOCOL";
        public const string ServiceNameEnvironmentVariable = "OTEL_SERVICE_NAME";
        public const string SamplerEnvironmentVariable = "OTEL_TRACES_SAMPLER";
        public const string SamplerArgEnvironmentVariable = "OTEL_TRACES_SAMPLER_ARG";

        /// <summary>The SDK's own default when no endpoint is configured.</summary>
        public const string DefaultGrpcEndpoint = "http://localhost:4317";
        public const string DefaultHttpEndpoint = "http://localhost:4318";

        private static readonly string[] s_signals = { "TRACES", "METRICS", "LOGS" };

        private TelemetryOptions() { }

        public TelemetryHost Host { get; private init; }
        public bool Enabled { get; private init; }

        /// <summary>Why telemetry is off, for the startup line; null when it is on.</summary>
        public string? DisabledReason { get; private init; }

        public TelemetrySinks Traces { get; private init; }
        public TelemetrySinks Metrics { get; private init; }
        public TelemetrySinks Logs { get; private init; }

        public string ServiceName { get; private init; } = "";
        public bool ServiceNameFromEnvironment { get; private init; }

        /// <summary>The OTLP endpoint as the SDK will use it, credentials removed; null when no signal uses OTLP.</summary>
        public string? OtlpEndpointDescription { get; private init; }
        public string OtlpProtocol { get; private init; } = "grpc";

        public string? AzureMonitorConnectionString { get; private init; }
        public bool AzureMonitorUseEntraId { get; private init; }

        public bool CaptureMessageContent { get; private init; }
        public bool TraceSteps { get; private init; }

        /// <summary>
        /// Minimum level exported through OpenTelemetry, from <c>TENSORSHARP_OTEL_LOG_LEVEL</c>;
        /// null keeps the host's level (and, on the server, honours
        /// <c>Logging:OpenTelemetry:LogLevel</c> configuration).
        /// </summary>
        public LogLevel? LogLevel { get; private init; }

        /// <summary>Non-fatal problems found while resolving, e.g. an unknown exporter name.</summary>
        public IReadOnlyList<string> Warnings { get; private init; } = Array.Empty<string>();

        /// <summary>
        /// Per-signal OTLP overrides from <c>OTEL_EXPORTER_OTLP_{SIGNAL}_{ENDPOINT,HEADERS,PROTOCOL,TIMEOUT}</c>.
        /// The SDK's per-signal <c>AddOtlpExporter</c> reads only the generic variables, so
        /// these are applied in code; a null field leaves the generic variable (or the SDK
        /// default) in charge.
        /// </summary>
        public OtlpSignalSettings OtlpTraces { get; private init; } = OtlpSignalSettings.Empty;
        public OtlpSignalSettings OtlpMetrics { get; private init; } = OtlpSignalSettings.Empty;
        public OtlpSignalSettings OtlpLogs { get; private init; } = OtlpSignalSettings.Empty;

        /// <summary>
        /// Azure Monitor's trace sampler settings. <c>AddAzureMonitorTraceExporter</c> installs
        /// its own sampler on the whole tracer provider (by default a 5 traces/s rate limit
        /// that would also throttle OTLP), and the SDK then ignores <c>OTEL_TRACES_SAMPLER</c>.
        /// So the standard variable is mapped onto the Azure sampler: unset or always_on is
        /// 100%, traceidratio / parentbased_traceidratio take their argument, always_off is
        /// 0%, and the exporter's own <c>microsoft.fixed_percentage</c> /
        /// <c>microsoft.rate_limited</c> are honoured.
        /// </summary>
        public float AzureSamplingRatio { get; private init; } = 1.0f;
        public double? AzureTracesPerSecond { get; private init; }

        /// <summary>The sampler that will actually decide, for the startup line.</summary>
        public string EffectiveSampler { get; private init; } = "parentbased_always_on";

        public bool AnyAzureMonitor => ((Traces | Metrics | Logs) & TelemetrySinks.AzureMonitor) != 0;
        public bool AnyOtlp => ((Traces | Metrics | Logs) & TelemetrySinks.Otlp) != 0;

        /// <summary>A disabled instance, for hosts and tests that skip resolution.</summary>
        public static TelemetryOptions Disabled(TelemetryHost host, string reason) => new()
        {
            Host = host,
            Enabled = false,
            DisabledReason = reason,
            ServiceName = DefaultServiceName(host),
        };

        public static string DefaultServiceName(TelemetryHost host) =>
            host == TelemetryHost.Cli ? "tensorsharp-cli" : "tensorsharp-server";

        /// <summary>
        /// Resolve from the process environment. Throws <see cref="ArgumentException"/> for a
        /// value the SDK would reject or silently misread (a relative endpoint, an unknown
        /// protocol, a malformed log level), so the host reports it as a configuration error
        /// before it starts.
        /// </summary>
        public static TelemetryOptions FromEnvironment(TelemetryHost host) =>
            FromEnvironment(host, Environment.GetEnvironmentVariable);

        /// <summary><see cref="FromEnvironment(TelemetryHost)"/> over any variable source (tests).</summary>
        public static TelemetryOptions FromEnvironment(TelemetryHost host, Func<string, string?> getEnvironmentVariable)
        {
            ArgumentNullException.ThrowIfNull(getEnvironmentVariable);
            string? Get(string name)
            {
                string? value = getEnvironmentVariable(name);
                return string.IsNullOrWhiteSpace(value) ? null : value.Trim();
            }

            var warnings = new List<string>();
            string defaultName = DefaultServiceName(host);
            string? serviceName = Get(ServiceNameEnvironmentVariable);

            if (TensorSharpTelemetry.IsTruthy(Get(SdkDisabledEnvironmentVariable)))
                return Disabled(host, SdkDisabledEnvironmentVariable + "=true");

            string? generalEndpoint = Get(OtlpEndpointEnvironmentVariable);
            if (generalEndpoint != null)
                ValidateEndpoint(OtlpEndpointEnvironmentVariable, generalEndpoint);

            var signalEndpoints = new Dictionary<string, string?>(StringComparer.Ordinal);
            foreach (string signal in s_signals)
            {
                string variable = $"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT";
                string? value = Get(variable);
                if (value != null) ValidateEndpoint(variable, value);
                signalEndpoints[signal] = value;
            }

            // Read now, validated only once a signal is known to export over OTLP: the OTEL_*
            // variables are shared by every OpenTelemetry SDK on the machine, and a value set for
            // another program (http/json is valid in the specification, not in .NET) must not
            // stop TensorSharp from starting when TensorSharp exports nothing over OTLP.
            string protocol = (Get(OtlpProtocolEnvironmentVariable) ?? "grpc").ToLowerInvariant();

            bool forceOn = TensorSharpTelemetry.IsTruthy(Get(EnableEnvironmentVariable));
            string? connectionString = Get(AzureMonitorConnectionStringEnvironmentVariable);

            var sinks = new Dictionary<string, TelemetrySinks>(StringComparer.Ordinal);
            bool explicitSinkNamed = false;
            foreach (string signal in s_signals)
            {
                string variable = $"OTEL_{signal}_EXPORTER";
                string? list = Get(variable);
                TelemetrySinks selected;
                bool off = false;
                if (list != null)
                {
                    selected = TelemetrySinks.None;
                    foreach (string raw in list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        switch (raw.ToLowerInvariant())
                        {
                            case "otlp": selected |= TelemetrySinks.Otlp; explicitSinkNamed = true; break;
                            case "console": selected |= TelemetrySinks.Console; explicitSinkNamed = true; break;
                            case "none": off = true; break;
                            default:
                                warnings.Add($"{variable}: '{raw}' is not an exporter TensorSharp supports (otlp, console, none); ignored.");
                                break;
                        }
                    }
                    if (off) selected = TelemetrySinks.None;
                }
                else
                {
                    bool otlp = generalEndpoint != null || signalEndpoints[signal] != null || forceOn;
                    selected = otlp ? TelemetrySinks.Otlp : TelemetrySinks.None;
                }
                if (!off && connectionString != null)
                    selected |= TelemetrySinks.AzureMonitor;
                sinks[signal] = selected;
            }

            bool requested = forceOn || connectionString != null || explicitSinkNamed
                || generalEndpoint != null || signalEndpoints.Values.Any(v => v != null);
            if (!requested)
                return Disabled(host, "no OTLP endpoint, APPLICATIONINSIGHTS_CONNECTION_STRING or TENSORSHARP_OTEL set");

            TelemetrySinks traces = sinks["TRACES"], metrics = sinks["METRICS"], logs = sinks["LOGS"];
            if ((traces | metrics | logs) == TelemetrySinks.None)
                return Disabled(host, "every signal's OTEL_*_EXPORTER is none");

            LogLevel? logLevel = null;
            if (Get(LogLevelEnvironmentVariable) is { } rawLevel)
            {
                if (!Enum.TryParse(rawLevel, ignoreCase: true, out LogLevel parsed) || !Enum.IsDefined(parsed))
                {
                    throw new ArgumentException(
                        $"{LogLevelEnvironmentVariable}={rawLevel} is not a log level (Trace, Debug, Information, Warning, Error, Critical, None).");
                }
                logLevel = parsed;
            }

            bool anyOtlp = ((traces | metrics | logs) & TelemetrySinks.Otlp) != 0;
            if (anyOtlp && protocol is not ("grpc" or "http/protobuf"))
            {
                throw new ArgumentException(
                    $"{OtlpProtocolEnvironmentVariable}={protocol} is not supported; use grpc or http/protobuf.");
            }

            // Per-signal overrides are read (and validated) only for the signals that use OTLP,
            // for the same reason as the protocol above.
            var otlpSignals = new Dictionary<string, OtlpSignalSettings>(StringComparer.Ordinal);
            foreach (string signal in s_signals)
            {
                TelemetrySinks signalSinks = sinks[signal];
                otlpSignals[signal] = (signalSinks & TelemetrySinks.Otlp) != 0
                    ? OtlpSignalSettings.Resolve(signal, Get, signalEndpoints[signal])
                    : OtlpSignalSettings.Empty;
            }

            bool azureTraces = (traces & TelemetrySinks.AzureMonitor) != 0;
            ResolveSampler(Get(SamplerEnvironmentVariable), Get(SamplerArgEnvironmentVariable), azureTraces, warnings,
                out float azureRatio, out double? azureRate, out string effectiveSampler);

            string? endpointDescription = null;
            if (anyOtlp)
            {
                string shown = generalEndpoint
                    ?? signalEndpoints.Values.FirstOrDefault(v => v != null)
                    ?? (protocol == "grpc" ? DefaultGrpcEndpoint : DefaultHttpEndpoint);
                endpointDescription = StripCredentials(shown);
            }

            return new TelemetryOptions
            {
                Host = host,
                Enabled = true,
                Traces = traces,
                Metrics = metrics,
                Logs = logs,
                ServiceName = serviceName ?? defaultName,
                ServiceNameFromEnvironment = serviceName != null,
                OtlpEndpointDescription = endpointDescription,
                OtlpProtocol = protocol,
                AzureMonitorConnectionString = connectionString,
                AzureMonitorUseEntraId = connectionString != null
                    && TensorSharpTelemetry.IsTruthy(Get(AzureMonitorEntraIdEnvironmentVariable)),
                CaptureMessageContent = TensorSharpTelemetry.IsTruthy(Get(TensorSharpTelemetry.CaptureMessageContentEnvironmentVariable)),
                TraceSteps = TensorSharpTelemetry.IsTruthy(Get(TensorSharpTelemetry.TraceStepsEnvironmentVariable)),
                LogLevel = logLevel,
                Warnings = warnings,
                OtlpTraces = otlpSignals["TRACES"],
                OtlpMetrics = otlpSignals["METRICS"],
                OtlpLogs = otlpSignals["LOGS"],
                AzureSamplingRatio = azureRatio,
                AzureTracesPerSecond = azureRate,
                EffectiveSampler = effectiveSampler,
            };
        }

        private static void ResolveSampler(string? sampler, string? arg, bool azureTraces, List<string> warnings,
            out float azureRatio, out double? azureRate, out string effective)
        {
            azureRatio = 1.0f;
            azureRate = null;
            string name = (sampler ?? "parentbased_always_on").Trim().ToLowerInvariant();
            double? argValue = double.TryParse(arg, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out double parsed) ? parsed : null;

            if (!azureTraces)
            {
                // The SDK reads OTEL_TRACES_SAMPLER / _ARG itself; this only describes it.
                effective = argValue is double a && name.Contains("traceidratio", StringComparison.Ordinal)
                    ? $"{name}({a.ToString(System.Globalization.CultureInfo.InvariantCulture)})"
                    : name;
                return;
            }

            switch (name)
            {
                case "always_on":
                case "parentbased_always_on":
                    break;
                case "always_off":
                case "parentbased_always_off":
                    azureRatio = 0f;
                    break;
                case "traceidratio":
                case "parentbased_traceidratio":
                case "microsoft.fixed_percentage":
                    azureRatio = (float)Math.Clamp(argValue ?? 1.0, 0.0, 1.0);
                    break;
                case "microsoft.rate_limited":
                    azureRate = argValue is double rate && rate > 0 ? rate : 5.0;
                    break;
                default:
                    warnings.Add($"{SamplerEnvironmentVariable}={sampler} is not supported while Application Insights is on; sampling 100%.");
                    break;
            }
            effective = azureRate is double r
                ? $"azuremonitor rate_limited({r.ToString(System.Globalization.CultureInfo.InvariantCulture)}/s)"
                : $"azuremonitor fixed_percentage({azureRatio.ToString(System.Globalization.CultureInfo.InvariantCulture)})";
        }

        /// <summary>
        /// One line for the startup log, e.g.
        /// <c>Telemetry: traces(otlp,azuremonitor) metrics(otlp) logs(otlp) -> OTLP http://collector:4317 (grpc), Azure Monitor; service tensorsharp-server; content capture off; sampler parentbased_always_on</c>.
        /// Never includes the connection string or OTLP headers.
        /// </summary>
        public string Describe(Func<string, string?>? getEnvironmentVariable = null)
        {
            if (!Enabled)
                return "Telemetry: off (" + (DisabledReason ?? "not requested") + ")";

            getEnvironmentVariable ??= Environment.GetEnvironmentVariable;
            var sb = new StringBuilder("Telemetry: ");
            AppendSignal(sb, "traces", Traces);
            sb.Append(' ');
            AppendSignal(sb, "metrics", Metrics);
            sb.Append(' ');
            AppendSignal(sb, "logs", Logs);

            var destinations = new List<string>();
            if (AnyOtlp) destinations.Add($"OTLP {OtlpEndpointDescription} ({OtlpProtocol}{DescribeSignalOverrides()})");
            if (AnyAzureMonitor) destinations.Add(AzureMonitorUseEntraId ? "Azure Monitor (Entra ID)" : "Azure Monitor");
            if (((Traces | Metrics | Logs) & TelemetrySinks.Console) != 0) destinations.Add("console");
            if (destinations.Count > 0) sb.Append(" -> ").Append(string.Join(", ", destinations));

            sb.Append("; service ").Append(ServiceName);
            sb.Append("; content capture ").Append(CaptureMessageContent ? "ON" : "off");
            if (Traces != TelemetrySinks.None)
                sb.Append("; sampler ").Append(EffectiveSampler);
            if (TraceSteps) sb.Append("; engine step spans ON");
            return sb.ToString();
        }

        /// <summary>
        /// The per-signal OTLP endpoints and protocols that differ from the ones the line shows,
        /// as <c>"; metrics: http/protobuf"</c> or <c>"; traces: http://tempo:4318/v1/traces http/protobuf"</c>;
        /// empty when every OTLP signal uses them.
        /// </summary>
        private string DescribeSignalOverrides()
        {
            var sb = new StringBuilder();
            AppendOverride(sb, "traces", Traces, OtlpTraces);
            AppendOverride(sb, "metrics", Metrics, OtlpMetrics);
            AppendOverride(sb, "logs", Logs, OtlpLogs);
            return sb.ToString();
        }

        private void AppendOverride(StringBuilder sb, string name, TelemetrySinks sinks, OtlpSignalSettings settings)
        {
            if ((sinks & TelemetrySinks.Otlp) == 0)
                return;
            string? endpoint = settings.Endpoint is { } uri ? StripCredentials(uri.OriginalString) : null;
            string? protocol = settings.Protocol switch
            {
                OpenTelemetry.Exporter.OtlpExportProtocol.Grpc => "grpc",
                OpenTelemetry.Exporter.OtlpExportProtocol.HttpProtobuf => "http/protobuf",
                _ => null,
            };
            if (string.Equals(endpoint, OtlpEndpointDescription, StringComparison.Ordinal))
                endpoint = null;
            if (string.Equals(protocol, OtlpProtocol, StringComparison.Ordinal))
                protocol = null;
            if (endpoint is null && protocol is null)
                return;
            sb.Append("; ").Append(name).Append(':');
            if (endpoint != null) sb.Append(' ').Append(endpoint);
            if (protocol != null) sb.Append(' ').Append(protocol);
        }

        private static void AppendSignal(StringBuilder sb, string name, TelemetrySinks sinks)
        {
            sb.Append(name).Append('(');
            if (sinks == TelemetrySinks.None)
            {
                sb.Append("off");
            }
            else
            {
                var parts = new List<string>(3);
                if ((sinks & TelemetrySinks.Otlp) != 0) parts.Add("otlp");
                if ((sinks & TelemetrySinks.AzureMonitor) != 0) parts.Add("azuremonitor");
                if ((sinks & TelemetrySinks.Console) != 0) parts.Add("console");
                sb.Append(string.Join(',', parts));
            }
            sb.Append(')');
        }

        private static void ValidateEndpoint(string variable, string value)
        {
            if (!Uri.TryCreate(value, UriKind.Absolute, out Uri? uri) || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
            {
                throw new ArgumentException(
                    $"{variable}={StripCredentials(value)} is not an absolute http:// or https:// URL.");
            }
        }

        /// <summary>Remove a <c>user:password@</c> part so an endpoint can be logged.</summary>
        /// <remarks>Also used by <see cref="TelemetryCliFlags"/>.</remarks>
        internal static string StripCredentials(string endpoint)
        {
            if (Uri.TryCreate(endpoint, UriKind.Absolute, out Uri? uri) && !string.IsNullOrEmpty(uri.UserInfo))
                return new UriBuilder(uri) { UserName = string.Empty, Password = string.Empty }.Uri.ToString();
            return endpoint;
        }
    }
}
