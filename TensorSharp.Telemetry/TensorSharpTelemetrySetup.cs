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
using System.Diagnostics;
using Azure.Core;
using Azure.Identity;
using Azure.Monitor.OpenTelemetry.Exporter;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Logs;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Telemetry
{
    /// <summary>
    /// How TensorSharp configures the OpenTelemetry SDK, shared by the server (dependency
    /// injection) and the CLI (no host). Signal-specific exporters only: the cross-cutting
    /// <c>UseOtlpExporter</c> / <c>UseAzureMonitor</c> helpers each own the whole pipeline and do
    /// not compose with each other, while per-signal OTLP and Azure Monitor exporters run side
    /// by side.
    /// </summary>
    public static class TensorSharpTelemetrySetup
    {
        /// <summary>Wildcard that subscribes to every TensorSharp activity source.</summary>
        public const string SourceWildcard = TensorSharpTelemetry.NamePrefix + ".*";

        /// <summary>Built-in .NET meters worth exporting from every host.</summary>
        public static readonly IReadOnlyList<string> RuntimeMeters = new[] { "System.Runtime", "System.Net.Http" };

        /// <summary>
        /// Process-wide switches the libraries read (content capture on spans, engine step
        /// spans), pinned to the resolved options so the span side and the log redaction side
        /// agree, and the observable instruments created before the first collection.
        /// </summary>
        public static void ApplyProcessWide(TelemetryOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            TensorSharpTelemetry.ConfigureCaptureMessageContent(options.Enabled && options.CaptureMessageContent);
            TensorSharpTelemetry.ConfigureTraceSteps(options.Enabled && options.TraceSteps);
            TensorSharpTelemetry.ConfigureLogsExported(options.Enabled && options.Logs != TelemetrySinks.None);
            if (options.Enabled)
                TensorSharpMetrics.EnsureInitialized();
        }

        /// <summary>
        /// Service name, version and instance, then the SDK's own environment detectors last,
        /// so <c>OTEL_SERVICE_NAME</c> and <c>OTEL_RESOURCE_ATTRIBUTES</c> win over TensorSharp's
        /// defaults.
        /// </summary>
        public static ResourceBuilder ConfigureResource(ResourceBuilder resource, TelemetryOptions options)
        {
            ArgumentNullException.ThrowIfNull(resource);
            ArgumentNullException.ThrowIfNull(options);
            string machine = SafeMachineName();
            int pid = Environment.ProcessId;
            var attributes = new List<KeyValuePair<string, object>>
            {
                new("host.name", machine),
                new("process.pid", pid),
            };
            if (Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID") is { Length: > 0 } nodeId)
                attributes.Add(new(TelemetryAttributes.TensorParallelNodeId, nodeId.Trim()));
            if (Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE") is { Length: > 0 } degree)
                attributes.Add(new(TelemetryAttributes.TensorParallelDegree, degree.Trim()));

            return resource
                .Clear()
                .AddService(
                    serviceName: TelemetryOptions.DefaultServiceName(options.Host),
                    serviceVersion: TensorSharpTelemetry.Version,
                    autoGenerateServiceInstanceId: false,
                    serviceInstanceId: $"{machine}:{pid}")
                .AddTelemetrySdk()
                .AddAttributes(attributes)
                .AddEnvironmentVariableDetector();
        }

        /// <summary>The Entra ID credential for Azure Monitor, or null for connection-string auth.</summary>
        public static TokenCredential? CreateAzureCredential(TelemetryOptions options) =>
            options.AzureMonitorUseEntraId ? new DefaultAzureCredential() : null;

        /// <summary>TensorSharp sources, outgoing HTTP, and the trace exporters.</summary>
        public static TracerProviderBuilder ConfigureTracing(
            TracerProviderBuilder tracing, TelemetryOptions options, TokenCredential? credential,
            Action<TracerProviderBuilder>? configure = null)
        {
            tracing.AddSource(SourceWildcard);
            tracing.AddHttpClientInstrumentation();
            configure?.Invoke(tracing);
            // Before the exporters: processors run in the order added.
            if (!options.CaptureMessageContent)
                tracing.AddProcessor(new ContentSpanProcessor());
            if ((options.Traces & TelemetrySinks.Otlp) != 0) tracing.AddOtlpExporter(o => options.OtlpTraces.ApplyTo(o));
            if ((options.Traces & TelemetrySinks.Console) != 0) tracing.AddConsoleExporter();
            if ((options.Traces & TelemetrySinks.AzureMonitor) != 0)
                tracing.AddAzureMonitorTraceExporter(o => ConfigureAzure(o, options), credential);
            return tracing;
        }

        /// <summary>TensorSharp meters, .NET runtime and HTTP meters, and the metric exporters.</summary>
        public static MeterProviderBuilder ConfigureMetrics(
            MeterProviderBuilder metrics, TelemetryOptions options, TokenCredential? credential,
            Action<MeterProviderBuilder>? configure = null)
        {
            metrics.AddMeter(TensorSharpTelemetry.MeterName, TensorSharpTelemetry.StudioMeterName, SourceWildcard);
            foreach (string meter in RuntimeMeters)
                metrics.AddMeter(meter);
            configure?.Invoke(metrics);
            if ((options.Metrics & TelemetrySinks.Otlp) != 0) metrics.AddOtlpExporter(o => options.OtlpMetrics.ApplyTo(o));
            if ((options.Metrics & TelemetrySinks.Console) != 0) metrics.AddConsoleExporter();
            if ((options.Metrics & TelemetrySinks.AzureMonitor) != 0)
                metrics.AddAzureMonitorMetricExporter(o => ConfigureAzure(o, options), credential);
            return metrics;
        }

        /// <summary>
        /// The content redaction processor first (processors run in the order added, and the
        /// exporters must see the redacted record), then the log exporters.
        /// </summary>
        public static LoggerProviderBuilder ConfigureLogging(
            LoggerProviderBuilder logging, TelemetryOptions options, TokenCredential? credential)
        {
            logging.AddProcessor(new LogContextEnrichmentProcessor());
            if (!options.CaptureMessageContent)
                logging.AddProcessor(new ContentRedactionProcessor());
            if ((options.Logs & TelemetrySinks.Otlp) != 0) logging.AddOtlpExporter(o => options.OtlpLogs.ApplyTo(o));
            if ((options.Logs & TelemetrySinks.Console) != 0) logging.AddConsoleExporter();
            if ((options.Logs & TelemetrySinks.AzureMonitor) != 0)
                logging.AddAzureMonitorLogExporter(o => ConfigureAzure(o, options), credential);
            return logging;
        }

        /// <summary>
        /// Log scopes are not exported. A processor cannot rewrite or drop one scope item, and
        /// the scopes carry things that must not leave the process or are simply wrong for the
        /// record: ASP.NET Core's hosting scope holds the raw request path (an <c>/uploads</c> file
        /// name, a code-artifact run id) and its own <c>RequestId</c> under the same key as
        /// TensorSharp's; and long-lived worker threads (the inference engine, the diffusion
        /// scheduler) inherit the scope of whichever request created them, so every line they
        /// write would carry that request's ids. The request id and protocol are added by
        /// <see cref="LogContextEnrichmentProcessor"/> under their own names, and every record
        /// written inside a request carries its trace and span id, which lead to the spans that
        /// hold the model, backend and session. Local console and file logs keep their scopes.
        /// The rendered message is kept for records that carry no content, and removed by
        /// <see cref="ContentRedactionProcessor"/> from those that do.
        /// </summary>
        public static void ConfigureLoggerOptions(OpenTelemetryLoggerOptions options)
        {
            options.IncludeScopes = false;
            options.IncludeFormattedMessage = true;
        }

        /// <summary>
        /// Filters for the OpenTelemetry logger provider. The host's ASP.NET Core filter names no
        /// provider, and <c>Microsoft.Extensions.Logging</c> prefers rules that name a provider,
        /// so as soon as any OpenTelemetry-specific rule exists (from
        /// <c>Logging:OpenTelemetry:LogLevel</c> configuration or <c>TENSORSHARP_OTEL_LOG_LEVEL</c>)
        /// that filter would stop applying here. It is therefore repeated for this provider.
        /// </summary>
        public static ILoggingBuilder AddLoggingFilters(ILoggingBuilder logging, TelemetryOptions options)
        {
            logging.AddFilter<OpenTelemetryLoggerProvider>("Microsoft.AspNetCore", LogLevel.Warning);
            if (options.LogLevel is { } level)
                logging.AddFilter<OpenTelemetryLoggerProvider>(category: null, level);
            return logging;
        }

        internal static void ConfigureAzure(AzureMonitorExporterOptions azure, TelemetryOptions options)
        {
            azure.ConnectionString = options.AzureMonitorConnectionString;
            // The per-signal exporters do not support Live Metrics (only the cross-cutting
            // UseAzureMonitorExporter does), and leaving it at its default of true only makes
            // each registration log a warning.
            azure.EnableLiveMetrics = false;
            // AddAzureMonitorTraceExporter installs this sampler on the whole tracer provider,
            // OTLP included, and the SDK then ignores OTEL_TRACES_SAMPLER; its default is a
            // 5 traces/s rate limit. TelemetryOptions maps OTEL_TRACES_SAMPLER onto it instead.
            azure.TracesPerSecond = options.AzureTracesPerSecond;
            azure.SamplingRatio = options.AzureSamplingRatio;
        }

        /// <summary>
        /// Export what each provider holds, metrics first (their gauges need the services that
        /// are about to be disposed), then spans, then logs, and wait for that at most
        /// <paramref name="timeoutMilliseconds"/> in all. Never throws.
        ///
        /// <para>
        /// The flush runs on a worker thread because the deadline cannot be left to the SDK: a
        /// meter provider's flush collects and exports on the calling thread and checks its
        /// timeout only before the export starts, so a stalled metric exporter would hold the
        /// caller for the exporter's own timeout (10 s for OTLP). When the deadline passes the
        /// caller goes on (the server to dispose its models) and the export finishes or times out
        /// in the background; the providers' own flush at disposal still runs afterwards.
        /// </para>
        /// </summary>
        /// <returns>Whether the flush finished within the deadline.</returns>
        internal static bool Flush(MeterProvider? meter, TracerProvider? tracer, LoggerProvider? logger, int timeoutMilliseconds)
        {
            if (meter is null && tracer is null && logger is null)
                return true;
            var flush = new System.Threading.Thread(() =>
            {
                var watch = Stopwatch.StartNew();
                int Remaining() => (int)Math.Max(0, timeoutMilliseconds - watch.ElapsedMilliseconds);
                try { meter?.ForceFlush(Remaining()); } catch (Exception) { }
                try { tracer?.ForceFlush(Remaining()); } catch (Exception) { }
                try { logger?.ForceFlush(Remaining()); } catch (Exception) { }
            })
            {
                // A dedicated background thread: the thread pool can be starved at shutdown, and
                // a stalled export must not keep the process alive.
                IsBackground = true,
                Name = "TensorSharp telemetry flush",
            };
            flush.Start();
            return flush.Join(Math.Max(0, timeoutMilliseconds));
        }

        private static string SafeMachineName()
        {
            try { return Environment.MachineName; }
            catch (InvalidOperationException) { return "unknown"; }
        }
    }

    /// <summary>Dependency-injection entry point, used by <c>TensorSharp.Server.Host</c>.</summary>
    public static class TensorSharpTelemetryServiceCollectionExtensions
    {
        /// <summary>
        /// Build the tracer and meter providers now. The hosting package otherwise creates them
        /// when the host starts, and the server loads its startup model, warms the prefix cache
        /// and recovers Studio jobs before that; resolving them right after
        /// <c>builder.Build()</c> traces that work too, and makes them outlive (be disposed after)
        /// every service resolved later. A no-op when telemetry is off.
        ///
        /// <para>
        /// Also exports everything collected so far once the host has stopped serving
        /// (<see cref="IHostApplicationLifetime.ApplicationStopped"/>: the server and the hosted
        /// services have stopped, and dependency injection has not yet disposed the models).
        /// The providers' own flush comes later, when they are disposed, after the models, and
        /// two things depend on the earlier one. The engine and KV-cache gauges read only live
        /// engines, so a process that lived less than one export interval (always 60 s under
        /// Azure Monitor, whose reader ignores <c>OTEL_METRIC_EXPORT_INTERVAL</c>) would never
        /// report them. And a model's teardown can take long enough for an orchestrator's
        /// grace period to run out, which would lose the last requests' spans, logs and metric
        /// deltas. The flush is bounded by <see cref="StopFlushTimeoutMilliseconds"/> in all.
        /// </para>
        /// </summary>
        public static void StartTensorSharpTelemetry(this IServiceProvider services)
        {
            ArgumentNullException.ThrowIfNull(services);
            TracerProvider? tracer = services.GetService<TracerProvider>();
            MeterProvider? meter = services.GetService<MeterProvider>();
            LoggerProvider? logger = services.GetService<LoggerProvider>();
            if ((tracer is not null || meter is not null || logger is not null)
                && services.GetService<IHostApplicationLifetime>() is { } lifetime)
            {
                lifetime.ApplicationStopped.Register(
                    () => TensorSharpTelemetrySetup.Flush(meter, tracer, logger, StopFlushTimeoutMilliseconds));
            }
        }

        /// <summary>The most the flush when the host stops may take, all three signals together.</summary>
        public const int StopFlushTimeoutMilliseconds = 5000;

        /// <summary>
        /// Register traces, metrics and logs as <paramref name="options"/> says. Does nothing
        /// when telemetry is off, so the host can call it unconditionally. The callbacks add
        /// host-specific instrumentation (the server adds ASP.NET Core's) before the exporters.
        /// </summary>
        public static IServiceCollection AddTensorSharpTelemetry(
            this IServiceCollection services,
            ILoggingBuilder logging,
            TelemetryOptions options,
            Action<TracerProviderBuilder>? configureTracing = null,
            Action<MeterProviderBuilder>? configureMetrics = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(logging);
            ArgumentNullException.ThrowIfNull(options);
            TensorSharpTelemetrySetup.ApplyProcessWide(options);
            if (!options.Enabled)
                return services;

            TokenCredential? credential = TensorSharpTelemetrySetup.CreateAzureCredential(options);
            var otel = services.AddOpenTelemetry()
                .ConfigureResource(r => TensorSharpTelemetrySetup.ConfigureResource(r, options));
            if (options.Traces != TelemetrySinks.None)
                otel.WithTracing(t => TensorSharpTelemetrySetup.ConfigureTracing(t, options, credential, configureTracing));
            if (options.Metrics != TelemetrySinks.None)
                otel.WithMetrics(m => TensorSharpTelemetrySetup.ConfigureMetrics(m, options, credential, configureMetrics));
            if (options.Logs != TelemetrySinks.None)
            {
                otel.WithLogging(
                    l => TensorSharpTelemetrySetup.ConfigureLogging(l, options, credential),
                    TensorSharpTelemetrySetup.ConfigureLoggerOptions);
                TensorSharpTelemetrySetup.AddLoggingFilters(logging, options);
            }
            return services;
        }
    }

    /// <summary>
    /// Telemetry for a process without a host (the CLI): builds the tracer and meter
    /// providers, adds the logger provider to a <see cref="ILoggingBuilder"/>, and flushes
    /// everything on <see cref="Dispose"/>, which a short run must call before it exits.
    /// </summary>
    public sealed class TelemetrySession : IDisposable
    {
        private readonly TelemetryOptions _options;
        private readonly TokenCredential? _credential;
        private TracerProvider? _tracerProvider;
        private MeterProvider? _meterProvider;

        private TelemetrySession(TelemetryOptions options)
        {
            _options = options;
            _credential = TensorSharpTelemetrySetup.CreateAzureCredential(options);
            if (options.Traces != TelemetrySinks.None)
            {
                var tracing = Sdk.CreateTracerProviderBuilder()
                    .ConfigureResource(r => TensorSharpTelemetrySetup.ConfigureResource(r, options));
                _tracerProvider = TensorSharpTelemetrySetup.ConfigureTracing(tracing, options, _credential).Build();
            }
            if (options.Metrics != TelemetrySinks.None)
            {
                var metrics = Sdk.CreateMeterProviderBuilder()
                    .ConfigureResource(r => TensorSharpTelemetrySetup.ConfigureResource(r, options));
                _meterProvider = TensorSharpTelemetrySetup.ConfigureMetrics(metrics, options, _credential).Build();
            }
        }

        public TelemetryOptions Options => _options;

        /// <summary>Start telemetry, or return null when it is off.</summary>
        public static TelemetrySession? Start(TelemetryOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            TensorSharpTelemetrySetup.ApplyProcessWide(options);
            return options.Enabled ? new TelemetrySession(options) : null;
        }

        /// <summary>
        /// Add the OpenTelemetry logger provider to a logger factory under construction. The
        /// factory owns it and flushes it when the factory is disposed.
        /// </summary>
        public void AddLogging(ILoggingBuilder logging)
        {
            ArgumentNullException.ThrowIfNull(logging);
            if (_options.Logs == TelemetrySinks.None)
                return;
            logging.AddOpenTelemetry(o =>
            {
                TensorSharpTelemetrySetup.ConfigureLoggerOptions(o);
                o.SetResourceBuilder(TensorSharpTelemetrySetup.ConfigureResource(ResourceBuilder.CreateEmpty(), _options));
                o.AddProcessor(new LogContextEnrichmentProcessor());
                if (!_options.CaptureMessageContent)
                    o.AddProcessor(new ContentRedactionProcessor());
                if ((_options.Logs & TelemetrySinks.Otlp) != 0) o.AddOtlpExporter(e => _options.OtlpLogs.ApplyTo(e));
                if ((_options.Logs & TelemetrySinks.Console) != 0) o.AddConsoleExporter();
                if ((_options.Logs & TelemetrySinks.AzureMonitor) != 0)
                    o.AddAzureMonitorLogExporter(a => TensorSharpTelemetrySetup.ConfigureAzure(a, _options), _credential);
            });
            TensorSharpTelemetrySetup.AddLoggingFilters(logging, _options);
        }

        /// <summary>
        /// Export the metrics and spans collected so far, without stopping anything. The CLI
        /// calls it before it disposes its inference engine: the engine and KV-cache gauges read
        /// only live engines, so a run shorter than one export interval would otherwise never
        /// report them. Bounded by <paramref name="timeoutMilliseconds"/> in all; never throws.
        /// </summary>
        public void Flush(int timeoutMilliseconds = TensorSharpTelemetryServiceCollectionExtensions.StopFlushTimeoutMilliseconds) =>
            TensorSharpTelemetrySetup.Flush(System.Threading.Volatile.Read(ref _meterProvider),
                System.Threading.Volatile.Read(ref _tracerProvider), logger: null, timeoutMilliseconds);

        /// <summary>Flush and stop the tracer and meter providers (bounded by the SDK's export timeouts).</summary>
        public void Dispose()
        {
            var tracer = System.Threading.Interlocked.Exchange(ref _tracerProvider, null);
            var meter = System.Threading.Interlocked.Exchange(ref _meterProvider, null);
            try { tracer?.ForceFlush(5000); } catch (Exception) { }
            try { meter?.ForceFlush(5000); } catch (Exception) { }
            tracer?.Dispose();
            meter?.Dispose();
        }
    }
}
