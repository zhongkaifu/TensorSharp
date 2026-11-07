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
using Microsoft.Extensions.Logging;
using OpenTelemetry.Exporter;
using TensorSharp.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// The on/off and exporter rules of the opt-in OpenTelemetry pipeline, over an explicit
/// variable map (never the process environment), so every combination is checked without
/// touching global state.
/// </summary>
public class TelemetryOptionsTests
{
    private static TelemetryOptions Resolve(TelemetryHost host = TelemetryHost.Server, params (string Name, string Value)[] variables)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (name, value) in variables)
            map[name] = value;
        return TelemetryOptions.FromEnvironment(host, n => map.TryGetValue(n, out var v) ? v : null);
    }

    [Fact]
    public void NothingSetMeansOff()
    {
        var options = Resolve();
        Assert.False(options.Enabled);
        Assert.Equal(TelemetrySinks.None, options.Traces | options.Metrics | options.Logs);
        Assert.StartsWith("Telemetry: off", options.Describe());
    }

    [Fact]
    public void AnOtlpEndpointTurnsEverySignalOnOverOtlp()
    {
        var options = Resolve(variables: ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"));
        Assert.True(options.Enabled);
        Assert.Equal(TelemetrySinks.Otlp, options.Traces);
        Assert.Equal(TelemetrySinks.Otlp, options.Metrics);
        Assert.Equal(TelemetrySinks.Otlp, options.Logs);
        Assert.Contains("OTLP http://collector:4317 (grpc)", options.Describe());
        Assert.Equal("tensorsharp-server", options.ServiceName);
    }

    [Fact]
    public void TheStartupLineShowsPerSignalEndpointsAndProtocols()
    {
        var options = Resolve(variables: new[]
        {
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://user:secret@tempo:4318/v1/traces"),
            ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf"),
            ("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "http/protobuf"),
            // The same as the generic protocol: not an override worth showing.
            ("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "grpc"),
        });
        string line = options.Describe();
        Assert.Contains("OTLP http://collector:4317 (grpc; traces: http://tempo:4318/v1/traces http/protobuf; metrics: http/protobuf)", line);
        Assert.DoesNotContain("secret", line);
        Assert.DoesNotContain("logs:", line);
    }

    [Fact]
    public void ApplicationInsightsAloneSendsEverySignalToAzureMonitorAndNeverPrintsTheConnectionString()
    {
        const string connection = "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://example.invalid/";
        var options = Resolve(variables: ("APPLICATIONINSIGHTS_CONNECTION_STRING", connection));
        Assert.True(options.Enabled);
        Assert.Equal(TelemetrySinks.AzureMonitor, options.Traces);
        Assert.Equal(TelemetrySinks.AzureMonitor, options.Logs);
        Assert.Equal(connection, options.AzureMonitorConnectionString);
        string line = options.Describe();
        Assert.Contains("Azure Monitor", line);
        Assert.DoesNotContain("InstrumentationKey", line);
        Assert.DoesNotContain("00000000", line);
    }

    [Fact]
    public void OtlpAndApplicationInsightsCombine()
    {
        var options = Resolve(variables: new[]
        {
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=x"),
        });
        Assert.Equal(TelemetrySinks.Otlp | TelemetrySinks.AzureMonitor, options.Traces);
        Assert.Equal(TelemetrySinks.Otlp | TelemetrySinks.AzureMonitor, options.Metrics);
    }

    [Theory]
    [InlineData("true")]
    [InlineData("TRUE")]
    [InlineData("1")]
    public void SdkDisabledWinsOverEverything(string value)
    {
        var options = Resolve(variables: new[]
        {
            ("OTEL_SDK_DISABLED", value),
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=x"),
            ("TENSORSHARP_OTEL", "1"),
        });
        Assert.False(options.Enabled);
        Assert.Contains("OTEL_SDK_DISABLED", options.Describe());
    }

    [Theory]
    [InlineData(null, "http://localhost:4317 (grpc)")]
    [InlineData("http/protobuf", "http://localhost:4318 (http/protobuf)")]
    public void TensorSharpOtelUsesTheSdkDefaultEndpointForTheProtocol(string? protocol, string expected)
    {
        var variables = new List<(string, string)> { ("TENSORSHARP_OTEL", "1") };
        if (protocol != null) variables.Add(("OTEL_EXPORTER_OTLP_PROTOCOL", protocol));
        var options = Resolve(variables: variables.ToArray());
        Assert.True(options.Enabled);
        Assert.Equal(TelemetrySinks.Otlp, options.Traces);
        Assert.Contains(expected, options.Describe());
    }

    [Fact]
    public void ASignalSpecificEndpointTurnsOnlyThatSignalOnAndIsUsedAsGiven()
    {
        var options = Resolve(variables: ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces.example:4318/v1/traces"));
        Assert.True(options.Enabled);
        Assert.Equal(TelemetrySinks.Otlp, options.Traces);
        Assert.Equal(TelemetrySinks.None, options.Metrics);
        Assert.Equal(TelemetrySinks.None, options.Logs);
        Assert.Equal(new Uri("http://traces.example:4318/v1/traces"), options.OtlpTraces.Endpoint);
        Assert.True(options.OtlpMetrics.IsEmpty);
    }

    [Fact]
    public void SignalSpecificHeadersProtocolAndTimeoutReachTheExporterOptions()
    {
        var options = Resolve(variables: new[]
        {
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "x-otlp-api-key=secret"),
            ("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf"),
            ("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT", "2500"),
        });
        var exporter = new OtlpExporterOptions();
        options.OtlpLogs.ApplyTo(exporter);
        Assert.Equal("x-otlp-api-key=secret", exporter.Headers);
        Assert.Equal(OtlpExportProtocol.HttpProtobuf, exporter.Protocol);
        Assert.Equal(2500, exporter.TimeoutMilliseconds);
        // Headers are credentials: never in the startup line.
        Assert.DoesNotContain("secret", options.Describe());
    }

    [Fact]
    public void ExporterNoneTurnsOneSignalOffForEverySinkIncludingAzureMonitor()
    {
        var options = Resolve(variables: new[]
        {
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=x"),
            ("OTEL_LOGS_EXPORTER", "none"),
        });
        Assert.Equal(TelemetrySinks.None, options.Logs);
        Assert.Equal(TelemetrySinks.Otlp | TelemetrySinks.AzureMonitor, options.Traces);
        Assert.Contains("logs(off)", options.Describe());
    }

    [Fact]
    public void ExporterConsoleAloneIsEnoughToTurnTelemetryOn()
    {
        var options = Resolve(variables: ("OTEL_TRACES_EXPORTER", "console"));
        Assert.True(options.Enabled);
        Assert.Equal(TelemetrySinks.Console, options.Traces);
        Assert.Equal(TelemetrySinks.None, options.Metrics);
    }

    [Fact]
    public void AnUnknownExporterNameIsAWarningNotAnError()
    {
        var options = Resolve(variables: new[]
        {
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("OTEL_METRICS_EXPORTER", "prometheus,otlp"),
        });
        Assert.Equal(TelemetrySinks.Otlp, options.Metrics);
        Assert.Contains(options.Warnings, w => w.Contains("prometheus", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("OTEL_EXPORTER_OTLP_ENDPOINT", "collector:4317")]
    [InlineData("OTEL_EXPORTER_OTLP_ENDPOINT", "ftp://collector")]
    [InlineData("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "/v1/traces")]
    [InlineData("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")]
    [InlineData("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "udp")]
    [InlineData("OTEL_EXPORTER_OTLP_METRICS_TIMEOUT", "soon")]
    [InlineData("TENSORSHARP_OTEL_LOG_LEVEL", "Loud")]
    public void AValueTheSdkWouldMisreadIsAConfigurationError(string name, string value)
    {
        var variables = new List<(string, string)> { ("TENSORSHARP_OTEL", "1"), (name, value) };
        Assert.Throws<ArgumentException>(() => Resolve(variables: variables.ToArray()));
    }

    [Theory]
    [InlineData("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")]
    [InlineData("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "udp")]
    [InlineData("OTEL_EXPORTER_OTLP_METRICS_TIMEOUT", "soon")]
    public void AnOtlpValueSetForAnotherProgramNeverStopsStartupWhenTelemetryIsOff(string name, string value)
    {
        // The OTEL_* variables are shared by every OpenTelemetry SDK on the machine.
        var options = Resolve(variables: (name, value));
        Assert.False(options.Enabled);
    }

    [Fact]
    public void AnOtlpProtocolIsNotValidatedWhenOnlyApplicationInsightsIsUsed()
    {
        var options = Resolve(variables: new[]
        {
            ("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=x"),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "udp"),
        });
        Assert.True(options.Enabled);
        Assert.Equal(TelemetrySinks.AzureMonitor, options.Traces);
        Assert.True(options.OtlpTraces.IsEmpty);
    }

    [Fact]
    public void EndpointCredentialsAreNotPrinted()
    {
        var options = Resolve(variables: ("OTEL_EXPORTER_OTLP_ENDPOINT", "https://user:pa55@collector.example:4318"));
        string line = options.Describe();
        Assert.DoesNotContain("pa55", line);
        Assert.Contains("collector.example", line);
    }

    [Fact]
    public void TheServiceNameDefaultsPerHostAndHonoursOtelServiceName()
    {
        Assert.Equal("tensorsharp-cli", Resolve(TelemetryHost.Cli, ("TENSORSHARP_OTEL", "1")).ServiceName);
        var named = Resolve(TelemetryHost.Server, ("TENSORSHARP_OTEL", "1"), ("OTEL_SERVICE_NAME", "inference-eu"));
        Assert.Equal("inference-eu", named.ServiceName);
        Assert.True(named.ServiceNameFromEnvironment);
    }

    [Fact]
    public void ContentCaptureIsOffUnlessAsked()
    {
        Assert.False(Resolve(variables: ("TENSORSHARP_OTEL", "1")).CaptureMessageContent);
        var on = Resolve(variables: new[] { ("TENSORSHARP_OTEL", "1"), ("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true") });
        Assert.True(on.CaptureMessageContent);
        Assert.Contains("content capture ON", on.Describe());
    }

    [Fact]
    public void TheExportLogLevelIsParsed()
    {
        var options = Resolve(variables: new[] { ("TENSORSHARP_OTEL", "1"), ("TENSORSHARP_OTEL_LOG_LEVEL", "warning") });
        Assert.Equal(LogLevel.Warning, options.LogLevel);
    }

    [Fact]
    public void EntraIdNeedsAConnectionString()
    {
        Assert.False(Resolve(variables: new[] { ("TENSORSHARP_OTEL", "1"), ("TENSORSHARP_AZURE_MONITOR_AAD", "1") }).AzureMonitorUseEntraId);
        var entra = Resolve(variables: new[]
        {
            ("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=x"),
            ("TENSORSHARP_AZURE_MONITOR_AAD", "1"),
        });
        Assert.True(entra.AzureMonitorUseEntraId);
        Assert.Contains("Azure Monitor (Entra ID)", entra.Describe());
    }

    [Fact]
    public void WithoutApplicationInsightsTheSdkSamplerIsDescribedAsIs()
    {
        var options = Resolve(variables: new[]
        {
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317"),
            ("OTEL_TRACES_SAMPLER", "parentbased_traceidratio"),
            ("OTEL_TRACES_SAMPLER_ARG", "0.25"),
        });
        Assert.Equal("parentbased_traceidratio(0.25)", options.EffectiveSampler);
    }

    [Theory]
    [InlineData(null, null, 1.0f, null)]
    [InlineData("parentbased_always_on", null, 1.0f, null)]
    [InlineData("parentbased_traceidratio", "0.1", 0.1f, null)]
    [InlineData("traceidratio", "2", 1.0f, null)]
    [InlineData("always_off", null, 0f, null)]
    [InlineData("microsoft.fixed_percentage", "0.5", 0.5f, null)]
    [InlineData("microsoft.rate_limited", "3", 1.0f, 3.0)]
    public void WithApplicationInsightsTheStandardSamplerIsMappedOntoTheAzureSampler(
        string? sampler, string? arg, float ratio, double? perSecond)
    {
        var variables = new List<(string, string)> { ("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=x") };
        if (sampler != null) variables.Add(("OTEL_TRACES_SAMPLER", sampler));
        if (arg != null) variables.Add(("OTEL_TRACES_SAMPLER_ARG", arg));
        var options = Resolve(variables: variables.ToArray());
        Assert.Equal(ratio, options.AzureSamplingRatio, 3);
        Assert.Equal(perSecond, options.AzureTracesPerSecond);
        Assert.StartsWith("azuremonitor ", options.EffectiveSampler);
    }

    [Fact]
    public void AnUnsupportedSamplerUnderApplicationInsightsWarnsAndSamplesEverything()
    {
        var options = Resolve(variables: new[]
        {
            ("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=x"),
            ("OTEL_TRACES_SAMPLER", "jaeger_remote"),
        });
        Assert.Equal(1.0f, options.AzureSamplingRatio);
        Assert.Contains(options.Warnings, w => w.Contains("jaeger_remote", StringComparison.Ordinal));
    }
}

public class TelemetryCliFlagsTests
{
    [Fact]
    public void TheFlagIsConsumedInBothSpellingsAndTheLastOneWins()
    {
        string[] rest = TelemetryCliFlags.Consume(
            new[] { "--model", "m.gguf", "--otel-endpoint", "http://a:4317", "--port", "5000", "--otel-endpoint=http://b:4318" },
            out string? endpoint);
        Assert.Equal("http://b:4318", endpoint);
        Assert.Equal(new[] { "--model", "m.gguf", "--port", "5000" }, rest);
    }

    [Fact]
    public void WithoutTheFlagArgumentsPassThroughUnchanged()
    {
        string[] args = { "--model", "m.gguf", "--otel" };
        Assert.Equal(args, TelemetryCliFlags.Consume(args, out string? endpoint));
        Assert.Null(endpoint);
    }

    [Theory]
    [InlineData("--otel-endpoint")]
    [InlineData("--otel-endpoint", "--port")]
    [InlineData("--otel-endpoint", "collector:4317")]
    [InlineData("--otel-endpoint=")]
    public void AMissingOrRelativeUrlIsAConfigurationError(params string[] args)
    {
        Assert.Throws<ArgumentException>(() => TelemetryCliFlags.Consume(args, out _));
    }
}
