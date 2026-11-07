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
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Logs;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Host.Hosting;
using TensorSharp.Server.Logging;
using TensorSharp.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>A logger that keeps what the request logger hands it: each entry's state and each scope.</summary>
internal sealed class RequestLogCapture : ILogger<RequestLoggingMiddleware>
{
    public List<Dictionary<string, object>> Scopes { get; } = new();

    public List<(LogLevel Level, EventId EventId, IReadOnlyList<KeyValuePair<string, object?>> State)> Entries { get; } = new();

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
    {
        if (state is IEnumerable<KeyValuePair<string, object>> pairs)
            Scopes.Add(pairs.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal));
        return null;
    }

    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        var values = state is IReadOnlyList<KeyValuePair<string, object?>> list
            ? list.ToList()
            : new List<KeyValuePair<string, object?>>();
        Entries.Add((logLevel, eventId, values));
    }

    public static object? Value(IReadOnlyList<KeyValuePair<string, object?>> state, string key) =>
        Assert.Single(state, p => p.Key == key).Value;

    public static DefaultHttpContext Request(string method, string path)
    {
        var context = new DefaultHttpContext();
        context.Request.Method = method;
        context.Request.Path = path;
        return context;
    }
}

/// <summary>
/// The request logger's telemetry facts: the protocol label, the request id on the server span
/// and in <see cref="TelemetryContext"/>, and capability paths marked as content. None of these
/// touch process-wide switches (<see cref="TelemetryContext"/> is an AsyncLocal the middleware
/// sets inside its own call), so they run in parallel.
/// </summary>
public class RequestLoggingTelemetryTests
{
    /// <summary>The <c>IHttpActivityFeature</c> ASP.NET Core hosting sets; a test context has none.</summary>
    private sealed class ActivityFeature : IHttpActivityFeature
    {
        public ActivityFeature(Activity activity) => Activity = activity;
        public Activity Activity { get; set; }
    }

    [Theory]
    [InlineData("/api/embed", "ollama")]
    [InlineData("/API/EMBED", "ollama")]
    [InlineData("/api/chat/ollama", "ollama")]
    [InlineData("/api/generate", "ollama")]
    [InlineData("/api/tags", "ollama")]
    [InlineData("/v1/embeddings", "openai")]
    [InlineData("/v1/chat/completions", "openai")]
    [InlineData("/api/chat", "webui")]
    [InlineData("/api/queue/status", "webui")]
    [InlineData("/uploads/9f2c.png", null)]
    [InlineData("/health", null)]
    public void TheClientLabelNamesTheProtocolAndApiEmbedIsOllama(string path, string? expected)
    {
        // /api/embed shares the /api/ prefix with the Web UI routes, but it is Ollama's
        // embedding endpoint; tensorsharp.protocol on its spans and metrics must say so.
        Assert.Equal(expected, RequestLoggingMiddleware.ClassifyClient(new PathString(path)));
    }

    [Fact]
    public async Task ALongInboundRequestIdIsEchoedWholeButTelemetryGetsABoundedPrefix()
    {
        string inbound = string.Concat(Enumerable.Range(0, 100).Select(i => (char)('a' + i % 26)));
        Assert.Equal(100, inbound.Length);
        Assert.Equal(64, RequestLoggingMiddleware.MaxTelemetryRequestIdLength);

        // A recorded server span, as the ASP.NET Core instrumentation creates one. A private
        // source and listener, so no other test's listener is involved.
        using var source = new ActivitySource("InferenceWeb.Tests.ServerSpan." + Guid.NewGuid().ToString("N"));
        using var listener = new ActivityListener
        {
            ShouldListenTo = s => ReferenceEquals(s, source),
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
        };
        ActivitySource.AddActivityListener(listener);
        using var span = source.StartActivity("POST /v1/chat/completions", ActivityKind.Server);
        Assert.NotNull(span);

        var context = RequestLogCapture.Request("POST", "/v1/chat/completions");
        context.Request.Headers[RequestLoggingMiddleware.RequestIdHeader] = inbound;
        context.Features.Set<IHttpActivityFeature>(new ActivityFeature(span!));
        string? seenRequestId = null, seenProtocol = null;
        var logger = new RequestLogCapture();
        var middleware = new RequestLoggingMiddleware(_ =>
        {
            // What the generation spans and metrics further down read.
            seenRequestId = TelemetryContext.RequestId;
            seenProtocol = TelemetryContext.Protocol;
            return Task.CompletedTask;
        }, logger);

        await middleware.InvokeAsync(context);

        // The client gets back exactly what it sent, and local logs keep it whole...
        Assert.Equal(inbound, context.Response.Headers[RequestLoggingMiddleware.RequestIdHeader].ToString());
        Assert.Equal(inbound, Assert.Single(logger.Scopes)[LogScopeKeys.RequestId]);
        // ...while everything exported is cut to 64 characters.
        Assert.Equal(inbound[..64], seenRequestId);
        Assert.Equal(inbound[..64], span!.GetTagItem(TelemetryAttributes.RequestId));
        Assert.Equal("openai", seenProtocol);
        Assert.Equal("openai", span.GetTagItem(TelemetryAttributes.Protocol));
    }

    [Fact]
    public async Task AGeneratedRequestIdReachesTelemetryUnchanged()
    {
        // No inbound header and no recorded server span (telemetry off): the generated
        // 12-character id is used as is, and the middleware must not need a span.
        var context = RequestLogCapture.Request("GET", "/api/chat");
        string? seenRequestId = null;
        var middleware = new RequestLoggingMiddleware(_ => { seenRequestId = TelemetryContext.RequestId; return Task.CompletedTask; },
            new RequestLogCapture());

        await middleware.InvokeAsync(context);

        string generated = context.Response.Headers[RequestLoggingMiddleware.RequestIdHeader].ToString();
        Assert.Equal(12, generated.Length);
        Assert.Equal(generated, seenRequestId);
    }

    [Fact]
    public void TelemetryTruncationNeverSplitsASurrogatePair()
    {
        // The 64th character is the high half of an emoji: cutting there would leave a lone
        // surrogate, which some exporters reject and others turn into U+FFFD.
        string id = new string('a', 63) + "\U0001F600" + "tail";
        Assert.True(char.IsHighSurrogate(id[63]));
        Assert.Equal(new string('a', 63), RequestLoggingMiddleware.TruncateForTelemetry(id));

        string exact = new string('b', 64);
        Assert.Same(exact, RequestLoggingMiddleware.TruncateForTelemetry(exact));
        Assert.Equal(new string('c', 64), RequestLoggingMiddleware.TruncateForTelemetry(new string('c', 65)));
    }

    [Theory]
    [InlineData("/uploads/9f2c4e1ab37d.png")]
    [InlineData("/api/code/artifacts/run-7f3a/report.csv")]
    [InlineData("/v1/responses/resp_4b9e2c")]
    public async Task ACapabilityPathIsLoggedAsContent(string path)
    {
        var logger = new RequestLogCapture();
        var middleware = new RequestLoggingMiddleware(_ => Task.CompletedTask, logger);

        await middleware.InvokeAsync(RequestLogCapture.Request("GET", path));

        Assert.Equal(2, logger.Entries.Count); // started, completed
        foreach (var entry in logger.Entries)
        {
            var marked = Assert.IsType<LogContent>(RequestLogCapture.Value(entry.State, "Path"));
            Assert.Equal(path, marked.Text);
        }
    }

    [Theory]
    [InlineData("/v1/chat/completions")]
    [InlineData("/v1/responses")]
    [InlineData("/uploads")]
    [InlineData("/api/chat")]
    public async Task AnOrdinaryRouteIsLoggedAsPlainText(string path)
    {
        var logger = new RequestLogCapture();
        var middleware = new RequestLoggingMiddleware(_ => Task.CompletedTask, logger);

        await middleware.InvokeAsync(RequestLogCapture.Request("POST", path));

        foreach (var entry in logger.Entries)
            Assert.Equal(path, Assert.IsType<string>(RequestLogCapture.Value(entry.State, "Path")));
    }

    [Fact]
    public async Task AFailedCapabilityRequestStillMarksItsPath()
    {
        const string path = "/uploads/0a1b2c3d.mp4";
        var logger = new RequestLogCapture();
        var middleware = new RequestLoggingMiddleware(_ => throw new IOException("disk"), logger);

        await Assert.ThrowsAsync<IOException>(() => middleware.InvokeAsync(RequestLogCapture.Request("GET", path)));

        var failed = Assert.Single(logger.Entries, e => e.EventId == LogEventIds.HttpRequestFailed);
        Assert.Equal(path, Assert.IsType<LogContent>(RequestLogCapture.Value(failed.State, "Path")).Text);
    }

    /// <summary>What an exporter would see, copied in OnEnd (LogRecord instances are pooled).</summary>
    private sealed record Exported(
        string? FormattedMessage, object? Body,
        IReadOnlyList<KeyValuePair<string, object?>> Attributes,
        IReadOnlyList<KeyValuePair<string, object?>> Scopes);

    private sealed class ExportCapture : BaseProcessor<LogRecord>
    {
        public List<Exported> Records { get; } = new();

        public override void OnEnd(LogRecord data)
        {
            var scopes = new List<KeyValuePair<string, object?>>();
            data.ForEachScope((scope, list) =>
            {
                foreach (var pair in scope)
                    list.Add(pair);
            }, scopes);
            Records.Add(new Exported(data.FormattedMessage, data.Body,
                data.Attributes?.ToList() ?? new List<KeyValuePair<string, object?>>(), scopes));
        }
    }

    private static async Task<(List<Exported> Records, string RequestId)> ExportUploadRequest(string path, bool includeScopes)
    {
        var exported = new ExportCapture();
        var context = RequestLogCapture.Request("GET", path);
        using (var factory = LoggerFactory.Create(builder => builder.AddOpenTelemetry(o =>
        {
            TensorSharpTelemetrySetup.ConfigureLoggerOptions(o);
            if (includeScopes)
                o.IncludeScopes = true; // the control: what would leak without the setup's choice
            o.AddProcessor(new LogContextEnrichmentProcessor());
            o.AddProcessor(new ContentRedactionProcessor());
            o.AddProcessor(exported);
        })))
        {
            var middleware = new RequestLoggingMiddleware(_ => Task.CompletedTask, factory.CreateLogger<RequestLoggingMiddleware>());
            await middleware.InvokeAsync(context);
        }
        return (exported.Records, context.Response.Headers[RequestLoggingMiddleware.RequestIdHeader].ToString());
    }

    [Fact]
    public async Task AnExportedRequestLogCarriesNeitherTheUploadNameNorTheScopeThatHoldsIt()
    {
        string file = Guid.NewGuid().ToString("N") + ".png";
        string path = "/uploads/" + file;

        var (records, requestId) = await ExportUploadRequest(path, includeScopes: false);

        Assert.Equal(2, records.Count);
        foreach (var record in records)
        {
            Assert.Equal($"[redacted: {path.Length} chars]", Assert.Single(record.Attributes, a => a.Key == "Path").Value);
            Assert.DoesNotContain(record.Attributes, a => a.Value?.ToString()?.Contains(file, StringComparison.Ordinal) == true);
            // The rendered message quotes the path, so it is dropped and the template is the body.
            Assert.Null(record.FormattedMessage);
            Assert.DoesNotContain(file, record.Body?.ToString() ?? string.Empty, StringComparison.Ordinal);
            // The Operation scope ("GET /uploads/<file>") is never exported.
            Assert.Empty(record.Scopes);
            Assert.Equal(requestId, Assert.Single(record.Attributes, a => a.Key == TelemetryAttributes.RequestId).Value);
        }

        // Control: with scopes included the Operation scope would carry the file name, so the
        // setup's IncludeScopes = false is what keeps it out.
        var (leaky, _) = await ExportUploadRequest(path, includeScopes: true);
        Assert.Contains(leaky.SelectMany(r => r.Scopes),
            s => s.Key == LogScopeKeys.Operation && Equals(s.Value, "GET " + path));
    }
}

/// <summary>
/// Request-logger and server-span behaviour that depends on the process-wide content-capture
/// and logs-exported switches, so these run in the non-parallel telemetry collection and
/// restore both switches.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public sealed class ServerTelemetryContentSwitchTests : IDisposable
{
    private readonly bool _previousLogsExported = TensorSharpTelemetry.LogsExported;

    public void Dispose()
    {
        TensorSharpTelemetry.ConfigureLogsExported(_previousLogsExported);
        TensorSharpTelemetry.ConfigureCaptureMessageContent(null);
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(true, true)]
    [InlineData(false, true)]
    public async Task AnUploadPathIsAlwaysMarkedAndTheLocalOperationScopeIsUnchanged(bool logsExported, bool capture)
    {
        TensorSharpTelemetry.ConfigureLogsExported(logsExported);
        TensorSharpTelemetry.ConfigureCaptureMessageContent(capture);
        string path = "/uploads/" + Guid.NewGuid().ToString("N") + ".webp";
        var logger = new RequestLogCapture();
        var middleware = new RequestLoggingMiddleware(_ => Task.CompletedTask, logger);

        await middleware.InvokeAsync(RequestLogCapture.Request("GET", path));

        // The marker does not depend on the switches: the exporter decides what to do with it.
        Assert.Equal(2, logger.Entries.Count);
        foreach (var entry in logger.Entries)
            Assert.Equal(path, Assert.IsType<LogContent>(RequestLogCapture.Value(entry.State, "Path")).Text);
        // Scopes are not exported (TensorSharpTelemetrySetup.ConfigureLoggerOptions), so the
        // console and file logs keep the real path whatever the switches say.
        var scope = Assert.Single(logger.Scopes);
        Assert.Equal("GET " + path, scope[LogScopeKeys.Operation]);
        Assert.False(scope.ContainsKey(LogScopeKeys.Client));
    }

    private static Activity ServerSpanFor(string method, string path)
    {
        var activity = new Activity(method + " " + path);
        activity.SetTag("url.path", path);
        return activity;
    }

    [Theory]
    [InlineData("/uploads/9f2c4e1ab37d.png", "/uploads/{file}")]
    [InlineData("/UPLOADS/9f2c4e1ab37d.png", "/uploads/{file}")]
    [InlineData("/api/code/artifacts/run-7f3a/report.csv", "/api/code/artifacts/{run}/{path}")]
    [InlineData("/api/code/artifacts/run-7f3a", "/api/code/artifacts/{run}")]
    [InlineData("/v1/responses/resp_4b9e2c", "/v1/responses/{id}")]
    [InlineData("/v1/responses/resp_4b9e2c/input_items", "/v1/responses/{id}/input_items")]
    public void WithCaptureOffACapabilityPathIsReplacedInUrlPathAndTheSpanName(string path, string masked)
    {
        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);
        using var activity = ServerSpanFor("GET", path);

        ServerTelemetry.Enrich(activity, RequestLogCapture.Request("GET", path).Request);

        Assert.Equal(masked, activity.GetTagItem("url.path"));
        Assert.Equal("GET " + masked, activity.DisplayName);
    }

    [Fact]
    public void AnUploadSpanUsesThePlaceholderTheRequestLoggerDocuments()
    {
        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);
        using var activity = ServerSpanFor("GET", "/uploads/a.png");
        ServerTelemetry.Enrich(activity, RequestLogCapture.Request("GET", "/uploads/a.png").Request);
        Assert.Equal(UploadStaticFiles.RedactedFilePath, activity.GetTagItem("url.path"));
    }

    [Fact]
    public void WithCaptureOnTheSpanKeepsTheRealPath()
    {
        TensorSharpTelemetry.ConfigureCaptureMessageContent(true);
        const string path = "/uploads/9f2c4e1ab37d.png";
        using var activity = ServerSpanFor("GET", path);

        ServerTelemetry.Enrich(activity, RequestLogCapture.Request("GET", path).Request);

        Assert.Equal(path, activity.GetTagItem("url.path"));
        Assert.Equal("GET " + path, activity.DisplayName);
    }

    [Theory]
    [InlineData("/v1/chat/completions")]
    [InlineData("/v1/responses")]
    [InlineData("/uploads")]
    [InlineData("/uploads/")]
    [InlineData("/uploadsx/a.png")]
    [InlineData("/api/chat")]
    public void AnOrdinaryRouteIsLeftAloneWithCaptureOff(string path)
    {
        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);
        using var activity = ServerSpanFor("POST", path);

        ServerTelemetry.Enrich(activity, RequestLogCapture.Request("POST", path).Request);

        Assert.Equal(path, activity.GetTagItem("url.path"));
        Assert.Equal("POST " + path, activity.DisplayName);
    }
}

/// <summary>
/// <c>--otel-endpoint</c> on the server command line: Program.cs consumes it before
/// <see cref="ServerOptionsBuilder.Build"/>, but Build must still accept it (an embedder or a
/// test hands it the raw command line), suggest it for a typo, and the usage page must name it.
/// Build reads the environment, so these run in the non-parallel collection with the
/// variables the other server-option tests clear.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public sealed class ServerOtelEndpointFlagTests : IDisposable
{
    private readonly string _baseDir;
    private readonly EnvScope _env = new();

    public ServerOtelEndpointFlagTests()
    {
        _env.ClearSpeculationVars();
        foreach (string variable in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TENSORSHARP_TP_NODE_ID", "TENSORSHARP_TP_PEERS" })
            _env.Set(variable, null);
        _env.Set(TensorSharp.Runtime.LoraCliFlags.EnvironmentVariable, null);
        _env.Set(TextLoraHostOptions.EnvironmentVariable, null);
        // Build creates an uploads folder under its base directory.
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-otel-flag-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    [Theory]
    [InlineData("--otel-endpoint", "http://localhost:4317")]
    [InlineData("--otel-endpoint=http://localhost:4317")]
    [InlineData("--OTEL-ENDPOINT", "https://collector.example:4318")]
    public void BuildAcceptsTheEndpointFlagInBothSpellings(params string[] otelArgs)
    {
        // A flag after it must still be parsed: the URL was consumed as the flag's value,
        // not left behind as a positional argument.
        string[] args = otelArgs.Concat(new[] { "--video-mode", "ref" }).ToArray();
        var options = ServerOptionsBuilder.Build(args, _baseDir);
        Assert.Equal("ref", options.DefaultVideoMode);
    }

    [Fact]
    public void ATypoOfTheEndpointFlagSuggestsIt()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.Build(new[] { "--otel-endpiont", "http://localhost:4317" }, _baseDir));
        Assert.Contains("Unknown option '--otel-endpiont'", ex.Message);
        Assert.Contains("Did you mean '--otel-endpoint'?", ex.Message);
    }

    [Fact]
    public void TheServerAndTheTelemetryPackageNameTheSameFlag()
    {
        // TensorSharp.Server cannot reference TensorSharp.Telemetry, so the name is a literal
        // in both; Program.cs consumes it with one and Build skips it with the other.
        Assert.Equal(TelemetryCliFlags.EndpointFlag, ServerOptionsBuilder.OtelEndpointFlag);
    }

    [Fact]
    public void TheUsagePageDocumentsTheEndpointFlag()
    {
        Assert.Contains(TelemetryCliFlags.EndpointFlag, ServerUsage.DocumentedFlags());
        var writer = new StringWriter();
        ServerUsage.PrintUsage(writer);
        string usage = writer.ToString();
        Assert.Contains("--otel-endpoint <url>", usage);
        Assert.Contains("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", usage);
    }
}
