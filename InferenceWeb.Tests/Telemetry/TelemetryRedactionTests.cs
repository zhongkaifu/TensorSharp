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
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Logs;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// The content contract: what a call site marks with <see cref="LogContent"/> never leaves the
/// process through OpenTelemetry while content capture is off, and local logs print exactly
/// what they printed before.
/// </summary>
public class TelemetryRedactionTests
{
    /// <summary>What an exporter would see, copied in OnEnd (LogRecord instances are pooled).</summary>
    private sealed record Captured(
        string? FormattedMessage, object? Body, IReadOnlyList<KeyValuePair<string, object?>> Attributes, Exception? Exception);

    private sealed class CapturingProcessor : BaseProcessor<LogRecord>
    {
        public List<Captured> Records { get; } = new();

        public override void OnEnd(LogRecord data) =>
            Records.Add(new Captured(data.FormattedMessage, data.Body,
                data.Attributes?.ToList() ?? new List<KeyValuePair<string, object?>>(), data.Exception));
    }

    private static (ILoggerFactory Factory, CapturingProcessor Exported) CreatePipeline(bool redact = true)
    {
        var exported = new CapturingProcessor();
        var factory = LoggerFactory.Create(builder => builder.AddOpenTelemetry(o =>
        {
            TensorSharpTelemetrySetup.ConfigureLoggerOptions(o);
            o.AddProcessor(new LogContextEnrichmentProcessor());
            if (redact)
                o.AddProcessor(new ContentRedactionProcessor());
            o.AddProcessor(exported);
        }));
        return (factory, exported);
    }

    private static object? Attribute(Captured record, string key) =>
        record.Attributes.FirstOrDefault(a => a.Key == key).Value;

    [Fact]
    public void LogContentIsRedactedWhateverItsKeyAndTheRenderedMessageIsDropped()
    {
        var (factory, exported) = CreatePipeline();
        using (factory)
        {
            factory.CreateLogger("test").LogInformation(
                "chat.start tokens={Tokens} userInput=\"{LastUserContent}\" path={Path}",
                42, LogContent.Of("what is my password?"), LogContent.Of("/uploads/abc.png"));
        }

        var record = Assert.Single(exported.Records);
        Assert.Null(record.FormattedMessage);
        Assert.Equal("[redacted: 20 chars]", Attribute(record, "LastUserContent"));
        Assert.Equal("[redacted: 16 chars]", Attribute(record, "Path"));
        Assert.Equal(42, Attribute(record, "Tokens"));
        // With the rendered message gone, the body an exporter sends is the template.
        Assert.Equal("chat.start tokens={Tokens} userInput=\"{LastUserContent}\" path={Path}", record.Body);
        Assert.DoesNotContain(record.Attributes, a => a.Value is LogContent);
    }

    [Fact]
    public void ARecordWithoutContentKeepsItsRenderedMessage()
    {
        var (factory, exported) = CreatePipeline();
        using (factory)
            factory.CreateLogger("test").LogInformation("model loaded in {Seconds} s", 3);

        var record = Assert.Single(exported.Records);
        Assert.Equal("model loaded in 3 s", record.FormattedMessage);
    }

    [Fact]
    public void ANamedPlaceholderThatIsNotMarkedIsLeftAlone()
    {
        // A token count named Prompt is not content; only the LogContent wrapper decides.
        var (factory, exported) = CreatePipeline();
        using (factory)
            factory.CreateLogger("test").LogInformation("A {Prompt}-token prompt", 1200);

        var record = Assert.Single(exported.Records);
        Assert.Equal(1200, Attribute(record, "Prompt"));
        Assert.Equal("A 1200-token prompt", record.FormattedMessage);
    }

    [Fact]
    public void WithContentCaptureOnTheProcessorIsLeftOutAndContentIsExported()
    {
        var (factory, exported) = CreatePipeline(redact: false);
        using (factory)
            factory.CreateLogger("test").LogInformation("userInput=\"{LastUserContent}\"", LogContent.Of("hello"));

        var record = Assert.Single(exported.Records);
        Assert.Equal("userInput=\"hello\"", record.FormattedMessage);
        Assert.Equal("hello", Attribute(record, "LastUserContent")?.ToString());
    }

    [Fact]
    public void LocalLogsPrintExactlyWhatTheyPrintedBefore()
    {
        var plain = new List<string>();
        var wrapped = new List<string>();
        using (var factory = LoggerFactory.Create(b => b.AddProvider(new FormattingProvider(plain))))
            factory.CreateLogger("t").LogInformation("in=\"{Input}\" n={N}", "a \"quoted\" prompt", 3);
        using (var factory = LoggerFactory.Create(b => b.AddProvider(new FormattingProvider(wrapped))))
            factory.CreateLogger("t").LogInformation("in=\"{Input}\" n={N}", LogContent.Of("a \"quoted\" prompt"), 3);
        Assert.Equal(plain, wrapped);

        Assert.Null(LogContent.Of((string?)null));
        Assert.Equal("x", LogContent.Of("x")!.ToString());
        Assert.Equal("12", LogContent.Of((object)12)!.ToString());
    }

    [Fact]
    public void TheTensorSharpRequestIdIsAddedUnderItsOwnName()
    {
        var (factory, exported) = CreatePipeline();
        string? previousId = TelemetryContext.RequestId;
        string? previousProtocol = TelemetryContext.Protocol;
        try
        {
            TelemetryContext.RequestId = "abc123def456";
            TelemetryContext.Protocol = "openai";
            using (factory)
                factory.CreateLogger("test").LogInformation("hello");
        }
        finally
        {
            TelemetryContext.RequestId = previousId;
            TelemetryContext.Protocol = previousProtocol;
        }

        var record = Assert.Single(exported.Records);
        Assert.Equal("abc123def456", Attribute(record, TelemetryAttributes.RequestId));
        Assert.Equal("openai", Attribute(record, TelemetryAttributes.Protocol));
    }

    [Fact]
    public void AnExceptionThatQuotesAnUploadPathIsReplacedButKeepsItsTypeAndStack()
    {
        string uploads = Path.Combine(Path.GetTempPath(), "ts-otel-test-uploads-" + Guid.NewGuid().ToString("N"));
        TensorSharpTelemetry.RegisterSensitivePath(uploads);
        Exception thrown;
        try { throw new IOException("Failed to open video file: " + Path.Combine(uploads, "clip-1.mp4")); }
        catch (IOException ex) { thrown = ex; }

        var (factory, exported) = CreatePipeline();
        using (factory)
            factory.CreateLogger("test").LogError(thrown, "decode failed");

        var record = Assert.Single(exported.Records);
        var scrubbed = Assert.IsType<ScrubbedException>(record.Exception);
        Assert.Equal("System.IO.IOException", scrubbed.OriginalType);
        Assert.DoesNotContain(uploads, scrubbed.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("<uploads>", scrubbed.Message);
        Assert.Equal(thrown.StackTrace, scrubbed.StackTrace);
    }

    [Fact]
    public void AnExceptionWithNothingSensitiveIsExportedAsIs()
    {
        var thrown = new InvalidOperationException("CUDA out of memory");
        var (factory, exported) = CreatePipeline();
        using (factory)
            factory.CreateLogger("test").LogError(thrown, "load failed");

        Assert.Same(thrown, Assert.Single(exported.Records).Exception);
    }

    [Fact]
    public void UploadUrlsAndRegisteredDirectoriesAreScrubbedFromText()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ts-otel-scrub-" + Guid.NewGuid().ToString("N"));
        TensorSharpTelemetry.RegisterSensitivePath(dir);
        string text = $"saved {Path.Combine(dir, "a.png")} served at /uploads/9f2c.png, see '/uploads/x y'";
        string scrubbed = TensorSharpTelemetry.ScrubSensitivePaths(text);
        Assert.DoesNotContain(dir, scrubbed, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("9f2c", scrubbed);
        Assert.Contains("/uploads/{file}", scrubbed);

        const string clean = "nothing to see here";
        Assert.Same(clean, TensorSharpTelemetry.ScrubSensitivePaths(clean));
    }

    [Fact]
    public void ARegisteredDirectoryIsReplacedThroughTheEndOfThePath()
    {
        // Below the directory are the names that matter: an upload's unguessable name, an
        // artifact's run id, a workspace file's own name. None of it may survive.
        string dir = Path.Combine(Path.GetTempPath(), "ts-otel-whole-" + Guid.NewGuid().ToString("N"));
        TensorSharpTelemetry.RegisterSensitivePath(dir);

        Assert.Equal("Failed to open video file: <uploads>. Retry.",
            TensorSharpTelemetry.ScrubSensitivePaths($"Failed to open video file: {Path.Combine(dir, "clip-1.mp4")}. Retry."));
        // Quoted, as .NET's IO messages quote paths: to the closing quote, spaces and all.
        Assert.Equal("Could not find file '<uploads>'.",
            TensorSharpTelemetry.ScrubSensitivePaths($"Could not find file '{Path.Combine(dir, "run-7", "Q3 layoffs plan.xlsx")}'."));
        Assert.Equal("open \"<uploads>\" failed",
            TensorSharpTelemetry.ScrubSensitivePaths($"open \"{Path.Combine(dir, "my report.pdf")}\" failed"));
        // The other separator.
        Assert.Equal("open <uploads> failed",
            TensorSharpTelemetry.ScrubSensitivePaths($"open {dir.Replace('\\', '/')}/a.png failed"));

        // The default upload directory is <base>/uploads: its paths must not come out as
        // "<uploads>/<name>", which the /uploads/ URL pass would no longer recognise.
        string uploads = Path.Combine(Path.GetTempPath(), "ts-otel-base-" + Guid.NewGuid().ToString("N"), "uploads");
        TensorSharpTelemetry.RegisterSensitivePath(uploads);
        string scrubbed = TensorSharpTelemetry.ScrubSensitivePaths($"decode {Path.Combine(uploads, "clip.mp4")} failed");
        Assert.Equal("decode <uploads> failed", scrubbed);
    }

    [Fact]
    public void UploadStorageNamesAreReplacedWherever()
    {
        string id = Guid.NewGuid().ToString("N");
        // A bare storage name, as a media provider quotes Path.GetFileName(path).
        Assert.Equal("cannot decode {id}.mp4", TensorSharpTelemetry.ScrubSensitivePaths($"cannot decode {id}.mp4"));
        // A generated file named after one.
        Assert.Equal("saved video-{id}.mp4 (3 MB)", TensorSharpTelemetry.ScrubSensitivePaths($"saved video-{id}.mp4 (3 MB)"));
        // A 32-digit id with no extension (a trace id) is not a file name.
        string trace = $"trace {id} failed";
        Assert.Same(trace, TensorSharpTelemetry.ScrubSensitivePaths(trace));
        // Nor is a longer hex string (a hash).
        string hash = $"sha256 {id}{id}.done";
        Assert.Same(hash, TensorSharpTelemetry.ScrubSensitivePaths(hash));
    }

    [Fact]
    public void MarkedContentIsRemovedFromTheRecordsOtherAttributesAndException()
    {
        // The CLI's PDF failure line marks the path, but ex.Message quotes the same path and the
        // exception is attached; the path is under no registered directory.
        string pdf = Path.Combine(Path.GetTempPath(), "settlement-" + Guid.NewGuid().ToString("N")[..8], "divorce settlement.pdf");
        Exception thrown;
        try { throw new IOException($"The process cannot access the file '{pdf}' because it is being used by another process."); }
        catch (IOException ex) { thrown = ex; }

        var (factory, exported) = CreatePipeline();
        using (factory)
        {
            factory.CreateLogger("test").LogError(thrown,
                "Failed to extract text from PDF {PdfPath}: {Error}", LogContent.Of(pdf), thrown.Message);
        }

        var record = Assert.Single(exported.Records);
        Assert.Equal($"[redacted: {pdf.Length} chars]", Attribute(record, "PdfPath"));
        string error = Assert.IsType<string>(Attribute(record, "Error"));
        Assert.DoesNotContain(pdf, error, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("[redacted]", error);
        var scrubbed = Assert.IsType<ScrubbedException>(record.Exception);
        Assert.DoesNotContain(pdf, scrubbed.Message, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(pdf, scrubbed.ToString(), StringComparison.OrdinalIgnoreCase);
        // The template is the exported body; it is code, never content, and is left alone.
        Assert.Equal("Failed to extract text from PDF {PdfPath}: {Error}", record.Body);
    }

    [Fact]
    public void AShortOrTemplateLikeMarkedValueLeavesTheTemplateAndShortMatchesAlone()
    {
        var (factory, exported) = CreatePipeline();
        using (factory)
        {
            var logger = factory.CreateLogger("test");
            // A prompt that happens to be a word of the template.
            logger.LogInformation("chat.start input={Input} model={Model}", LogContent.Of("chat"), "chat-model-7b");
            // Two characters are not searched for elsewhere: they would match unrelated text.
            logger.LogInformation("reply={Reply} status={Status}", LogContent.Of("ok"), "ok, done");
        }

        Assert.Equal(2, exported.Records.Count);
        Assert.Equal("chat.start input={Input} model={Model}", exported.Records[0].Body);
        Assert.Equal("[redacted]-model-7b", Attribute(exported.Records[0], "Model"));
        Assert.Equal("ok, done", Attribute(exported.Records[1], "Status"));
    }

    [Fact]
    public void AScrubbedExceptionStartsWithTheOriginalTypeName()
    {
        string uploads = Path.Combine(Path.GetTempPath(), "ts-otel-type-" + Guid.NewGuid().ToString("N"));
        TensorSharpTelemetry.RegisterSensitivePath(uploads);
        Exception thrown;
        try { throw new FileNotFoundException($"Could not find file '{Path.Combine(uploads, "a.png")}'."); }
        catch (FileNotFoundException ex) { thrown = ex; }

        var (factory, exported) = CreatePipeline();
        using (factory)
            factory.CreateLogger("test").LogError(thrown, "read failed");

        var scrubbed = Assert.IsType<ScrubbedException>(Assert.Single(exported.Records).Exception);
        // Exporters report the type of the exception they are given; the original's leads the message.
        Assert.Equal("TensorSharp.Telemetry.ScrubbedException", scrubbed.GetType().FullName);
        Assert.Equal("System.IO.FileNotFoundException: Could not find file '<uploads>'.", scrubbed.Message);
        Assert.StartsWith("System.IO.FileNotFoundException: Could not find file '<uploads>'.", scrubbed.ToString());
        Assert.DoesNotContain("FileNotFoundException: System.IO.FileNotFoundException", scrubbed.ToString());
    }

    [Fact]
    public void LogScopesAreNeverExported()
    {
        // ASP.NET Core's hosting scope carries the raw request path (an /uploads file name, a
        // code-artifact run id) and worker threads inherit the scope of the request that made
        // them; a processor cannot drop one scope item, so scopes are not exported at all.
        var options = new OpenTelemetryLoggerOptions();
        TensorSharpTelemetrySetup.ConfigureLoggerOptions(options);
        Assert.False(options.IncludeScopes);
        Assert.True(options.IncludeFormattedMessage);
    }

    [Fact]
    public void ClearingTheContextForgetsEveryRequestFact()
    {
        string? id = TelemetryContext.RequestId, protocol = TelemetryContext.Protocol, origin = TelemetryContext.Origin;
        int? round = TelemetryContext.Round;
        try
        {
            TelemetryContext.RequestId = "abc";
            TelemetryContext.Protocol = "openai";
            TelemetryContext.Origin = "knowledge-rerank";
            TelemetryContext.Round = 2;
            TelemetryContext.Clear();
            Assert.Null(TelemetryContext.RequestId);
            Assert.Null(TelemetryContext.Protocol);
            Assert.Null(TelemetryContext.Origin);
            Assert.Null(TelemetryContext.Round);
        }
        finally
        {
            TelemetryContext.RequestId = id;
            TelemetryContext.Protocol = protocol;
            TelemetryContext.Origin = origin;
            TelemetryContext.Round = round;
        }
    }

    [Theory]
    [InlineData("/uploads/3f9c0e1.png", "/uploads/{file}")]
    [InlineData("/api/code/artifacts/1a2b3c4d5e6f7a8b", "/api/code/artifacts/{run}")]
    [InlineData("/api/code/artifacts/1a2b3c4d5e6f7a8b/", "/api/code/artifacts/{run}")]
    [InlineData("/api/code/artifacts/1a2b3c4d5e6f7a8b/report.pptx", "/api/code/artifacts/{run}/{path}")]
    [InlineData("/v1/responses/resp_9f2c", "/v1/responses/{id}")]
    [InlineData("/v1/responses/resp_9f2c/input_items", "/v1/responses/{id}/input_items")]
    [InlineData("/v1/responses", null)]
    [InlineData("/v1/chat/completions", null)]
    [InlineData("/api/code/artifacts", null)]
    [InlineData("/health", null)]
    public void CapabilityPathsAreMaskedAndNothingElseIs(string path, string? masked)
    {
        Assert.Equal(masked, TensorSharp.Server.Logging.CapabilityPaths.Mask(new Microsoft.AspNetCore.Http.PathString(path)));
    }

    [Fact]
    public void TheSpanBackstopRemovesContentAttributes()
    {
        using var activity = new Activity("chat test");
        activity.SetTag(TelemetryAttributes.GenAiInputMessages, "[{\"role\":\"user\",\"content\":\"secret\"}]");
        activity.SetTag(TelemetryAttributes.VideoPrompt, "a cat");
        activity.SetTag(TelemetryAttributes.ImagePrompt, "a dog");
        activity.SetTag(TelemetryAttributes.GenAiRequestModel, "m");
        new ContentSpanProcessor().OnEnd(activity);
        Assert.Null(activity.GetTagItem(TelemetryAttributes.GenAiInputMessages));
        Assert.Null(activity.GetTagItem(TelemetryAttributes.VideoPrompt));
        Assert.Null(activity.GetTagItem(TelemetryAttributes.ImagePrompt));
        Assert.Equal("m", activity.GetTagItem(TelemetryAttributes.GenAiRequestModel));
    }

    private sealed class FormattingProvider : ILoggerProvider
    {
        private readonly List<string> _lines;
        public FormattingProvider(List<string> lines) => _lines = lines;
        public ILogger CreateLogger(string categoryName) => new Logger(_lines);
        public void Dispose() { }

        private sealed class Logger : ILogger
        {
            private readonly List<string> _lines;
            public Logger(List<string> lines) => _lines = lines;
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
            public bool IsEnabled(LogLevel logLevel) => true;
            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            {
                _lines.Add(formatter(state, exception));
                if (state is IReadOnlyList<KeyValuePair<string, object?>> values)
                    foreach (var pair in values)
                        _lines.Add(pair.Key + "=" + pair.Value);
            }
        }
    }
}
