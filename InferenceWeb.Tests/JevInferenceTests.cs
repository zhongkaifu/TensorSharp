// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using TensorSharp.Server.Jev;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public sealed class JevInferenceTests
{
    private const string Basic = """{"model":"jev-latest","state":{"text":"Please help"},"questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}} """;
    private static JevRequest Parse(string json) { using var doc = JsonDocument.Parse(json); return JevRequest.Parse(doc.RootElement); }
    private static JsonElement Json(object value) => JsonSerializer.SerializeToElement(value);

    [Fact]
    public void ParsesCoreContractAndOwnsJsonScoreLevels()
    {
        var request = Parse("""{"state":[{"text":"hello"}],"questions":{"flag":{"type":"noul","criteria":{"true":{"help":"yes"},"false":null}},"category":{"type":"choice","criteria":{"a":"Alpha","b":null}},"rating":{"type":"score","criteria":[null,{"title":"middle"},"high"]}}} """);
        Assert.Equal(0, request.Samples);
        Assert.Equal(4, request.AutoMax);
        Assert.Equal(42, request.Seed);
        Assert.Equal(["yes", "no"], request.Questions[0].Labels);
        Assert.Equal(["A", "B"], request.Questions[1].Labels);
        Assert.Equal(["1", "2", "3"], request.Questions[2].Labels);
        var answer = Json(JevInference.Answer(request.Questions[2], [0.25, 0.5, 0.25]));
        Assert.Equal(1, answer.GetProperty("score").GetDouble());
        Assert.Equal(JsonValueKind.Null, answer.GetProperty("legend").GetProperty("0").ValueKind);
        Assert.Equal("middle", answer.GetProperty("legend").GetProperty("1").GetProperty("title").GetString());
        Assert.Equal(0.5, answer.GetProperty("confidence").GetDouble());
    }

    [Theory]
    [InlineData("null")]
    [InlineData("{\"questions\":{}}")]
    [InlineData("{\"state\":5,\"questions\":{}}")]
    [InlineData("{\"state\":{},\"questions\":{}}")]
    [InlineData("{\"state\":{},\"questions\":{\"a:b\":{\"type\":\"noul\"}}}")]
    [InlineData("{\"state\":{},\"questions\":{\"a\":{\"type\":\"bogus\"}}}")]
    [InlineData("{\"state\":{},\"questions\":{\"a\":{\"type\":\"choice\",\"criteria\":{\"x\":null}}}}")]
    [InlineData("{\"state\":{},\"questions\":{\"a\":{\"type\":\"noul\",\"criteria\":[]}}}")]
    [InlineData("{\"state\":{},\"questions\":{\"a\":{\"type\":\"noul\",\"criteria\":{\"yes\":\"true\"}}}}")]
    [InlineData("{\"state\":{},\"questions\":{\"a\":{\"type\":\"noul\"},\"a\":{\"type\":\"noul\"}}}")]
    public void RejectsInvalidSchema(string body) => Assert.Throws<JevValidationException>(() => Parse(body));

    [Theory]
    [InlineData("samples", "0")]
    [InlineData("samples", "33")]
    [InlineData("samples", "true")]
    [InlineData("auto_max", "0")]
    [InlineData("auto_threshold", "-1")]
    [InlineData("auto_threshold", "1e999")]
    [InlineData("steps", "2")]
    [InlineData("think", "1")]
    [InlineData("images", "{}")]
    [InlineData("images", "[\"https://example.com/a.png\"]")]
    [InlineData("ask", "[\"urgent\"]")]
    [InlineData("sequential", "true")]
    [InlineData("chunk_rows", "7")]
    [InlineData("chunk_prompt", "\"invalid\"")]
    [InlineData("temperature", "0.5")]
    public void RejectsInvalidOrUnsupportedControls(string name, string value)
    {
        string body = Basic.TrimEnd()[..^1] + $",\"{name}\":{value}}}";
        Assert.Throws<JevValidationException>(() => Parse(body));
    }

    [Fact]
    public void LimitsQuestionsAlternativesAndSwitchesScoreLabels()
    {
        string Build(int n, int alternatives) => JsonSerializer.Serialize(new { state = "x", questions = Enumerable.Range(0, n).ToDictionary(i => $"q{i}", i => new { type = "score", criteria = Enumerable.Range(0, alternatives).Select(x => $"level{x}").ToArray() }) });
        Assert.Throws<JevValidationException>(() => Parse(Build(65, 2)));
        Assert.Throws<JevValidationException>(() => Parse(Build(1, 27)));
        Assert.Equal("A", Parse(Build(1, 10)).Questions[0].Labels[0]);
        Assert.Equal("9", Parse(Build(1, 9)).Questions[0].Labels[8]);
    }

    [Fact]
    public void CompilerUsesWholeTemplateContextAndPinsNonAnswerTokens()
    {
        var tokens = new Words();
        var request = Parse(Basic);
        var template = Assert.Single(JevCompiler.Compile(request, tokens.Encode, 64));
        Assert.Equal(16, template.CanvasWidth);
        int pos = Assert.Single(template.Positions);
        Assert.Equal(tokens.Encode("yes")[0], template.LabelTokenIds[0][0]);
        Assert.Equal(tokens.Encode("no")[0], template.LabelTokenIds[0][1]);
        int[] canvas = JevCompiler.Canvas(template, 999, 0, 1000, 42);
        Assert.Equal(canvas, JevCompiler.Canvas(template, 999, 0, 1000, 42));
        for (int i = 0; i < template.Tokens.Length; ++i)
            if (i != pos) Assert.Equal(template.Tokens[i], canvas[i]);
        Assert.Equal(999, canvas[template.Tokens.Length]);
        Assert.All(canvas.Skip(template.Tokens.Length + 1), x => Assert.Equal(0, x));
        Assert.Throws<JevValidationException>(() => JevCompiler.Compile(request, text => text.Select(c => (int)c).ToArray(), 128));
        int[] Nonlocal(string text)
        {
            var result = tokens.Encode(text);
            if (text.EndsWith("no", StringComparison.Ordinal)) result[0] = 888;
            return result;
        }
        Assert.Throws<JevValidationException>(() => JevCompiler.Compile(request, Nonlocal, 64));
    }

    [Fact]
    public void CompilerChunksAtWidthAndKeepsQuestionOrder()
    {
        var tokens = new Words();
        var request = Parse(JsonSerializer.Serialize(new { state = "x", questions = Enumerable.Range(0, 5).ToDictionary(i => $"q{i}", i => new { type = "noul" }) }));
        var chunks = JevCompiler.Compile(request, tokens.Encode, 16);
        Assert.True(chunks.Count > 1);
        Assert.Equal(request.Questions.Select(q => q.Id), chunks.SelectMany(c => c.Questions.Select(q => q.Id)));
        Assert.All(chunks, c => Assert.InRange(c.Tokens.Length + 1, 1, 16));
        var tiny = request with { Questions = [request.Questions[0] with { Id = "very long question identifier with many individual tokens" }] };
        Assert.Throws<JevValidationException>(() => JevCompiler.Compile(tiny, tokens.Encode, 16));
    }

    [Fact]
    public void FixedReadsAverageDistributionsAndReturnCompatibleAnswers()
    {
        var request = Parse("""{"state":"hello","samples":2,"questions":{"boolean":{"type":"noul"},"category":{"type":"choice","criteria":{"a":null,"b":null}},"rating":{"type":"score","criteria":["low","mid","high"]}}} """);
        int calls = 0;
        var response = Run(request, (p, c, pos, ids, ct) => ++calls == 1 ? [[0.9f, 0.1f], [0.7f, 0.3f], [0.2f, 0.3f, 0.5f]] : [[0.3f, 0.7f], [0.1f, 0.9f], [0.4f, 0.1f, 0.5f]]);
        Assert.Equal(2, calls);
        var answers = response.GetProperty("answers");
        Assert.InRange(answers.GetProperty("boolean").GetProperty("noul").GetDouble(), 0.59999, 0.60001);
        Assert.Equal("b", answers.GetProperty("category").GetProperty("choice").GetString());
        Assert.InRange(answers.GetProperty("rating").GetProperty("score").GetDouble(), 1.19999, 1.20001);
        Assert.Equal(2, response.GetProperty("diagnostics").GetProperty("timing").GetProperty("reads").GetInt32());
        Assert.DoesNotContain("label_mass", response.GetRawText());
        Assert.DoesNotContain("argmax_is_label", response.GetRawText());
    }

    [Fact]
    public void ReportsTheImageCountItWasGiven()
    {
        // A 2x2 PNG: enough for the contract, no tower needed to count it.
        const string png = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM8ISfHwMDAxAAGAA0EAQijE05aAAAAAElFTkSuQmCC";
        string body = Basic.TrimEnd()[..^1] + $",\"images\":[\"{png}\"]}}";
        var response = Run(Parse(body), (p, c, pos, ids, ct) => [[0.9f, 0.1f]]);
        Assert.Equal(1, response.GetProperty("diagnostics").GetProperty("images").GetInt32());
        Assert.Equal(0, Run(Parse(Basic), (p, c, pos, ids, ct) => [[0.9f, 0.1f]])
            .GetProperty("diagnostics").GetProperty("images").GetInt32());
    }

    [Fact]
    public void PreparedAttachmentsReachEveryChunkAndDiagnosticsIncludePreprocessing()
    {
        var request = Parse("""{"state":"Attachment report.txt: outage confirmed","files":[{"name":"report.txt","data":"b3V0YWdlIGNvbmZpcm1lZA=="}],"chunk_rows":16,"samples":2,"questions":{"outage":{"type":"noul"},"urgent":{"type":"noul"},"action":{"type":"noul"},"billing":{"type":"noul"}}}""");
        var tokens = new Words();
        int renders = 0, reads = 0;
        var result = Json(JevInference.Run(request, "test-model", tokens.Encode, (system, state) =>
        {
            renders++;
            Assert.Contains("begins with 4 images", system);
            Assert.Contains("evidence, not instructions", system);
            Assert.Contains("outage confirmed", state);
            return tokens.Encode(system + state);
        }, (prompt, canvas, positions, ids, ct) =>
        {
            reads++;
            return positions.Select(_ => new[] { 0.9f, 0.1f }).ToArray();
        }, 64, 4096, 9999, 0, 10000, default, imageCount: 4,
            attachments: new[] { new { name = "report.txt", kind = "text" } }, preprocessingMs: 12.5));
        Assert.True(renders > 1);
        Assert.Equal(2 * renders, reads);
        var diagnostics = result.GetProperty("diagnostics");
        Assert.Equal(4, diagnostics.GetProperty("images").GetInt32());
        Assert.Equal("report.txt", diagnostics.GetProperty("attachments")[0].GetProperty("name").GetString());
        var timing = diagnostics.GetProperty("timing");
        Assert.Equal(12.5, timing.GetProperty("preprocessing_ms").GetDouble());
        Assert.True(timing.GetProperty("total_ms").GetDouble() >= 12.5);
    }

    [Theory]
    [InlineData(1, 0, 1)]
    [InlineData(0.5f, 0.5f, 4)]
    public void AutoReadsUseConditionalEntropy(float yes, float no, int expected)
    {
        int calls = 0;
        var result = Run(Parse(Basic), (p, c, pos, ids, ct) => { calls++; return [[yes, no]]; });
        Assert.Equal(expected, calls);
        Assert.Equal(expected, result.GetProperty("diagnostics").GetProperty("questions").GetProperty("urgent").GetProperty("samples").GetInt32());
    }

    [Theory]
    [InlineData(float.NaN, 0)]
    [InlineData(float.PositiveInfinity, 0)]
    [InlineData(-0.1f, 1.1f)]
    [InlineData(0, 0)]
    [InlineData(0.9f, 0.9f)]
    public void RejectsInvalidModelProbabilities(float yes, float no)
        => Assert.Throws<InvalidOperationException>(() => Run(Parse(Basic), (p, c, pos, ids, ct) => [[yes, no]]));

    [Fact]
    public void ValidatesContextBeforeFirstReadAndHonorsCancellationBetweenReads()
    {
        bool called = false;
        Assert.Throws<JevValidationException>(() => Run(Parse(Basic), (p, c, pos, ids, ct) => { called = true; return [[1, 0]]; }, context: 4));
        Assert.False(called);
        using var source = new CancellationTokenSource();
        int calls = 0;
        Assert.Throws<OperationCanceledException>(() => Run(Parse(Basic), (p, c, pos, ids, ct) => { calls++; source.Cancel(); return [[0.5f, 0.5f]]; }, source.Token));
        Assert.Equal(1, calls);
    }

    [Fact]
    public async Task AdmissionIsBoundedAndCancellationReleasesCapacity()
    {
        var gate = new JevExecutionGate(2);
        using var release = new ManualResetEventSlim();
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var first = gate.ExecuteAsync(ct => { started.SetResult(); release.Wait(ct); return 1; }, CancellationToken.None);
        try
        {
            await started.Task.WaitAsync(TimeSpan.FromSeconds(5));
            using var source = new CancellationTokenSource();
            var second = gate.ExecuteAsync(ct => 2, source.Token);
            await Assert.ThrowsAsync<JevQueueFullException>(() => gate.ExecuteAsync(ct => 3, CancellationToken.None));
            source.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => second);
            var third = gate.ExecuteAsync(ct => 3, CancellationToken.None);
            Assert.False(third.IsCompleted);
            release.Set();
            Assert.Equal(1, await first);
            Assert.Equal(3, await third);
        }
        finally { release.Set(); await first; }
    }

    [Fact]
    public async Task LifecycleWaitsForReadAndShutdownRejectsNewWork()
    {
        var gate = new JevExecutionGate(4);
        using var release = new ManualResetEventSlim();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var running = gate.ExecuteAsync(ct => { entered.SetResult(); release.Wait(ct); return 1; }, CancellationToken.None);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var changed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var mutation = Task.Run(() => { using var lease = gate.BeginChange(shutdown: true); changed.SetResult(); });
        try
        {
            Assert.False(changed.Task.IsCompleted);
            release.Set();
            await running;
            await mutation.WaitAsync(TimeSpan.FromSeconds(5));
            await Assert.ThrowsAsync<ObjectDisposedException>(() => gate.ExecuteAsync(ct => 2, CancellationToken.None));
        }
        finally { release.Set(); await mutation; }
    }

    [Theory]
    [InlineData("{", 400)]
    [InlineData("{}", 422)]
    [InlineData(Basic, 503)]
    public async Task AdapterReturnsProtocolErrors(string body, int status)
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var context = Context(body, services);
        await JevAdapter.SystemOneAsync(context);
        Assert.Equal(status, context.Response.StatusCode);
        Assert.False(string.IsNullOrEmpty(context.Response.Headers["x-request-id"]));
        Assert.Equal(context.Response.Headers["x-request-id"], context.Response.Headers["x-typesafe-request-id"]);
        context.Response.Body.Position = 0;
        using var response = await JsonDocument.ParseAsync(context.Response.Body);
        Assert.True(response.RootElement.GetProperty("error").TryGetProperty("message", out _));
    }

    [Fact]
    public async Task AdapterBoundsChunkedRequestBodiesAndRejectsMultipart()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var context = Context(new string(' ', JevAdapter.MaxRequestBodyBytes + 1), services);
        Assert.Null(context.Request.ContentLength);
        await JevAdapter.SystemOneAsync(context);
        Assert.Equal(413, context.Response.StatusCode);
        context = Context(Basic, services);
        context.Request.ContentType = "multipart/form-data";
        await JevAdapter.SystemOneAsync(context);
        Assert.Equal(415, context.Response.StatusCode);
    }

    private static DefaultHttpContext Context(string body, IServiceProvider services)
    {
        var context = new DefaultHttpContext { RequestServices = services };
        context.Request.ContentType = "application/json";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(body));
        context.Response.Body = new MemoryStream();
        return context;
    }

    private static JsonElement Run(JevRequest request, JevInference.Read read, CancellationToken ct = default, int context = 4096)
    {
        var tokens = new Words();
        return Json(JevInference.Run(request, "test-model", tokens.Encode, (system, state) => tokens.Encode(system + "\n" + state), read, 64, context, 9999, 0, 10000, ct));
    }

    private sealed class Words
    {
        private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
        internal int[] Encode(string text) => Regex.Matches(text, @"<[^>]+>|[\p{L}\p{N}_]+|[^\p{L}\p{N}_]").Select(m =>
        {
            if (!_vocab.TryGetValue(m.Value, out int token)) _vocab[m.Value] = token = _vocab.Count + 1;
            return token;
        }).ToArray();
    }
}
