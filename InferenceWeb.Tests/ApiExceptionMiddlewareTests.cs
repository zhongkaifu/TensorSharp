using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using TensorSharp.Server.Logging;

namespace InferenceWeb.Tests;

/// <summary>
/// No request may answer with a framework error page. Before this middleware
/// existed, an exception escaping an endpoint hit the Developer Exception Page
/// that <c>WebApplication.CreateBuilder</c> installs ahead of all application
/// middleware, and a client sending <c>Accept: text/html</c> got the throwing
/// file's verbatim source back — which is how issue #142 was reported, with the
/// source of <c>ToolFunctionParser.cs</c> arriving in place of a completion. In
/// Production the same request got a bodiless 500 instead.
/// </summary>
public class ApiExceptionMiddlewareTests
{
    private static DefaultHttpContext ContextFor(string path)
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Path = path;
        ctx.Response.Body = new MemoryStream();
        return ctx;
    }

    private static async Task<string> BodyOf(HttpContext ctx)
    {
        ctx.Response.Body.Seek(0, SeekOrigin.Begin);
        using var reader = new StreamReader(ctx.Response.Body, Encoding.UTF8);
        return await reader.ReadToEndAsync();
    }

    /// <summary>
    /// Raise the throw the reporter hit — a legal tool spec whose <c>enum</c>
    /// holds integers, read by an accessor that only accepts strings — from
    /// inside the pipeline, exactly as a request parser does. It has to be
    /// thrown here rather than caught and re-thrown by value: <c>throw ex;</c>
    /// resets <c>TargetSite</c>, which is how the middleware tells a caller's
    /// unreadable body from a fault of the server's own.
    /// </summary>
    private static Task ThrowJsonKindMismatch()
    {
        using var doc = JsonDocument.Parse("""{"enum":[1,2,3]}""");
        foreach (var v in doc.RootElement.GetProperty("enum").EnumerateArray())
            _ = v.GetString();
        throw new Xunit.Sdk.XunitException("GetString() on a Number was expected to throw.");
    }

    private static Exception JsonKindMismatch()
    {
        try
        {
            ThrowJsonKindMismatch().GetAwaiter().GetResult();
            throw new Xunit.Sdk.XunitException("unreachable");
        }
        catch (InvalidOperationException ex)
        {
            return ex;
        }
    }

    [Fact]
    public async Task JsonKindMismatch_OnOpenAiRoute_Becomes400NotAnHtmlPage()
    {
        var ctx = ContextFor("/v1/chat/completions");

        await ApiExceptionMiddleware.InvokeAsync(ctx, ThrowJsonKindMismatch);

        Assert.Equal(400, ctx.Response.StatusCode);
        string body = await BodyOf(ctx);
        using var doc = JsonDocument.Parse(body);
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        // The message describes the caller's own JSON, so it is safe to echo and
        // is the only thing that tells them which value the server choked on.
        Assert.Contains("Number", error.GetProperty("message").GetString());
        Assert.DoesNotContain("<html", body, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task MalformedJson_Becomes400()
    {
        var ctx = ContextFor("/v1/chat/completions");
        Exception parseFailure;
        try
        {
            using var bad = JsonDocument.Parse("{ not json");
            throw new Xunit.Sdk.XunitException("unreachable");
        }
        catch (JsonException ex)
        {
            parseFailure = ex;
        }

        await ApiExceptionMiddleware.InvokeAsync(ctx, () => throw parseFailure);

        Assert.Equal(400, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("error").GetProperty("type").GetString());
    }

    [Fact]
    public async Task ServerFault_Becomes500AndLeaksNothingAboutTheException()
    {
        var ctx = ContextFor("/v1/chat/completions");
        var fault = new NullReferenceException("Object reference not set — /home/dev/TensorSharp/Secret.cs:line 42");

        await ApiExceptionMiddleware.InvokeAsync(ctx, () => throw fault);

        Assert.Equal(500, ctx.Response.StatusCode);
        string body = await BodyOf(ctx);
        Assert.Equal("internal_error",
            JsonDocument.Parse(body).RootElement.GetProperty("error").GetProperty("type").GetString());

        // A server fault's detail belongs in the log, which RequestLoggingMiddleware
        // has already written by the time the exception reaches here.
        Assert.DoesNotContain("Secret.cs", body);
        Assert.DoesNotContain("NullReferenceException", body);
        Assert.DoesNotContain("Object reference", body);
    }

    [Fact]
    public async Task OnOllamaOrWebUiRoute_UsesTheFlatErrorShape()
    {
        var ctx = ContextFor("/api/chat");

        await ApiExceptionMiddleware.InvokeAsync(ctx, ThrowJsonKindMismatch);

        Assert.Equal(400, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("error").ValueKind);
    }

    [Fact]
    public async Task CorrelationIdSurvivesTheClearedResponse()
    {
        // RequestLoggingMiddleware sets this on the way in and documents it as
        // present on every response; Response.Clear() would otherwise drop it
        // from exactly the responses a user needs to quote.
        var ctx = ContextFor("/v1/chat/completions");
        ctx.Response.Headers[RequestLoggingMiddleware.RequestIdHeader] = "abc123";

        await ApiExceptionMiddleware.InvokeAsync(ctx, ThrowJsonKindMismatch);

        Assert.Equal("abc123", ctx.Response.Headers[RequestLoggingMiddleware.RequestIdHeader]);
    }

    /// <summary>
    /// A context whose response has begun as <paramref name="contentType"/>, as a
    /// stream has once its first event is out.
    /// </summary>
    private static DefaultHttpContext StartedContextFor(string path, string contentType)
    {
        var ctx = ContextFor(path);
        ctx.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpResponseFeature>(new StartedResponseFeature());
        ctx.Response.ContentType = contentType;
        return ctx;
    }

    /// <summary>
    /// The exception a Qwen3.5 server threw from a decode step after a 16K prompt
    /// had filled the device: the OpenAI stream used to stop without a
    /// finish_reason or an error, and the bench's stream_completed check could only
    /// report a truncated reply.
    /// </summary>
    private static Task ThrowMidStream() =>
        throw new InvalidOperationException("Native GGML RmsNorm failed: cudaMalloc failed: out of memory");

    [Fact]
    public async Task OpenAiChatStreamThatFails_EndsWithAnErrorChunkAndDone()
    {
        var ctx = StartedContextFor("/v1/chat/completions", "text/event-stream");

        await ApiExceptionMiddleware.InvokeAsync(ctx, ThrowMidStream);

        string body = await BodyOf(ctx);
        Assert.EndsWith("data: [DONE]\n\n", body);
        string[] events = body.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(2, events.Length);
        Assert.StartsWith("data: ", events[0]);
        using var doc = JsonDocument.Parse(events[0]["data: ".Length..]);
        JsonElement error = doc.RootElement.GetProperty("error");
        Assert.Equal(ApiExceptionMiddleware.StreamFailureMessage, error.GetProperty("message").GetString());
        Assert.Equal("internal_error", error.GetProperty("type").GetString());
        // As with a 500 body, the exception's detail stays in the server log.
        Assert.DoesNotContain("cudaMalloc", body);
    }

    [Fact]
    public async Task ResponsesStreamThatFails_EndsWithAnErrorEvent()
    {
        var ctx = StartedContextFor("/v1/responses", "text/event-stream");

        await ApiExceptionMiddleware.InvokeAsync(ctx, ThrowMidStream);

        string body = await BodyOf(ctx);
        Assert.StartsWith("event: error\ndata: ", body);
        using var doc = JsonDocument.Parse(body["event: error\ndata: ".Length..].TrimEnd('\n'));
        Assert.Equal("error", doc.RootElement.GetProperty("type").GetString());
        Assert.Equal("server_error", doc.RootElement.GetProperty("code").GetString());
        Assert.Equal(ApiExceptionMiddleware.StreamFailureMessage, doc.RootElement.GetProperty("message").GetString());
    }

    [Fact]
    public async Task WebUiStreamThatFails_EndsWithADoneErrorFrame()
    {
        var ctx = StartedContextFor("/api/chat", "text/event-stream");

        await ApiExceptionMiddleware.InvokeAsync(ctx, ThrowMidStream);

        string body = await BodyOf(ctx);
        Assert.StartsWith("data: ", body);
        using var doc = JsonDocument.Parse(body["data: ".Length..].TrimEnd('\n'));
        Assert.True(doc.RootElement.GetProperty("done").GetBoolean());
        Assert.Equal(ApiExceptionMiddleware.StreamFailureMessage, doc.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public async Task OllamaStreamThatFails_EndsWithAnErrorLine()
    {
        var ctx = StartedContextFor("/api/generate", "application/x-ndjson");

        await ApiExceptionMiddleware.InvokeAsync(ctx, ThrowMidStream);

        string body = await BodyOf(ctx);
        Assert.EndsWith("\n", body);
        using var doc = JsonDocument.Parse(body.TrimEnd('\n'));
        Assert.Equal(ApiExceptionMiddleware.StreamFailureMessage, doc.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public async Task AfterANonStreamBodyStarted_TheExceptionPropagates()
    {
        // A JSON body cut off mid-write has no frame left to add: dropping the
        // connection is the only signal it has.
        var ctx = StartedContextFor("/v1/chat/completions", "application/json");

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            ApiExceptionMiddleware.InvokeAsync(ctx, () => throw new InvalidOperationException("mid-body")));
        Assert.Equal(string.Empty, await BodyOf(ctx));
    }

    [Fact]
    public async Task ClientAbortMidStream_WritesNothing()
    {
        var ctx = StartedContextFor("/v1/chat/completions", "text/event-stream");
        using var aborted = new CancellationTokenSource();
        await aborted.CancelAsync();
        ctx.RequestAborted = aborted.Token;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            ApiExceptionMiddleware.InvokeAsync(ctx, () => Task.FromCanceled(aborted.Token)));
        Assert.Equal(string.Empty, await BodyOf(ctx));
    }

    [Fact]
    public async Task ClientAbort_IsNotAnsweredAtAll()
    {
        // The caller hung up before the response started: there is no socket
        // left to write an error body to, so the cancellation propagates rather
        // than turning into a 500 nobody can read.
        var ctx = ContextFor("/v1/chat/completions");
        using var aborted = new CancellationTokenSource();
        await aborted.CancelAsync();
        ctx.RequestAborted = aborted.Token;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            ApiExceptionMiddleware.InvokeAsync(ctx, () => Task.FromCanceled(aborted.Token)));

        Assert.Equal(200, ctx.Response.StatusCode);
        Assert.Equal(string.Empty, await BodyOf(ctx));
    }

    [Fact]
    public async Task CancellationThatIsNotAClientAbort_StillBecomesAnError()
    {
        // An internal timeout is a server fault, not a hang-up: the connection
        // is alive and deserves an answer.
        var ctx = ContextFor("/v1/chat/completions");
        using var unrelated = new CancellationTokenSource();
        await unrelated.CancelAsync();

        await ApiExceptionMiddleware.InvokeAsync(ctx, () => Task.FromCanceled(unrelated.Token));

        Assert.Equal(500, ctx.Response.StatusCode);
    }

    [Fact]
    public async Task NoException_PassesThroughUntouched()
    {
        var ctx = ContextFor("/v1/chat/completions");
        ctx.Response.StatusCode = 200;

        await ApiExceptionMiddleware.InvokeAsync(ctx, () => Task.CompletedTask);

        Assert.Equal(200, ctx.Response.StatusCode);
        Assert.Equal(string.Empty, await BodyOf(ctx));
    }

    [Fact]
    public void OnlyJsonReadFailuresCountAsAMalformedRequest()
    {
        Assert.True(ApiExceptionMiddleware.IsMalformedRequest(JsonKindMismatch()));
        Assert.False(ApiExceptionMiddleware.IsMalformedRequest(new NullReferenceException()));
        // An InvalidOperationException raised by our own code is a server fault,
        // not a client one — only System.Text.Json's are attributed to the body.
        Assert.False(ApiExceptionMiddleware.IsMalformedRequest(new InvalidOperationException("ours")));
    }

    private sealed class StartedResponseFeature : Microsoft.AspNetCore.Http.Features.IHttpResponseFeature
    {
        public int StatusCode { get; set; } = 200;
        public string ReasonPhrase { get; set; }
        public IHeaderDictionary Headers { get; set; } = new HeaderDictionary();
        public Stream Body { get; set; } = new MemoryStream();
        public bool HasStarted => true;
        public void OnStarting(System.Func<object, Task> callback, object state) { }
        public void OnCompleted(System.Func<object, Task> callback, object state) { }
    }
}
