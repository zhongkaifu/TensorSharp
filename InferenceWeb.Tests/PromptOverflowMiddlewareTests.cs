using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using TensorSharp.Server;
using TensorSharp.Memory;
using TensorSharp.Server.Logging;

namespace InferenceWeb.Tests;

/// <summary>
/// A prompt that does not fit the context window must surface as an HTTP 400
/// with a protocol-appropriate JSON body. The overflow is thrown during prompt
/// preparation, before any response body is written, so the middleware can
/// still set the status.
/// </summary>
public class PromptOverflowMiddlewareTests
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

    [Fact]
    public async Task Overflow_OnOpenAiRoute_Becomes400WithOpenAiErrorShape()
    {
        var ctx = ContextFor("/v1/chat/completions");

        await PromptOverflowMiddleware.InvokeAsync(ctx,
            () => throw new PromptContextOverflowException("Prompt (9000 tokens) exceeds the model's context limit (4096 tokens)."));

        Assert.Equal(400, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Contains("context limit", error.GetProperty("message").GetString());
    }

    [Fact]
    public async Task Overflow_OnOllamaRoute_Becomes400WithFlatErrorShape()
    {
        var ctx = ContextFor("/api/generate");

        await PromptOverflowMiddleware.InvokeAsync(ctx,
            () => throw new PromptContextOverflowException("Prompt exceeds the model's context limit."));

        Assert.Equal(400, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("error").ValueKind);
    }

    [Fact]
    public async Task MemoryRefusal_OnOpenAiRoute_Returns503AndPreservesRequestId()
    {
        var ctx = ContextFor("/v1/chat/completions");
        ctx.Response.Headers[RequestLoggingMiddleware.RequestIdHeader] = "memory-test";
        ctx.Response.Headers["X-Stale"] = "remove";
        ctx.Response.Body.Write(Encoding.UTF8.GetBytes("unfinished result"));
        // Test the real middleware order: the outer generic error boundary
        // must not turn an expected refusal into an opaque HTTP 500.
        await ApiExceptionMiddleware.InvokeAsync(ctx, () => PromptOverflowMiddleware.InvokeAsync(ctx,
            () => Task.FromException(new MemoryPressureException("Request exceeds available shared host/device capacity."))));
        Assert.Equal(503, ctx.Response.StatusCode);
        Assert.Equal("memory-test", ctx.Response.Headers[RequestLoggingMiddleware.RequestIdHeader]);
        Assert.False(ctx.Response.Headers.ContainsKey("X-Stale"));
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("resource_exhausted", error.GetProperty("type").GetString());
        Assert.Equal("resource_exhausted", error.GetProperty("code").GetString());
        Assert.Contains("shared", error.GetProperty("message").GetString());
    }

    [Theory]
    [InlineData("/api/generate")]
    [InlineData("/api/chat")]
    [InlineData("/v10/not-openai")]
    public async Task MemoryRefusal_OnOtherRoutes_Returns503WithFlatError(string path)
    {
        var ctx = ContextFor(path);
        await PromptOverflowMiddleware.InvokeAsync(ctx,
            () => throw new MemoryPressureException("Shared memory budget cannot admit this request."));
        Assert.Equal(503, ctx.Response.StatusCode);
        using var doc = JsonDocument.Parse(await BodyOf(ctx));
        Assert.Equal("Shared memory budget cannot admit this request.", doc.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public async Task MemoryRefusal_AfterResponseStarted_PreservesOriginalExceptionAndBody()
    {
        var ctx = ContextFor("/v1/chat/completions");
        var feature = new StartedResponseFeature();
        ctx.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpResponseFeature>(feature);
        ctx.Response.Body.Write(Encoding.UTF8.GetBytes("data: partial\n\n"));
        var refusal = new MemoryPressureException("Later allocation refused.");
        var thrown = await Assert.ThrowsAsync<MemoryPressureException>(() =>
            ApiExceptionMiddleware.InvokeAsync(ctx, () => PromptOverflowMiddleware.InvokeAsync(ctx, () => throw refusal)));
        Assert.Same(refusal, thrown);
        Assert.Equal(200, ctx.Response.StatusCode);
        Assert.Equal("data: partial\n\n", await BodyOf(ctx));
    }

    [Fact]
    public async Task OrdinaryOutOfMemory_IsNotMisclassifiedAsAdmissionRefusal()
    {
        var ctx = ContextFor("/v1/chat/completions");
        await Assert.ThrowsAsync<OutOfMemoryException>(() =>
            PromptOverflowMiddleware.InvokeAsync(ctx, () => throw new OutOfMemoryException("unexpected allocator failure")));
    }

    [Fact]
    public async Task NonOverflowException_IsNotSwallowed()
    {
        var ctx = ContextFor("/v1/chat/completions");

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            PromptOverflowMiddleware.InvokeAsync(ctx,
                () => throw new InvalidOperationException("something else")));
    }

    [Fact]
    public async Task Overflow_AfterResponseStarted_IsNotConvertedTo400()
    {
        // A stream already in flight: the body is committed, so the status
        // can't be rewritten and the exception must propagate.
        var ctx = ContextFor("/v1/chat/completions");
        ctx.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpResponseFeature>(new StartedResponseFeature());

        await Assert.ThrowsAsync<PromptContextOverflowException>(() =>
            PromptOverflowMiddleware.InvokeAsync(ctx,
                () => throw new PromptContextOverflowException("too long")));
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

    [Fact]
    public async Task NoException_PassesThroughUntouched()
    {
        var ctx = ContextFor("/v1/chat/completions");
        ctx.Response.StatusCode = 200;

        await PromptOverflowMiddleware.InvokeAsync(ctx, () => Task.CompletedTask);

        Assert.Equal(200, ctx.Response.StatusCode);
    }
}
