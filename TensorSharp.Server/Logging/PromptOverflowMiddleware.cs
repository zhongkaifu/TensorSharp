using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using TensorSharp.Memory;

namespace TensorSharp.Server.Logging;

/// <summary>
/// Maps a <see cref="PromptContextOverflowException"/> — a client input
/// error (the prompt does not fit the context window) — to a 400 with a
/// protocol-appropriate JSON body. The overflow surfaces during prompt
/// preparation, before any response body is written, so the status is still
/// settable. Once the response has started, the exception is left to
/// propagate.
/// </summary>
public static class PromptOverflowMiddleware
{
    public static IApplicationBuilder UsePromptOverflowHandling(this IApplicationBuilder app)
    {
        return app.Use(next => context => InvokeAsync(context, () => next(context)));
    }

    internal static async Task InvokeAsync(HttpContext context, System.Func<Task> next)
    {
        try
        {
            await next().ConfigureAwait(false);
        }
        catch (PromptContextOverflowException ex) when (!context.Response.HasStarted)
        {
            await WriteErrorAsync(context, ex.Message, StatusCodes.Status400BadRequest,
                "invalid_request_error", "context_length_exceeded").ConfigureAwait(false);
        }
        catch (MemoryPressureException ex) when (!context.Response.HasStarted)
        {
            await WriteErrorAsync(context, ex.Message, StatusCodes.Status503ServiceUnavailable,
                "resource_exhausted", "resource_exhausted").ConfigureAwait(false);
        }
    }

    private static async Task WriteErrorAsync(HttpContext context, string message, int status,
        string type, string code)
    {
        context.Response.Headers.TryGetValue(RequestLoggingMiddleware.RequestIdHeader, out var requestId);
        context.Response.Clear();
        if (requestId.Count > 0)
            context.Response.Headers[RequestLoggingMiddleware.RequestIdHeader] = requestId;
        context.Response.StatusCode = status;

        // OpenAI SDKs parse {error:{message,type}}; the Ollama and Web UI
        // clients expect a flat {error:"..."}. Shape by route prefix.
        if (context.Request.Path.StartsWithSegments("/v1"))
        {
            await context.Response.WriteAsJsonAsync(new
            {
                error = new { message, type, code }
            }).ConfigureAwait(false);
        }
        else
        {
            await context.Response.WriteAsJsonAsync(new { error = message }).ConfigureAwait(false);
        }
    }
}
