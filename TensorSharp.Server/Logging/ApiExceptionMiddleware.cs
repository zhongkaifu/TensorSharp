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
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace TensorSharp.Server.Logging;

/// <summary>
/// The server's outermost error boundary: an exception that escapes an
/// endpoint becomes a protocol-shaped JSON error instead of whatever the
/// framework would have produced.
/// <para>
/// Without it the server has two bad answers for the same event.
/// <c>WebApplication.CreateBuilder</c> installs the Developer Exception Page
/// ahead of all application middleware, so under
/// <c>ASPNETCORE_ENVIRONMENT=Development</c> an unhandled exception answers a
/// request that sent <c>Accept: text/html</c> with a ~30 KB HTML page
/// containing the throwing file's <i>verbatim source</i>, the full stack, the
/// absolute paths of the server's source tree and an echo of the request
/// headers — which is how issue #142 was reported: a tool spec the parser
/// could not read came back as the source of <c>ToolFunctionParser.cs</c>.
/// In Production the page is not registered at all and the same request gets
/// a bodiless 500, which an OpenAI-compatible client cannot interpret either.
/// </para>
/// <para>
/// Registered as the first application middleware, so it sits <i>below</i>
/// the Developer Exception Page and handles the exception before that page
/// ever sees it, and <i>above</i>
/// <see cref="RequestLoggingMiddleware"/>, which has already logged the
/// failure with its full detail server-side by the time it arrives here.
/// Nothing about the exception reaches the client.
/// </para>
/// </summary>
public static class ApiExceptionMiddleware
{
    /// <summary>
    /// Assembly prefix identifying a body the server could not read. Matched
    /// as a prefix because the runtime attributes the throw to
    /// <c>System.Text.Json</c> or, once it has passed through the rethrow
    /// helper, to <c>System.Text.Json.Rethrowable</c>.
    /// </summary>
    private const string JsonAssemblyPrefix = "System.Text.Json";

    public static IApplicationBuilder UseApiExceptionHandling(this IApplicationBuilder app)
    {
        return app.Use(next => context => InvokeAsync(context, () => next(context)));
    }

    internal static async Task InvokeAsync(HttpContext context, Func<Task> next)
    {
        try
        {
            await next().ConfigureAwait(false);
        }
        catch (Exception ex) when (!IsClientAbort(context, ex))
        {
            if (!context.Response.HasStarted)
            {
                await WriteErrorAsync(context, ex).ConfigureAwait(false);
                return;
            }
            // A streaming completion that fails mid-body (a device allocation
            // failure halfway through a decode, say) has already sent its
            // status line. Ending the stream there, or dropping the
            // connection, reads to a client as a reply that simply stopped:
            // an OpenAI client sees neither a finish_reason nor an error. The
            // stream gets its protocol's terminal error frame instead; a
            // started body that is not a stream keeps propagating, and the
            // connection drops.
            if (!await TryEndStreamWithErrorAsync(context).ConfigureAwait(false))
                throw;
        }
    }

    /// <summary>
    /// The message a failed stream ends with. Like a 500 body it says nothing
    /// about the exception, whose detail <see cref="RequestLoggingMiddleware"/>
    /// has already logged.
    /// </summary>
    internal const string StreamFailureMessage =
        "The server failed while generating this response; the output before this error is incomplete.";

    /// <summary>
    /// End a started stream with its protocol's error frame: an
    /// <c>{error:{message,type}}</c> chunk and the <c>[DONE]</c> sentinel on
    /// the OpenAI chat routes (OpenAI's own mid-stream error, which its SDKs
    /// raise on), the Responses API's <c>error</c> event on
    /// <c>/v1/responses</c>, a final <c>{done,error}</c> frame on the Web UI's
    /// event streams and an <c>{error}</c> line on Ollama's NDJSON. False when
    /// the body is not a stream, or the frame cannot be written (the client
    /// is gone).
    /// </summary>
    private static async Task<bool> TryEndStreamWithErrorAsync(HttpContext context)
    {
        string contentType = context.Response.ContentType ?? "";
        string? frame = null;
        if (contentType.StartsWith("text/event-stream", StringComparison.OrdinalIgnoreCase))
        {
            if (context.Request.Path.StartsWithSegments("/v1/responses"))
                frame = "event: error\ndata: " + JsonSerializer.Serialize(new
                {
                    type = "error",
                    code = "server_error",
                    message = StreamFailureMessage,
                    param = (string?)null,
                }) + "\n\n";
            else if (context.Request.Path.StartsWithSegments("/v1"))
                frame = "data: " + JsonSerializer.Serialize(new
                {
                    error = new { message = StreamFailureMessage, type = "internal_error" },
                }) + "\n\ndata: [DONE]\n\n";
            else
                frame = "data: " + JsonSerializer.Serialize(new { done = true, error = StreamFailureMessage }) + "\n\n";
        }
        else if (contentType.StartsWith("application/x-ndjson", StringComparison.OrdinalIgnoreCase))
        {
            frame = JsonSerializer.Serialize(new { error = StreamFailureMessage }) + "\n";
        }
        if (frame == null)
            return false;
        try
        {
            await context.Response.WriteAsync(frame).ConfigureAwait(false);
            await context.Response.Body.FlushAsync().ConfigureAwait(false);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// The caller hung up — a browser navigating away, a harness cancelling a
    /// generation. There is no longer a socket to answer on, so writing an
    /// error body would only throw again; let it propagate and be logged as
    /// the abort it is.
    /// </summary>
    private static bool IsClientAbort(HttpContext context, Exception ex)
        => ex is OperationCanceledException && context.RequestAborted.IsCancellationRequested;

    /// <summary>
    /// Is this the caller's error rather than the server's? Two failure
    /// modes are attributable to the request body:
    /// <see cref="JsonException"/> for text that is not valid JSON, and an
    /// <see cref="InvalidOperationException"/> raised inside System.Text.Json
    /// for valid JSON whose value is of a kind the reader did not expect —
    /// <c>GetString()</c> on a number, <c>TryGetProperty</c> on a non-object.
    /// An <see cref="InvalidOperationException"/> from our own code means
    /// something else entirely and stays a 500, so the provenance test is
    /// what separates them.
    /// </summary>
    internal static bool IsMalformedRequest(Exception ex)
    {
        if (ex is JsonException)
            return true;
        if (ex is not InvalidOperationException)
            return false;

        // TargetSite is the more precise of the two and survives a rethrow;
        // Source covers the case where the stack was not captured.
        string? origin = ex.TargetSite?.DeclaringType?.Assembly.GetName().Name ?? ex.Source;
        return origin != null && origin.StartsWith(JsonAssemblyPrefix, StringComparison.Ordinal);
    }

    private static async Task WriteErrorAsync(HttpContext context, Exception ex)
    {
        bool malformed = IsMalformedRequest(ex);

        // Response.Clear() drops headers as well as the body, and the
        // correlation id RequestLoggingMiddleware promises on every response
        // is one of them — put it back so an error is still the response a
        // user can quote in a bug report.
        context.Response.Headers.TryGetValue(RequestLoggingMiddleware.RequestIdHeader, out var requestId);
        context.Response.Clear();
        if (requestId.Count > 0)
            context.Response.Headers[RequestLoggingMiddleware.RequestIdHeader] = requestId;

        context.Response.StatusCode = malformed
            ? StatusCodes.Status400BadRequest
            : StatusCodes.Status500InternalServerError;

        // A malformed-body message describes the caller's own JSON ("the
        // target element has type 'Number'"), so it is safe and useful to
        // return. Anything else is a server fault whose detail stays in the
        // log, where RequestLoggingMiddleware has already written it.
        string message = malformed
            ? $"Could not read the request body: {ex.Message}"
            : "The server failed to handle the request.";

        // OpenAI SDKs parse {error:{message,type}}; the Ollama and Web UI
        // clients expect a flat {error:"..."}. Shape by route prefix, the
        // same rule PromptOverflowMiddleware uses.
        if (context.Request.Path.StartsWithSegments("/v1"))
        {
            await context.Response.WriteAsJsonAsync(new
            {
                error = new
                {
                    message,
                    type = malformed ? "invalid_request_error" : "internal_error"
                }
            }).ConfigureAwait(false);
        }
        else
        {
            await context.Response.WriteAsJsonAsync(new { error = message }).ConfigureAwait(false);
        }
    }
}
