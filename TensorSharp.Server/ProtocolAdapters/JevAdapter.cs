// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Jev;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>Jev's structured probability contract over native DiffusionGemma reads.</summary>
public static class JevAdapter
{
    /// <summary>
    /// Request-body ceiling. Inline attachments travel base64-encoded inside the JSON body, so the
    /// default is megabytes rather than the kilobytes a typed text decision needs;
    /// <c>TS_JEV_MAX_BODY_MB</c> (1 to 64) sets it for an operator who wants it tighter.
    /// </summary>
    public static int MaxRequestBodyBytes => _maxRequestBodyBytes ??= ResolveMaxRequestBodyBytes();

    private static int? _maxRequestBodyBytes;

    private static int ResolveMaxRequestBodyBytes()
    {
        string? raw = Environment.GetEnvironmentVariable("TS_JEV_MAX_BODY_MB");
        if (string.IsNullOrEmpty(raw)) return 8 * 1024 * 1024;
        if (!int.TryParse(raw, out int mb) || mb < 1 || mb > 64)
            throw new ArgumentException("TS_JEV_MAX_BODY_MB must be an integer from 1 to 64.");
        return mb * 1024 * 1024;
    }

    public static async Task SystemOneAsync(HttpContext context)
    {
        string requestId = Guid.NewGuid().ToString("N");
        context.Response.Headers["x-request-id"] = requestId;
        context.Response.Headers["x-typesafe-request-id"] = requestId;
        try
        {
            if (!context.Request.HasJsonContentType())
            {
                await Error(context, 415, "unsupported_media_type",
                    "Jev requests must use application/json; multipart uploads are not supported. " +
                    "Send inline attachments as base64, or upload to /api/upload and send its 'file' reference in files/documents/videos/audios.");
                return;
            }
            using var document = await ReadBody(context).ConfigureAwait(false);
            JevRequest request = JevRequest.Parse(document.RootElement);
            var service = context.RequestServices.GetService<ModelService>();
            if (service == null) throw new JevModelUnavailableException("Jev inference requires a loaded DiffusionGemma model.");
            var response = await service.JevAsync(request, context.RequestAborted).ConfigureAwait(false);
            await context.Response.WriteAsJsonAsync(response, context.RequestAborted).ConfigureAwait(false);
        }
        catch (JevBodyTooLargeException)
        {
            await Error(context, 413, "invalid_request_error", $"Jev request body exceeds {MaxRequestBodyBytes} bytes.");
        }
        catch (JsonException)
        {
            await Error(context, 400, "invalid_request_error", "Request body must be valid JSON.");
        }
        catch (JevValidationException error)
        {
            await Error(context, 422, "validation_error", error.Message);
        }
        catch (UploadLimitExceededException error)
        {
            // Attachment storage is governed by the operator's upload limits; answer with the
            // status those limits declare (413 over the per-file cap, 507 over the quota).
            await Error(context, error.StatusCode, "invalid_request_error", error.Message);
        }
        catch (JevQueueFullException error)
        {
            context.Response.Headers.RetryAfter = "1";
            await Error(context, 529, "overloaded_error", error.Message);
        }
        catch (JevModelNotFoundException error)
        {
            await Error(context, 404, "not_found_error", error.Message);
        }
        catch (JevModelUnavailableException error)
        {
            await Error(context, 503, "model_unavailable", error.Message);
        }
        catch (ObjectDisposedException)
        {
            await Error(context, 503, "model_unavailable", "The model service is shutting down.");
        }
    }

    private static async Task<JsonDocument> ReadBody(HttpContext context)
    {
        if (context.Request.ContentLength > MaxRequestBodyBytes) throw new JevBodyTooLargeException();
        using var buffer = new MemoryStream();
        var chunk = new byte[8192];
        int count;
        while ((count = await context.Request.Body.ReadAsync(chunk, context.RequestAborted).ConfigureAwait(false)) != 0)
        {
            if (buffer.Length + count > MaxRequestBodyBytes) throw new JevBodyTooLargeException();
            buffer.Write(chunk, 0, count);
        }
        return JsonDocument.Parse(buffer.GetBuffer().AsMemory(0, checked((int)buffer.Length)),
            new JsonDocumentOptions { MaxDepth = 64 });
    }

    private static Task Error(HttpContext context, int status, string type, string message)
    {
        context.Response.StatusCode = status;
        return context.Response.WriteAsJsonAsync(new { error = new { message, type } }, context.RequestAborted);
    }

    private sealed class JevBodyTooLargeException : Exception;
}
