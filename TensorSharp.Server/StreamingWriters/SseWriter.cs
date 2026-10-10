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
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace TensorSharp.Server.StreamingWriters;

/// <summary>
/// Tiny helpers for writing Server-Sent Events. Centralised so we never
/// have <c>"data: " + json + "\n\n"</c> inlined across handlers and so
/// flushes are always paired with the writes that need to be visible to
/// the client immediately.
/// </summary>
internal static class SseWriter
{
    public static void ApplyHeaders(HttpResponse response)
    {
        response.ContentType = "text/event-stream";
        response.Headers.CacheControl = "no-cache";
        response.Headers.Connection = "keep-alive";
        StreamKeepAlive.Stamp(response);
    }

    /// <summary>
    /// Serialise <paramref name="payload"/> as a single <c>data:</c> SSE
    /// frame and flush it. Honours <paramref name="cancellationToken"/> so
    /// callers can abort quickly when the client disappears.
    /// </summary>
    public static async Task WriteEventAsync(
        HttpResponse response,
        object payload,
        CancellationToken cancellationToken,
        JsonSerializerOptions? jsonOptions = null)
    {
        string json = JsonSerializer.Serialize(payload, jsonOptions);
        await response.WriteAsync($"data: {json}\n\n", cancellationToken).ConfigureAwait(false);
        await response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
        StreamKeepAlive.Stamp(response);
    }

    /// <summary>
    /// Same as <see cref="WriteEventAsync(HttpResponse, object, CancellationToken, JsonSerializerOptions)"/>
    /// but without a cancellation token; used by best-effort "final flush"
    /// paths that must not throw on already-cancelled requests.
    /// </summary>
    public static async Task WriteEventAsync(
        HttpResponse response,
        object payload,
        JsonSerializerOptions? jsonOptions = null)
    {
        string json = JsonSerializer.Serialize(payload, jsonOptions);
        await response.WriteAsync($"data: {json}\n\n").ConfigureAwait(false);
        await response.Body.FlushAsync().ConfigureAwait(false);
        StreamKeepAlive.Stamp(response);
    }

    /// <summary>
    /// Write the literal <c>data: [DONE]</c> sentinel used by the OpenAI
    /// streaming chat-completions API.
    /// </summary>
    public static Task WriteDoneSentinelAsync(HttpResponse response, CancellationToken cancellationToken)
    {
        return response.WriteAsync("data: [DONE]\n\n", cancellationToken);
    }

    /// <summary>
    /// Serialise <paramref name="payload"/> as a named SSE event (an
    /// <c>event: &lt;name&gt;</c> line followed by <c>data:</c>), matching the
    /// Responses API's typed event stream. Unlike chat-completions chunks,
    /// each event here carries an explicit type both in the SSE frame and
    /// in the JSON body's own <c>type</c> field.
    /// </summary>
    public static async Task WriteNamedEventAsync(
        HttpResponse response,
        string eventName,
        object payload,
        CancellationToken cancellationToken,
        JsonSerializerOptions? jsonOptions = null)
    {
        string json = JsonSerializer.Serialize(payload, jsonOptions);
        await response.WriteAsync($"event: {eventName}\ndata: {json}\n\n", cancellationToken).ConfigureAwait(false);
        await response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
        StreamKeepAlive.Stamp(response);
    }

    /// <summary>
    /// An SSE comment line (<c>: keep-alive</c>) when nothing has been written for
    /// <paramref name="interval"/> (see <see cref="StreamKeepAlive"/>). The SSE format says
    /// a line that starts with a colon is ignored, so every client skips it, and it carries
    /// no event an OpenAI or Responses client could mistake for output.
    /// </summary>
    public static async Task KeepAliveIfIdleAsync(
        HttpResponse response, TimeSpan interval, CancellationToken cancellationToken)
    {
        if (!StreamKeepAlive.IsDue(response, interval))
            return;
        await response.WriteAsync(": keep-alive\n\n", cancellationToken).ConfigureAwait(false);
        await response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
        StreamKeepAlive.Stamp(response);
    }
}
