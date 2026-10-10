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
/// Helpers for writing newline-delimited JSON streams. Used by the Ollama
/// streaming endpoints (<c>/api/generate</c>, <c>/api/chat/ollama</c>).
/// </summary>
internal static class NdJsonWriter
{
    public static void ApplyHeaders(HttpResponse response)
    {
        response.ContentType = "application/x-ndjson";
        response.Headers.CacheControl = "no-cache";
        StreamKeepAlive.Stamp(response);
    }

    /// <summary>
    /// Serialise <paramref name="payload"/> followed by a newline and flush
    /// the response so the next chunk can immediately reach the client.
    /// </summary>
    public static async Task WriteLineAsync(
        HttpResponse response,
        object payload,
        CancellationToken cancellationToken,
        JsonSerializerOptions? jsonOptions = null)
    {
        string json = JsonSerializer.Serialize(payload, jsonOptions);
        await response.WriteAsync(json + "\n", cancellationToken).ConfigureAwait(false);
        await response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
        StreamKeepAlive.Stamp(response);
    }

    /// <summary>
    /// The chunk <paramref name="keepAlive"/> makes, as one line, when nothing has been
    /// written for <paramref name="interval"/> (see <see cref="StreamKeepAlive"/>). NDJSON
    /// has no comment, and Ollama clients parse every line, so the caller makes an ordinary
    /// chunk with nothing in it: an empty message that is not done.
    /// </summary>
    public static Task KeepAliveIfIdleAsync(
        HttpResponse response, TimeSpan interval, Func<object> keepAlive, CancellationToken cancellationToken,
        JsonSerializerOptions? jsonOptions = null)
        => StreamKeepAlive.IsDue(response, interval)
            ? WriteLineAsync(response, keepAlive(), cancellationToken, jsonOptions)
            : Task.CompletedTask;
}
