// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace TensorSharp.Server.StreamingWriters
{
    /// <summary>
    /// Helpers for writing newline-delimited JSON streams. Used by the Ollama
    /// streaming endpoints (<c>/api/generate</c>, <c>/api/chat/ollama</c>).
    /// </summary>
    internal static class NdJsonWriter
    {
        public static void ApplyHeaders(HttpResponse response)
        {
            response.ContentType = "application/x-ndjson";
            response.Headers["Cache-Control"] = "no-cache";
        }

        /// <summary>
        /// Serialise <paramref name="payload"/> followed by a newline and flush
        /// the response so the next chunk can immediately reach the client.
        /// </summary>
        public static async Task WriteLineAsync(
            HttpResponse response,
            object payload,
            CancellationToken cancellationToken,
            JsonSerializerOptions jsonOptions = null)
        {
            string json = JsonSerializer.Serialize(payload, jsonOptions);
            await response.WriteAsync(json + "\n", cancellationToken);
            await response.Body.FlushAsync(cancellationToken);
        }
    }
}
