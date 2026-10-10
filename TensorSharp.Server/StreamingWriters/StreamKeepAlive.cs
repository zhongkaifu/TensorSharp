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
using System.Diagnostics;
using Microsoft.AspNetCore.Http;

namespace TensorSharp.Server.StreamingWriters;

/// <summary>
/// When a streamed response last had bytes written to it, so a stream loop can send a frame
/// its clients skip before a proxy gives up on the connection.
///
/// <para>
/// A stream can go quiet while the model is still writing: an output parser holds text it
/// cannot classify yet. Nemotron-H Reasoning-128K's thinking-off replies are held until a
/// <c>&lt;/think&gt;</c> within their first 2048 characters, a complete call or the end of
/// the reply says whether they were reasoning, which is about 30 s on the 8B on Metal and
/// well past nginx's default 60 s <c>proxy_read_timeout</c> on a slower model or backend.
/// Without a byte in between, the proxy closed the connection and the client got a 504 or a
/// truncated stream where it used to get the answer from the first token.
/// </para>
/// <para>
/// The writers stamp the response whenever they write (and when they set the stream's
/// headers); a stream loop asks on every update, so the check costs nothing while text
/// flows. Updates keep arriving while text is held: the adapters' own parsers see every
/// token, and the skills loop forwards an empty update for each piece it holds.
/// </para>
/// </summary>
internal static class StreamKeepAlive
{
    /// <summary>A quarter of the shortest common proxy idle timeout (60 s).</summary>
    public static readonly TimeSpan DefaultInterval = TimeSpan.FromSeconds(15);

    private static readonly object LastWriteKey = new();

    /// <summary>Record that <paramref name="response"/> was written to just now.</summary>
    public static void Stamp(HttpResponse response)
        => response.HttpContext.Items[LastWriteKey] = Stopwatch.GetTimestamp();

    /// <summary>
    /// Whether nothing has been written to <paramref name="response"/> for
    /// <paramref name="interval"/>. False for a response whose stream has not started: a
    /// buffered reply (json_schema) still chooses its status code at the end, and a
    /// keep-alive would commit it to 200.
    /// </summary>
    public static bool IsDue(HttpResponse response, TimeSpan interval)
        => response.HttpContext.Items.TryGetValue(LastWriteKey, out object? last)
           && last is long stamp
           && Stopwatch.GetElapsedTime(stamp) >= interval;
}
