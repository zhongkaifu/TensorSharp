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
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Forward the prompt every conversation shares, once, before anyone asks.
    ///
    /// <para>
    /// On models that support shared checkpoints, the engine retains state at the
    /// shared prompt boundary so later conversations can start from a copy. A host can
    /// run this request before serving users, and an attached checkpoint store can
    /// persist the payload when the model supports export. Completing the request does
    /// not itself prove that a reusable payload was retained.
    /// </para>
    /// <para>
    /// There is no prefill-only entry point anywhere in this codebase, and adding one
    /// would be a second way to build a prompt that could drift from the real one. So the
    /// warm-up is an ordinary chat request that asks for a single token, using the same
    /// service, renderer and default skills plan as a real request.
    /// </para>
    /// </summary>
    public static class PrefixCacheWarmup
    {
        /// <summary>How the warm-up ended, for a caller that wants to say so.</summary>
        public readonly record struct Result(bool Warmed, string Detail, TimeSpan Elapsed)
        {
            public override string ToString() =>
                (Warmed ? "warmed in " : "not warmed after ")
                + Elapsed.TotalSeconds.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)
                + " s: " + Detail;
        }

        /// <summary>
        /// The one-token request whose only purpose is the K/V it leaves behind.
        /// </summary>
        /// <param name="sessionId">
        /// REQUIRED, and the single most important argument. A request without a session
        /// is served by the shared default session, which has no workspace — and a chat
        /// with no workspace is offered a different set of tools. Reuse is a
        /// longest-common-PREFIX match, so a tool block that differs by one token makes
        /// every token after it unshareable: the warm-up would run for twenty seconds and
        /// save a real turn only the part before the tools.
        /// </param>
        /// <param name="think">
        /// Must match what real requests send to maximize reuse. Some templates render
        /// different system prefixes for thinking on and off.
        /// </param>
        /// <remarks>
        /// Every field of the body is load-bearing, and three of them are traps:
        /// <list type="bullet">
        /// <item><c>skills</c> is OMITTED, never sent as <c>[]</c>: an empty array is not
        /// "the default", it suppresses both the operator's selection and the discovery
        /// catalog, producing a materially shorter block than a real request renders.</item>
        /// <item><c>tools</c> is OMITTED for the same class of reason: supplying one
        /// disables the host skill router and changes the block for reasons unrelated to
        /// tools.</item>
        /// <item><c>newChat</c> is FALSE. True would reset the session and release its
        /// workspace — deleting a conversation's files to warm a cache.</item>
        /// </list>
        /// The message is "hi" to keep warm-up work small. Only the declared shared
        /// system prefix is public; the warm-up's user message and reply stay scoped to
        /// its own session.
        /// </remarks>
        public static JsonElement BuildRequest(string sessionId, bool think, int maxTokens = 1)
        {
            if (string.IsNullOrWhiteSpace(sessionId))
                throw new ArgumentException("a warm-up needs its own session id", nameof(sessionId));

            var body = new Dictionary<string, object>
            {
                ["sessionId"] = sessionId,
                ["messages"] = new[] { new Dictionary<string, string> { ["role"] = "user", ["content"] = "hi" } },
                ["maxTokens"] = maxTokens,
                ["think"] = think,
                ["newChat"] = false,
            };
            return JsonSerializer.SerializeToElement(body);
        }

        /// <summary>
        /// Drive one warm-up to completion and report what happened.
        /// </summary>
        /// <param name="frames">
        /// The host's own chat stream — <c>WebUiChatService.ChatStreamAsync</c>. Taken as a
        /// delegate so this is testable without a model and so a host with a different
        /// chat surface can pass its own.
        /// </param>
        /// <remarks>
        /// <para>
        /// Failure does NOT arrive as an exception. The chat service catches what it can
        /// and ends the stream with a <c>done</c> frame carrying <c>error</c>, so code that
        /// only watches for throws marks a warm-up that forwarded nothing as warm. Both
        /// shapes are handled here, and the refusals that DO throw before any frame is
        /// yielded — no model loaded, an unknown session, a rejected backend — are caught
        /// as well.
        /// </para>
        /// <para>
        /// Nothing reads the answer. The tokens are worthless; the K/V they leave behind
        /// is the point.
        /// </para>
        /// </remarks>
        public static async Task<Result> RunAsync(
            Func<JsonElement, CancellationToken, IAsyncEnumerable<object>> frames,
            string sessionId,
            bool think,
            ILogger logger,
            CancellationToken cancellationToken = default)
        {
            if (frames == null) throw new ArgumentNullException(nameof(frames));

            var clock = Stopwatch.StartNew();
            try
            {
                JsonElement body = BuildRequest(sessionId, think);
                string failure = null;
                bool completed = false;
                await foreach (object frame in frames(body, cancellationToken).ConfigureAwait(false))
                {
                    failure ??= ErrorIn(frame);
                    if (frame?.GetType().GetProperty("done")?.GetValue(frame) is true)
                    {
                        completed = true;
                        if (frame.GetType().GetProperty("aborted")?.GetValue(frame) is true)
                            failure ??= "warm-up request was aborted";
                    }
                }

                clock.Stop();
                if (cancellationToken.IsCancellationRequested)
                    return new Result(false, "cancelled", clock.Elapsed);
                if (!completed)
                    failure ??= "warm-up stream ended without completing the request";
                if (!string.IsNullOrEmpty(failure))
                {
                    logger?.LogWarning(LogEventIds.HostConfiguration,
                        "Prefix cache warm-up failed: {Error}", failure);
                    return new Result(false, failure, clock.Elapsed);
                }
                return new Result(true, "shared-prompt warm-up request completed", clock.Elapsed);
            }
            catch (OperationCanceledException)
            {
                clock.Stop();
                return new Result(false, "cancelled", clock.Elapsed);
            }
            catch (Exception ex)
            {
                // Broad on purpose. This runs at startup for a latency benefit; there is
                // no failure here worth refusing to serve over, and the chat service
                // throws several refusal shapes before it yields a single frame.
                clock.Stop();
                logger?.LogWarning(LogEventIds.HostConfiguration,
                    "Prefix cache warm-up failed: {Error}", ex.Message);
                return new Result(false, ex.Message, clock.Elapsed);
            }
        }

        /// <summary>The error a <c>done</c> frame carries, or null for every other frame.</summary>
        private static string ErrorIn(object frame)
        {
            if (frame == null)
                return null;
            System.Reflection.PropertyInfo property = frame.GetType().GetProperty("error");
            object value = property?.GetValue(frame);
            string text = value as string;
            return string.IsNullOrWhiteSpace(text) ? null : text;
        }
    }
}
