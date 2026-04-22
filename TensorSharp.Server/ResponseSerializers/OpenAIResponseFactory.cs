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
using System.Linq;
using System.Text.Json;
using TensorSharp.Models;

namespace TensorSharp.Server.ResponseSerializers
{
    /// <summary>
    /// Builders for the OpenAI-compatible chat-completions response shapes.
    /// Splits the streaming and non-streaming forms into separate methods so
    /// the wire protocol details (chunk vs full completion, optional usage
    /// block, structured-output handling) live in one place.
    /// </summary>
    internal static class OpenAIResponseFactory
    {
        private const string ChunkObject = "chat.completion.chunk";
        private const string CompletionObject = "chat.completion";

        public static string NewRequestId()
        {
            return $"chatcmpl-{Guid.NewGuid():N}".Substring(0, 30);
        }

        public static object QueueChunk(string requestId, string model, int position, int pending) => new
        {
            id = requestId,
            @object = ChunkObject,
            model,
            created = UnixNow(),
            choices = new[] { new { index = 0, delta = new { role = "assistant", content = "" }, finish_reason = (string)null } },
            queue_position = position,
            queue_pending = pending,
        };

        public static object ContentChunk(string requestId, string model, string contentChunk) => new
        {
            id = requestId,
            @object = ChunkObject,
            created = UnixNow(),
            model,
            choices = new[]
            {
                new
                {
                    index = 0,
                    delta = new { role = (string)null, content = contentChunk },
                    finish_reason = (string)null,
                },
            },
        };

        public static object StructuredContentChunk(string requestId, string model, string normalizedContent) => new
        {
            id = requestId,
            @object = ChunkObject,
            created = UnixNow(),
            model,
            choices = new[]
            {
                new
                {
                    index = 0,
                    delta = new { role = "assistant", content = normalizedContent },
                    finish_reason = (string)null,
                },
            },
        };

        public static object EndChunk(
            string requestId,
            string model,
            string finishReason,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens) => new
        {
            id = requestId,
            @object = ChunkObject,
            created = UnixNow(),
            model,
            choices = new[]
            {
                new
                {
                    index = 0,
                    delta = new { role = (string)null, content = (string)null },
                    finish_reason = finishReason,
                },
            },
            usage = BuildUsage(promptTokens, evalTokens, kvCacheReusedTokens),
        };

        public static object ErrorContentChunk(string requestId, string model, string errorMessage) => new
        {
            id = requestId,
            @object = ChunkObject,
            model,
            created = UnixNow(),
            choices = new[]
            {
                new
                {
                    index = 0,
                    delta = new { content = $"Error: {errorMessage}" },
                    finish_reason = "stop",
                },
            },
        };

        public static object Completion(
            string requestId,
            string model,
            object responseMessage,
            string finishReason,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens) => new
        {
            id = requestId,
            @object = CompletionObject,
            created = UnixNow(),
            model,
            choices = new object[]
            {
                new
                {
                    index = 0,
                    message = responseMessage,
                    finish_reason = finishReason,
                },
            },
            usage = BuildUsage(promptTokens, evalTokens, kvCacheReusedTokens),
        };

        public static object ParsedAssistantMessage(
            string content,
            string thinking,
            IReadOnlyList<ToolCall> toolCalls)
        {
            return new
            {
                role = "assistant",
                content = content ?? "",
                thinking,
                tool_calls = BuildToolCalls(toolCalls),
            };
        }

        public static object PlainAssistantMessage(string content) => new
        {
            role = "assistant",
            content,
        };

        public static object StructuredAssistantMessage(string normalizedContent) => new
        {
            role = "assistant",
            content = normalizedContent,
        };

        private static IReadOnlyList<object> BuildToolCalls(IReadOnlyList<ToolCall> toolCalls)
        {
            if (toolCalls == null || toolCalls.Count == 0)
                return null;

            return toolCalls.Select(tc => (object)new
            {
                id = $"call_{Guid.NewGuid():N}".Substring(0, 24),
                type = "function",
                function = new { name = tc.Name, arguments = JsonSerializer.Serialize(tc.Arguments) },
            }).ToArray();
        }

        private static long UnixNow() => DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        /// <summary>
        /// Build the OpenAI <c>usage</c> block including the
        /// <c>prompt_tokens_details.cached_tokens</c> extension that callers use to
        /// reason about KV cache hit rates (matches the official OpenAI shape so
        /// SDKs that already parse <c>cached_tokens</c> work unchanged).
        /// </summary>
        private static object BuildUsage(int promptTokens, int evalTokens, int kvCacheReusedTokens) => new
        {
            prompt_tokens = promptTokens,
            completion_tokens = evalTokens,
            total_tokens = promptTokens + evalTokens,
            prompt_tokens_details = new
            {
                cached_tokens = kvCacheReusedTokens,
            },
        };
    }
}
