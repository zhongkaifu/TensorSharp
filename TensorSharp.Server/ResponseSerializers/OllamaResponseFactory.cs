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
using TensorSharp.Models;

namespace TensorSharp.Server.ResponseSerializers
{
    /// <summary>
    /// Builders for the response shapes used by the Ollama-compatible
    /// endpoints. Returns anonymous-typed objects which the JSON serializer
    /// emits as Ollama clients expect (notably: stable property ordering and
    /// ISO-8601 <c>created_at</c> timestamps).
    /// </summary>
    internal static class OllamaResponseFactory
    {
        public static object QueueGenerateChunk(string model, int position, int pending) => new
        {
            model,
            created_at = TimestampNow(),
            response = "",
            done = false,
            queue_position = position,
            queue_pending = pending,
        };

        public static object GenerateTokenChunk(string model, string piece) => new
        {
            model,
            created_at = TimestampNow(),
            response = piece,
            done = false,
        };

        public static object GenerateFinalChunk(
            string model,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            response = "",
            done = true,
            done_reason = "stop",
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        public static object GenerateError(string model, string error) => new
        {
            model,
            created_at = TimestampNow(),
            response = "",
            done = true,
            done_reason = "error",
            error,
        };

        public static object GenerateNonStreamingResponse(
            string model,
            string content,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            response = content,
            done = true,
            done_reason = "stop",
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        public static object QueueChatChunk(string model, int position, int pending) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = "" },
            done = false,
            queue_position = position,
            queue_pending = pending,
        };

        public static object ChatRawTokenChunk(string model, string piece) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = piece },
            done = false,
        };

        public static object ChatParsedChunk(string model, string contentChunk, string thinkingChunk) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = contentChunk, thinking = thinkingChunk },
            done = false,
        };

        public static object ChatRawFinalChunk(
            string model,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = "" },
            done = true,
            done_reason = "stop",
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        public static object ChatParsedFinalChunk(
            string model,
            IReadOnlyList<ToolCall> collectedToolCalls,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs)
        {
            var toolCallsJson = ConvertToolCalls(collectedToolCalls);
            return new
            {
                model,
                created_at = TimestampNow(),
                message = new
                {
                    role = "assistant",
                    content = "",
                    tool_calls = toolCallsJson,
                },
                done = true,
                done_reason = collectedToolCalls != null ? "tool_calls" : "stop",
                total_duration = totalNs,
                prompt_eval_count = promptTokens,
                prompt_eval_duration = promptNs,
                eval_count = evalTokens,
                eval_duration = evalNs,
                prompt_cache_hit_tokens = kvCacheReusedTokens,
                prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
            };
        }

        public static object ChatErrorChunk(string model, string error) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = "" },
            done = true,
            done_reason = "error",
            error,
        };

        public static object ChatNonStreamingResponse(
            string model,
            object message,
            string doneReason,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            message,
            done = true,
            done_reason = doneReason,
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        public static object ChatNonStreamingMessage(string content, string thinking, IReadOnlyList<ToolCall> toolCalls) => new
        {
            role = "assistant",
            content = content ?? "",
            thinking,
            tool_calls = ConvertToolCalls(toolCalls),
        };

        public static object ChatPlainMessage(string content) => new
        {
            role = "assistant",
            content,
        };

        private static IReadOnlyList<object> ConvertToolCalls(IReadOnlyList<ToolCall> toolCalls)
        {
            if (toolCalls == null || toolCalls.Count == 0)
                return null;

            var result = new object[toolCalls.Count];
            for (int i = 0; i < toolCalls.Count; i++)
            {
                var tc = toolCalls[i];
                result[i] = new
                {
                    function = new { name = tc.Name, arguments = tc.Arguments },
                };
            }
            return result;
        }

        private static string TimestampNow() => DateTime.UtcNow.ToString("o");

        /// <summary>
        /// Fraction of the prompt that was served from the prior turn's KV cache.
        /// Returns 0.0 when the prompt is empty so consumers can render the value
        /// uniformly without special-casing the no-tokens path.
        /// </summary>
        private static double ComputeRatio(int reused, int total)
        {
            return total > 0 ? (double)reused / total : 0.0;
        }
    }
}
