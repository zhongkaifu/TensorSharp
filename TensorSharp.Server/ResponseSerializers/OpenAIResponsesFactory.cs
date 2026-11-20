// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Text.Json;

namespace TensorSharp.Server.ResponseSerializers;

/// <summary>
/// Builders for the OpenAI-compatible Responses API surface
/// (<c>POST /v1/responses</c>). Separate from <see cref="OpenAIResponseFactory"/>
/// because the wire shape is materially different: output is a typed item
/// list (<c>reasoning</c>/<c>message</c>/<c>function_call</c>) instead of <c>choices</c>, and
/// streaming emits named semantic events instead of per-token chunks.
/// </summary>
internal static class OpenAIResponsesFactory
{
    public static string NewResponseId() => $"resp_{Guid.NewGuid():N}";

    public static string NewMessageItemId() => $"msg_{Guid.NewGuid():N}";

    public static string NewFunctionCallItemId() => $"fc_{Guid.NewGuid():N}";

    public static string NewReasoningItemId() => $"rs_{Guid.NewGuid():N}";

    public static string NewCallId() => $"call_{Guid.NewGuid():N}"[..24];

    // ---- Output items -----------------------------------------------------

    /// <param name="status">Item-level status. Defaults to <c>completed</c>; pass
    /// <c>incomplete</c> when the message was cut short (e.g. the output-token
    /// budget ran out), so the item does not claim to be whole while the
    /// enclosing response says otherwise.</param>
    public static object OutputMessageItem(string itemId, string text, string status = "completed") => new
    {
        id = itemId,
        type = "message",
        status,
        role = "assistant",
        content = new[]
        {
            new { type = "output_text", text = text ?? "", annotations = Array.Empty<object>() },
        },
    };

    /// <summary>
    /// The model's reasoning as a <c>reasoning</c> output item. The whole reasoning is its one
    /// <c>summary_text</c> part: there is no separate summary, and <c>summary</c> is the field
    /// OpenAI's own API fills and its SDKs read (Microsoft.Extensions.AI's Responses client
    /// turns it into <c>TextReasoningContent</c>; it does not read <c>content</c>). The item
    /// goes before the message it led to, as OpenAI orders them. An empty
    /// <paramref name="text"/> is the item as <c>response.output_item.added</c> announces it,
    /// with no summary part yet.
    /// </summary>
    public static object ReasoningItem(string itemId, string text, string status = "completed") => new
    {
        id = itemId,
        type = "reasoning",
        status,
        summary = string.IsNullOrEmpty(text)
            ? Array.Empty<object>()
            : new object[] { new { type = "summary_text", text } },
    };

    public static object FunctionCallItem(string itemId, string callId, ToolCall toolCall) => new
    {
        id = itemId,
        type = "function_call",
        status = "completed",
        call_id = callId,
        name = toolCall.Name,
        arguments = ToolCallArgumentsJson.Serialize(toolCall.Arguments),
    };

    // ---- Non-streaming / final response object ----------------------------

    public static object Response(
        string responseId,
        string model,
        string status,
        string? instructions,
        int? maxOutputTokens,
        IReadOnlyList<object> output,
        bool store,
        SamplingConfig samplingConfig,
        int promptTokens,
        int evalTokens,
        int kvCacheReusedTokens,
        string? errorMessage = null,
        string? incompleteReason = null,
        int reasoningTokens = 0) => new
        {
            id = responseId,
            @object = "response",
            created_at = UnixNow(),
            status,
            error = errorMessage != null ? new { message = errorMessage, code = "server_error" } : null,
            // Populated only for status "incomplete" — this is where the Responses
            // API puts the fact that generation was cut off ("max_output_tokens"),
            // the equivalent of chat completions' finish_reason "length".
            incomplete_details = incompleteReason != null ? new { reason = incompleteReason } : null,
            instructions,
            max_output_tokens = maxOutputTokens,
            model,
            output,
            parallel_tool_calls = true,
            previous_response_id = (string?)null,
            store,
            temperature = samplingConfig?.Temperature,
            top_p = samplingConfig?.TopP,
            truncation = "disabled",
            // An "incomplete" response still generated (and billed) tokens — in fact
            // its usage is the whole point, since output_tokens is what proves the
            // budget was the thing that stopped it. Only a failed response has none.
            usage = status != "failed" ? BuildUsage(promptTokens, evalTokens, kvCacheReusedTokens, reasoningTokens) : null,
            metadata = new Dictionary<string, string>(),
        };

    // ---- Streaming events ---------------------------------------------------
    // Each payload carries its own "type" field (matching the real API) in
    // addition to the SSE "event:" line, since some client libraries key off
    // the JSON body rather than the frame.

    public static object Created(string responseId, string model) => new
    {
        type = "response.created",
        response = new
        {
            id = responseId,
            @object = "response",
            created_at = UnixNow(),
            status = "in_progress",
            model,
            output = Array.Empty<object>(),
        },
    };

    public static object OutputItemAdded(int outputIndex, object item) => new
    {
        type = "response.output_item.added",
        output_index = outputIndex,
        item,
    };

    public static object ContentPartAdded(string itemId, int outputIndex, int contentIndex) => new
    {
        type = "response.content_part.added",
        item_id = itemId,
        output_index = outputIndex,
        content_index = contentIndex,
        part = new { type = "output_text", text = "", annotations = Array.Empty<object>() },
    };

    public static object OutputTextDelta(string itemId, int outputIndex, int contentIndex, string delta) => new
    {
        type = "response.output_text.delta",
        item_id = itemId,
        output_index = outputIndex,
        content_index = contentIndex,
        delta,
    };

    public static object OutputTextDone(string itemId, int outputIndex, int contentIndex, string text) => new
    {
        type = "response.output_text.done",
        item_id = itemId,
        output_index = outputIndex,
        content_index = contentIndex,
        text,
    };

    public static object ContentPartDone(string itemId, int outputIndex, int contentIndex, string text) => new
    {
        type = "response.content_part.done",
        item_id = itemId,
        output_index = outputIndex,
        content_index = contentIndex,
        part = new { type = "output_text", text, annotations = Array.Empty<object>() },
    };

    // The reasoning item streams as its one summary part: part added, text deltas, text
    // done, part done (OpenAI's event sequence for a reasoning summary).

    public static object ReasoningSummaryPartAdded(string itemId, int outputIndex, int summaryIndex) => new
    {
        type = "response.reasoning_summary_part.added",
        item_id = itemId,
        output_index = outputIndex,
        summary_index = summaryIndex,
        part = new { type = "summary_text", text = "" },
    };

    public static object ReasoningSummaryTextDelta(string itemId, int outputIndex, int summaryIndex, string delta) => new
    {
        type = "response.reasoning_summary_text.delta",
        item_id = itemId,
        output_index = outputIndex,
        summary_index = summaryIndex,
        delta,
    };

    public static object ReasoningSummaryTextDone(string itemId, int outputIndex, int summaryIndex, string text) => new
    {
        type = "response.reasoning_summary_text.done",
        item_id = itemId,
        output_index = outputIndex,
        summary_index = summaryIndex,
        text,
    };

    public static object ReasoningSummaryPartDone(string itemId, int outputIndex, int summaryIndex, string text) => new
    {
        type = "response.reasoning_summary_part.done",
        item_id = itemId,
        output_index = outputIndex,
        summary_index = summaryIndex,
        part = new { type = "summary_text", text },
    };

    public static object OutputItemDone(int outputIndex, object item) => new
    {
        type = "response.output_item.done",
        output_index = outputIndex,
        item,
    };

    public static object Completed(object response) => new
    {
        type = "response.completed",
        response,
    };

    public static object Failed(string responseId, string model, string? errorMessage) => new
    {
        type = "response.failed",
        response = new
        {
            id = responseId,
            @object = "response",
            created_at = UnixNow(),
            status = "failed",
            model,
            output = Array.Empty<object>(),
            error = new { message = errorMessage, code = "server_error" },
        },
    };

    private static long UnixNow() => DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    // reasoning_tokens: the generated tokens that were reasoning (ChatStreamUpdate.ReasoningTokens), counted with the
    // output tokens as OpenAI counts them; the text itself is the response's reasoning item.
    private static object BuildUsage(int promptTokens, int evalTokens, int kvCacheReusedTokens, int reasoningTokens) => new
    {
        input_tokens = promptTokens,
        output_tokens = evalTokens,
        total_tokens = promptTokens + evalTokens,
        input_tokens_details = new { cached_tokens = kvCacheReusedTokens },
        output_tokens_details = new { reasoning_tokens = Math.Min(Math.Max(reasoningTokens, 0), evalTokens) },
    };
}
