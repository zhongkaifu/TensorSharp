// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Regression tests for streaming reasoning ("analysis"/thinking channel) as
// incremental OpenAI `reasoning_content` deltas. Reasoning-first models such as
// gpt-oss emit their whole analysis block before the first visible content
// token. The streaming adapter used to PARSE that analysis (into
// ParsedOutput.Thinking) and then DROP it (emitting a null-content delta), so a
// streaming client saw no token until the `final` channel began — which inflated
// measured time-to-first-token from real prefill (~0.3s) to the full reasoning
// decode (~3s). The fix emits ParsedOutput.Thinking as a `reasoning_content`
// delta. These tests lock the wire format (OpenAIResponseFactory) and the
// adapter's emission rule (reasoning streamed, and emitted before content).

using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

public class ReasoningContentStreamingTests
{
    // The exact per-piece emission decision the streaming adapter makes for a
    // parsed chunk. Mirrors OpenAIChatAdapter.StreamCompletionAsync: emit a
    // reasoning_content delta for any non-empty Thinking, then a content delta
    // for any non-empty Content. Returned as serialized SSE delta payloads so a
    // test can assert ordering and field shape without standing up a model/HTTP.
    private static List<string> EmitDeltasForStream(string[] chunks)
    {
        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: true, tools: new List<ToolFunction>());

        var deltas = new List<string>();
        for (int i = 0; i < chunks.Length; i++)
        {
            bool done = i == chunks.Length - 1;
            var parsed = parser.Add(chunks[i], done);

            if (!string.IsNullOrEmpty(parsed.Thinking))
                deltas.Add(JsonSerializer.Serialize(
                    OpenAIResponseFactory.ReasoningContentChunk("id", "model", parsed.Thinking)));

            if (!string.IsNullOrEmpty(parsed.Content))
                deltas.Add(JsonSerializer.Serialize(
                    OpenAIResponseFactory.ContentChunk("id", "model", parsed.Content)));
        }
        return deltas;
    }

    [Fact]
    public void ReasoningContentChunk_SerializesReasoningContentField()
    {
        string json = JsonSerializer.Serialize(
            OpenAIResponseFactory.ReasoningContentChunk("req-1", "gpt-oss", "thinking out loud"));

        using var doc = JsonDocument.Parse(json);
        var delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");

        Assert.Equal("thinking out loud", delta.GetProperty("reasoning_content").GetString());
        // Visible content must be null on a pure-reasoning delta so clients render
        // it as reasoning, not answer text.
        Assert.Equal(JsonValueKind.Null, delta.GetProperty("content").ValueKind);
        Assert.Equal(JsonValueKind.Null, doc.RootElement
            .GetProperty("choices")[0].GetProperty("finish_reason").ValueKind);
    }

    [Fact]
    public void AnalysisThenFinal_StreamsReasoningBeforeContent()
    {
        // A reasoning-first Harmony completion split into streaming chunks, exactly
        // as gpt-oss emits it after the "<|start|>assistant" generation marker.
        string[] chunks =
        {
            "<|channel|>analysis<|message|>",
            "The user wants a short answer.",
            " Keep it concise.",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
            "Hello",
            " there!",
        };

        var deltas = EmitDeltasForStream(chunks);

        // At least one reasoning delta and one content delta were emitted.
        Assert.Contains(deltas, d => d.Contains("reasoning_content"));
        Assert.Contains(deltas, d => DeltaContentText(d) is { Length: > 0 });

        // The FIRST emitted delta carries reasoning — i.e. time-to-first-token now
        // fires on the first analysis token (right after prefill), not after the
        // whole reasoning block has been decoded.
        Assert.Contains("reasoning_content", deltas[0]);

        // Reasoning is fully emitted before the first visible content delta.
        int firstReasoning = deltas.FindIndex(d => d.Contains("reasoning_content"));
        int firstContent = deltas.FindIndex(d => DeltaContentText(d) is { Length: > 0 });
        Assert.True(firstReasoning >= 0 && firstContent >= 0);
        Assert.True(firstReasoning < firstContent,
            "reasoning_content must be streamed before the first content token");

        // The reasoning text is reassembled from the reasoning deltas, and the
        // analysis text never leaks into visible content.
        string reasoning = ConcatReasoning(deltas);
        string content = ConcatContent(deltas);
        Assert.Equal("The user wants a short answer. Keep it concise.", reasoning);
        Assert.Equal("Hello there!", content);
        Assert.DoesNotContain("user wants", content);
        Assert.DoesNotContain("<|channel|>", content);
    }

    [Fact]
    public void PlainFinalOnly_EmitsNoReasoningDelta()
    {
        // No analysis channel: a content-only stream must not fabricate reasoning.
        string[] chunks =
        {
            "<|channel|>final<|message|>",
            "Just the answer.",
        };

        var deltas = EmitDeltasForStream(chunks);

        Assert.DoesNotContain(deltas, d => d.Contains("reasoning_content"));
        Assert.Equal("Just the answer.", ConcatContent(deltas));
    }

    // ---- helpers ---------------------------------------------------------

    private static string DeltaContentText(string deltaJson)
    {
        using var doc = JsonDocument.Parse(deltaJson);
        var content = doc.RootElement.GetProperty("choices")[0]
            .GetProperty("delta").GetProperty("content");
        return content.ValueKind == JsonValueKind.String ? content.GetString() : null;
    }

    private static string DeltaReasoningText(string deltaJson)
    {
        using var doc = JsonDocument.Parse(deltaJson);
        var delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");
        if (!delta.TryGetProperty("reasoning_content", out var r) || r.ValueKind != JsonValueKind.String)
            return null;
        return r.GetString();
    }

    private static string ConcatReasoning(List<string> deltas)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var d in deltas)
            if (DeltaReasoningText(d) is { } r) sb.Append(r);
        return sb.ToString();
    }

    private static string ConcatContent(List<string> deltas)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var d in deltas)
            if (DeltaContentText(d) is { } c) sb.Append(c);
        return sb.ToString();
    }
}
