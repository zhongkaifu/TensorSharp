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
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

/// <summary>
/// The OpenAI chat route names the reasoning the same way whether or not the reply is
/// streamed: <c>reasoning_content</c>, which OpenAI-compatible clients read from a
/// complete message as well as from deltas. A non-streamed reply also keeps
/// <c>thinking</c>, its earlier name.
/// </summary>
public class OpenAIReasoningContentTests
{
    private static JsonElement Message(string? thinking)
    {
        object completion = OpenAIResponseFactory.Completion("chatcmpl-test", "probe.gguf",
            OpenAIResponseFactory.ParsedAssistantMessage("42", thinking, null), "stop", 10, 2, 0);
        // The route serializes the completion with IgnoreNulls (OpenAIChatAdapter).
        using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(completion, JsonOptions.IgnoreNulls));
        return doc.RootElement.GetProperty("choices")[0].GetProperty("message").Clone();
    }

    [Fact]
    public void ANonStreamedReplyCarriesTheReasoningAsReasoningContent()
    {
        JsonElement message = Message("Six times seven.");

        Assert.Equal("42", message.GetProperty("content").GetString());
        Assert.Equal("Six times seven.", message.GetProperty("reasoning_content").GetString());
        Assert.Equal("Six times seven.", message.GetProperty("thinking").GetString());
    }

    [Fact]
    public void AReplyWithoutReasoningCarriesNeitherField()
    {
        JsonElement message = Message(null);

        Assert.False(message.TryGetProperty("reasoning_content", out _));
        Assert.False(message.TryGetProperty("thinking", out _));
    }

    [Fact]
    public void StreamedAndNonStreamedRepliesUseTheSameFieldName()
    {
        object chunk = OpenAIResponseFactory.ReasoningContentChunk("chatcmpl-test", "probe.gguf", "Six times seven.");
        using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(chunk, JsonOptions.IgnoreNulls));
        JsonElement delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");

        Assert.Equal(delta.GetProperty("reasoning_content").GetString(),
            Message("Six times seven.").GetProperty("reasoning_content").GetString());
    }
}
