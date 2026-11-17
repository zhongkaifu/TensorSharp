using System.Text.Json;
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

/// <summary>
/// The non-streaming chat completion carries the reasoning as <c>reasoning_content</c>, the field
/// its streaming deltas use and OpenAI-compatible clients read (found by eng/lfm3/e2e: a client of
/// LFM3 with thinking on got the reasoning only under <c>thinking</c>).
/// </summary>
public class OpenAIReasoningContentTests
{
    [Fact]
    public void NonStreamingMessage_CarriesReasoningContentLikeTheStream()
    {
        JsonElement message = JsonSerializer.SerializeToElement(OpenAIResponseFactory.Completion("id", "m",
            OpenAIResponseFactory.ParsedAssistantMessage("391", "17*23 = 340 + 51", null), "stop", 10, 5, 0))
            .GetProperty("choices")[0].GetProperty("message");
        Assert.Equal("391", message.GetProperty("content").GetString());
        Assert.Equal("17*23 = 340 + 51", message.GetProperty("reasoning_content").GetString());
        // The earlier spelling stays for clients that read it.
        Assert.Equal("17*23 = 340 + 51", message.GetProperty("thinking").GetString());

        JsonElement delta = JsonSerializer.SerializeToElement(OpenAIResponseFactory.ReasoningContentChunk("id", "m", "17*23"))
            .GetProperty("choices")[0].GetProperty("delta");
        Assert.Equal("17*23", delta.GetProperty("reasoning_content").GetString());
    }
}
