using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Gemma 4's generation prompt primes the thinking channel, so a thinking block
/// the model produces is closed by a <c>&lt;channel|&gt;</c> whose opening
/// <c>&lt;|channel&gt;</c> is in the PROMPT, not in the generated text. The parser
/// has to recognise that stray close, or the whole chain of thought plus the raw
/// marker is delivered as the assistant's answer (issue #113's log shows exactly
/// that: <c>thought\n…&lt;channel|&gt;Hello! How can I help you today?</c>).
/// </summary>
public class Gemma4PrimedThinkingParserTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PromptOpenedChannel_IsKnownBeforeAnyStreamingContent(bool thinkingEnabled)
    {
        IOutputParser parser = new Gemma4OutputParser();
        parser.Init(thinkingEnabled, null);
        parser.SetGenerationPromptSuffix("<|channel>thought\n");
        var content = new System.Text.StringBuilder();
        var thinking = new System.Text.StringBuilder();
        const string thought = "Inspecting the tool result carefully.";
        foreach (char value in thought)
        {
            var delta = parser.Add(value.ToString(), false);
            Assert.Empty(delta.Content);
            thinking.Append(delta.Thinking);
        }
        foreach (char value in "<channel|>42")
        {
            var delta = parser.Add(value.ToString(), false);
            content.Append(delta.Content);
            thinking.Append(delta.Thinking);
        }
        var final = parser.Add("", true);
        content.Append(final.Content);
        thinking.Append(final.Thinking);
        Assert.Equal("42", content.ToString());
        Assert.Equal(thinkingEnabled ? thought : "", thinking.ToString());

        parser.Init(false, null);
        parser.SetGenerationPromptSuffix("<|channel>thought\n<channel|>");
        Assert.Equal("A plain answer", parser.Add("A plain answer", false).Content);
    }

    private static ParsedOutput ParseWhole(string raw, bool thinkingEnabled)
    {
        var parser = new Gemma4OutputParser();
        parser.Init(thinkingEnabled, null);
        return parser.Add(raw, true);
    }

    [Fact]
    public void StrayChannelCloseSplitsThinkingFromAnswer()
    {
        // Verbatim shape from gemma-4-E4B (thinking disabled, 400-token budget).
        const string raw =
            "The user wants a two-sentence explanation for why the sky is blue.\n" +
            "Drafting:\n1. Sunlight enters the atmosphere.\n" +
            "<channel|>Sunlight enters Earth's atmosphere and is scattered by gas molecules.";

        var parsed = ParseWhole(raw, thinkingEnabled: false);

        Assert.Equal("Sunlight enters Earth's atmosphere and is scattered by gas molecules.", parsed.Content);
        Assert.DoesNotContain("channel", parsed.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("Drafting", parsed.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void StrayChannelCloseKeepsThinkingWhenThinkingEnabled()
    {
        const string raw = "thought\nWeighing the options.<channel|>The answer is 42.";

        var parsed = ParseWhole(raw, thinkingEnabled: true);

        Assert.Equal("The answer is 42.", parsed.Content);
        // The re-emitted channel name is not part of the reasoning.
        Assert.Equal("Weighing the options.", parsed.Thinking);
    }

    [Fact]
    public void NormalAnswerWithoutMarkersIsUntouched()
    {
        const string raw = "As sunlight reaches Earth's atmosphere, it scatters in all directions.";

        var parsed = ParseWhole(raw, thinkingEnabled: false);

        Assert.Equal(raw, parsed.Content);
        Assert.Equal("", parsed.Thinking);
    }

    [Fact]
    public void FullyDelimitedThinkingBlockStillParses()
    {
        // The opener is present in the generated text: the pre-existing path.
        const string raw = "<|channel>thought\nShort deliberation.<channel|>Final answer.";

        var parsed = ParseWhole(raw, thinkingEnabled: true);

        Assert.Equal("Final answer.", parsed.Content);
        Assert.Equal("Short deliberation.", parsed.Thinking);
    }

    [Fact]
    public void StrayChannelCloseSplitAcrossStreamingChunksNeverLeaksTheMarker()
    {
        const string raw = "Deliberating quietly.<channel|>The answer is 42.";
        var parser = new Gemma4OutputParser();
        parser.Init(false, null);

        var content = new System.Text.StringBuilder();
        for (int i = 0; i < raw.Length; i += 3)
            content.Append(parser.Add(raw.Substring(i, Math.Min(3, raw.Length - i)), false).Content);
        content.Append(parser.Add("", true).Content);

        Assert.DoesNotContain("channel", content.ToString(), StringComparison.Ordinal);
        Assert.EndsWith("The answer is 42.", content.ToString(), StringComparison.Ordinal);
    }
}
