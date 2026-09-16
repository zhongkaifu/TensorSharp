using System.Text;
using TensorSharp.Cli;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class CliOutputParserTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PromptOpenedThought_IsSeparatedBeforeItsClosingMarker(bool enableThinking)
    {
        var tokenizer = new CharacterTokenizer();
        var prompt = tokenizer.Encode(new string('x', 10000) + "<|channel>thought\n");
        var parser = CliOutputParser.Create("gemma4", enableThinking, null, tokenizer, prompt);
        var thinking = new StringBuilder();
        const string reasoning = "Inspecting the tool result before answering.";
        foreach (char ch in reasoning)
        {
            var delta = parser.Add(ch.ToString(), false);
            Assert.Empty(delta.Content);
            thinking.Append(delta.Thinking);
        }

        var final = parser.Add("<channel|>42", true);
        thinking.Append(final.Thinking);
        Assert.Equal("42", final.Content);
        Assert.Equal(enableThinking ? reasoning : "", thinking.ToString());
        Assert.Equal(new[] { 64 }, tokenizer.DecodeLengths);
    }

    [Fact]
    public void BufferedGeneration_TruncatedInsidePromptOpenedThought_HasNoAnswer()
    {
        var tokenizer = new CharacterTokenizer();
        var parser = CliOutputParser.Create("gemma4", false, null, tokenizer,
            tokenizer.Encode("<|channel>thought\n"));

        var output = parser.Add("Still inspecting the tool result.", true);

        Assert.Empty(output.Content);
        Assert.Empty(output.Thinking);
    }

    [Theory]
    [InlineData("<|channel>thought\n<channel|>")]
    [InlineData("<|channel>thought\nOld reasoning<channel|>\n<|turn>model\n")]
    [InlineData("")]
    public void AClosedOrAbsentThoughtChannel_DoesNotConsumeTheAnswer(string promptText)
    {
        var tokenizer = new CharacterTokenizer();
        var parser = CliOutputParser.Create("gemma4", false, null, tokenizer,
            tokenizer.Encode(promptText));

        var output = parser.Add("An immediate answer", false);

        Assert.Equal("An immediate answer", output.Content);
        Assert.Empty(output.Thinking);
    }

    [Fact]
    public void OtherParsers_DoNotDecodeThePromptAgain()
    {
        var tokenizer = new CharacterTokenizer();
        var parser = CliOutputParser.Create("unknown", false, null, tokenizer,
            tokenizer.Encode("<|channel>thought\n"));

        Assert.Equal("The answer", parser.Add("The answer", true).Content);
        Assert.Empty(tokenizer.DecodeLengths);
    }

    private sealed class CharacterTokenizer : ITokenizer
    {
        public List<int> DecodeLengths { get; } = new();
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => -1;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids)
        {
            DecodeLengths.Add(ids.Count);
            return new string(ids.Select(id => (char)id).ToArray());
        }
        public void AppendTokenBytes(int tokenId, List<byte> buffer) =>
            buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }
}
