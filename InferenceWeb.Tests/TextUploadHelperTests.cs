namespace InferenceWeb.Tests;

public class TextUploadHelperTests
{
    [Fact]
    public void PrepareTextContent_UsesHalfModelContextAsTokenLimit()
    {
        var tokenizer = new CharacterTokenizer();

        var result = TextUploadHelper.PrepareTextContent(
            "abcdef",
            tokenizer,
            modelContextLimit: 8,
            fallbackCharLimit: 100);

        Assert.True(result.Truncated);
        Assert.Equal("abcd", result.TextContent);
        Assert.Equal(4, result.TruncateLimit);
        Assert.Equal("tokens", result.TruncateUnit);
        Assert.Equal(8, result.ModelContextLimit);
        Assert.Equal(6, result.OriginalTokenCount);
        Assert.Equal(4, result.ReturnedTokenCount);
    }

    [Fact]
    public void PrepareTextContent_FallsBackToCharacterLimitWithoutModelTokenizer()
    {
        var result = TextUploadHelper.PrepareTextContent(
            "abcdef",
            tokenizer: null,
            modelContextLimit: 0,
            fallbackCharLimit: 3);

        Assert.True(result.Truncated);
        Assert.Equal("abc", result.TextContent);
        Assert.Equal(3, result.TruncateLimit);
        Assert.Equal("characters", result.TruncateUnit);
        Assert.Null(result.ModelContextLimit);
        Assert.Null(result.OriginalTokenCount);
        Assert.Null(result.ReturnedTokenCount);
    }

    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 0;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => 0;

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var ids = new List<int>();
            foreach (char ch in text ?? string.Empty)
                ids.Add(ch);
            return ids;
        }

        public string Decode(List<int> ids)
        {
            return new string(ids.ConvertAll(id => (char)id).ToArray());
        }

        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            throw new NotSupportedException();
        }

        public bool IsEos(int tokenId)
        {
            return false;
        }

        public int LookupToken(string tokenStr)
        {
            return -1;
        }
    }
}
