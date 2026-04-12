
namespace InferenceWeb.Tests;

public class KVCacheTests
{
    [Fact]
    public void FindTokenPrefixLength_NullCached_ReturnsZero()
    {
        var newTokens = new List<int> { 1, 2, 3 };
        Assert.Equal(0, ModelService.FindTokenPrefixLength(null, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_EmptyCached_ReturnsZero()
    {
        var cached = new List<int>();
        var newTokens = new List<int> { 1, 2, 3 };
        Assert.Equal(0, ModelService.FindTokenPrefixLength(cached, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_NullNewTokens_ReturnsZero()
    {
        var cached = new List<int> { 1, 2, 3 };
        Assert.Equal(0, ModelService.FindTokenPrefixLength(cached, null));
    }

    [Fact]
    public void FindTokenPrefixLength_NoCommonPrefix_ReturnsZero()
    {
        var cached = new List<int> { 1, 2, 3 };
        var newTokens = new List<int> { 4, 5, 6 };
        Assert.Equal(0, ModelService.FindTokenPrefixLength(cached, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_IdenticalSequences_ReturnsZero()
    {
        // If new tokens == cached tokens exactly, there's nothing new to forward
        var cached = new List<int> { 1, 2, 3 };
        var newTokens = new List<int> { 1, 2, 3 };
        Assert.Equal(0, ModelService.FindTokenPrefixLength(cached, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_NewIsPrefixOfCached_ReturnsZero()
    {
        // New tokens are a subset of cached — nothing new to forward
        var cached = new List<int> { 1, 2, 3, 4, 5 };
        var newTokens = new List<int> { 1, 2, 3 };
        Assert.Equal(0, ModelService.FindTokenPrefixLength(cached, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_CachedIsPrefixOfNew_ReturnsCachedLength()
    {
        // Typical multi-turn: cached=[prompt1+response1], new=[prompt1+response1+user2+genPrompt]
        var cached = new List<int> { 1, 2, 3, 4, 5 };
        var newTokens = new List<int> { 1, 2, 3, 4, 5, 6, 7 };
        Assert.Equal(5, ModelService.FindTokenPrefixLength(cached, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_PartialMatch_ReturnsCommonLength()
    {
        // Cached and new diverge partway through (e.g., different template rendering)
        var cached = new List<int> { 1, 2, 3, 10, 11 };
        var newTokens = new List<int> { 1, 2, 3, 20, 21, 22 };
        Assert.Equal(3, ModelService.FindTokenPrefixLength(cached, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_SingleTokenDifference_ReturnsZero()
    {
        // First token differs (e.g., different BOS)
        var cached = new List<int> { 99, 2, 3 };
        var newTokens = new List<int> { 1, 2, 3 };
        Assert.Equal(0, ModelService.FindTokenPrefixLength(cached, newTokens));
    }

    [Fact]
    public void FindTokenPrefixLength_SimulatesMultiTurnConversation()
    {
        // Turn 1: prompt tokens [BOS=1, sys, user1, genPrompt] + generated [resp1, resp2]
        var turn1Prompt = new List<int> { 1, 100, 200, 300 };
        var turn1Generated = new List<int> { 500, 501 };
        var cached = new List<int>(turn1Prompt);
        cached.AddRange(turn1Generated);
        // cached = [1, 100, 200, 300, 500, 501]

        // Turn 2: re-tokenized prompt includes turn1 + assistant response + user2 + genPrompt
        // The tokenizer processes the full rendered text including the assistant's raw output
        var turn2Tokens = new List<int> { 1, 100, 200, 300, 500, 501, 600, 700, 800 };
        // [BOS, sys, user1, genPrompt, resp1, resp2, closingTag, user2Content, genPrompt2]

        int common = ModelService.FindTokenPrefixLength(cached, turn2Tokens);
        Assert.Equal(6, common); // All of cached is a prefix
    }

    [Fact]
    public void FindTokenPrefixLength_ThinkingModel_PartialMatch()
    {
        // For thinking models: raw output includes <think> tags that template drops
        // Turn 1 cached: [BOS, sys, user1, genPrompt, thinkStart, thinking, thinkEnd, content]
        var cached = new List<int> { 1, 100, 200, 300, 50, 51, 52, 53 };

        // Turn 2 re-rendered: template renders assistant without thinking tags
        // [BOS, sys, user1, genPrompt(which is now assistant msg opening), content, closingTag, user2, genPrompt2]
        // The tokens diverge at position 4 (where thinking was vs where content starts)
        var newTokens = new List<int> { 1, 100, 200, 300, 53, 600, 700, 800 };

        int common = ModelService.FindTokenPrefixLength(cached, newTokens);
        // Common prefix is 4 tokens: [1, 100, 200, 300]
        Assert.Equal(4, common);
    }

    [Fact]
    public void FindTokenPrefixLength_ThinkingModelWithContentInContext()
    {
        // If we include thinking in the context, the template renders the full output
        // Turn 1 cached: [BOS, sys, user1, genPrompt, thinkStart, thinking, thinkEnd, content]
        var cached = new List<int> { 1, 100, 200, 300, 50, 51, 52, 53 };

        // Turn 2 re-rendered with thinking in context: same tokens up to the end of cached
        var newTokens = new List<int> { 1, 100, 200, 300, 50, 51, 52, 53, 600, 700, 800 };

        int common = ModelService.FindTokenPrefixLength(cached, newTokens);
        Assert.Equal(8, common); // Full cached is prefix
    }
}

