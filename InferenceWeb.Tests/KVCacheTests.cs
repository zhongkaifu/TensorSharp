
namespace InferenceWeb.Tests;

public class KVCacheTests
{
    private sealed class FakeModelArchitecture : IModelArchitecture
    {
        public ModelConfig Config { get; init; } = new();
        public ITokenizer Tokenizer => throw new NotSupportedException();
        public IKVCachePolicy KVCachePolicy => DefaultKvCachePolicy.Shared;
        public IMultimodalInjector MultimodalInjector => throw new NotSupportedException();
        public IBackendExecutionPlan ExecutionPlan => throw new NotSupportedException();
        public bool SupportsKVCacheTruncation { get; init; } = true;

        public float[] Forward(int[] tokens) => throw new NotSupportedException();
        public void ResetKVCache() => throw new NotSupportedException();
        public void TruncateKVCache(int tokenCount) => throw new NotSupportedException();
        public void Dispose() { }
    }

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

    [Fact]
    public void ResolvePrefillChunkSize_CudaLongPrompt_UsesSafeChunkSize()
    {
        int chunkSize = ModelService.ResolvePrefillChunkSize(TensorSharp.Runtime.BackendType.GgmlCuda, 11573);
        Assert.Equal(5120, chunkSize);
    }

    [Fact]
    public void ResolvePrefillChunkSize_NonCudaPrompt_DoesNotChunk()
    {
        int chunkSize = ModelService.ResolvePrefillChunkSize(TensorSharp.Runtime.BackendType.GgmlCpu, 11573);
        Assert.Equal(11573, chunkSize);
    }

    [Fact]
    public void DefaultKvCachePolicy_ExactMatch_ReusesFullPrompt()
    {
        var model = new FakeModelArchitecture();
        var cached = new List<int> { 1, 2, 3, 4 };
        var input = new List<int> { 1, 2, 3, 4 };

        int reusable = DefaultKvCachePolicy.Shared.ComputeReusablePrefix(model, cached, input, hasMultimodal: false);

        Assert.Equal(4, reusable);
    }

    [Fact]
    public void DefaultKvCachePolicy_MultimodalPrompt_CanStillReusePrefix()
    {
        var model = new FakeModelArchitecture();
        var cached = new List<int> { 10, 11, 12, 13 };
        var input = new List<int> { 10, 11, 12, 13, 14, 15 };

        int reusable = DefaultKvCachePolicy.Shared.ComputeReusablePrefix(model, cached, input, hasMultimodal: true);

        Assert.Equal(4, reusable);
    }

    [Fact]
    public void DefaultKvCachePolicy_NonCircularSlidingWindow_DoesNotBacktrack()
    {
        var model = new FakeModelArchitecture
        {
            Config = new ModelConfig
            {
                Architecture = "gptoss",
                SlidingWindow = 128,
                UsesCircularKvCache = false,
            }
        };

        var cached = new List<int> { 1, 2, 3, 4, 5 };
        var input = new List<int> { 1, 2, 3, 4, 5, 6, 7 };

        int reusable = DefaultKvCachePolicy.Shared.ComputeReusablePrefix(model, cached, input, hasMultimodal: false);

        Assert.Equal(5, reusable);
    }

    [Fact]
    public void DefaultKvCachePolicy_CircularSlidingWindow_ReplaysWindow()
    {
        var model = new FakeModelArchitecture
        {
            Config = new ModelConfig
            {
                Architecture = "gemma4",
                SlidingWindow = 4,
                UsesCircularKvCache = true,
            }
        };

        var cached = new List<int> { 1, 2, 3, 4, 5, 6 };
        var input = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8 };

        int reusable = DefaultKvCachePolicy.Shared.ComputeReusablePrefix(model, cached, input, hasMultimodal: false);

        Assert.Equal(2, reusable);
    }

    [Theory]
    [InlineData(6, 6, false, 5)]
    [InlineData(6, 6, true, 6)]
    [InlineData(4, 6, false, 4)]
    [InlineData(1, 1, false, 0)]
    public void ResolveReusablePrefixForInference_MaximizesReuseWithoutLosingLogits(
        int reusablePrefix, int inputCount, bool hasExactCachedLogits, int expected)
    {
        int resolved = ModelService.ResolveReusablePrefixForInference(reusablePrefix, inputCount, hasExactCachedLogits);
        Assert.Equal(expected, resolved);
    }
}

