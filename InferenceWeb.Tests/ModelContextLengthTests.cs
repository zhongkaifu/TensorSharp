namespace InferenceWeb.Tests;

public class ModelContextLengthTests
{
    [Fact]
    public void ResolveConfiguredContextLength_PrefersExplicitOverride()
    {
        var metadata = new Dictionary<string, object>
        {
            ["qwen3.context_length"] = 32768u,
            ["qwen3.rope.scaling.original_context_length"] = 4096u
        };

        int resolved = ModelBase.ResolveConfiguredContextLength("qwen3", metadata, 4096, 8192, out string source);

        Assert.Equal(8192, resolved);
        Assert.Equal("MAX_CONTEXT", source);
    }

    [Fact]
    public void ResolveConfiguredContextLength_UsesStandardContextLengthBeforeOriginalContext()
    {
        var metadata = new Dictionary<string, object>
        {
            ["gptoss.context_length"] = 131072u,
            ["gptoss.rope.scaling.original_context_length"] = 4096u
        };

        int resolved = ModelBase.ResolveConfiguredContextLength("gptoss", metadata, 4096, null, out string source);

        Assert.Equal(131072, resolved);
        Assert.Equal("gptoss.context_length", source);
    }

    [Fact]
    public void ResolveConfiguredContextLength_FallsBackWhenMetadataIsMissing()
    {
        int resolved = ModelBase.ResolveConfiguredContextLength(
            "nemotron_h",
            new Dictionary<string, object>(),
            4096,
            null,
            out string source);

        Assert.Equal(4096, resolved);
        Assert.Equal("fallback", source);
    }
}
