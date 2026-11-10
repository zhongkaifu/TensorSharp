using TensorSharp.Distributed;

namespace InferenceWeb.Tests;

public sealed class ModelParallelismOptionsTests : IDisposable
{
    private readonly Dictionary<string, string?> _saved = new();

    public ModelParallelismOptionsTests()
    {
        foreach (string variable in new[] { ModelParallelismOptions.TpDegreeVariable,
                     ModelParallelismOptions.LayerSplitDegreeVariable, ModelParallelismOptions.NodeIdVariable,
                     ModelParallelismOptions.PeersVariable })
        {
            _saved[variable] = Environment.GetEnvironmentVariable(variable);
            Environment.SetEnvironmentVariable(variable, null);
        }
    }

    public void Dispose()
    {
        foreach (var pair in _saved)
            Environment.SetEnvironmentVariable(pair.Key, pair.Value);
    }

    [Theory]
    [InlineData("--tp", 2, 1)]
    [InlineData("--layer-split", 1, 2)]
    public void ModesHaveIndependentDegreesAndConsumeOnlyTheirOwnArguments(string flag, int tp, int layers)
    {
        foreach (var args in new[] { new[] { "--model", "m.gguf", flag, "2", "--test" },
                     new[] { "--model", "m.gguf", flag + "=2", "--test" } })
        {
            var remaining = new List<string>();
            var options = ModelParallelismOptions.Parse(args, remaining);
            Assert.Equal(tp, options.TpDegree);
            Assert.Equal(layers, options.LayerSplitDegree);
            Assert.Equal(new[] { "--model", "m.gguf", "--test" }, remaining);
            Assert.Null(options.Distributed);
        }
    }

    [Theory]
    [InlineData("--tp")]
    [InlineData("--layer-split")]
    [InlineData("--tp-node-id")]
    [InlineData("--tp-peers")]
    public void MissingValuesNameTheFlag(string flag)
    {
        foreach (var args in new[] { new[] { flag }, new[] { flag + "=" }, new[] { flag, "--model", "m.gguf" } })
            Assert.Contains(flag, Assert.Throws<ArgumentException>(() => ModelParallelismOptions.Parse(args)).Message);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("two")]
    [InlineData("2147483648")]
    public void InvalidDegreesAreRejectedInBothModes(string value)
    {
        foreach (string flag in new[] { "--tp", "--layer-split" })
            Assert.Contains(flag, Assert.Throws<ArgumentException>(() =>
                ModelParallelismOptions.Parse(new[] { flag, value })).Message);
    }

    [Fact]
    public void BothDegreesSelectHybridPlacementRegardlessOfArgumentOrder()
    {
        // --layer-split P stages of --tp T devices each; whether the architecture supports it
        // is the loader's decision (its serving contract), not the option parser's.
        foreach (var args in new[] { new[] { "--tp", "2", "--layer-split", "3" },
                     new[] { "--layer-split=3", "--tp=2" } })
        {
            var options = ModelParallelismOptions.Parse(args);
            Assert.Equal(2, options.TpDegree);
            Assert.Equal(3, options.LayerSplitDegree);
            Assert.False(options.IsLayerSplit);
            Assert.Null(options.Distributed);
        }
    }

    [Fact]
    public void HybridPlacementIsRefusedAcrossNodes()
        => Assert.Contains("one process", Assert.Throws<ArgumentException>(() => ModelParallelismOptions.Parse(new[]
        {
            "--tp=2", "--layer-split=2", "--tp-node-id=0", "--tp-peers=127.0.0.1:9500,127.0.0.1:9501",
        })).Message);

    [Fact]
    public void CliOverridesEnvironmentIncludingAnExplicitDegreeOfOne()
    {
        Environment.SetEnvironmentVariable(ModelParallelismOptions.TpDegreeVariable, "4");
        var options = ModelParallelismOptions.Parse(new[] { "--tp=1", "--layer-split=2" });
        Assert.Equal(1, options.TpDegree);
        Assert.Equal(2, options.LayerSplitDegree);
        Assert.Equal("4", Environment.GetEnvironmentVariable(ModelParallelismOptions.TpDegreeVariable));
        Assert.True(options.ApplyEnvironment());
        Assert.Equal("1", Environment.GetEnvironmentVariable(ModelParallelismOptions.TpDegreeVariable));
        Assert.Equal("2", Environment.GetEnvironmentVariable(ModelParallelismOptions.LayerSplitDegreeVariable));
    }

    [Fact]
    public void InvalidConfigurationDoesNotPartiallyMutateEnvironment()
    {
        Environment.SetEnvironmentVariable(ModelParallelismOptions.TpDegreeVariable, "1");
        Assert.Throws<ArgumentException>(() => ModelParallelismOptions.Parse(new[] { "--tp=2", "--layer-split=0" }));
        Assert.Equal("1", Environment.GetEnvironmentVariable(ModelParallelismOptions.TpDegreeVariable));
        Assert.Null(Environment.GetEnvironmentVariable(ModelParallelismOptions.LayerSplitDegreeVariable));
    }

    [Fact]
    public void EnvironmentModesAreValidatedEvenWithoutCliFlags()
    {
        Environment.SetEnvironmentVariable(ModelParallelismOptions.LayerSplitDegreeVariable, "3");
        var options = ModelParallelismOptions.Parse(Array.Empty<string>());
        Assert.Equal(3, options.LayerSplitDegree);
        Assert.False(options.ApplyEnvironment());
        Environment.SetEnvironmentVariable(ModelParallelismOptions.TpDegreeVariable, "2");
        var hybrid = ModelParallelismOptions.Parse(Array.Empty<string>());
        Assert.Equal((2, 3), (hybrid.TpDegree, hybrid.LayerSplitDegree));
        Environment.SetEnvironmentVariable(ModelParallelismOptions.TpDegreeVariable, "zero");
        Assert.Throws<ArgumentException>(() => ModelParallelismOptions.Parse(Array.Empty<string>()));
    }

    [Fact]
    public void DistributedTensorParallelismUsesLocalAndGlobalDegrees()
    {
        var options = ModelParallelismOptions.Parse(new[] { "--tp=2", "--tp-node-id=1", "--tp-peers=127.0.0.1:9500,127.0.0.1:9501" });
        Assert.NotNull(options.Distributed);
        Assert.Equal(2, options.Distributed.LocalDegree);
        Assert.Equal(4, options.Distributed.GlobalDegree);
        Assert.Equal(1, options.Distributed.NodeId);
    }

    [Theory]
    [InlineData("--tp-node-id=0")]
    [InlineData("--tp-peers=127.0.0.1:9500,127.0.0.1:9501")]
    public void PartialDistributedConfigurationNeverSilentlyBecomesLocal(string flag)
        => Assert.Throws<ArgumentException>(() => ModelParallelismOptions.Parse(new[] { flag }));

    [Theory]
    [InlineData("0", "127.0.0.1:9500")]
    [InlineData("2", "127.0.0.1:9500,127.0.0.1:9501")]
    [InlineData("0", "not-an-endpoint")]
    public void InvalidClusterIsRejectedBeforeConnecting(string node, string peers)
        => Assert.Throws<ArgumentException>(() => ModelParallelismOptions.Parse(new[] { "--tp-node-id", node, "--tp-peers", peers }));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void DistributedLayerSplitUsesTheRequestedLocalDegree(int degree)
    {
        var options = ModelParallelismOptions.Parse(new[] {
            $"--layer-split={degree}", "--tp-node-id=0", "--tp-peers=127.0.0.1:9500,127.0.0.1:9501" });
        Assert.True(options.IsLayerSplit);
        Assert.Equal(1, options.TpDegree);
        Assert.Equal(degree, options.Distributed.LocalDegree);
        Assert.Equal(2 * degree, options.Distributed.GlobalDegree);
    }

    [Fact]
    public void EnvironmentSelectsLayerSplitWithOneGpuPerNode()
    {
        Environment.SetEnvironmentVariable(ModelParallelismOptions.LayerSplitDegreeVariable, "1");
        var options = ModelParallelismOptions.Parse(new[] {
            "--tp-node-id=1", "--tp-peers=127.0.0.1:9500,127.0.0.1:9501" });
        Assert.True(options.IsLayerSplit);
        Assert.Equal(1, options.Distributed.LocalDegree);
    }

    [Fact]
    public void ExplicitLayerDegreeOneCanDisableLayerSplittingForTensorParallelism()
    {
        Environment.SetEnvironmentVariable(ModelParallelismOptions.LayerSplitDegreeVariable, "2");
        var options = ModelParallelismOptions.Parse(new[] { "--layer-split=1", "--tp=2" });
        Assert.False(options.IsLayerSplit);
    }

    [Theory]
    [InlineData("127.0.0.1:9500,,127.0.0.1:9501")]
    [InlineData("127.0.0.1:9500,")]
    [InlineData("127.0.0.1:9500,127.0.0.1:9500")]
    [InlineData("::1:9500,::1:9501")]
    public void InvalidPeerListsAreRejected(string peers)
        => Assert.Throws<ArgumentException>(() => ModelParallelismOptions.Parse(new[] {
            "--tp-node-id=0", "--tp-peers=" + peers }));

    [Fact]
    public void BracketedIpv6PeersAreParsed()
    {
        var options = ModelParallelismOptions.Parse(new[] {
            "--layer-split=2", "--tp-node-id=0", "--tp-peers=[::1]:9500,[::1]:9501" });
        Assert.Equal(System.Net.IPAddress.IPv6Loopback, options.Distributed.PeerEndpoints[0].Address);
    }
}
