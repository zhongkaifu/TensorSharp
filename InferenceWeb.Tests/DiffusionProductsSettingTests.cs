// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Design D4 (docs/design/diffusion-runtime.md): the int8 products of a diffusion model on ggml_cuda are a setting with one
// spelling for MiniMax-H3-Image and MiniMax-H3 video, on the CLI, the server and in a --config file
// (--diffusion-products bf16|int8|int8-all, the key diffusion-products) and in the environment the native products read
// (TS_DIFFUSION_PRODUCTS). TS_H3_INT8_MATMUL, the variable it replaces, is refused by name. Nothing here sets the process
// environment: the server's reading of the flag and of an inherited variable is a pure function
// (ServerOptionsBuilder.ReadDiffusionProductsCliFlag), and the refusal of the old name is RemovedCliFlags' (its own tests
// set the variable through EnvScope).
using TensorSharp.Graph;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed class DiffusionProductsSettingTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-diffusion-products-" + Guid.NewGuid().ToString("N"));

    public DiffusionProductsSettingTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Theory]
    [InlineData("bf16", DiffusionProductsPath.Bf16)]
    [InlineData("int8", DiffusionProductsPath.Int8)]
    [InlineData("int8-all", DiffusionProductsPath.Int8All)]
    [InlineData(" INT8 ", DiffusionProductsPath.Int8)]
    public void EachRouteHasOneSpelling(string value, DiffusionProductsPath expected)
    {
        Assert.True(DiffusionProductsConfig.TryParse(value, out DiffusionProductsPath path));
        Assert.Equal(expected, path);
        Assert.Equal(value.Trim().ToLowerInvariant(), DiffusionProductsConfig.Name(path));
        Assert.Equal(expected, DiffusionProductsConfig.FromValue(value));
    }

    /// <summary>The old variable's values are not spellings of the new one: <c>1</c>, <c>all</c> and <c>vae</c> are refused.</summary>
    [Theory]
    [InlineData("1")]
    [InlineData("all")]
    [InlineData("vae")]
    [InlineData("fp8")]
    [InlineData("int8_all")]
    public void OtherSpellingsAreRefused(string value)
    {
        Assert.False(DiffusionProductsConfig.TryParse(value, out _));
        var refused = Assert.Throws<ArgumentException>(() => DiffusionProductsConfig.FromValue(value));
        Assert.Contains(DiffusionProductsConfig.EnvironmentVariable, refused.Message);
        Assert.Contains("bf16, int8, int8-all", refused.Message);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    public void NothingChosenIsBf16(string? value)
    {
        Assert.Null(DiffusionProductsConfig.FromValue(value!));
        Assert.Equal(DiffusionProductsPath.Bf16, DiffusionProductsConfig.Default);
    }

    [Fact]
    public void TheServerReadsTheFlagLastOneWinning_AndTheFlagBeatsTheVariable()
    {
        Assert.Null(ServerOptionsBuilder.ReadDiffusionProductsCliFlag(new[] { "--port", "5000" }, inherited: null));
        Assert.Equal(DiffusionProductsPath.Int8,
            ServerOptionsBuilder.ReadDiffusionProductsCliFlag(new[] { "--diffusion-products", "int8" }, inherited: null));
        Assert.Equal(DiffusionProductsPath.Int8All,
            ServerOptionsBuilder.ReadDiffusionProductsCliFlag(new[] { "--diffusion-products=int8-all" }, inherited: null));
        Assert.Equal(DiffusionProductsPath.Bf16, ServerOptionsBuilder.ReadDiffusionProductsCliFlag(
            new[] { "--diffusion-products", "int8", "--diffusion-products", "bf16" }, inherited: null));
        Assert.Equal(DiffusionProductsPath.Int8,
            ServerOptionsBuilder.ReadDiffusionProductsCliFlag(new[] { "--diffusion-products", "int8" }, inherited: "nonsense"));
        Assert.Null(ServerOptionsBuilder.ReadDiffusionProductsCliFlag(Array.Empty<string>(), inherited: "int8"));
    }

    [Fact]
    public void TheServerRefusesAnUnknownValueOfTheFlagOrOfAnInheritedVariable()
    {
        var flag = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ReadDiffusionProductsCliFlag(new[] { "--diffusion-products", "1" }, inherited: null));
        Assert.Contains("--diffusion-products", flag.Message);
        Assert.Contains("'1'", flag.Message);
        var inherited = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ReadDiffusionProductsCliFlag(new[] { "--port", "5000" }, inherited: "all"));
        Assert.Contains(DiffusionProductsConfig.EnvironmentVariable, inherited.Message);
        Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.ReadDiffusionProductsCliFlag(new[] { "--diffusion-products" }, inherited: null));
    }

    [Fact]
    public void TheServerOptionsAcceptTheFlag()
    {
        foreach (string[] args in new[] { new[] { "--diffusion-products", "int8" }, new[] { "--diffusion-products=int8-all" } })
        {
            Exception? ex = Record.Exception(() => ServerOptionsBuilder.Build(args, _dir));
            Assert.False(ex is ArgumentException { Message: var m } && m.StartsWith("Unknown option", StringComparison.Ordinal),
                $"[{string.Join(' ', args)}]: {ex?.Message}");
        }
    }

    [Fact]
    public void AConfigFileCarriesTheKey()
    {
        string path = Path.Combine(_dir, "preset.json");
        File.WriteAllText(path, "{ \"diffusion-products\": \"int8\" }");
        string[] expanded = ConfigFileArgs.Expand(new[] { "--config", path });
        int at = Array.IndexOf(expanded, DiffusionProductsConfig.Flag);
        Assert.True(at >= 0, $"expanded to [{string.Join(' ', expanded)}]");
        Assert.Equal("int8", expanded[at + 1]);
        Assert.Equal(DiffusionProductsPath.Int8, ServerOptionsBuilder.ReadDiffusionProductsCliFlag(expanded, inherited: null));
    }

    /// <summary>The int8 routes are ggml_cuda's: elsewhere the products run as stored (as TS_H3_INT8_MATMUL's did), and the
    /// hosts say so once instead of running bf16 without a word.</summary>
    [Fact]
    public void AnInt8RouteOffGgmlCudaIsReportedAndBf16IsNot()
    {
        Assert.Null(DiffusionProductsConfig.IgnoredOn(DiffusionProductsPath.Int8, "ggml_cuda"));
        Assert.Null(DiffusionProductsConfig.IgnoredOn(DiffusionProductsPath.Int8All, "GGML_CUDA"));
        Assert.Null(DiffusionProductsConfig.IgnoredOn(DiffusionProductsPath.Bf16, "ggml_metal"));
        Assert.Null(DiffusionProductsConfig.IgnoredOn(null, "ggml_metal"));
        string warning = DiffusionProductsConfig.IgnoredOn(DiffusionProductsPath.Int8, "ggml_metal");
        Assert.Contains("--diffusion-products int8", warning);
        Assert.Contains("ggml_metal", warning);
        Assert.Contains("ggml_cuda", warning);
    }

    /// <summary>The native products read the variable with getenv, which on Unix never sees
    /// Environment.SetEnvironmentVariable: every place that publishes it goes through NativeEnvironment (the C runtime's
    /// table too). A flag published the other way ran bf16 silently (found by the S10 CUDA smoke).</summary>
    [Fact]
    public void EveryPublisherSetsTheVariableWhereTheNativeProductsReadIt()
    {
        string root = AppContext.BaseDirectory;
        while (root != null && !File.Exists(Path.Combine(root, "TensorSharp.slnx"))) root = Path.GetDirectoryName(root)!;
        Assert.NotNull(root);
        foreach (string file in new[]
        {
            Path.Combine(root, "TensorSharp.Cli", "Program.cs"),
            Path.Combine(root, "TensorSharp.Server", "Hosting", "ServerOptionsBuilder.cs"),
            Path.Combine(root, "eng", "validation", "MiniMaxH3ImageAB", "Program.cs"),
        })
        {
            string text = File.ReadAllText(file);
            Assert.Contains("NativeEnvironment.Set(TensorSharp.Graph.DiffusionProductsConfig.EnvironmentVariable", text);
            Assert.DoesNotContain("SetEnvironmentVariable(TensorSharp.Graph.DiffusionProductsConfig.EnvironmentVariable", text);
        }
    }

    /// <summary>The variable it replaces is refused by name, and the refusal names the option and its values for the old
    /// variable's <c>1</c> and <c>all</c>.</summary>
    [Fact]
    public void TheEarlierVariableIsRefusedNamingTheOption()
    {
        (string name, string advice) = Assert.Single(RemovedCliFlags.RemovedEnvironmentVariables, e => e.Name == "TS_H3_INT8_MATMUL");
        Assert.Contains("--diffusion-products int8", advice);
        Assert.Contains("int8-all", advice);
        Assert.Contains("diffusion-products", advice);
        Assert.Equal("TS_H3_INT8_MATMUL", name);
    }
}
