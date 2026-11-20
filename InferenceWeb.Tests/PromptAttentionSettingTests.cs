// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #383, problem 3: the prompt attention of --backend cuda is a setting, one spelling on the CLI, the server and in a
// --config file (--prompt-attention f16|f32acc|int8) and in the environment (TS_PROMPT_ATTENTION), with the default f16. An
// unknown value is refused, never run as another path. Nothing here sets the process environment: the server's reading of the
// flag and of an inherited variable is a pure function (ServerOptionsBuilder.ReadPromptAttentionCliFlag).
using TensorSharp.Graph;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed class PromptAttentionSettingTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-prompt-attention-" + Guid.NewGuid().ToString("N"));

    public PromptAttentionSettingTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Theory]
    [InlineData("f16", PromptAttentionPath.F16)]
    [InlineData("f32acc", PromptAttentionPath.F32Acc)]
    [InlineData("int8", PromptAttentionPath.Int8)]
    [InlineData(" F32ACC ", PromptAttentionPath.F32Acc)]
    public void EachPathHasOneSpelling(string value, PromptAttentionPath expected)
    {
        Assert.True(PromptAttentionConfig.TryParse(value, out PromptAttentionPath path));
        Assert.Equal(expected, path);
        Assert.Equal(value.Trim().ToLowerInvariant(), PromptAttentionConfig.Name(path));
        Assert.Equal(expected, PromptAttentionConfig.FromValue(value));
    }

    [Theory]
    [InlineData("fp16")]
    [InlineData("f32")]
    [InlineData("8bit")]
    [InlineData("q8_0")]
    [InlineData("default")]
    public void OtherSpellingsAreRefused(string value)
    {
        Assert.False(PromptAttentionConfig.TryParse(value, out _));
        var refused = Assert.Throws<ArgumentException>(() => PromptAttentionConfig.FromValue(value));
        Assert.Contains(PromptAttentionConfig.EnvironmentVariable, refused.Message);
        Assert.Contains("f16, f32acc, int8", refused.Message);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    public void NothingChosenIsTheDefaultF16(string? value)
    {
        Assert.Equal(PromptAttentionPath.F16, PromptAttentionConfig.Default);
        Assert.Equal(PromptAttentionPath.F16, PromptAttentionConfig.FromValue(value!));
    }

    [Fact]
    public void TheServerReadsTheFlagLastOneWinning_AndTheFlagBeatsTheVariable()
    {
        Assert.Null(ServerOptionsBuilder.ReadPromptAttentionCliFlag(new[] { "--port", "5000" }, inherited: null));
        Assert.Equal(PromptAttentionPath.F32Acc,
            ServerOptionsBuilder.ReadPromptAttentionCliFlag(new[] { "--prompt-attention", "f32acc" }, inherited: null));
        Assert.Equal(PromptAttentionPath.Int8,
            ServerOptionsBuilder.ReadPromptAttentionCliFlag(new[] { "--prompt-attention=int8" }, inherited: null));
        Assert.Equal(PromptAttentionPath.F16, ServerOptionsBuilder.ReadPromptAttentionCliFlag(
            new[] { "--prompt-attention", "int8", "--prompt-attention", "f16" }, inherited: null));
        // The flag wins over the variable, even one that names no path.
        Assert.Equal(PromptAttentionPath.F32Acc,
            ServerOptionsBuilder.ReadPromptAttentionCliFlag(new[] { "--prompt-attention", "f32acc" }, inherited: "nonsense"));
        Assert.Null(ServerOptionsBuilder.ReadPromptAttentionCliFlag(Array.Empty<string>(), inherited: "int8"));
    }

    [Fact]
    public void TheServerRefusesAnUnknownValueOfTheFlagOrOfAnInheritedVariable()
    {
        var flag = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ReadPromptAttentionCliFlag(new[] { "--prompt-attention", "8bit" }, inherited: null));
        Assert.Contains("--prompt-attention", flag.Message);
        Assert.Contains("'8bit'", flag.Message);
        var inherited = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ReadPromptAttentionCliFlag(new[] { "--port", "5000" }, inherited: "fp32"));
        Assert.Contains(PromptAttentionConfig.EnvironmentVariable, inherited.Message);
        Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.ReadPromptAttentionCliFlag(new[] { "--prompt-attention" }, inherited: null));
    }

    [Fact]
    public void TheServerOptionsAcceptTheFlag()
    {
        // ParseArgs skips what the earlier pass consumed (no unknown-option refusal), in both spellings of a value.
        foreach (string[] args in new[] { new[] { "--prompt-attention", "f32acc" }, new[] { "--prompt-attention=int8" } })
        {
            Exception? ex = Record.Exception(() => ServerOptionsBuilder.Build(args, _dir));
            Assert.False(ex is ArgumentException { Message: var m } && m.StartsWith("Unknown option", StringComparison.Ordinal),
                $"[{string.Join(' ', args)}]: {ex?.Message}");
        }
    }

    [Fact]
    public void AConfigFileCarriesTheFlag()
    {
        string path = Path.Combine(_dir, "preset.json");
        File.WriteAllText(path, "{ \"prompt-attention\": \"f32acc\" }");
        string[] expanded = ConfigFileArgs.Expand(new[] { "--config", path });
        int at = Array.IndexOf(expanded, PromptAttentionConfig.Flag);
        Assert.True(at >= 0, $"expanded to [{string.Join(' ', expanded)}]");
        Assert.Equal("f32acc", expanded[at + 1]);
        Assert.Equal(PromptAttentionPath.F32Acc, ServerOptionsBuilder.ReadPromptAttentionCliFlag(expanded, inherited: null));
    }
}
