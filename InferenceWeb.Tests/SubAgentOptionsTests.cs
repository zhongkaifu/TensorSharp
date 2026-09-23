// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

/// <summary>
/// The sub-agent flags: parsing, bounds, and environment layering with the command line
/// winning. The environment tests save and restore the three variables; the assembly
/// runs its tests serially, so nothing else observes them in between.
/// </summary>
public class SubAgentOptionsTests
{
    private sealed class EnvironmentScope : IDisposable
    {
        private readonly string? _enable = Environment.GetEnvironmentVariable(SubAgentOptions.EnableEnvVar);
        private readonly string? _threads = Environment.GetEnvironmentVariable(SubAgentOptions.MaxThreadsEnvVar);
        private readonly string? _depth = Environment.GetEnvironmentVariable(SubAgentOptions.MaxDepthEnvVar);

        public EnvironmentScope(string? enable = null, string? threads = null, string? depth = null)
        {
            Environment.SetEnvironmentVariable(SubAgentOptions.EnableEnvVar, enable);
            Environment.SetEnvironmentVariable(SubAgentOptions.MaxThreadsEnvVar, threads);
            Environment.SetEnvironmentVariable(SubAgentOptions.MaxDepthEnvVar, depth);
        }

        public void Dispose()
        {
            Environment.SetEnvironmentVariable(SubAgentOptions.EnableEnvVar, _enable);
            Environment.SetEnvironmentVariable(SubAgentOptions.MaxThreadsEnvVar, _threads);
            Environment.SetEnvironmentVariable(SubAgentOptions.MaxDepthEnvVar, _depth);
        }
    }

    [Fact]
    public void FlagTables_NameEveryFlagExactlyOnce()
    {
        Assert.Equal(new[] { "--sub-agents" }, SubAgentOptions.SwitchFlags);
        Assert.Equal(new[] { "--sub-agents-max-threads", "--sub-agents-max-depth" }, SubAgentOptions.ValueFlags);
        Assert.Equal("TS_SUB_AGENTS", SubAgentOptions.EnableEnvVar);
        Assert.Equal("TS_SUB_AGENTS_MAX_THREADS", SubAgentOptions.MaxThreadsEnvVar);
        Assert.Equal("TS_SUB_AGENTS_MAX_DEPTH", SubAgentOptions.MaxDepthEnvVar);
    }

    [Fact]
    public void Defaults_AreOffWithFourThreadsAndDepthOne()
    {
        var options = new SubAgentOptions();
        Assert.False(options.Enabled);
        Assert.Equal(4, options.MaxThreads);
        Assert.Equal(1, options.MaxDepth);
        Assert.Equal("off", options.Describe());

        SubAgentOptions parsed = SubAgentOptions.Parse(null);
        Assert.False(parsed.Enabled);
        Assert.Equal(4, parsed.MaxThreads);
        Assert.Equal(1, parsed.MaxDepth);
    }

    [Fact]
    public void Parse_TheSwitchEnables_AndUnknownArgumentsAreIgnored()
    {
        SubAgentOptions options = SubAgentOptions.Parse(new[] { "--model", "m.gguf", "--sub-agents", "--port", "5000" });
        Assert.True(options.Enabled);
        Assert.Equal("on (max 4 open, depth 1)", options.Describe());

        Assert.True(SubAgentOptions.Parse(new[] { "--SUB-AGENTS" }).Enabled);
        Assert.False(SubAgentOptions.Parse(new[] { "--sub-agent" }).Enabled);
        Assert.False(SubAgentOptions.Parse(new[] { "--model", "m.gguf" }).Enabled);
    }

    [Theory]
    [InlineData(new[] { "--sub-agents-max-threads", "8" }, 8, 1)]
    [InlineData(new[] { "--sub-agents-max-threads=8" }, 8, 1)]
    [InlineData(new[] { "--sub-agents-max-threads", "1" }, 1, 1)]
    [InlineData(new[] { "--sub-agents-max-threads", "16" }, 16, 1)]
    [InlineData(new[] { "--sub-agents-max-depth", "3" }, 4, 3)]
    [InlineData(new[] { "--sub-agents-max-depth=4" }, 4, 4)]
    [InlineData(new[] { "--sub-agents-max-threads", " 2 ", "--sub-agents-max-depth", "2" }, 2, 2)]
    public void Parse_ReadsTheLimits(string[] args, int threads, int depth)
    {
        SubAgentOptions options = SubAgentOptions.Parse(args);
        Assert.Equal(threads, options.MaxThreads);
        Assert.Equal(depth, options.MaxDepth);
        Assert.False(options.Enabled); // a limit alone does not turn the feature on
    }

    [Theory]
    [InlineData("--sub-agents-max-threads", "0", "Invalid value for --sub-agents-max-threads: '0'. Expected a whole number between 1 and 16.")]
    [InlineData("--sub-agents-max-threads", "17", "Invalid value for --sub-agents-max-threads: '17'. Expected a whole number between 1 and 16.")]
    [InlineData("--sub-agents-max-threads", "-1", "Invalid value for --sub-agents-max-threads: '-1'. Expected a whole number between 1 and 16.")]
    [InlineData("--sub-agents-max-threads", "four", "Invalid value for --sub-agents-max-threads: 'four'. Expected a whole number between 1 and 16.")]
    [InlineData("--sub-agents-max-threads", "2.5", "Invalid value for --sub-agents-max-threads: '2.5'. Expected a whole number between 1 and 16.")]
    [InlineData("--sub-agents-max-depth", "0", "Invalid value for --sub-agents-max-depth: '0'. Expected a whole number between 1 and 4.")]
    [InlineData("--sub-agents-max-depth", "5", "Invalid value for --sub-agents-max-depth: '5'. Expected a whole number between 1 and 4.")]
    public void Parse_OutOfRangeValues_FailWithTheExactMessage(string flag, string value, string message)
    {
        ArgumentException separate = Assert.Throws<ArgumentException>(() => SubAgentOptions.Parse(new[] { flag, value }));
        Assert.Equal(message, separate.Message);

        ArgumentException joined = Assert.Throws<ArgumentException>(() => SubAgentOptions.Parse(new[] { flag + "=" + value }));
        Assert.Equal(message, joined.Message);
    }

    [Theory]
    [InlineData("--sub-agents-max-threads")]
    [InlineData("--sub-agents-max-depth")]
    public void Parse_AMissingValue_Fails(string flag)
    {
        string message = $"Missing value for option {flag}.";
        Assert.Equal(message, Assert.Throws<ArgumentException>(() => SubAgentOptions.Parse(new[] { flag })).Message);
        Assert.Equal(message, Assert.Throws<ArgumentException>(() => SubAgentOptions.Parse(new[] { flag, "  " })).Message);
        Assert.Equal(message, Assert.Throws<ArgumentException>(() => SubAgentOptions.Parse(new[] { flag + "=" })).Message);
    }

    [Fact]
    public void ApplyEnvironment_TurnsOnAndSetsLimits()
    {
        using var env = new EnvironmentScope(enable: "1", threads: "6", depth: "2");
        SubAgentOptions options = SubAgentOptions.Parse(Array.Empty<string>()).ApplyEnvironment();
        Assert.True(options.Enabled);
        Assert.Equal(6, options.MaxThreads);
        Assert.Equal(2, options.MaxDepth);
    }

    [Theory]
    [InlineData("0", false)]
    [InlineData(" 0 ", false)]
    [InlineData("", false)]
    [InlineData("1", true)]
    [InlineData("true", true)]
    public void ApplyEnvironment_EnableIsAnythingButZero(string value, bool expected)
    {
        using var env = new EnvironmentScope(enable: value);
        Assert.Equal(expected, new SubAgentOptions().ApplyEnvironment().Enabled);
    }

    [Fact]
    public void ApplyEnvironment_TheCommandLineWins()
    {
        using var env = new EnvironmentScope(enable: "0", threads: "6", depth: "3");

        SubAgentOptions options = SubAgentOptions.Parse(new[]
        {
            "--sub-agents", "--sub-agents-max-threads", "3", "--sub-agents-max-depth", "1",
        }).ApplyEnvironment();

        Assert.True(options.Enabled);          // TS_SUB_AGENTS=0 cannot switch off an explicit flag
        Assert.Equal(3, options.MaxThreads);
        Assert.Equal(1, options.MaxDepth);     // explicitly the default, and still the command line's
    }

    [Fact]
    public void ApplyEnvironment_OnlyTheUnsetLimitComesFromTheEnvironment()
    {
        using var env = new EnvironmentScope(threads: "6", depth: "3");
        SubAgentOptions options = SubAgentOptions.Parse(new[] { "--sub-agents-max-threads=2" }).ApplyEnvironment();
        Assert.Equal(2, options.MaxThreads);
        Assert.Equal(3, options.MaxDepth);
    }

    [Fact]
    public void Clone_KeepsWhatTheCommandLineSet()
    {
        using var env = new EnvironmentScope(threads: "6");
        SubAgentOptions parsed = SubAgentOptions.Parse(new[] { "--sub-agents", "--sub-agents-max-threads", "3" });

        SubAgentOptions copy = parsed.Clone();
        Assert.NotSame(parsed, copy);
        Assert.True(copy.Enabled);
        Assert.Equal(3, copy.ApplyEnvironment().MaxThreads);
    }

    [Theory]
    [InlineData(SubAgentOptions.MaxThreadsEnvVar, "lots", "Invalid value for TS_SUB_AGENTS_MAX_THREADS: 'lots'. Expected a whole number between 1 and 16.")]
    [InlineData(SubAgentOptions.MaxThreadsEnvVar, "99", "Invalid value for TS_SUB_AGENTS_MAX_THREADS: '99'. Expected a whole number between 1 and 16.")]
    [InlineData(SubAgentOptions.MaxDepthEnvVar, "0", "Invalid value for TS_SUB_AGENTS_MAX_DEPTH: '0'. Expected a whole number between 1 and 4.")]
    public void ApplyEnvironment_AMalformedLimitIsFatal(string variable, string value, string message)
    {
        using var env = new EnvironmentScope(
            threads: variable == SubAgentOptions.MaxThreadsEnvVar ? value : null,
            depth: variable == SubAgentOptions.MaxDepthEnvVar ? value : null);

        ArgumentException error = Assert.Throws<ArgumentException>(() => new SubAgentOptions().ApplyEnvironment());
        Assert.Equal(message, error.Message);
    }

    [Fact]
    public void ApplyEnvironment_AMalformedLimitTheCommandLineOverridesIsNeverRead()
    {
        using var env = new EnvironmentScope(threads: "lots");
        SubAgentOptions options = SubAgentOptions.Parse(new[] { "--sub-agents-max-threads", "5" }).ApplyEnvironment();
        Assert.Equal(5, options.MaxThreads);
    }
}
